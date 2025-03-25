from torch import nn

from classes.classes import ZeroShotModelConfig
from models.zeroshot.subplan_runtime_helper import compute_subplan_runtimes
from models.zeroshot.utils.fc_out_model import FcOutModel
from models.zeroshot.zero_shot_model import ZeroShotModelEncoder
from training.losses import create_loss


class PostgresZeroShotModel(nn.Module):
    def __init__(self, model_config: ZeroShotModelConfig, feature_statistics: dict, **zero_shot_kwargs):
        super(PostgresZeroShotModel, self).__init__()

        self.label_norm = None
        self.device = model_config.device

        self.encoder = PostgresZeroShotModelEncoder(model_config, feature_statistics,
                                                    return_all_node_states=model_config.subplan_monotonicity_loss,
                                                    **zero_shot_kwargs)
        self.decoder = FcOutModel(output_dim=model_config.output_dim, input_dim=model_config.hidden_dim,
                                  final_out_layer=True, **model_config.final_mlp_kwargs)

        # if model_config.loss_class_name is not None:
        self.loss_fxn = create_loss(model_config.loss_class_name, loss_kwargs=model_config.loss_class_kwargs)

    def forward(self, model_input):
        g, features = model_input

        if self.encoder.return_all_node_states:
            hidden_states, all_node_states = self.encoder((g, features))

            # compute predictions of full plans
            root_predictions = self.decoder.fcout(hidden_states)

            # compute subplan predictions
            subplan_graphs, subplan_predictions_dict = compute_subplan_runtimes(g, all_node_states, self.decoder.fcout)

            return root_predictions, subplan_graphs, subplan_predictions_dict
        else:
            hidden_states = self.encoder((g, features))
            return self.decoder.fcout(hidden_states)


class PostgresZeroShotModelEncoder(ZeroShotModelEncoder):
    """
    Zero-shot cost estimation model encoder for postgres.
    """

    def __init__(self, model_config: ZeroShotModelConfig, feature_statistics: dict, return_all_node_states: bool,
                 **zero_shot_kwargs):
        plan_featurization, encoders = None, None
        if model_config.featurization is not None:
            plan_featurization = model_config.featurization

            if model_config.filter_inline_col_feats:
                # inline column features into filter features
                filter_feats = plan_featurization.FILTER_FEATURES + plan_featurization.COLUMN_FEATURES
            else:
                # do not inline column features into filter features - message passing between column and filter nodes
                filter_feats = plan_featurization.FILTER_FEATURES

            # define the MLPs for the different node types in the graph representation of queries
            encoders = [
                ('column', plan_featurization.COLUMN_FEATURES),
                ('table', plan_featurization.TABLE_FEATURES),
                ('output_column', plan_featurization.OUTPUT_COLUMN_FEATURES),
                ('filter_column', filter_feats),
                ('plan', plan_featurization.PLAN_FEATURES),
                ('logical_pred', plan_featurization.FILTER_FEATURES),
            ]

        # define messages passing which is peculiar for postgres
        prepasses = [dict(model_name='column_output_column', e_name='col_output_col')]
        tree_model_types = ['column_output_column']

        if not model_config.filter_inline_col_feats:
            print(
                f'Filter inline column features is disabled - message passing between column and filter nodes will be performed',
                flush=True)
            # in case feature inlining is deactivated: we are doing message passing between column and filter nodes as a prepassing step
            prepasses.append(dict(model_name='col_filter', e_name='col_filter'))
            tree_model_types.append('col_filter')

        super().__init__(plan_featurization=plan_featurization,
                         encoders=encoders,
                         prepasses=prepasses,
                         add_tree_model_types=tree_model_types,
                         model_config=model_config,
                         feature_statistics=feature_statistics,
                         return_all_node_states=return_all_node_states,
                         run_with_assertions=True,
                         **zero_shot_kwargs)
