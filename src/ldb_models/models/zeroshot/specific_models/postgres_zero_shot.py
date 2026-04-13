import torch
from classes import ZeroShotModelConfig
from models.zeroshot.subplan_runtime_helper import (
    compute_subplan_runtimes,
    extract_smaller_larger_preds_tensor_for_subplan_graphs,
)
from models.zeroshot.utils.fc_out_model import FcOutModel
from models.zeroshot.zero_shot_model import ZeroShotModelEncoder
from torch import nn
from torch.nn import functional as F
from training.losses.factory import create_loss
from training.losses.monotonicity import LessThanLoss, RelativeLessThanLoss
from training.training.outputs import LossOutput, ModelOutput


class PostgresZeroShotModel(nn.Module):
    def __init__(
        self,
        model_config: ZeroShotModelConfig,
        feature_statistics: dict,
        **zero_shot_kwargs,
    ):
        super(PostgresZeroShotModel, self).__init__()

        self.label_norm = None
        self.device = model_config.device
        self.hidden_dim = model_config.hidden_dim

        return_all_node_states = (
            model_config.subplan_monotonicity_loss or model_config.subplan_estimation
        )

        print(
            f"Subplan monotonicity loss: {model_config.subplan_monotonicity_loss} Subplan loss: {model_config.subplan_estimation} Return all node states: {return_all_node_states}",
            flush=True,
        )

        self.encoder = PostgresZeroShotModelEncoder(
            model_config,
            feature_statistics,
            return_all_node_states=return_all_node_states,
            **zero_shot_kwargs,
        )
        self.decoder = FcOutModel(
            output_dim=model_config.output_dim,
            input_dim=model_config.hidden_dim,
            final_out_layer=True,
            **model_config.final_mlp_kwargs,
        )

        # if model_config.loss_class_name is not None:
        self.loss_fxn = create_loss(
            model_config.loss_class_name, loss_kwargs=model_config.loss_class_kwargs
        )

        if (
            hasattr(model_config, "subplan_monotonicity_loss")
            and model_config.subplan_monotonicity_loss
        ):
            self.perform_subplan_monotonicity_loss = True

            # select loss function
            assert hasattr(model_config, "subplan_monotonicity_loss_fn")
            if model_config.subplan_monotonicity_loss_fn == "lt_loss":
                self.subplan_monotonicity_loss = LessThanLoss()
            elif model_config.subplan_monotonicity_loss_fn == "relative_lt_loss":
                self.subplan_monotonicity_loss = RelativeLessThanLoss()
            else:
                raise ValueError(
                    f"Unknown subplan monotonicity loss function: {model_config.subplan_monotonicity_loss_fn}"
                )
        else:
            self.perform_subplan_monotonicity_loss = False
            self.subplan_monotonicity_loss = None

    def forward(self, model_input, plan_dict=None, **kwargs) -> ModelOutput:
        if self.encoder.return_all_node_states and (plan_dict is not None):
            hidden_states, all_node_states, g = self.get_representation(model_input)

            # compute subplan predictions
            subplan_graphs, subplan_predictions_tensor = compute_subplan_runtimes(
                g, all_node_states, self.decoder.fcout, plan_dict
            )

            plan_predictions = F.softplus(self.decoder.fcout(hidden_states))

            subplan_predictions_tensor = F.softplus(subplan_predictions_tensor)

            return ModelOutput(
                plan_predictions=plan_predictions,
                subplan_graphs=subplan_graphs,
                subplan_predictions=subplan_predictions_tensor,
            )
        else:
            hidden_states = self.get_representation(model_input, subplan=False)

            plan_predictions = F.softplus(self.decoder.fcout(hidden_states))
            # compute predictions
            return ModelOutput(plan_predictions=plan_predictions)

    def get_representation(self, model_input, subplan=True, **kwargs):
        g, features = model_input

        encoder_output = self.encoder((g, features))

        if self.encoder.return_all_node_states:
            hidden_states, all_node_states = encoder_output
            if subplan:
                return hidden_states, all_node_states, g
            else:
                return hidden_states
        else:
            hidden_states = encoder_output
            return hidden_states

    def compute_loss(
        self, output: ModelOutput, query_stats: dict, labels: torch.Tensor, **kwargs
    ) -> LossOutput:
        # Loss computation
        plan_predictions = output.plan_predictions
        assert len(plan_predictions) == len(labels), (
            f"Output length {len(plan_predictions)} does not match label length {len(labels)}"
        )
        loss = plan_loss = self.loss_fxn(plan_predictions, labels)
        # co
        monotonicity_loss = None
        if self.perform_subplan_monotonicity_loss:
            assert hasattr(output, "subplan_graphs")
            assert hasattr(output, "subplan_predictions")
            # extract smaller and larger tensors - input for our monotonicity loss
            smaller_tensor, larger_tensor = (
                extract_smaller_larger_preds_tensor_for_subplan_graphs(
                    output.subplan_graphs, output.subplan_predictions
                )
            )

            # Loss computation
            monotonicity_loss = self.subplan_monotonicity_loss(
                smaller_tensor, larger_tensor
            )
            loss = loss + monotonicity_loss
        return LossOutput(
            loss=loss,
            plan_loss=plan_loss,
            monotonicity_loss=monotonicity_loss,
        )


class PostgresZeroShotModelEncoder(ZeroShotModelEncoder):
    """
    Zero-shot cost estimation model encoder for postgres.
    """

    def __init__(
        self,
        model_config: ZeroShotModelConfig,
        feature_statistics: dict,
        return_all_node_states: bool,
        **zero_shot_kwargs,
    ):
        plan_featurization, encoders = None, None
        if model_config.featurization is not None:
            plan_featurization = model_config.featurization

            if model_config.filter_inline_col_feats:
                # inline column features into filter features
                filter_feats = (
                    plan_featurization.FILTER_FEATURES
                    + plan_featurization.COLUMN_FEATURES
                )
            else:
                # do not inline column features into filter features - message passing between column and filter nodes
                filter_feats = plan_featurization.FILTER_FEATURES

            # define the MLPs for the different node types in the graph representation of queries
            encoders = [
                ("column", plan_featurization.COLUMN_FEATURES),
                ("table", plan_featurization.TABLE_FEATURES),
                ("output_column", plan_featurization.OUTPUT_COLUMN_FEATURES),
                ("filter_column", filter_feats),
                ("plan", plan_featurization.PLAN_FEATURES),
                ("logical_pred", plan_featurization.FILTER_FEATURES),
            ]

        # define messages passing which is peculiar for postgres
        prepasses = [dict(model_name="column_output_column", e_name="col_output_col")]
        tree_model_types = ["column_output_column"]

        if not model_config.filter_inline_col_feats:
            print(
                "Filter inline column features is disabled - message passing between column and filter nodes will be performed",
                flush=True,
            )
            # in case feature inlining is deactivated: we are doing message passing between column and filter nodes as a prepassing step
            prepasses.append(dict(model_name="col_filter", e_name="col_filter"))
            tree_model_types.append("col_filter")

        super().__init__(
            plan_featurization=plan_featurization,
            encoders=encoders,
            prepasses=prepasses,
            add_tree_model_types=tree_model_types,
            model_config=model_config,
            feature_statistics=feature_statistics,
            return_all_node_states=return_all_node_states,
            run_with_assertions=True,
            **zero_shot_kwargs,
        )
