from classes.classes import ZeroShotModelConfig
from models.zeroshot.utils.fc_out_model import FcOutModel
from models.zeroshot.zero_shot_model import ZeroShotModelEncoder

class PostgresZeroShotModel:
    def __init__(self, model_config: ZeroShotModelConfig, feature_statistics: dict, **zero_shot_kwargs):
        self.encoder = PostgresZeroShotModelEncoder(model_config, feature_statistics, **zero_shot_kwargs)
        self.decoder = FcOutModel(output_dim=model_config.output_dim, input_dim=model_config.hidden_dim, **model_config.final_mlp_kwargs)

    def forward(self, g, features):
        hidden_states = self.encoder(g, features)
        return self.decoder(hidden_states)

class PostgresZeroShotModelEncoder(ZeroShotModelEncoder):
    """
    Zero-shot cost estimation model encoder for postgres.
    """
    def __init__(self, model_config: ZeroShotModelConfig, feature_statistics: dict, **zero_shot_kwargs):
        plan_featurization, encoders = None, None
        if model_config.featurization is not None:
            plan_featurization = model_config.featurization

            # define the MLPs for the different node types in the graph representation of queries
            encoders = [
                ('column', plan_featurization.COLUMN_FEATURES),
                ('table', plan_featurization.TABLE_FEATURES),
                ('output_column', plan_featurization.OUTPUT_COLUMN_FEATURES),
                ('filter_column', plan_featurization.FILTER_FEATURES + plan_featurization.COLUMN_FEATURES),
                ('plan', plan_featurization.PLAN_FEATURES),
                ('logical_pred', plan_featurization.FILTER_FEATURES),
            ]

        # define messages passing which is peculiar for postgres
        prepasses = [dict(model_name='column_output_column', e_name='col_output_col')]
        tree_model_types = ['column_output_column']

        super().__init__(plan_featurization=plan_featurization,
                         encoders=encoders,
                         prepasses=prepasses,
                         add_tree_model_types=tree_model_types,
                         model_config=model_config,
                         feature_statistics=feature_statistics, **zero_shot_kwargs)
