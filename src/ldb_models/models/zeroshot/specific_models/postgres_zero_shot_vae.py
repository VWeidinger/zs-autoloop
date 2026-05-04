import torch
from ldb_models.classes import ZeroShotVAEModelConfig
from ldb_models.models.zeroshot.subplan_runtime_helper import compute_subplan_runtimes
from ldb_models.models.zeroshot.utils.fc_out_model import FcOutModel
from ldb_models.models.zeroshot.zero_shot_model import ZeroShotModelEncoder
from torch import nn
from ldb_models.training.losses.factory import create_loss
from ldb_models.training.training.outputs import LossOutput, ModelOutput


class PostgresZeroShotModelVAE(nn.Module):
    def __init__(
        self,
        model_config: ZeroShotVAEModelConfig,
        feature_statistics: dict,
        **zero_shot_kwargs,
    ):
        super(PostgresZeroShotModelVAE, self).__init__()

        self.label_norm = None
        self.device = model_config.device

        self.encoder = PostgresZeroShotModelEncoder(
            model_config,
            feature_statistics,
            return_all_node_states=model_config.subplan_monotonicity_loss,
            **zero_shot_kwargs,
        )
        self.decoder = FcOutModel(
            output_dim=model_config.output_dim,
            input_dim=model_config.vae_dim,
            final_out_layer=True,
            **model_config.final_mlp_kwargs,
        )

        self.mu_proj = nn.Linear(model_config.hidden_dim, model_config.vae_dim)
        self.log_var_proj = nn.Linear(model_config.hidden_dim, model_config.vae_dim)
        self.beta = model_config.beta
        self.num_samples = model_config.num_samples

        # if model_config.loss_class_name is not None:
        self.loss_fcn = create_loss(
            model_config.loss_class_name, loss_kwargs=model_config.loss_class_kwargs
        )
        self.loss_fxn = self.compute_loss

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, -10, 10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, model_input, **kwargs):
        g, features = model_input

        if self.encoder.return_all_node_states:
            hidden_states, all_node_states = self.encoder((g, features))

            self.mu = self.mu_proj(hidden_states)
            self.logvar = self.log_var_proj(hidden_states)
            z = self.reparameterize(self.mu, self.logvar)

            # compute predictions of full plans
            root_predictions = self.decoder.fcout(z)

            # compute subplan predictions
            subplan_graphs, subplan_predictions_tensor = compute_subplan_runtimes(
                g, all_node_states, self.decoder.fcout
            )

            return ModelOutput(
                plan_predictions=root_predictions,
                subplan_graphs=subplan_graphs,
                subplan_predictions=subplan_predictions_tensor,
            )
        else:
            hidden_states = self.encoder((g, features))

            self.mu = self.mu_proj(hidden_states)
            self.logvar = self.log_var_proj(hidden_states)
            z = self.reparameterize(self.mu, self.logvar)

            return ModelOutput(plan_predictions=self.decoder.fcout(z))

    def predict(self, model_input, num_samples: int = 0):
        if num_samples == 0:
            num_samples = self.num_samples

        g, features = model_input
        hidden_states = self.encoder((g, features))
        mu = self.mu_proj(hidden_states)
        logvar = self.log_var_proj(hidden_states)

        mu_expanded = mu.unsqueeze(0).expand(num_samples, -1, -1)
        logvar_expanded = logvar.unsqueeze(0).expand(num_samples, -1, -1)

        eps = torch.randn_like(mu_expanded)
        z = mu_expanded + eps * torch.exp(0.5 * logvar_expanded)

        batch_size = z.size(1)
        z_reshaped = z.reshape(num_samples * batch_size, -1)

        all_preds = self.decoder.fcout(z_reshaped)

        predictions = all_preds.view(num_samples, batch_size, -1)
        predictions = torch.mean(predictions, dim=0)

        return ModelOutput(plan_predictions=predictions)

    def compute_loss(
        self, output: ModelOutput, query_stats: dict, labels: torch.Tensor, **kwargs
    ) -> LossOutput:
        recon_loss = self.loss_fcn(output.plan_predictions, labels)
        kl_loss = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        loss = recon_loss + self.beta * kl_loss
        return LossOutput(loss=loss, plan_loss=recon_loss)


class PostgresZeroShotModelEncoder(ZeroShotModelEncoder):
    """
    Zero-shot cost estimation model encoder for postgres.
    """

    def __init__(
        self,
        model_config: ZeroShotVAEModelConfig,
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
