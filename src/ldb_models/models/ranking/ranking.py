import torch
import torch.nn as nn
from classes import (
    CascadeRankingModelConfig,
    LatentZeroShotRankingModelConfig,
    ModelConfig,
)
from models.ranking.latent_ranker import (
    AttentionRanker,
    TournamentAttentionRanker,
    UnifiedAttentionRanker,
)
from models.ranking.relevancy_generator import RelevancyGenerator
from training.losses import create_loss
from training.training.outputs import LossOutput, ModelOutput


class RankModel(nn.Module):
    """"""

    def __init__(self, model, model_config: ModelConfig, **kwargs):
        super(RankModel, self).__init__()
        self.model = model
        self.device = model.device
        self.label_norm = model.label_norm
        self.config = model_config

        self.group_size = model_config.group_size
        self.batch_group_size = model_config.batch_group_size
        self.batch_size = model_config.batch_size

        if (
            hasattr(model_config, "use_relevancy_generator")
            and model_config.use_relevancy_generator
        ):
            self.predict_relevancy_scores = model_config.predict_relevancy_scores
            self.use_relevancy_generator = model_config.use_relevancy_generator
            self.relevancy_generator = RelevancyGenerator(
                model_config.relevancy_generator_args, device=self.device
            )
        else:
            self.predict_relevancy_scores = False
            self.use_relevancy_generator = False

        self.cardinality_estimation = (
            getattr(model_config, "cardinality_estimation", False)
            if hasattr(model_config, "cardinality_estimation")
            else False
        )
        self.cardinality_weight = (
            getattr(model_config, "cardinality_weight", 0.0)
            if hasattr(model_config, "cardinality_weight")
            else 0.0
        )

        self.subplan_estimation = (
            getattr(model_config, "subplan_estimation", False)
            if hasattr(model_config, "subplan_estimation")
            else False
        )
        self.subplan_weight = (
            getattr(model_config, "subplan_weight", 0.0)
            if hasattr(model_config, "subplan_weight")
            else 0.0
        )
        self.subplans_reg_loss = (
            getattr(model_config, "subplans_reg_loss", False)
            if hasattr(model_config, "subplans_reg_loss")
            else False
        )

        self.cascade = isinstance(model_config, CascadeRankingModelConfig)
        self.latent = isinstance(model_config, LatentZeroShotRankingModelConfig)
        if self.latent:
            self.latent = True
            self.representation_dim = model.hidden_dim
            latent_model_dict = {
                "tournament": TournamentAttentionRanker,
                "unified": UnifiedAttentionRanker,
                "grouped": AttentionRanker,
            }
            # Instantiate our new attention ranker
            self.rank_model = latent_model_dict[model_config.latent_strategy](
                representation_dim=self.representation_dim,
                num_heads=model_config.attention_heads,
                num_layers=model_config.attention_layers,
                use_layernorm=getattr(model_config, "use_layernorm", True),
                group_size=model_config.group_size
                if hasattr(model_config, "group_size")
                else None,
                device=self.device,
            )
            if self.cascade:
                self.filter_model = kwargs.get("filter_model", None)
                self.use_pretrained_embeddings = (
                    model_config.use_pretrained_embeddings
                    if hasattr(model_config, "use_pretrained_embeddings")
                    else False
                )
                print(
                    f"Cascade Ranking Model initialized. Using pretrained embeddings: {self.use_pretrained_embeddings}"
                )
                assert self.filter_model is not None, (
                    "Filter model must be provided for cascade ranking models."
                )

        loss_kwargs = model_config.loss_class_kwargs
        loss_kwargs["device"] = self.device
        self.loss_fxn = create_loss(
            model_config.loss_class_name, loss_kwargs=loss_kwargs
        )

    def forward(self, x, plan_dict=None, debug=False, **kwargs) -> ModelOutput:
        # Execute forward pass based on model type
        # CASCADE RANKING MODEL
        if self.cascade:
            model_output: ModelOutput = self.cascade_forward(
                x, debug=debug, plan_dict=plan_dict, **kwargs
            )
        # LATENT RANKING MODEL
        elif self.latent:
            latent_reps = self.model.get_representation(
                x, subplan=False, plan_dict=plan_dict
            )
            model_output: ModelOutput = self.rank_model(
                latent_representations=latent_reps,
                tournament_selection=not self.training,
                debug=debug,
            )
        # STANDARD RANKING MODEL
        else:
            model_output: ModelOutput = self.model(
                x,
                plan_dict=plan_dict,
                debug=debug,
                subplan_labels=kwargs.get("subplan_labels", None),
                card_labels=kwargs.get("card_labels", None),
            )
        # Revert predictions if predicting relevancy scores
        if self.predict_relevancy_scores and not self.training:
            model_output.plan_predictions = self.relevancy_generator.revert(
                model_output.plan_predictions
            )
        return model_output

    def get_representation(self, x, subplan=False, **kwargs) -> torch.Tensor:
        if self.cascade and self.use_pretrained_embeddings:
            with torch.no_grad():
                output = self.filter_model.get_representation(x, subplan=subplan)
            if isinstance(output, tuple):
                hidden_states, all_node_states, g = output
                if subplan:
                    return hidden_states.detach(), all_node_states.detach(), g.detach()
                else:
                    return hidden_states.detach()
        else:
            return self.model.get_representation(x, subplan=subplan)

    def group_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        n = tensor.shape[0] // self.group_size  # Number of complete groups
        if n == 0:
            # form a single group with the remaining elements
            return (
                tensor.view(1, -1)
                if tensor.numel() > 0
                else torch.empty(0, self.group_size, device=tensor.device)
            )
        return tensor[: n * self.group_size].view(n, self.group_size)

    def compute_loss(
        self, output: ModelOutput, labels: torch.Tensor, query_stats: dict
    ) -> LossOutput:
        loss = plan_loss = subplan_loss = card_loss = None
        # Form groups for ranking loss
        grouped_output = self.group_tensor(output.plan_predictions)
        grouped_labels = self.group_tensor(labels)
        # Compute relevancy scores if required
        relevancy_scores_output, relevancy_scores_labels = None, None
        if self.use_relevancy_generator:
            relevancy_scores_labels = self.relevancy_generator(grouped_labels)
            if self.predict_relevancy_scores:
                relevancy_scores_output = grouped_output
                grouped_output = self.relevancy_generator.revert(grouped_output)
            else:
                relevancy_scores_output = self.relevancy_generator(grouped_output)
        # Calculate main ranking loss
        # It uses relevancy scores for ranking and runtimes for regression
        loss = plan_loss = self.loss_fxn(
            grouped_output,
            grouped_labels,
            relevancy_scores_output,
            relevancy_scores_labels,
        )
        # Calculate and add subplan operation time loss
        if self.subplan_estimation and output.subplan_predictions is not None:
            grouped_subplan_output = self.group_tensor(output.subplan_predictions)
            grouped_subplan_labels = self.group_tensor(query_stats["subplan_labels"])
            subplan_loss = self.loss_fxn(
                grouped_subplan_output,
                grouped_subplan_labels,
                subplan=True,
                only_regression=self.subplans_reg_loss,
            )
            loss += self.subplan_weight * subplan_loss
        # Calculate and add cardinality estimation loss
        if self.cardinality_estimation and output.subplan_card_predictions is not None:
            grouped_card_output = self.group_tensor(output.subplan_card_predictions)
            grouped_card_labels = self.group_tensor(query_stats["subplan_card_labels"])
            card_loss = self.loss_fxn(
                grouped_card_output,
                grouped_card_labels,
                subplan=True,
                only_regression=self.subplans_reg_loss,
            )
            loss += self.cardinality_weight * card_loss
        return LossOutput(
            loss=loss,
            plan_loss=plan_loss,
            subplan_loss=subplan_loss,
            card_loss=card_loss,
            gouped_output=grouped_output,
            grouped_labels=grouped_labels,
        )

    def cascade_forward(self, x, debug=False, plan_dict=None, **kwargs) -> ModelOutput:
        """
        Executes the forward pass. If in eval mode, applies cascade logic.
        Otherwise, passes the call directly to the main rank_model.
        """
        # Cascade logic is only used during evaluation/testing
        if self.training:
            representations = self.get_representation(x, plan_dict=plan_dict, **kwargs)
            return self.rank_model(
                latent_representations=representations, debug=debug, **kwargs
            )
        else:
            with torch.no_grad():
                plans: torch.Tensor = (
                    self.filter_model(x, plan_dict=plan_dict)
                    .plan_predictions.detach()
                    .squeeze()
                )
            # Filter to get the top `group_size` candidates (lowest runtimes)
            top_k = min(self.group_size, plans.shape[0])
            _, top_k_indices = torch.topk(plans, k=top_k, largest=False)
            # Get the representations for the top-k candidates
            representations = self.get_representation(x, **kwargs)
            top_k_representations = representations[top_k_indices]
            # Pass the top-k representations through the rank model
            y_output: ModelOutput = self.rank_model(top_k_representations, **kwargs)
            y = y_output.plan_predictions.detach()
            # Rescale 'y' to get final runtime predictions 'r_hat' for the top-k candidates
            sigma_y = torch.softmax(y, dim=0)
            candidates = plans[top_k_indices]
            C_min, C_max = torch.min(candidates), torch.max(candidates)
            r_hat = sigma_y * (C_max - C_min) + C_min
            # Construct the final predictions tensor for all original candidates

            if debug:
                self._cascade_debug_print(
                    top_k, top_k_indices, plans, y, r_hat, **kwargs
                )

            plans[top_k_indices] = r_hat

            return ModelOutput(plan_predictions=plans)

    def _cascade_debug_print(self, top_k, top_k_indices, plans, y, r_hat, **kwargs):
        print(f"Cascade Ranking - Filtered Top-{top_k}: ")
        rows = []
        ordered = r_hat.cpu().argsort().tolist()

        for pos, idx in enumerate(top_k_indices):
            filter_pred = float(plans[idx])
            filter_rank = int(torch.argsort(plans).cpu().argsort()[idx])
            rank_model_pred = float(y[pos])
            rescaled_pred = float(r_hat[pos])
            predicted_rank_in_batch = ordered.index(pos)

            entry = {
                "rescaled": rescaled_pred,
                "filter_pred": filter_pred,
                "filter_rank": filter_rank,
                "final": rank_model_pred,
                "pred_rank": predicted_rank_in_batch,
            }

            plan_labels = kwargs.get("plan_labels", None)
            if plan_labels is not None:
                true_label = float(plan_labels[idx])
                true_rank = int(torch.argsort(plan_labels).cpu().argsort()[idx])
                entry["true_label"] = true_label
                entry["true_rank"] = true_rank

            rows.append(entry)

        for r in sorted(rows, key=lambda x: x["rescaled"]):
            line = (
                f"  rescaled={r['rescaled']:.3f}\t| "
                f"filter={r['filter_pred']:.3f} | "
                f"rank={r['filter_rank']}\t| "
                f"final={r['final']:.3} | "
                f"rank={r['pred_rank']}\t"
            )
            if "true_label" in r:
                line += f"| true={r['true_label']:.3f} | rank={r['true_rank']}"
            print(line)
