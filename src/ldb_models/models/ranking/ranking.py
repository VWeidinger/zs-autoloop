import torch
import torch.nn as nn
from ldb_models.classes import (
    CascadeRankingModelConfig,
    LatentZeroShotRankingModelConfig,
    ModelConfig,
)
from ldb_models.models.ranking.latent_ranker import (
    AttentionRanker,
    TournamentAttentionRanker,
    UnifiedAttentionRanker,
)
from ldb_models.models.zeroshot.relevancy_generator import RelevancyGenerator
from ldb_models.training.losses.factory import create_loss
from ldb_models.training.training.outputs import LossOutput, ModelOutput


class RankModel(nn.Module):
    """
    Wrapper class for ranking models.
    """

    def __init__(
        self,
        model,
        model_config: ModelConfig,
        **kwargs,
    ):
        super(RankModel, self).__init__()
        self.model = model
        self.device = model.device
        self.label_norm = model.label_norm
        self.config = model_config

        self.group_size = model_config.group_size
        self.batch_group_size = model_config.batch_group_size
        self.batch_size = model_config.batch_size

        self.use_relevancy_generator = (
            hasattr(model_config, "use_relevancy_generator")
            and model_config.use_relevancy_generator
        )
        if self.use_relevancy_generator:
            # add group size and single plan mask to model config for use in relevancy generator
            relevancy_generator_train_args = (
                model_config.relevancy_generator_args.copy()
            )
            relevancy_generator_train_args["group_size"] = self.group_size
            self.relevancy_generator_train = RelevancyGenerator(
                relevancy_generator_train_args,
                device=self.device,
                single_plan_mask=kwargs.get("single_plan_mask", None),
                query_min_tensor=kwargs.get("query_min_tensor", None),
                query_max_tensor=kwargs.get("query_max_tensor", None),
            ).to(self.device)
            relevancy_generator_eval_args = relevancy_generator_train_args.copy()
            relevancy_generator_eval_args["rl_scope"] = "global"
            relevancy_generator_eval_args["mask_single_plan_queries"] = False
            self.relevancy_generator_eval = RelevancyGenerator(
                relevancy_generator_eval_args, device=self.device
            ).to(self.device)
        else:
            self.relevancy_generator_train = None
            self.relevancy_generator_eval = None

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

        self.log_transform_card_labels = (
            getattr(model_config, "log_transform_card_labels", True)
            if hasattr(model_config, "log_transform_card_labels")
            else True
        )
        self.dont_log_if_log_transformed = (
            getattr(model_config, "dont_log_if_log_transformed", True)
            if hasattr(model_config, "dont_log_if_log_transformed")
            else True
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

    def forward(self, x, debug=False, **kwargs) -> ModelOutput:
        # Execute forward pass based on model type
        # CASCADE RANKING MODEL
        if self.cascade:
            model_output: ModelOutput = self.cascade_forward(x, debug=debug, **kwargs)
        # LATENT RANKING MODEL
        elif self.latent:
            latent_reps = self.model.get_representation(x, subplan=False, **kwargs)
            model_output: ModelOutput = self.rank_model(
                latent_representations=latent_reps,
                tournament_selection=not self.training,
                debug=debug,
            )
        # STANDARD RANKING MODEL
        else:
            model_output: ModelOutput = self.model(
                x,
                debug=debug,
                **kwargs,
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
        self,
        output: ModelOutput,
        labels: torch.Tensor,
        query_stats: dict,
        debug=False,
    ) -> LossOutput:
        loss = plan_loss = subplan_loss = card_loss = None
        loss_breakdown = {}

        # Compute relevancy scores for both labels and predictions if using relevancy generator
        relevancy_scores = None
        relevancy_scores_output = None
        if self.use_relevancy_generator:
            # Compute relevancy for labels (targets)
            if self.training:
                relevancy_scores = self.relevancy_generator_train(
                    labels,
                    query_stats=query_stats,
                )
                # Compute relevancy for predictions
                relevancy_scores_output = self.relevancy_generator_train(
                    output.plan_predictions, query_stats=query_stats
                )
            else:
                relevancy_scores = self.relevancy_generator_eval(
                    labels, query_stats=query_stats
                )
                relevancy_scores_output = self.relevancy_generator_eval(
                    output.plan_predictions, query_stats=query_stats
                )

        # Calculate main ranking loss
        # It uses relevancy scores for ranking and runtimes for regression
        plan_loss = self.loss_fxn(
            output.plan_predictions,
            labels,
            relevancy_scores_output,
            relevancy_scores,
            group_function=self.group_tensor,
            debug=debug,
        )
        loss = plan_loss.clone()

        if hasattr(self.loss_fxn, "last_breakdown"):
            for k, v in self.loss_fxn.last_breakdown.items():
                loss_breakdown[f"plan_{k}"] = v
        loss_breakdown["plan_loss"] = plan_loss.item()

        # Calculate and add subplan operation time loss
        if self.subplan_estimation and output.subplan_predictions is not None:
            raw_subplan_loss = self.loss_fxn(
                output.subplan_predictions,
                query_stats["subplan_labels"],
                subplan=True,
                only_regression_on_subplans=self.subplans_reg_loss,
                group_function=self.group_tensor,
            )
            subplan_loss = raw_subplan_loss * self.subplan_weight
            loss += subplan_loss

            loss_breakdown["subplan_loss_unweighted"] = raw_subplan_loss.item()
            loss_breakdown["subplan_loss"] = subplan_loss.item()
            if hasattr(self.loss_fxn, "last_breakdown"):
                for k, v in self.loss_fxn.last_breakdown.items():
                    loss_breakdown[f"subplan_{k}"] = v

        # Calculate and add cardinality estimation loss
        if self.cardinality_estimation and output.subplan_card_predictions is not None:
            raw_card_loss = self.loss_fxn(
                output.subplan_card_predictions,
                query_stats["subplan_card_labels"],
                subplan=True,
                only_regression_on_subplans=self.subplans_reg_loss,
                inputs_are_already_log=self.log_transform_card_labels
                and self.dont_log_if_log_transformed,
                group_function=self.group_tensor,
            )
            card_loss = raw_card_loss * self.cardinality_weight
            loss += card_loss

            loss_breakdown["card_loss_unweighted"] = raw_card_loss.item()
            loss_breakdown["card_loss"] = card_loss.item()
            if hasattr(self.loss_fxn, "last_breakdown"):
                for k, v in self.loss_fxn.last_breakdown.items():
                    loss_breakdown[f"card_{k}"] = v

        return LossOutput(
            loss=loss,
            plan_loss=plan_loss,
            subplan_loss=subplan_loss,
            card_loss=card_loss,
            relevancy_scores_preds=(
                relevancy_scores_output.detach().cpu()
                if relevancy_scores_output is not None
                else None
            ),
            relevancy_scores_labels=(
                relevancy_scores.detach().cpu()
                if relevancy_scores is not None
                else None
            ),
            loss_breakdown=loss_breakdown,
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
