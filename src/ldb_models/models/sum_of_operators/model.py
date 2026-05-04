from typing import Dict

import dgl
import torch.nn as nn
from ldb_models.classes import SumOfOperatorsModelConfig
from dgl import DGLHeteroGraph
from ldb_models.models.zeroshot.specific_models.postgres_zero_shot import (
    PostgresZeroShotModelEncoder,
)
from ldb_models.models.zeroshot.utils.fc_out_model import FcOutModel
from torch import Tensor
from ldb_models.training.losses.factory import create_loss
from ldb_models.training.training.outputs import ModelOutput

from ldb_models.models.sum_of_operators.aggregation import aggregate_op_times
from ldb_models.models.sum_of_operators.debug import print_debug_info
from ldb_models.models.sum_of_operators.prediction import (
    apply_loop_multipliers,
    augment_features_with_cardinality,
    predict_per_node,
)


class SumOfOperatorsModel(nn.Module):
    """
    Zero-shot plan runtime model built from per-operator predictions.

    The model uses the PostgreSQL zero-shot encoder to produce node-level hidden
    states, then applies one or two MLP decoders to predict operator runtime and
    optional cardinality. The final plan prediction is the sum of all operator
    runtimes in plan order.

    Config flags that affect behavior:
    - ``use_card_feature``: append predicted cardinality to the runtime decoder input.
    - ``cardinality_estimation``: enable the auxiliary cardinality head.
    - ``log_transform_card_labels``: interpret cardinality tensors in log1p space.

    SUMO-LOOP modes for loop-aware operator runtime prediction:
    - Base ``sumo_loop=True``: Scale predictions by sibling cardinality,
      use labels divided by actual sibling cardinality (adjusted in collator).
    - ``sumo_loop_multiply_est_error=True``: Scale by predicted/estimated cardinality
      ratio, stabilizes training by not applying loss using the low per-loop runtimes
      from optimizer.
    - ``sumo_loop_loss_on_scaled=True``: Scale predictions by sibling predicted
      cardinality and supervise scaled predictions with original labels.
    """

    def __init__(
        self,
        model_config: SumOfOperatorsModelConfig,
        feature_statistics: Dict,
        **zero_shot_kwargs: object,
    ) -> None:
        """
        Build the encoders, decoders, and loss function.

        Args:
            model_config: Model configuration object. The relevant attributes are
                ``device``, ``hidden_dim``, ``final_mlp_kwargs``,
                ``loss_class_name``, ``loss_class_kwargs``, and the SUMO feature
                flags listed in the class docstring.
            feature_statistics: Feature statistics dictionary consumed by the
                PostgreSQL encoder.
            **zero_shot_kwargs: Extra keyword arguments forwarded to the encoder.

        Returns:
            None. The module stores all state on ``self``.
        """
        super().__init__()
        self.config = model_config
        self.label_norm = None
        self.device = model_config.device

        self.encoder = PostgresZeroShotModelEncoder(
            model_config,
            feature_statistics,
            return_all_node_states=True,
            **zero_shot_kwargs,
        )

        self.decoders = nn.ModuleDict()

        self.use_card_feature = getattr(self.config, "use_card_feature", False)
        self.cardinality_estimation = getattr(
            self.config, "cardinality_estimation", False
        )
        self.log_transform_card_labels = getattr(
            self.config, "log_transform_card_labels", True
        )

        # SUMO-LOOP: Loop-aware operator runtime prediction for join operators.
        # Three modes based on supervision strategy:
        # 1. Base (no flags): Loss on labels divided by actual sibling cardinality.
        # 2. multiply_est_error: Loss on original labels, scaled by pred/est ratio.
        # 3. loss_on_scaled: Loss on predictions scaled by sibling predicted cardinality.
        self.sumo_loop = getattr(self.config, "sumo_loop", False)
        if self.sumo_loop:
            self.multiply_est_error = getattr(
                self.config, "sumo_loop_multiply_est_error", False
            )
            self.loss_on_scaled = getattr(
                self.config, "sumo_loop_loss_on_scaled", False
            )
            # Use scaled predictions for loss only in modes 2 and 3.
            self.use_scaled_subplan_predictions = (
                self.multiply_est_error or self.loss_on_scaled
            )
        else:
            self.multiply_est_error = False
            self.loss_on_scaled = False
            self.use_scaled_subplan_predictions = False

        op_time_input_dim = model_config.hidden_dim
        if self.use_card_feature:
            # Cardinality is appended as a single scalar feature.
            op_time_input_dim += 1

        self.decoders["op_time"] = FcOutModel(
            output_dim=1,
            input_dim=op_time_input_dim,
            final_out_layer=True,
            **model_config.final_mlp_kwargs,
        )

        if self.cardinality_estimation:
            self.decoders["card"] = FcOutModel(
                output_dim=1,
                input_dim=model_config.hidden_dim,
                final_out_layer=True,
                **model_config.final_mlp_kwargs,
            )

        self.loss_fxn = create_loss(
            model_config.loss_class_name, loss_kwargs=model_config.loss_class_kwargs
        )

    def get_representation(
        self,
        model_input: tuple[dgl.DGLGraph | DGLHeteroGraph, dict],
        subplan: bool = True,
    ) -> Tensor | tuple[Tensor, dict, dgl.DGLGraph | DGLHeteroGraph]:
        """
        Encode a graph batch and optionally return node-level states.

        Args:
            model_input: Tuple of ``(graph, features)``.
            subplan: When ``True``, return both the plan-level representation and
                the per-node states. When ``False``, return only the plan-level
                representation.

        Returns:
            If ``subplan`` is ``True``, returns ``(hidden_states, all_node_states, g)``.
            Otherwise returns only ``hidden_states``.
        """
        g, features = model_input
        encoder_output = self.encoder((g, features))
        hidden_states, all_node_states = encoder_output
        if subplan:
            return hidden_states, all_node_states, g
        return hidden_states

    def forward(
        self,
        model_input: tuple[dgl.DGLGraph | DGLHeteroGraph, dict],
        debug: bool = False,
        plan_dict: dict | None = None,
        subplan_labels: Tensor | None = None,
        subplan_card_labels: Tensor | None = None,
        subplan_est_card_labels: Tensor | None = None,
        loop_multiplier_map: tuple[list[int], list[int]] | None = None,
        **kwargs: object,
    ) -> ModelOutput:
        """
        Predict plan runtime and optional operator-level auxiliary outputs.

        Args:
            model_input: Tuple of ``(graph, features)``.
            debug: When ``True``, print a per-operator breakdown for the first plan.
            plan_dict: Mapping or container used to determine the number of plan
                nodes in the batch.
            subplan_labels: Optional operator runtime labels used for debug output.
            subplan_card_labels: Optional cardinality labels used for debug output.
            subplan_est_card_labels: Optional estimated cardinality labels used by
                SUMO-LOOP scaling when ``sumo_loop_multiply_est_error`` is enabled.
            loop_multiplier_map: Optional ``(node_ids, sibling_ids)`` pair that
                identifies loop operators and the sibling cardinality source.
            **kwargs: Reserved for compatibility with the wider training pipeline.

        Returns:
            ``ModelOutput`` with plan predictions, subplan predictions, optional
            cardinality predictions, and subplan lengths.
        """
        g, features = model_input
        _, all_node_states = self.encoder((g, features))
        num_total_plan_nodes = len(plan_dict)

        # Predict cardinality first when the runtime head consumes it as a feature.
        subplan_card_predictions = None
        if self.cardinality_estimation:
            subplan_card_predictions = predict_per_node(
                self.decoders["card"],
                all_node_states,
                g,
                num_total_plan_nodes,
                self.device,
            )

        op_time_input_states = all_node_states
        if self.use_card_feature:
            if subplan_card_predictions is None:
                raise ValueError(
                    "`use_card_feature` is True, but cardinality was not predicted."
                )
            # The runtime head sees the detached prediction as an extra feature.
            op_time_input_states = augment_features_with_cardinality(
                all_node_states, subplan_card_predictions, g
            )

        raw_op_time_preds = predict_per_node(
            self.decoders["op_time"],
            op_time_input_states,
            g,
            num_total_plan_nodes,
            self.device,
        )

        # Scale loop operators after the single-iteration runtime prediction.
        if self.sumo_loop:
            final_op_time_preds, subplan_card_predictions = apply_loop_multipliers(
                raw_op_time_preds,
                subplan_card_predictions,
                subplan_est_card_labels,
                loop_multiplier_map,
                self.device,
                self.log_transform_card_labels,
                self.multiply_est_error,
            )
        else:
            final_op_time_preds = raw_op_time_preds

        # Choose which predictions to use for loss based on SUMO-LOOP mode:
        # - Base mode or multiply_est_error: use raw_op_time_preds (labels adjusted in collator)
        # - loss_on_scaled: use final_op_time_preds (scaled, original labels)
        subplan_preds_for_loss = (
            final_op_time_preds
            if self.use_scaled_subplan_predictions
            else raw_op_time_preds
        )

        total_predictions, lengths = aggregate_op_times(
            final_op_time_preds, g, num_total_plan_nodes, self.device
        )
        subplan_op_time_predictions = subplan_preds_for_loss

        if debug:
            print_debug_info(
                total_predictions,
                subplan_op_time_predictions,
                lengths,
                subplan_card_predictions,
                subplan_labels,
                subplan_card_labels,
                subplan_est_card_labels,
                loop_multiplier_map,
                multiply_est_error=self.multiply_est_error,
                log_transform_card_labels=self.log_transform_card_labels,
            )

        return ModelOutput(
            plan_predictions=total_predictions,
            subplan_predictions=subplan_op_time_predictions,
            subplan_card_predictions=subplan_card_predictions,
            subplan_lengths=lengths,
        )
