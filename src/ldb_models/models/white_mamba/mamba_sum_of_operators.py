from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from ldb_models.classes import MambaSumOfOperatorsModelConfig

# These imports might need path adjustments depending on your project structure
from ldb_models.models.white_mamba.mamba_model import (
    MambaModelEncoder,
    augment_feature_stats_for_mamba,
)
from ldb_models.models.zeroshot.utils.fc_out_model import FcOutModel
from torch_scatter import scatter_sum
from ldb_models.training.preprocessing.feature_statistics import FeatureType
from ldb_models.training.training.outputs import ModelOutput


class MambaSumOfOperatorsModel(nn.Module):
    """
    A Mamba-based model that predicts total query plan runtime
    by summing the predicted "operation time" of each individual operator (node).
    This version uses separate decoders for scan and join nodes.
    """

    def __init__(
        self,
        model_config: MambaSumOfOperatorsModelConfig,
        feature_statistics: Dict,
        **kwargs,
    ):
        super().__init__()
        self.config = model_config
        self.label_norm = None
        self.device = model_config.device

        local_feature_statistics = feature_statistics.copy()
        local_feature_statistics["filter_flat_vector"] = {
            "type": str(FeatureType.numeric)
        }
        local_feature_statistics["outcols_flat_vector"] = {
            "type": str(FeatureType.numeric)
        }
        augment_feature_stats_for_mamba(local_feature_statistics)

        self.encoder = MambaModelEncoder(model_config, local_feature_statistics)

        self.use_card_feature = getattr(self.config, "use_card_feature", False)
        self.cardinality_estimation = (
            getattr(self.config, "cardinality_estimation", False)
            or self.use_card_feature
        )

        # Determine input dimension for op_time decoders
        op_time_input_dim = model_config.hidden_dim
        if self.use_card_feature:
            op_time_input_dim += 1

        # Define separate decoders for join and scan nodes
        op_time_decoders = nn.ModuleDict(
            {
                "join": FcOutModel(
                    output_dim=1,
                    input_dim=op_time_input_dim,  # Use correct input dim
                    final_out_layer=True,
                    **model_config.final_mlp_kwargs,
                ),
                "scan": FcOutModel(
                    output_dim=1,
                    input_dim=op_time_input_dim,  # Use correct input dim
                    final_out_layer=True,
                    **model_config.final_mlp_kwargs,
                ),
            }
        )

        self.decoders = nn.ModuleDict({"op_time": op_time_decoders})

        if self.cardinality_estimation:
            card_decoders = nn.ModuleDict(
                {
                    "join": FcOutModel(
                        output_dim=1,
                        input_dim=model_config.hidden_dim,  # Card estimation uses base hidden_dim
                        final_out_layer=True,
                        **model_config.final_mlp_kwargs,
                    ),
                    "scan": FcOutModel(
                        output_dim=1,
                        input_dim=model_config.hidden_dim,
                        final_out_layer=True,
                        **model_config.final_mlp_kwargs,
                    ),
                }
            )
            self.decoders["card"] = card_decoders

    def apply_decoder(self, hidden_states, node_type_lookup, mode="op_time"):
        # node_type_lookup: 0=scan, 1=join
        scan_mask = node_type_lookup == 0
        join_mask = node_type_lookup == 1

        # Get output dimension from one of the decoders (assuming both have same output dim)
        output_dim = self.decoders[mode]["join"].output_dim
        outputs = torch.zeros(
            (hidden_states.shape[0], output_dim),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        if join_mask.any():
            outputs[join_mask] = self.decoders[mode]["join"].fcout(
                hidden_states[join_mask]
            )
        if scan_mask.any():
            outputs[scan_mask] = self.decoders[mode]["scan"].fcout(
                hidden_states[scan_mask]
            )

        return outputs

    def forward(
        self,
        model_input,
        plan_dict=None,
        debug=False,
        subplan_labels=None,
        card_labels=None,
    ) -> ModelOutput:
        # Unpack inputs
        global_plan_ids = model_input[2]
        node_type_lookup = model_input[
            4
        ]  # FIX 3: Use the correct tensor for node types

        # 1. Encode the graph to get hidden states for all nodes.
        _, _, all_node_states = self.encoder(*model_input)

        # Create batch-local indices for the scatter operation
        unique_plan_ids, batch_local_plan_ids = torch.unique(
            global_plan_ids, return_inverse=True
        )
        num_plans_in_batch = unique_plan_ids.size(0)

        # 2. Predict cardinality for each node if configured.
        subplan_card_predictions = None
        if self.cardinality_estimation:
            raw_card_preds = self.apply_decoder(
                all_node_states, node_type_lookup, mode="card"
            ).squeeze(-1)
            subplan_card_predictions = F.softplus(raw_card_preds)

        # 3. Prepare input for the op_time decoder, augmenting with cardinality predictions if configured.
        op_time_input_states = all_node_states
        if self.use_card_feature:
            if subplan_card_predictions is None:
                raise ValueError(
                    "`use_card_feature` is True, but cardinality was not predicted."
                )
            detached_card_preds = subplan_card_predictions.detach().unsqueeze(-1)
            op_time_input_states = torch.cat(
                [all_node_states, detached_card_preds], dim=1
            )

        # 4. Predict the operation time for each node.
        raw_op_time_preds = self.apply_decoder(
            op_time_input_states, node_type_lookup, mode="op_time"
        ).squeeze(-1)
        subplan_op_time_predictions = F.softplus(raw_op_time_preds)

        # 5. Aggregate per-operator times using scatter_sum
        total_predictions = scatter_sum(
            src=subplan_op_time_predictions,
            index=batch_local_plan_ids,
            dim=0,
            dim_size=num_plans_in_batch,
        )

        if debug:
            lengths = torch.bincount(batch_local_plan_ids, minlength=num_plans_in_batch)
            self._print_debug_info(
                total_predictions,
                lengths,
                subplan_op_time_predictions,
                subplan_card_predictions,
                global_plan_ids,
                node_type_lookup,
                subplan_labels,
                card_labels,
            )

        # 6. Return all predictions.
        return ModelOutput(
            plan_predictions=total_predictions,
            subplan_predictions=subplan_op_time_predictions,
            subplan_card_predictions=subplan_card_predictions,
        )

    def _print_debug_info(
        self,
        total_predictions,
        lengths,
        subplan_op_time_predictions,
        subplan_card_predictions,
        global_plan_ids,
        node_type_lookup,
        subplan_labels=None,
        card_labels=None,
    ):
        print("\n--- DEBUG INFO ---")
        print(f"Total Predictions (per plan): {total_predictions.squeeze().tolist()}")
        print(f"Number of Operators per Plan: {lengths.tolist()}")

        num_plans_to_print = 1
        for plan_id in torch.unique(global_plan_ids)[:num_plans_to_print]:
            print(f"\nPlan ID: {plan_id.item()}")
            mask = global_plan_ids == plan_id
            op_times = subplan_op_time_predictions[mask].squeeze().tolist()
            node_types = node_type_lookup[mask].tolist()
            if subplan_card_predictions is not None:
                card_preds = subplan_card_predictions[mask].squeeze().tolist()
                for i, (op_time, node_type, card_pred) in enumerate(
                    zip(op_times, node_types, card_preds)
                ):
                    node_type_str = "Scan" if node_type == 0 else "Join"
                    print(
                        f"  Node {i}: Type={node_type_str}, Op P={op_time:.4f}, Card P={card_pred:.4f}"
                    )
            else:
                for i, (op_time, node_type) in enumerate(zip(op_times, node_types)):
                    node_type_str = "Scan" if node_type == 0 else "Join"
                    print(f"  Node {i}: Type={node_type_str}, Op P={op_time:.4f}")

            if subplan_labels is not None:
                true_op_times = subplan_labels[mask].squeeze().tolist()
                print(f" Op L: {true_op_times}")

            if card_labels is not None and subplan_card_predictions is not None:
                true_cards = card_labels[mask].squeeze().tolist()
                print(f" Card L: {true_cards}")

        print("--- END DEBUG INFO ---\n")
