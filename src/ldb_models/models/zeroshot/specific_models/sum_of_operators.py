import torch
import torch.nn as nn
import torch.nn.functional as F
from models.zeroshot.specific_models.postgres_zero_shot import (
    PostgresZeroShotModelEncoder,
)
from models.zeroshot.utils.fc_out_model import FcOutModel
from training.losses import create_loss
from training.training.outputs import ModelOutput


class SumOfOperatorsModel(nn.Module):
    """
    A zero-shot model that predicts the total runtime of a query plan
    by summing the predicted "operation time" of each individual operator (node).

    It can also be configured to predict the "cardinality" for each operator and use that
    prediction as a feature for the runtime estimation.
    """

    def __init__(self, model_config, feature_statistics, **zero_shot_kwargs):
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
        self.cardinality_estimation = (
            getattr(self.config, "cardinality_estimation", False)
            or self.use_card_feature
        )

        op_time_input_dim = model_config.hidden_dim
        if self.use_card_feature:
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

    def get_representation(self, model_input, subplan=True):
        g, features = model_input
        encoder_output = self.encoder((g, features))
        hidden_states, all_node_states = encoder_output
        if subplan:
            return hidden_states, all_node_states, g
        else:
            return hidden_states

    def forward(
        self,
        model_input,
        plan_dict=None,
        debug=False,
        subplan_labels=None,
        card_labels=None,
    ) -> ModelOutput:
        """
        Predicts total runtime and per-operator metrics (operation time, cardinality).
        """
        # 1. Encode the graph to get hidden states for all nodes.
        g, features = model_input
        _, all_node_states = self.encoder((g, features))
        num_total_plan_nodes = len(plan_dict)

        # 2. Predict cardinality if the model is configured to do so.
        subplan_card_predictions = None
        if self.cardinality_estimation:
            subplan_card_predictions = self._predict_per_node(
                self.decoders["card"], all_node_states, g, num_total_plan_nodes
            )

        # 3. Prepare input for the runtime decoder, augmenting with cardinality if configured.
        op_time_input_states = all_node_states
        if self.use_card_feature:
            if subplan_card_predictions is None:
                raise ValueError(
                    "`use_card_feature` is True, but cardinality was not predicted."
                )
            op_time_input_states = self._augment_features_with_cardinality(
                all_node_states, subplan_card_predictions, g
            )

        # 4. Predict the operation time for each node.
        raw_op_time_preds = self._predict_per_node(
            self.decoders["op_time"], op_time_input_states, g, num_total_plan_nodes
        )

        # 5. Aggregate per-operator times to get total plan runtime.
        total_predictions, subplan_op_time_predictions, lengths = (
            self._aggregate_op_times(raw_op_time_preds, g, num_total_plan_nodes)
        )

        # 6. (Optional) Print debug information.
        if debug:
            self._print_debug_info(
                total_predictions,
                subplan_op_time_predictions,
                lengths,
                subplan_card_predictions,
                subplan_labels,
                card_labels,
            )

        return ModelOutput(
            plan_predictions=total_predictions,
            subplan_predictions=subplan_op_time_predictions,
            subplan_card_predictions=subplan_card_predictions,
            subplan_lengths=lengths,
        )

    def _predict_per_node(self, decoder, node_states_dict, g, num_total_nodes):
        """
        Runs a decoder on a dictionary of node states and returns a single, ordered tensor.
        """
        pred_tensor = torch.zeros(num_total_nodes, device=self.device)
        for ntype, node_states in node_states_dict.items():
            if ntype.startswith("plan") and g.num_nodes(ntype) > 0:
                node_preds = decoder.fcout(node_states).squeeze(-1)
                original_ids = g.nodes[ntype].data["original_id"]
                pred_tensor[original_ids] = node_preds
        pred_tensor = F.softplus(pred_tensor)
        return pred_tensor

    def _augment_features_with_cardinality(self, all_node_states, card_predictions, g):
        """
        Attaches detached cardinality predictions as features to the node hidden states.
        """
        augmented_states = {}
        # DETACH: Stop gradients from flowing from runtime loss to card decoder.
        detached_card_preds = card_predictions.detach()

        for ntype, node_states in all_node_states.items():
            if ntype.startswith("plan") and g.num_nodes(ntype) > 0:
                original_ids = g.nodes[ntype].data["original_id"]
                card_feature = detached_card_preds[original_ids].unsqueeze(-1)
                # ATTACH: Concatenate the feature.
                augmented_states[ntype] = torch.cat([node_states, card_feature], dim=1)
            else:
                augmented_states[ntype] = node_states
        return augmented_states

    def _aggregate_op_times(self, subplan_op_time_predictions, g, num_total_plan_nodes):
        """
        Aggregates per-operator time predictions into total plan runtime predictions.
        """
        plan0_original_ids = g.nodes["plan0"].data["original_id"]
        sorted_plan0_orig_ids, _ = torch.sort(plan0_original_ids)

        lengths = torch.diff(
            sorted_plan0_orig_ids,
            append=torch.tensor(
                [num_total_plan_nodes], device=self.device, dtype=torch.long
            ),
        )

        total_predictions = torch.segment_reduce(
            data=subplan_op_time_predictions,
            reduce="sum",
            lengths=lengths,
        )
        return total_predictions, subplan_op_time_predictions, lengths

    def _print_debug_info(
        self,
        total_predictions,
        subplan_op_time_predictions,
        lengths,
        subplan_card_predictions,
        subplan_labels,
        card_labels,
    ):
        """Prints a detailed breakdown of predictions and labels for the first plan."""
        print("\n--- DEBUG: Breakdown for First Plan in Batch ---", flush=True)
        print(
            f"Total Prediction (Sum of OpTimes): {total_predictions[0].item():.4f}",
            flush=True,
        )
        num_ops = lengths[0].item()
        print(f"Number of Operators: {num_ops}", flush=True)
        for i in range(num_ops):
            op_time_pred = subplan_op_time_predictions[i].item()
            if subplan_card_predictions is not None:
                card_pred = subplan_card_predictions[i].item()
                print(
                    f" - Op[{i}]: OpT P={op_time_pred:.4f}, Card P={card_pred:.2f}",
                    flush=True,
                    end="",
                )
            else:
                print(f" - Op[{i}]: OpT P={op_time_pred:.4f}", flush=True, end="")

            if subplan_labels is not None and i < len(subplan_labels):
                op_time_label = subplan_labels[i].item()
                if (
                    subplan_card_predictions is not None
                    and card_labels is not None
                    and i < len(card_labels)
                ):
                    card_label = card_labels[i].item()
                    print(
                        f" | OpT L={op_time_label:.4f}, Card L={card_label:.2f}",
                        flush=True,
                    )
                else:
                    print(f" | OpT L={op_time_label:.4f}", flush=True)
            else:
                print("", flush=True)
