import torch
import torch.nn as nn
import torch.nn.functional as F
from models.zeroshot.specific_models.postgres_zero_shot import (
    PostgresZeroShotModelEncoder,
)
from models.zeroshot.utils.fc_out_model import FcOutModel
from models.zeroshot.zero_shot_model import PassDirection
from training.losses import create_loss
from training.training.outputs import ModelOutput


class SUMO(nn.Module):
    """
    A model that predicts per-operator metrics during the encoding phase,
    optionally using those predictions as features for subsequent operators.

    The final plan prediction is the sum of all its per-operator runtime predictions.
    """

    def __init__(self, model_config, feature_statistics, **zero_shot_kwargs):
        super().__init__()
        self.config = model_config
        self.label_norm = None
        self.device = model_config.device

        self.encoder = SUMOEncoder(
            model_config,
            feature_statistics,
            **zero_shot_kwargs,
        )

        self.loss_fxn = create_loss(
            model_config.loss_class_name, loss_kwargs=model_config.loss_class_kwargs
        )

    def forward(
        self,
        model_input,
        plan_dict=None,
        debug=False,
        subplan_labels=None,
        card_labels=None,
    ) -> ModelOutput:
        """
        Encodes the query plan, makes intermediate predictions, and sums
        per-operator runtimes to get the total plan runtime.
        """
        g, features = model_input
        num_total_plan_nodes = len(plan_dict)

        # The encoder returns flattened tensors for predictions and the lengths of each plan.
        op_time_preds, card_preds, lengths = self.encoder(
            (g, features), num_total_plan_nodes=num_total_plan_nodes
        )

        # Aggregate per-operator times to get the total plan runtime for each plan in the batch.
        total_predictions = torch.segment_reduce(
            data=op_time_preds,
            reduce="sum",
            lengths=lengths,
        )

        if debug:
            self._print_debug_info(
                total_predictions,
                op_time_preds,
                lengths,
                card_preds,
                subplan_labels,
                card_labels,
            )

        return ModelOutput(
            plan_predictions=total_predictions,
            subplan_predictions=op_time_preds,
            subplan_card_predictions=card_preds,
            subplan_lengths=lengths,
        )

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
        num_ops = lengths[0].item()
        print(f"Number of Operators: {num_ops}", flush=True)

        # --- START: Modified Code ---
        for i in range(num_ops):
            op_time_pred = subplan_op_time_predictions[i].item()
            op_time_label = (
                subplan_labels[i].item()
                if subplan_labels is not None and i < len(subplan_labels)
                else "N/A"
            )

            line_parts = [f"Op[{i}]:"]

            # Format OpTime: Label, Prediction
            if isinstance(op_time_label, float):
                line_parts.append(f"OpTime: {op_time_label:.4f}, {op_time_pred:.4f}")
            else:
                line_parts.append(f"OpTime: {op_time_label}, {op_time_pred:.4f}")

            # Format Card: Label, Prediction
            if subplan_card_predictions is not None:
                card_pred = subplan_card_predictions[i].item()
                card_label = (
                    card_labels[i].item()
                    if card_labels is not None and i < len(card_labels)
                    else "N/A"
                )
                if isinstance(card_label, float):
                    line_parts.append(f"| Card: {card_label:.2f}, {card_pred:.2f}")
                else:
                    line_parts.append(f"| Card: {card_label}, {card_pred:.2f}")
            print(" ".join(line_parts), flush=True)
        print(
            f"Total Prediction (Sum of OpTimes): {total_predictions[0].item():.4f}",
            flush=True,
        )


class SUMOEncoder(PostgresZeroShotModelEncoder):
    """
    An encoder that performs message passing while also making and appending
    per-node (operator) predictions for runtime and cardinality.
    """

    def __init__(self, model_config, feature_statistics, **zero_shot_kwargs):
        super().__init__(
            model_config,
            feature_statistics,
            return_all_node_states=True,
            **zero_shot_kwargs,
        )

        self.config = model_config
        self.device = model_config.device
        self.predict_cardinality = getattr(self.config, "cardinality_estimation", False)
        self.append_cardinality_feature = getattr(
            self.config, "use_card_feature", False
        )
        self.append_runtime_feature = getattr(self.config, "use_runtime_feature", False)

        self.prediction_heads = nn.ModuleDict()

        runtime_input_dim = self.hidden_dim
        if self.append_cardinality_feature:
            runtime_input_dim += 1

        # This encoder only predicts operator time, not total time.
        self.prediction_heads["op_time"] = FcOutModel(
            output_dim=1, input_dim=runtime_input_dim, **model_config.final_mlp_kwargs
        )

        if self.predict_cardinality:
            self.prediction_heads["cardinality"] = FcOutModel(
                output_dim=1, input_dim=self.hidden_dim, **model_config.final_mlp_kwargs
            )

    def _flatten_predictions(self, g, preds_dict, num_total_nodes):
        """Converts a dictionary of predictions into a single, ordered tensor."""
        if not preds_dict:
            return None
        pred_tensor = torch.zeros(num_total_nodes, device=self.device)
        for ntype, node_preds in preds_dict.items():
            if ntype.startswith("plan") and g.num_nodes(ntype) > 0:
                original_ids = g.nodes[ntype].data["original_id"]
                pred_tensor[original_ids] = node_preds.squeeze(-1)
        return pred_tensor

    def message_passing(self, g, feat_dict):
        """
        Overrides the base message passing to inject prediction and feature augmentation.
        Returns dictionaries of per-node predictions.
        """
        all_runtime_predictions = {}
        all_cardinality_predictions = {}

        pass_directions = self._get_pass_directions(g)

        for pd in pass_directions:
            if not pd.etypes:
                continue

            out_dict = self.tree_models[pd.model_name](
                g,
                etypes=pd.etypes,
                in_node_types=pd.in_types,
                out_node_types=pd.out_types,
                feat_dict=feat_dict,
            )

            for out_type, hidden_out in out_dict.items():
                feat_dict[out_type] = hidden_out
                if not out_type.startswith("plan"):
                    continue

                current_states = feat_dict[out_type]
                states_for_runtime_pred = current_states

                if self.predict_cardinality:
                    card_preds = F.softplus(
                        self.prediction_heads["cardinality"].fcout(current_states)
                    )
                    all_cardinality_predictions[out_type] = card_preds
                    if self.append_cardinality_feature:
                        states_for_runtime_pred = torch.cat(
                            [current_states, card_preds.detach()], dim=1
                        )

                op_time_preds = F.softplus(
                    self.prediction_heads["op_time"].fcout(states_for_runtime_pred)
                )
                all_runtime_predictions[out_type] = op_time_preds
                if self.append_runtime_feature:
                    feat_dict[out_type] = torch.cat(
                        [feat_dict[out_type], op_time_preds.detach()], dim=1
                    )

        return all_runtime_predictions, all_cardinality_predictions

    def _get_pass_directions(self, g):
        """Helper to construct the message passing sequence."""
        pass_directions = [
            PassDirection(g=g, **prepass_kwargs) for prepass_kwargs in self.prepasses
        ]
        if hasattr(g, "max_pred_depth") and g.max_pred_depth is not None:
            for d in reversed(range(g.max_pred_depth)):
                pass_directions.append(
                    PassDirection(
                        model_name="intra_pred",
                        g=g,
                        e_name="intra_predicate",
                        n_dest=f"logical_pred_{d}",
                    )
                )
        pass_directions.append(
            PassDirection(model_name="to_plan", g=g, e_name="to_plan")
        )
        if hasattr(g, "max_depth") and g.max_depth is not None:
            for d in reversed(range(g.max_depth)):
                pass_directions.append(
                    PassDirection(
                        model_name="intra_plan",
                        g=g,
                        e_name="intra_plan",
                        n_dest=f"plan{d}",
                    )
                )
        return pass_directions

    def forward(self, input, num_total_plan_nodes):
        """
        Encodes the graph, returns flattened predictions and plan lengths.
        """
        graph, features = input
        encoded_features = self.encode_node_types(graph, features)

        runtime_preds_dict, card_preds_dict = self.message_passing(
            graph, encoded_features
        )

        # Convert dictionary outputs to single, ordered tensors
        flat_runtime_preds = self._flatten_predictions(
            graph, runtime_preds_dict, num_total_plan_nodes
        )
        flat_card_preds = self._flatten_predictions(
            graph, card_preds_dict, num_total_plan_nodes
        )

        # Calculate the number of operators for each plan in the batch
        plan0_original_ids = graph.nodes["plan0"].data["original_id"]
        sorted_plan0_orig_ids, _ = torch.sort(plan0_original_ids)
        lengths = torch.diff(
            sorted_plan0_orig_ids,
            append=torch.tensor(
                [num_total_plan_nodes], device=self.device, dtype=torch.long
            ),
        )

        return flat_runtime_preds, flat_card_preds, lengths
