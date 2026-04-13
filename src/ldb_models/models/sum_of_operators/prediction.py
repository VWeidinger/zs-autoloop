import dgl
import torch
import torch.nn.functional as F
from models.zeroshot.utils.fc_out_model import FcOutModel
from torch import Tensor

from .cardinality import from_linear_cardinality, to_linear_cardinality


def predict_per_node(
    decoder: FcOutModel,
    node_states_dict: dict[str, Tensor],
    g: dgl.DGLGraph | dgl.DGLHeteroGraph,
    num_total_nodes: int,
    device: torch.device | str,
) -> Tensor:
    """
    Run a decoder over grouped node states and restore original plan order.

    Args:
        decoder: Decoder module with an ``fcout`` layer that maps hidden states to
            a single scalar per node.
        node_states_dict: Dictionary keyed by node type, where each value is a
            tensor of node hidden states for that type.
        g: DGL graph containing ``original_id`` for each plan node type.
        num_total_nodes: Total number of plan nodes across the full batch.
        device: Device used to allocate the output tensor.

    Returns:
        Tensor of shape ``[num_total_nodes]`` containing softplus-transformed
        predictions in original plan order.
    """
    # Rebuild the plan order after running the decoder on grouped node states.
    pred_tensor = torch.zeros(num_total_nodes, device=device)
    for ntype, node_states in node_states_dict.items():
        if ntype.startswith("plan") and g.num_nodes(ntype) > 0:
            node_preds = decoder.fcout(node_states).squeeze(-1)
            original_ids = g.nodes[ntype].data["original_id"]
            pred_tensor[original_ids] = node_preds
    return F.softplus(pred_tensor)


def augment_features_with_cardinality(
    all_node_states: dict[str, Tensor],
    card_predictions: Tensor,
    g: dgl.DGLGraph | dgl.DGLHeteroGraph,
) -> dict[str, Tensor]:
    """
    Append cardinality predictions to the corresponding plan node features.

    Args:
        all_node_states: Dictionary of node-type tensors produced by the encoder.
        card_predictions: Flattened tensor of predicted cardinalities in plan order.
        g: DGL graph used to map plan-node local indices back to global order.

    Returns:
        Dictionary mirroring ``all_node_states`` with the plan node tensors
        expanded by one column when cardinality features are enabled.
    """
    augmented_states = {}
    # Detach here so runtime supervision does not backprop into the card head.
    detached_card_preds = card_predictions.detach()

    for ntype, node_states in all_node_states.items():
        if ntype.startswith("plan") and g.num_nodes(ntype) > 0:
            original_ids = g.nodes[ntype].data["original_id"]
            card_feature = detached_card_preds[original_ids].unsqueeze(-1)
            augmented_states[ntype] = torch.cat([node_states, card_feature], dim=1)
        else:
            augmented_states[ntype] = node_states
    return augmented_states


def apply_loop_multipliers(
    per_op_predictions: Tensor,
    card_predictions: Tensor,
    est_card_labels: Tensor | None,
    loop_multiplier_map: tuple[list[int], list[int]],
    device: torch.device | str,
    log_transform_card_labels: bool,
    multiply_est_error: bool,
) -> tuple[Tensor, Tensor | None]:
    """
    Apply SUMO-LOOP scaling to right-side loop operator predictions.

    Implements 2 scaling modes for join operators based on sibling cardinality:

    1. **Multiply estimated error mode** (``multiply_est_error=True``):
       - Multiplier = sibling_pred_card / sibling_est_card (predicted/estimated ratio)
       - Subplan Loss = Loss(right_pred_est_loop_runtime * multiplier, op_time)
       - Stabilizes training when per-loop runtime from optimizer is extremely low.
       - right_pred_runtime = multiplier * right_pred_est_loop_runtime

    2. **Loss on scaled mode** (``sumo_loop_loss_on_scaled=True``):
       - Multiplier = sibling_pred_card
       - Subplan Loss = Loss(pred * sibling_pred_card, op_time)
       - Supervises the scaled prediction instead of unscaled.
       - right_pred_runtime = sibling_pred_card * right_pred_single_loop_runtime

    Args:
        per_op_predictions: Per-operator runtime predictions before loop scaling.
        card_predictions: Per-operator cardinality predictions in model space.
        est_card_labels: Estimated cardinality labels from the query statistics,
            required only for ``multiply_est_error=True`` mode.
        loop_multiplier_map: ``(node_ids, sibling_ids)`` tuple identifying
            loop operators and the sibling cardinality source.
        device: Device used for tensor conversion.
        log_transform_card_labels: Whether cardinalities are stored in log1p space.
        multiply_est_error: Whether to use predicted/estimated sibling cardinality
            ratio instead of only the predicted sibling cardinality.

    Returns:
        Tuple ``(scaled_runtime_predictions, adjusted_card_predictions)``.
    """

    # Convert node id lists once so the loop update stays vectorized.
    node_ids_list, sibling_ids_list = loop_multiplier_map
    node_ids = torch.tensor(node_ids_list, device=device, dtype=torch.long)
    sibling_ids = torch.tensor(sibling_ids_list, device=device, dtype=torch.long)

    scaled_runt_predictions = per_op_predictions.clone()
    scaled_card_predictions = card_predictions.clone()

    if multiply_est_error:
        if est_card_labels is None:
            raise ValueError(
                "SUMO-LOOP multiply_est_error mode requires subplan_est_card_labels in query_stats."
            )
        est_card_labels = torch.as_tensor(
            est_card_labels,
            device=device,
            dtype=per_op_predictions.dtype,
        )
        # Multiply estimated error mode: use predicted/estimated cardinality ratio.
        # pg_off_multiplier = sibling_pred_card / sibling_est_card
        sibling_card_preds = to_linear_cardinality(
            card_predictions[sibling_ids], log_transform_card_labels
        )
        sibling_card_ests = to_linear_cardinality(
            est_card_labels[sibling_ids], log_transform_card_labels
        )
        multiplier = sibling_card_preds / sibling_card_ests
    else:
        # Use sibling predicted cardinality as the multiplier.
        multiplier = to_linear_cardinality(
            card_predictions[sibling_ids], log_transform_card_labels
        )

    scaled_runt_predictions[node_ids] = per_op_predictions[node_ids] * multiplier

    # Keep the adjusted cardinality aligned with the scaled runtime path.

    scaled_card_predictions[node_ids] = from_linear_cardinality(
        to_linear_cardinality(card_predictions[node_ids], log_transform_card_labels)
        * multiplier,
        log_transform_card_labels,
    )

    return scaled_runt_predictions, scaled_card_predictions
