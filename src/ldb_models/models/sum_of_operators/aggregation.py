import dgl
import torch
from torch import Tensor


def aggregate_op_times(
    subplan_op_time_predictions: Tensor,
    g: dgl.DGLGraph | dgl.DGLHeteroGraph,
    num_total_plan_nodes: int,
    device: torch.device | str,
) -> tuple[Tensor, Tensor]:
    """
    Aggregate operator-level runtime predictions into plan-level totals.

    Args:
        subplan_op_time_predictions: Flat tensor of per-operator runtime predictions
            in original plan order.
        g: DGL graph containing the ``plan0`` node type and ``original_id`` fields.
        num_total_plan_nodes: Number of plan nodes in the flattened batch.
        device: Device used to build the segment lengths tensor.

    Returns:
        Tuple ``(total_predictions, lengths)`` where ``total_predictions`` contains
        the per-plan sums and ``lengths`` describes how many operators belong to
        each plan.
    """
    plan0_original_ids = g.nodes["plan0"].data["original_id"]
    sorted_plan0_orig_ids, _ = torch.sort(plan0_original_ids)

    # Segment lengths are derived from the sorted root-node ids.
    lengths = torch.diff(
        sorted_plan0_orig_ids,
        append=torch.tensor([num_total_plan_nodes], device=device, dtype=torch.long),
    )

    total_predictions = torch.segment_reduce(
        data=subplan_op_time_predictions,
        reduce="sum",
        lengths=lengths,
    )
    return total_predictions, lengths
