import torch
from torch import Tensor

from .cardinality import to_linear_cardinality


def print_debug_info(
    total_predictions: Tensor,
    subplan_op_time_predictions: Tensor,
    lengths: Tensor,
    subplan_card_predictions: Tensor | None,
    subplan_labels: Tensor | None,
    subplan_card_labels: Tensor | None,
    subplan_est_card_labels: Tensor | None,
    loop_multiplier_map: tuple[list[int], list[int]] | None = None,
    multiply_est_error: bool = False,
    log_transform_card_labels: bool = True,
) -> None:
    """
    Print a human-readable breakdown of one batch element.

    Args:
        total_predictions: Per-plan runtime totals returned by the model.
        subplan_op_time_predictions: Flattened operator runtime predictions.
        lengths: Number of operators per plan, as produced by aggregation.
        subplan_card_predictions: Optional flattened cardinality predictions.
        subplan_labels: Optional operator runtime labels for comparison.
        subplan_card_labels: Optional operator cardinality labels for comparison.
        subplan_est_card_labels: Optional estimated cardinality labels, used for
            SUMO-LOOP diagnostics.
        loop_multiplier_map: Optional ``(node_ids, sibling_ids)`` pair identifying
            loop operators.
        multiply_est_error: Whether loop scaling uses predicted/estimated sibling
            cardinalities.
        log_transform_card_labels: Whether the cardinality tensors are stored in
            log1p space.

    Returns:
        None. The function prints directly to stdout.
    """
    # Keep the debug dump scoped to the first plan in the batch.
    print("\n--- DEBUG: Breakdown for First Plan in Batch ---", flush=True)
    print(
        f"Total Prediction (Sum of OpTimes): {total_predictions[0].item():.4f}",
        flush=True,
    )
    num_ops = lengths[0].item()
    print(f"Number of Operators: {num_ops}", flush=True)
    for i in range(num_ops):
        op_time_label = subplan_labels[i].item()
        op_time_pred = subplan_op_time_predictions[i].item()
        print(
            f" - Op[{i}] {op_time_label:.2f}, Pre={op_time_pred:.2f}",
            flush=True,
            end="",
        )

        if subplan_card_predictions is not None:
            # Convert displayed cardinalities back to linear space first.
            card_pred = to_linear_cardinality(
                subplan_card_predictions[i], log_transform_card_labels
            ).item()
            card_est = to_linear_cardinality(
                subplan_est_card_labels[i], log_transform_card_labels
            ).item()
            card_label = to_linear_cardinality(
                torch.tensor(subplan_card_labels[i].item()), log_transform_card_labels
            ).item()

            print(
                f"\t| Card {int(card_label)}, Pre={card_pred:.1f}, Est={int(card_est)}",
                flush=True,
                end="",
            )

        is_loop_node = loop_multiplier_map is not None and i in loop_multiplier_map[0]
        if is_loop_node:
            sibling_id = loop_multiplier_map[1][loop_multiplier_map[0].index(i)]
            sibling_card = to_linear_cardinality(
                subplan_card_predictions[sibling_id], log_transform_card_labels
            ).item()
            if multiply_est_error:
                # Report the ratio used by the PG-offset-style loop multiplier.
                sibling_est_card = to_linear_cardinality(
                    torch.tensor(subplan_est_card_labels[sibling_id].item()),
                    log_transform_card_labels,
                ).item()
                sibling_act_card = to_linear_cardinality(
                    torch.tensor(subplan_card_labels[sibling_id].item()),
                    log_transform_card_labels,
                ).item()
                print(
                    f" [LOOP M={sibling_card:.1f}/{int(sibling_est_card)}={sibling_card / sibling_est_card:.2f} Act={int(sibling_act_card)}]",
                    flush=True,
                    end="",
                )
            else:
                print(f" [LOOP: {sibling_card:.1f}]", flush=True, end="")
        print("", flush=True)
