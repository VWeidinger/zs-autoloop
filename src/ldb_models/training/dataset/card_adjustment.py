import math
from typing import Dict


def apply_cardinality_estimation_logic(
    plan: Dict,
    enable_bu: bool = True,
    enable_td: bool = True,
    update_est_loops: bool = True,
) -> Dict:
    """
    We assume there is repetition in the workload, and we have a seen a plan of the same sql query beforehand.
    This means we know (independent of phys-op choices, or join order):
    1) The Actual Cardinality at the root node
    2) The Actual Selectivities of all base relations (Scans)

    The goal of this logic is to propagate this knowledge through the plan tree to adjust estimates
    at all nodes, so that they better reflect reality.
    """

    """
    Orchestrates the Bottom-Up and Top-Down estimation logic.
    """
    # 1. Bottom-Up Pass (Calculates BU estimates and tree depth)
    _, max_depth = annotate_bottom_up(plan)
    max_depth = max_depth + 1

    # 2. Top-Down Pass (Calculates TD estimates starting from top join)
    annotate_top_down(plan)

    # 3. Combination Pass (Weights and updates plan_params['est_card'])
    combine_estimates(plan, 0, max_depth, enable_bu, enable_td, update_est_loops)

    return plan


def annotate_bottom_up(node):
    """
    Recursively calculates estimates from leaves (Scans) up to Root.
    Returns: (est_bu, act_card, depth)
    """
    children = node.get("children", [])

    # Recurse first
    child_results = [annotate_bottom_up(child) for child in children]

    plan_params = node["plan_parameters"]
    op_type = get_operator_type(plan_params.get("op_name", ""))
    orig_est = float(plan_params.get("est_card", 1.0))

    # Calculate max depth for later weighting
    current_max_depth = 0
    if child_results:
        current_max_depth = max(c[1] for c in child_results) + 1

    if op_type == "SCAN":
        # Base case: Bottom-up estimate is the Actual Card (perfect knowledge at leaf)
        est_bu = float(plan_params.get("act_card", 1.0))
    elif op_type == "JOIN":
        # Re-implementation of the logic provided in original code snippet
        # Calculate error contribution from children
        est_bu = orig_est
        combined_error_rate = 1.0

        for i, c in enumerate(children):
            c_params = c["plan_parameters"]
            c_est_bu = c_params["est_card_bu"]
            c_orig = c_params["est_card_original"]  # We need to ensure we save original

            rate = c_est_bu / c_orig
            combined_error_rate *= rate

        # cap error rate to avoid extreme values
        combined_error_rate = min(combined_error_rate, 10e6)

        # Final BU estimate
        est_bu = orig_est * combined_error_rate

    else:
        # For Aggregates or Pass-through, usually we inherit the error from the primary child
        # or reset if it's a Scalar Agg.
        if op_type == "SCALAR_AGG":
            est_bu = 1.0
        else:
            # Pass through error from first child (e.g. Sort, GroupBy)
            if children:
                assert len(children) == 1, (
                    "Expected single child for non-join/non-scan operators"
                )
                c_params = children[0]["plan_parameters"]
                c_est_bu = c_params["est_card_bu"]
                c_orig = c_params["est_card_original"]
                rate = c_est_bu / max(c_orig, 1e-6)
                est_bu = orig_est * rate

                if "est_loops" not in plan_params or plan_params["est_loops"] <= 1.0:
                    # only in case this is not a nested loop join inner child
                    assert math.isclose(est_bu, c_est_bu, rel_tol=1e-3), (
                        f"Pass-through operator's BU estimate should match child's BU estimate. Got {est_bu} vs {c_est_bu} / orig est: {orig_est}, child orig: {c_orig}, rate: {rate}, op_type: {op_type}\n {node}"
                    )
            else:
                est_bu = orig_est

    # Sanity check
    est_bu = max(est_bu, 1.0)

    # Store
    plan_params["est_card_bu"] = est_bu
    plan_params["est_card_original"] = orig_est  # Backup original if not present
    plan_params["tree_max_depth"] = current_max_depth  # Store temporarily

    return est_bu, current_max_depth


def annotate_top_down(node, error_factor=1.0, inside_top_join_scope=False):
    """
    Propagates error from the top-most join downwards.

    :param error_factor: The multiplier to apply to the current node's estimate.
    :param inside_top_join_scope: Boolean, true if we have passed the top-most join.
    """
    plan_params = node["plan_parameters"]
    op_type = get_operator_type(plan_params.get("op_name", ""))
    orig_est = plan_params.get("est_card_original", plan_params.get("est_card", 1.0))

    # 1. Determine current node's Top-Down Estimate
    if "AGG" in op_type:
        # Reset error factor to 1.0 (or calculate local error if this is the root of analysis)
        # But we simply don't propagate the row-count error through a group-by.
        current_error_factor = 1.0
        est_td = orig_est  # Reset to original estimate
    else:
        current_error_factor = error_factor
        est_td = orig_est * current_error_factor

    # 2. Check if this is the "Top Most Join" to start propagation
    # If we haven't found the top join yet, and this IS a join:
    if not inside_top_join_scope and op_type == "JOIN":
        inside_top_join_scope = True
        # "We assume we know its card". Correct the factor to match Actual exactly.
        # This becomes the source of truth.
        current_error_factor = float(plan_params["act_card"]) / orig_est
        est_td = float(plan_params["act_card"])

    # Store TD estimate
    plan_params["est_card_td"] = est_td

    # 3. Propagate to children
    children = node.get("children", [])
    if not children:
        return

    # "Distribute this error while traversing down"
    if inside_top_join_scope:
        if op_type == "JOIN":
            child_factor = math.pow(current_error_factor, 1.0 / 2.0)
        else:
            # Linear operators (Filter, Sort, etc.) -> Error propagates 1:1
            child_factor = current_error_factor
    else:
        child_factor = 1.0

    for child in children:
        annotate_top_down(child, child_factor, inside_top_join_scope)


def combine_estimates(
    node, depth, max_depth, enable_bu, enable_td, update_est_loops: bool = True
):
    plan_params = node["plan_parameters"]

    # Retrieve estimates
    est_bu = plan_params.get("est_card_bu", plan_params.get("est_card"))
    est_td = plan_params.get("est_card_td", plan_params.get("est_card"))

    if not enable_bu:
        est_bu = plan_params.get("est_card_original", plan_params.get("est_card"))
    if not enable_td:
        est_td = plan_params.get("est_card_original", plan_params.get("est_card"))

    # Weighting by depth
    # Depth 0 (Top) -> Weight 1.0 for Top-Down (Since TD starts from Top knowledge)
    # Depth Max (Bottom) -> Weight 1.0 for Bottom-Up (Since BU starts from Leaf knowledge)

    td_weight = 1.0 - (depth / max_depth)

    # Clamp weight
    td_weight = max(1e-6, min(1.0, td_weight))

    if enable_bu and enable_td:
        final_est = (td_weight * est_td) + ((1.0 - td_weight) * est_bu)
    elif enable_td:
        final_est = est_td
    elif enable_bu:
        final_est = est_bu
    else:
        final_est = plan_params["est_card_original"]

    plan_params["est_card_pre"] = plan_params.get(
        "est_card", 1.0
    )  # Store previous estimate
    # Update the main est_card field which is used for featurization
    plan_params["est_card"] = final_est

    for child in node.get("children", []):
        combine_estimates(
            child, depth + 1, max_depth, enable_bu, enable_td, update_est_loops
        )

    # update est_children_card field
    children = node.get("children", [])
    if children:
        est_children_card = 1.0
        assert 2 >= len(children) >= 1, f"Plan node has more than 2 children: {node}"
        for c in children:
            c_params = c["plan_parameters"]
            est_children_card *= c_params.get("est_card", 1.0)
        plan_params["est_children_card"] = est_children_card

    # Optionally update est_loops for nested loop joins:
    if update_est_loops:
        op_name = plan_params.get("op_name", "").lower()
        if "nested loop" in op_name:
            assert len(node.get("children", [])) == 2, (
                f"Nested Loop Join does not have 2 children: {node}"
            )
            children_list = node.get("children", [])
            # Prefer the child whose operator type is a SCAN (more likely the inner/looped child)
            children_list[1]["plan_parameters"]["est_loops"] = children_list[0][
                "plan_parameters"
            ]["est_card"]


def get_operator_type(op_name):
    op_name = op_name.lower()
    # Scans (Base cases)
    if "scan" in op_name:
        # Index Scan, Seq Scan, Index Only Scan
        return "SCAN"

    # Joins (Multiplicative error propagation)
    elif "join" in op_name or "nested loop" in op_name:
        # Hash Join, Merge Join, Nested Loop Join, Index Nested Loop Join
        return "JOIN"

    # Aggregates (Cardinality reset or dependent on grouping)
    elif "aggregate" in op_name or "group" in op_name:
        if "simple" in op_name:
            # e.g., "Simple Aggregate" (COUNT(*)) -> Always returns 1 row
            return "SCALAR_AGG"
        return "GROUP_AGG"

    elif "sort" in op_name or "materialize" in op_name or "hash" in op_name:
        # Sort, Hash, Materialize in op_name:
        return "Pass-through"

    else:
        raise NotImplementedError(f"Unknown operator type for op_name: {op_name}")
