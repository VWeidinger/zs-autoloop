from typing import Dict, List, Tuple


class NoJoinFoundError(Exception):
    """Exception raised when no join operation is found in the plan."""

    pass


def get_first_join_op(plan: Dict) -> Dict:
    operator = plan["plan_parameters"]["op_name"]

    if operator in ["Nested Loop", "Index Nested Loop", "Hash Join", "Merge Join"]:
        return plan
    else:
        # Recursively search in subplans
        if "children" not in plan or len(plan["children"]) == 0:
            raise NoJoinFoundError("No join operation found in the plan.")

        assert len(plan["children"]) == 1, (
            f"Expected at least one child plan. Found: {len(plan['children'])} \n{plan}"
        )
        return get_first_join_op(plan["children"][0])


def get_first_join_op_json_plan(plan: Dict) -> Dict:
    operator = plan["Node Type"]

    if operator in ["Nested Loop", "Index Nested Loop", "Hash Join", "Merge Join"]:
        return plan
    else:
        # recursively search in subplans
        if "Plans" not in plan or len(plan["Plans"]) == 0:
            raise NoJoinFoundError("No join operation found in the plan.")

        assert len(plan["Plans"]) == 1, (
            f"Expected at least one child plan. Found: {len(plan['Plans'])} \n{plan}"
        )
        return get_first_join_op_json_plan(plan["Plans"][0])


def get_first_scan_op(plan: Dict) -> Dict:
    operator = plan["plan_parameters"]["op_name"]

    if operator in ["Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan"]:
        return plan
    else:
        # Recursively search in subplans
        assert "children" in plan, f"Expected 'children' key in plan: {plan}"
        assert len(plan["children"]) == 1, (
            f"Expected at least one child plan. Found: {len(plan['children'])} \n{plan}"
        )
        return get_first_scan_op(plan["children"][0])


def get_first_scan_op_json_plan(plan: Dict) -> Dict:
    operator = plan["Node Type"]

    if operator in [
        "Seq Scan",
        "Index Scan",
        "Index Only Scan",
        "Bitmap Heap Scan",
        "Result",  # this should happen rarely - invalid filters,...
    ]:
        return plan
    else:
        # recursively search in subplans
        if "Plans" not in plan or len(plan["Plans"]) == 0:
            raise NoJoinFoundError("No join operation found in the plan.")

        assert len(plan["Plans"]) == 1, (
            f"Expected at least one child plan. Found: {len(plan['Plans'])} \n{plan}"
        )
        return get_first_scan_op_json_plan(plan["Plans"][0])


def extract_card_from_all_operators(plan: Dict) -> Tuple[List[float], List[float]]:
    actual_card = []
    estimated_card = []

    assert "plan_parameters" in plan, f"Plan does not have 'plan_parameters': {plan}"
    params = plan["plan_parameters"]
    actual_card.append(params["act_card"])
    estimated_card.append(params["est_card"])

    if "children" in plan and plan["children"]:
        for child in plan["children"]:
            # traverse down to join or scan operators - skip nodes in between
            try:
                child_node = get_first_join_op(child)
            except NoJoinFoundError:
                child_node = get_first_scan_op(child)

            act_tmp, est_tmp = extract_card_from_all_operators(child_node)
            actual_card.extend(act_tmp)
            estimated_card.extend(est_tmp)

    return actual_card, estimated_card


def extract_card_from_all_operators_json_plan(
    plan: Dict,
) -> Tuple[List[float], List[float]]:
    actual_card = []
    estimated_card = []

    actual_card.append(plan["Actual Rows"])
    estimated_card.append(plan["Plan Rows"])

    if "Plans" in plan and plan["Plans"]:
        for child in plan["Plans"]:
            # traverse down to join or scan operators - skip nodes in between
            try:
                child_node = get_first_join_op_json_plan(child)
            except NoJoinFoundError:
                child_node = get_first_scan_op_json_plan(child)

            act_tmp, est_tmp = extract_card_from_all_operators_json_plan(child_node)
            actual_card.extend(act_tmp)
            estimated_card.extend(est_tmp)

    return actual_card, estimated_card


def extract_card_error_from_plan(
    plan: Dict, mode: str, is_parsed_plan: bool = True
) -> float:
    if not is_parsed_plan:
        plan = plan["analyze_plans"][0]["Plan"]

    if mode == "root":
        try:
            if is_parsed_plan:
                root_node = get_first_join_op(plan)
            else:
                root_node = get_first_join_op_json_plan(plan)
        except NoJoinFoundError:
            # no join in the plan - take the first scan op instead
            if is_parsed_plan:
                root_node = get_first_scan_op(plan)
            else:
                root_node = get_first_scan_op_json_plan(plan)

        # extract cardinality
        if is_parsed_plan:
            op_parameters = root_node["plan_parameters"]
            act_card = op_parameters["act_card"]
            est_card = op_parameters["est_card"]
        else:
            act_card = root_node["Actual Rows"]
            est_card = root_node["Plan Rows"]

        # ensure that the cardinalities are at least 1
        est_card_clipped = max(est_card, 1)
        act_card_clipped = max(act_card, 1)

        # compute mse & q-error
        mse = (est_card - act_card) ** 2
        q_error = max(
            est_card_clipped / act_card_clipped, act_card_clipped / est_card_clipped
        )

    elif mode == "avg":
        # retrieve all cards from all operators
        if is_parsed_plan:
            actual_card, estimated_card = extract_card_from_all_operators(plan)
        else:
            actual_card, estimated_card = extract_card_from_all_operators_json_plan(
                plan
            )

        # ensure that the cardinalities are at least 1
        estimated_card_clipped = [max(est, 1) for est in estimated_card]
        actual_card_clipped = [max(act, 1) for act in actual_card]

        # compute avg mse & q-error
        mse = sum(
            (est - act) ** 2 for est, act in zip(estimated_card, actual_card)
        ) / len(actual_card)
        q_error = sum(
            max(est / act, act / est)
            for est, act in zip(estimated_card_clipped, actual_card_clipped)
        ) / len(actual_card)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'root' or 'avg'.")

    return mse, q_error
