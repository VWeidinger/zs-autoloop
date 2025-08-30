from typing import Dict, List

from models.white_mamba.extract_card_error_utils import extract_card_error_from_plan


def extract_query_stats_for_plan(
    query_stats_dict: Dict[str, List], plan: Dict, plan_id: int, node_depth: int = None
):
    # store query stat information
    query_stats_dict["sample_idxs"].append(plan_id)  # store the sample index
    query_stats_dict["num_tables"].append(plan["num_tables"])
    assert not isinstance(plan["selectivity"], list), (
        f"Selectivity should not be a list, but got {plan['selectivity']}"
    )
    query_stats_dict["selectivity"].append(plan["selectivity"])
    query_stats_dict["col_correlation"].append(plan["col_correlation"])
    query_stats_dict["scan_op"].append(plan["scan_op"])
    query_stats_dict["join_op"].append(plan["join_op"])
    query_stats_dict["graph_depth"].append(node_depth)
    query_stats_dict["sql"].append(plan["sql"])
    query_stats_dict["hint"].append(plan["hint"])

    # add card error
    mse, qerror = extract_card_error_from_plan(plan, mode="root")
    mse_avg, qerror_avg = extract_card_error_from_plan(plan, mode="avg")

    query_stats_dict["card_error_root_mse"].append(mse)
    query_stats_dict["card_error_root_qerror"].append(qerror)
    query_stats_dict["card_error_avg_mse"].append(mse_avg)
    query_stats_dict["card_error_avg_qerror"].append(qerror_avg)
