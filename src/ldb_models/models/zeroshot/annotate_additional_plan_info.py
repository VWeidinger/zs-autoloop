from models.white_mamba.col_info_extractor import get_col_stats_for_idx_name


def assign_additional_plan_info(plan: dict, db_statistics: dict, db_schema_info: dict):
    selectivity_list = []
    scan_op_list = []
    join_op_list = []
    col_correlation_list = []

    # extract additional info
    if plan["num_tables"] == 1:
        # compute selectivity
        def get_leaf_node(p):
            if len(p["children"]) == 0:
                return p
            else:
                return get_leaf_node(p["children"][0])

        leaf_node = get_leaf_node(plan)
        scan_op = leaf_node["plan_parameters"]["op_name"]
        out_card = leaf_node["plan_parameters"]["act_card"]

        # compute base table size
        base_table_name = leaf_node["plan_parameters"]["table_name"]
        base_table_id = leaf_node["plan_parameters"]["table_id"]

        assert "database_id" in plan, f"No database_id in plan: {plan}"
        assert plan["database_id"] in db_statistics
        assert "table_stats" in db_statistics[plan["database_id"]]

        assert (
            "reltuples"
            in db_statistics[plan["database_id"]]["table_stats"][base_table_id]
        ), (
            f"table_size not in {db_statistics[plan['database_id']]['table_stats'][base_table_id].keys()}"
        )

        base_table_size = db_statistics[plan["database_id"]]["table_stats"][
            base_table_id
        ]["reltuples"]

        if scan_op in ["Index Scan", "Index Only Scan"]:
            index_name = leaf_node["plan_parameters"]["index_name"]
            col_stats = get_col_stats_for_idx_name(
                index_name,
                leaf_node["plan_parameters"],
                db_schema_info,
                plan["database_id"],
                db_statistics,
            )
            col_correlation_list.append(col_stats["correlation"])
        else:
            assert scan_op == "Seq Scan", (
                f"Expected scan_op to be 'Seq Scan' or 'Index Scan', but got {scan_op}"
            )
            col_correlation_list.append(None)

        # compute selectivity
        selectivity_list.append(out_card / base_table_size)
        scan_op_list.append(scan_op)
        join_op_list.append(None)

    elif plan["num_tables"] == 2:
        # compute join info
        selectivity_list.append(None)
        scan_op_list.append(None)
        col_correlation_list.append(None)

        def get_top_join_node(p):
            if len(p["children"]) == 2:
                return p
            elif len(p["children"]) == 1:
                return get_top_join_node(p["children"][0])
            else:
                raise ValueError("Invalid plan structure")

        join_node = get_top_join_node(plan)
        join_op_list.append(join_node["plan_parameters"]["op_name"])

    else:
        selectivity_list.append(None)
        scan_op_list.append(None)
        join_op_list.append(None)
        col_correlation_list.append(None)

    plan["selectivity"] = selectivity_list
    plan["scan_op"] = scan_op_list
    plan["join_op"] = join_op_list
    plan["col_correlation"] = col_correlation_list
    return
