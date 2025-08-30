from typing import Dict, Tuple

from models.white_mamba.col_info_extractor import get_col_stats_for_idx_name


def assign_additional_plan_info(
    plan: dict, db_statistics: dict, db_schema_info: dict, is_parsed_plan: bool = True
):
    num_tables = None
    if is_parsed_plan:
        selectivity, scan_op, join_op, col_correlation = (
            _get_add_plan_stats_parsed_plan(plan, db_statistics, db_schema_info)
        )
    else:
        selectivity, scan_op, join_op, col_correlation, num_tables = (
            _get_add_plan_stats_json_plan(plan, db_statistics, db_schema_info)
        )

    plan["selectivity"] = selectivity
    plan["scan_op"] = scan_op
    plan["join_op"] = join_op
    plan["col_correlation"] = col_correlation

    if num_tables is not None:
        plan["num_tables"] = num_tables

    return


def _get_add_plan_stats_parsed_plan(
    plan: Dict, db_statistics: Dict, db_schema_info: Dict
) -> Tuple[float, str, str, float]:
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
            assert "index_name" in leaf_node["plan_parameters"], (
                f"Index name not found in {leaf_node}"
            )
            index_name = leaf_node["plan_parameters"]["index_name"]
            col_stats = get_col_stats_for_idx_name(
                index_name,
                base_table_name,
                db_schema_info,
                plan["database_id"],
                db_statistics,
            )
            col_correlation = col_stats["correlation"]
        else:
            assert scan_op == "Seq Scan", (
                f"Expected scan_op to be 'Seq Scan' or 'Index Scan', but got {scan_op}"
            )
            col_correlation = None

        # compute selectivity
        selectivity = out_card / base_table_size
        scan_op = scan_op
        join_op = None

    elif plan["num_tables"] == 2:
        # compute join info
        selectivity = None
        scan_op = None
        col_correlation = None

        def get_top_join_node(p):
            if len(p["children"]) == 2:
                return p
            elif len(p["children"]) == 1:
                return get_top_join_node(p["children"][0])
            else:
                raise ValueError("Invalid plan structure")

        join_node = get_top_join_node(plan)
        join_op = join_node["plan_parameters"]["op_name"]

    else:
        selectivity = None
        scan_op = None
        join_op = None
        col_correlation = None

    return selectivity, scan_op, join_op, col_correlation


def _get_add_plan_stats_json_plan(
    plan: Dict, db_statistics: Dict, db_schema_info: Dict
) -> Tuple[float, str, str, float]:
    # get number of tables
    def get_leaf_nodes(p):
        if "Plans" not in p or len(p["Plans"]) == 0:
            return [p]

        return [leaf for child in p["Plans"] for leaf in get_leaf_nodes(child)]

    analyze_plan = plan["analyze_plans"][0]["Plan"]
    leaf_nodes = get_leaf_nodes(analyze_plan)

    # initialize return vals
    selectivity = None
    scan_op = None
    join_op = None
    col_correlation = None

    if len(leaf_nodes) == 1:
        # this is scan

        leaf_node = leaf_nodes[0]

        scan_op = leaf_node["Node Type"]
        out_card = leaf_node["Actual Rows"]

        # compute selectivity
        if scan_op == "Result":
            # sometimes because of invalid filters/... there is not scan executed
            selectivity = 0
        else:
            assert "Relation Name" in leaf_node, (
                f"Relation Name not found in {leaf_node}"
            )
            base_table_name = leaf_node["Relation Name"]
            base_table_alias = leaf_node["Alias"]

            # set database id to 0 - this should be workload driven (i.e. only one database involved in this run)
            assert len(db_statistics) == 1, (
                f"Expected 1 database, but got {len(db_statistics)}"
            )
            database_id = 0

            base_table_id = None
            for i, entry in enumerate(db_statistics[database_id]["table_stats"]):
                if entry["relname"] == base_table_name:
                    base_table_id = i
                    break

            assert database_id in db_statistics
            assert "table_stats" in db_statistics[database_id]

            assert (
                "reltuples" in db_statistics[database_id]["table_stats"][base_table_id]
            ), (
                f"table_size not in {db_statistics[database_id]['table_stats'][base_table_id].keys()}"
            )

            base_table_size = db_statistics[database_id]["table_stats"][base_table_id][
                "reltuples"
            ]

            if scan_op in ["Index Scan", "Index Only Scan"]:
                assert "Index Name" in leaf_node, f"Index name not found in {leaf_node}"
                index_name = leaf_node["Index Name"]
                col_stats = get_col_stats_for_idx_name(
                    index_name,
                    base_table_name,
                    db_schema_info,
                    database_id,
                    db_statistics,
                )
                col_correlation = col_stats["correlation"]
            else:
                assert scan_op == "Seq Scan", (
                    f"Expected scan_op to be 'Seq Scan' or 'Index Scan', but got {scan_op}"
                )

            # compute selectivity
            selectivity = out_card / base_table_size

    elif len(leaf_nodes) == 2:
        # compute join info

        def get_top_join_node(p):
            assert "Plans" in p, f"Expected 'Plans' key in plan: {p}"
            if len(p["Plans"]) == 2:
                return p
            elif len(p["Plans"]) == 1:
                return get_top_join_node(p["Plans"][0])
            else:
                raise ValueError("Invalid plan structure")

        join_node = get_top_join_node(analyze_plan)
        join_op = join_node["Node Type"]

    return selectivity, scan_op, join_op, col_correlation, len(leaf_nodes)
