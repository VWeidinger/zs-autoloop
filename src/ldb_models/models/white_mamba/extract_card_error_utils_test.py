import json
import unittest

from ldb_models.models.white_mamba import extract_card_error_utils


class TestExtractCardErrorUtils(unittest.TestCase):
    def test_get_first_join_op_returns_join(self):
        plan = {
            "plan_parameters": {"op_name": "Hash Join", "act_card": 10, "est_card": 12},
            "children": [],
        }
        result = extract_card_error_utils.get_first_join_op(plan)
        self.assertEqual(result["plan_parameters"]["op_name"], "Hash Join")

    def test_get_first_join_op_raises_no_join_found(self):
        plan = {
            "plan_parameters": {"op_name": "Seq Scan", "act_card": 5, "est_card": 6},
            "children": [],
        }
        with self.assertRaises(extract_card_error_utils.NoJoinFoundError):
            extract_card_error_utils.get_first_join_op(plan)

    def test_get_first_scan_op_returns_scan(self):
        plan = {
            "plan_parameters": {"op_name": "Seq Scan", "act_card": 5, "est_card": 6},
            "children": [],
        }
        result = extract_card_error_utils.get_first_scan_op(plan)
        self.assertEqual(result["plan_parameters"]["op_name"], "Seq Scan")

    def test_get_first_scan_op_nested(self):
        plan = {
            "plan_parameters": {"op_name": "Hash Join", "act_card": 10, "est_card": 12},
            "children": [
                {
                    "plan_parameters": {
                        "op_name": "Seq Scan",
                        "act_card": 5,
                        "est_card": 6,
                    },
                    "children": [],
                }
            ],
        }
        result = extract_card_error_utils.get_first_scan_op(plan["children"][0])
        self.assertEqual(result["plan_parameters"]["op_name"], "Seq Scan")

    def test_extract_card_from_all_operators_simple(self):
        plan = {
            "plan_parameters": {"op_name": "Seq Scan", "act_card": 5, "est_card": 6},
            "children": [],
        }
        actual, estimated = extract_card_error_utils.extract_card_from_all_operators(
            plan
        )
        self.assertEqual(actual, [5])
        self.assertEqual(estimated, [6])

    def test_extract_card_from_all_operators_nested(self):
        plan = {
            "plan_parameters": {"op_name": "Hash Join", "act_card": 10, "est_card": 12},
            "children": [
                {
                    "plan_parameters": {
                        "op_name": "Seq Scan",
                        "act_card": 5,
                        "est_card": 6,
                    },
                    "children": [],
                }
            ],
        }
        actual, estimated = extract_card_error_utils.extract_card_from_all_operators(
            plan
        )
        self.assertEqual(actual, [10, 5])
        self.assertEqual(estimated, [12, 6])

    def test_extract_card_error_from_plan_root_join(self):
        plan = {
            "plan_parameters": {"op_name": "Hash Join", "act_card": 10, "est_card": 12},
            "children": [],
        }
        mse, q_error = extract_card_error_utils.extract_card_error_from_plan(
            plan, mode="root"
        )
        self.assertEqual(mse, 4)
        self.assertEqual(q_error, 1.2)

    def test_extract_card_error_from_plan_root_scan(self):
        plan = {
            "plan_parameters": {"op_name": "Seq Scan", "act_card": 5, "est_card": 10},
            "children": [],
        }
        mse, q_error = extract_card_error_utils.extract_card_error_from_plan(
            plan, mode="root"
        )
        self.assertEqual(mse, 25)
        self.assertEqual(q_error, 2.0)

    def test_extract_card_error_from_plan_avg(self):
        plan = {
            "plan_parameters": {"op_name": "Hash Join", "act_card": 10, "est_card": 12},
            "children": [
                {
                    "plan_parameters": {
                        "op_name": "Seq Scan",
                        "act_card": 5,
                        "est_card": 6,
                    },
                    "children": [],
                }
            ],
        }
        mse, q_error = extract_card_error_utils.extract_card_error_from_plan(
            plan, mode="avg"
        )
        expected_mse = ((12 - 10) ** 2 + (6 - 5) ** 2) / 2
        expected_q_error = (max(12 / 10, 10 / 12) + max(6 / 5, 5 / 6)) / 2
        self.assertAlmostEqual(mse, expected_mse, places=2)
        self.assertAlmostEqual(q_error, expected_q_error, places=2)

    def test_extract_card_error_from_plan_test(self):
        plan = json.loads(
            """{"plan_parameters": {"act_card": 1, "act_time": 1706.952, "act_loops": 1, "est_card": 1, "est_width": 8, "pg_cost": 115954.18, "est_loops": 1, "op_name": "Simple Aggregate", "output_columns": [{"aggregation": "COUNT", "columns": [], "udf_name": null, "udf_output": "False", "child_ref": false}], "act_children_card": 283812, "est_children_card": 564222, "above_udf_filter": false, "is_udf_filter": false}, "children": [{"plan_parameters": {"act_card": 283812, "act_time": 1682.246, "act_loops": 1, "est_card": 564222, "est_width": 0, "pg_cost": 114543.62, "est_loops": 1, "join": {"table_name1": "movie_info_idx", "table_alias1": null, "column_name1": "movie_id", "col_id1": 9, "operator": "=", "table_name2": "title", "table_alias2": null, "column_name2": "id", "col_id2": 66, "is_index_cond": false}, "op_name": "Hash Join", "act_children_card": 1419972639040, "est_children_card": 1426508283406, "above_udf_filter": false, "is_udf_filter": false}, "children": [{"plan_parameters": {"act_card": 920110, "act_time": 263.124, "act_loops": 1, "est_card": 922183, "est_width": 4, "pg_cost": 25185.44, "est_loops": 1, "table_name": "movie_info_idx", "table_alias": "movie_info_idx", "table_id": 15, "filter": {"table_name": "movie_info_idx", "table_alias": null, "col_name": "info_type_id", "col_id": 10, "operator": ">=", "literal": "100.0", "is_index_cond": false, "filter_complexity": 0}, "op_name": "Seq Scan", "output_columns": [{"aggregation": "None", "columns": [{"col_id": 8, "child_ref": null, "table_name": "movie_info_idx", "table_alias": null, "col_name": "id"}], "udf_name": null, "udf_output": "False", "child_ref": false}, {"aggregation": "None", "columns": [{"col_id": 9, "child_ref": null, "table_name": "movie_info_idx", "table_alias": null, "col_name": "movie_id"}], "udf_name": null, "udf_output": "False", "child_ref": false}, {"aggregation": "None", "columns": [{"col_id": 10, "child_ref": null, "table_name": "movie_info_idx", "table_alias": null, "col_name": "info_type_id"}], "udf_name": null, "udf_output": "False", "child_ref": false}, {"aggregation": "None", "columns": [{"col_id": 46, "child_ref": null, "table_name": "movie_info_idx", "table_alias": null, "col_name": "info"}], "udf_name": null, "udf_output": "False", "child_ref": false}, {"aggregation": "None", "columns": [{"col_id": 47, "child_ref": null, "table_name": "movie_info_idx", "table_alias": null, "col_name": "note"}], "udf_name": null, "udf_output": "False", "child_ref": false}], "act_children_card": 1, "est_children_card": 1, "above_udf_filter": false, "is_udf_filter": false}, "children": []}, {"plan_parameters": {"act_card": 1543264, "act_time": 1192.693, "act_loops": 1, "est_card": 1546882, "est_width": 4, "pg_cost": 67601.43, "est_loops": 1, "op_name": "Hash", "output_columns": [{"aggregation": "None", "columns": [{"col_id": 66, "child_ref": null, "table_name": "title", "table_alias": null, "col_name": "id"}], "udf_name": null, "udf_output": "False", "child_ref": false}], "act_children_card": 1543264, "est_children_card": 1546882, "above_udf_filter": false, "is_udf_filter": false}, "children": [{"plan_parameters": {"act_card": 1543264, "act_time": 754.385, "act_loops": 1, "est_card": 1546882, "est_width": 4, "pg_cost": 67601.43, "est_loops": 1, "table_name": "title", "table_alias": "title", "table_id": 22, "filter": {"table_name": "title", "table_alias": null, "col_name": "kind_id", "col_id": 69, "operator": "=", "literal": "7", "is_index_cond": false, "filter_complexity": 0}, "op_name": "Seq Scan", "output_columns": [{"aggregation": "None", "columns": [{"col_id": 66, "child_ref": null, "table_name": "title", "table_alias": null, "col_name": "id"}], "udf_name": null, "udf_output": "False", "child_ref": false}], "act_children_card": 1, "est_children_card": 1, "above_udf_filter": false, "is_udf_filter": false}, "children": []}]}]}], "sql": "SELECT COUNT(*) FROM title,movie_info_idx WHERE title.id=movie_info_idx.movie_id AND title.kind_id=7 AND movie_info_idx.info_type_id>99;", "dataset": "imdb", "plan_runtime_ms": 1748.752, "num_tables": 2, "num_filters": 2, "hint": "/*+HashJoin(movie_info_idx title)*/", "invalid_hint": false, "zero_card": false, "timeout": false, "timeout_threshold": 180000}"""
        )

        mse, q_error = extract_card_error_utils.extract_card_error_from_plan(
            plan, mode="root"
        )

        self.assertEqual(mse, 78629768100)  # No error since act_card == est_card
        self.assertAlmostEqual(q_error, 1.988, places=3)


if __name__ == "__main__":
    unittest.main()
