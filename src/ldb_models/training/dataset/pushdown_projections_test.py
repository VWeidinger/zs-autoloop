import unittest

from training.dataset.pushdown_projections import extract_cols_from_join, extract_cols_from_filter, \
    extract_cols_from_aggr, apply_projection_pushdown


class ProjectionPushdownTest(unittest.TestCase):

    def test_extract_cols_from_join(self):
        join = {
            "table_name1": "client",
            "column_name1": "client_id",
            "col_id1": 0,
            "operator": "=",
            "table_name2": "probehnuto_mimo_penezenku",
            "column_name2": "client_id",
            "col_id2": 11,
            "is_index_cond": False
        }

        expected_output = [
            {
                "table_name": "client",
                "col_name": "client_id",
                "col_id": 0,
                "child_ref": None
            },
            {
                "table_name": "probehnuto_mimo_penezenku",
                "col_name": "client_id",
                "col_id": 11,
                "child_ref": None
            }
        ]

        self.assertEqual(expected_output, extract_cols_from_join(join))

        join = {'operator': 'AND', 'is_index_cond': False, 'children': [
            {'table_name1': 'partsupp', 'column_name1': 'ps_partkey', 'col_id1': 23, 'operator': '=',
             'table_name2': 'lineitem', 'column_name2': 'l_partkey', 'col_id2': 46, 'is_index_cond': True},
            {'table_name1': 'partsupp', 'column_name1': 'ps_suppkey', 'col_id1': 24, 'operator': '=',
             'table_name2': 'lineitem', 'column_name2': 'l_suppkey', 'col_id2': 47, 'is_index_cond': True}]}

        expected_output = [
            {'table_name': 'partsupp', 'col_name': 'ps_partkey', 'col_id': 23, 'child_ref': None},
            {'table_name': 'lineitem', 'col_name': 'l_partkey', 'col_id': 46, 'child_ref': None},
            {'table_name': 'partsupp', 'col_name': 'ps_suppkey', 'col_id': 24, 'child_ref': None},
            {'table_name': 'lineitem', 'col_name': 'l_suppkey', 'col_id': 47, 'child_ref': None}
        ]
        self.assertEqual(expected_output, extract_cols_from_join(join))

    def test_extract_cols_from_filter(self):
        filter = {
            "operator": "AND",
            "is_index_cond": False,
            "children": [
                {
                    "table_name": "client",
                    "col_name": "kraj",
                    "col_id": 1,
                    "operator": "=",
                    "literal": "'St\u0159edo\u010desk\u00fd kraj'",
                    "is_index_cond": False,
                    "filter_complexity": 0
                },
                {
                    "table_name": "client",
                    "col_name": "obor",
                    "col_id": 2,
                    "operator": "=",
                    "literal": "'Vilma'",
                    "is_index_cond": False,
                    "filter_complexity": 0
                }
            ]
        }

        expected_output = [
            {
                "table_name": "client",
                "col_name": "kraj",
                "col_id": 1,
                "child_ref": None
            },
            {
                "table_name": "client",
                "col_name": "obor",
                "col_id": 2,
                "child_ref": None
            }
        ]
        self.assertEqual(expected_output, extract_cols_from_filter(filter))

    def test_extract_cols_from_aggr(self):
        out_cols = [
            {
                "aggregation": "SUM",
                "columns": [
                    {
                        "col_id": 3,
                        "child_ref": None,
                        "table_name": "dobito",
                        "col_name": "client_id"
                    },
                    {
                        "col_id": 9,
                        "child_ref": None,
                        "table_name": "probehnuto",
                        "col_name": "kc_proklikano"
                    }
                ],
                "udf_name": None,
                "udf_output": "False",
                "child_ref": False
            },
            {
                "aggregation": "None",
                "columns": [
                    {
                        "col_id": 5,
                        "child_ref": None,
                        "table_name": "dobito",
                        "col_name": "client_id"
                    },
                    {
                        "col_id": 9,
                        "child_ref": None,
                        "table_name": "probehnuto",
                        "col_name": "kc_proklikano"
                    }
                ],
                "udf_name": None,
                "udf_output": "False",
                "child_ref": False
            }
        ]

        expected_output = [
            {
                "col_id": 3,
                "table_name": "dobito",
                "col_name": "client_id",
                'child_ref': None
            },
            {
                "col_id": 9,
                "table_name": "probehnuto",
                "col_name": "kc_proklikano",
                'child_ref': None
            }
        ]
        self.assertEqual(expected_output, extract_cols_from_aggr(out_cols))

    def test_apply_projection_pushdown(self):
        dataset_stats = {
            'column_stats': [
                {
                    "tablename": "title",
                    "attname": "col1",
                },
                {
                    "tablename": "title",
                    "attname": "id",
                },
            ]
        }

        plan = {
            "plan_parameters": {
                "act_card": 1,
                "act_time": 617.294,
                "act_loops": 1,
                "est_card": 1,
                "est_width": 32,
                "pg_cost": 61534.5,
                "est_loops": 1,
                "op_name": "Finalize Aggregate",
                "output_columns": [
                    {
                        "aggregation": "SUM",
                        "columns": [
                            {
                                "col_id": 3,
                                "child_ref": None,
                                "table_name": "dobito",
                                "col_name": "client_id"
                            },
                            {
                                "col_id": 9,
                                "child_ref": None,
                                "table_name": "probehnuto",
                                "col_name": "kc_proklikano"
                            }
                        ],
                        "udf_name": None,
                        "udf_output": "False",
                        "child_ref": False
                    }
                ],
                "act_children_card": 3,
                "est_children_card": 2,
                "above_udf_filter": False,
                "is_udf_filter": False
            },
            "children": [
                {
                    "plan_parameters": {
                        "act_card": 3,
                        "act_time": 617.269,
                        "act_loops": 1,
                        "est_card": 2,
                        "est_width": 32,
                        "pg_cost": 61534.48,
                        "est_loops": 1,
                        "op_name": "Gather",
                        "output_columns": [
                            {
                                "aggregation": "SUM",
                                "columns": [
                                    {
                                        "col_id": 3,
                                        "child_ref": None,
                                        "table_name": "dobito",
                                        "col_name": "client_id"
                                    },
                                    {
                                        "col_id": 9,
                                        "child_ref": None,
                                        "table_name": "probehnuto",
                                        "col_name": "kc_proklikano"
                                    }
                                ],
                                "udf_name": None,
                                "udf_output": "False",
                                "child_ref": False
                            }
                        ],
                        "act_children_card": 3,
                        "est_children_card": 1,
                        "above_udf_filter": False,
                        "is_udf_filter": False
                    },
                    "children": [
                        {
                            "plan_parameters": {
                                "act_card": 3,
                                "act_time": 1793.181,
                                "act_loops": 3,
                                "est_card": 1,
                                "est_width": 32,
                                "pg_cost": 60534.28,
                                "est_loops": 1,
                                "op_name": "Partial Aggregate",
                                "output_columns": [
                                    {
                                        "aggregation": "SUM",
                                        "columns": [
                                            {
                                                "col_id": 3,
                                                "child_ref": None,
                                                "table_name": "dobito",
                                                "col_name": "client_id"
                                            },
                                            {
                                                "col_id": 9,
                                                "child_ref": None,
                                                "table_name": "probehnuto",
                                                "col_name": "kc_proklikano"
                                            }
                                        ],
                                        "udf_name": None,
                                        "udf_output": "False",
                                        "child_ref": False
                                    }
                                ],
                                "act_children_card": 47880,
                                "est_children_card": 8629,
                                "above_udf_filter": False,
                                "is_udf_filter": False
                            },
                            "children": [
                                {
                                    "plan_parameters": {
                                        "act_card": 47880,
                                        "act_time": 1774.071,
                                        "act_loops": 3,
                                        "est_card": 8629,
                                        "est_width": 11,
                                        "pg_cost": 60469.55,
                                        "est_loops": 1,
                                        "join": {
                                            "table_name1": "client",
                                            "column_name1": "client_id",
                                            "col_id1": 0,
                                            "operator": "=",
                                            "table_name2": "probehnuto_mimo_penezenku",
                                            "column_name2": "client_id",
                                            "col_id2": 11,
                                            "is_index_cond": False
                                        },
                                        "op_name": "Nested Loop",
                                        "output_columns": [
                                            {
                                                "aggregation": "None",
                                                "columns": [
                                                    {
                                                        "col_id": 3,
                                                        "child_ref": None,
                                                        "table_name": "dobito",
                                                        "col_name": "client_id"
                                                    }
                                                ],
                                                "udf_name": None,
                                                "udf_output": "False",
                                                "child_ref": False
                                            },
                                            {
                                                "aggregation": "None",
                                                "columns": [
                                                    {
                                                        "col_id": 9,
                                                        "child_ref": None,
                                                        "table_name": "probehnuto",
                                                        "col_name": "kc_proklikano"
                                                    }
                                                ],
                                                "udf_name": None,
                                                "udf_output": "False",
                                                "child_ref": False
                                            }
                                        ],
                                        "act_children_card": 258952464,
                                        "est_children_card": 9436752,
                                        "above_udf_filter": False,
                                        "is_udf_filter": False
                                    },
                                    "children": [
                                        {
                                            "plan_parameters": {
                                                "act_card": 5364,
                                                "act_time": 1740.372,
                                                "act_loops": 3,
                                                "est_card": 852,
                                                "est_width": 19,
                                                "pg_cost": 59511.94,
                                                "est_loops": 1,
                                                "join": {
                                                    "table_name1": "probehnuto",
                                                    "column_name1": "client_id",
                                                    "col_id1": 7,
                                                    "operator": "=",
                                                    "table_name2": "client",
                                                    "column_name2": "client_id",
                                                    "col_id2": 0,
                                                    "is_index_cond": False
                                                },
                                                "op_name": "Hash Join",
                                                "output_columns": [
                                                    {
                                                        "aggregation": "None",
                                                        "columns": [
                                                            {
                                                                "col_id": 0,
                                                                "child_ref": None,
                                                                "table_name": "client",
                                                                "col_name": "client_id"
                                                            }
                                                        ],
                                                        "udf_name": None,
                                                        "udf_output": "False",
                                                        "child_ref": False
                                                    },
                                                    {
                                                        "aggregation": "None",
                                                        "columns": [
                                                            {
                                                                "col_id": 9,
                                                                "child_ref": None,
                                                                "table_name": "probehnuto",
                                                                "col_name": "kc_proklikano"
                                                            }
                                                        ],
                                                        "udf_name": None,
                                                        "udf_output": "False",
                                                        "child_ref": False
                                                    },
                                                    {
                                                        "aggregation": "None",
                                                        "columns": [
                                                            {
                                                                "col_id": 7,
                                                                "child_ref": None,
                                                                "table_name": "probehnuto",
                                                                "col_name": "client_id"
                                                            }
                                                        ],
                                                        "udf_name": None,
                                                        "udf_output": "False",
                                                        "child_ref": False
                                                    },
                                                    {
                                                        "aggregation": "None",
                                                        "columns": [
                                                            {
                                                                "col_id": 3,
                                                                "child_ref": None,
                                                                "table_name": "dobito",
                                                                "col_name": "client_id"
                                                            }
                                                        ],
                                                        "udf_name": None,
                                                        "udf_output": "False",
                                                        "child_ref": False
                                                    }
                                                ],
                                                "act_children_card": 1103129496,
                                                "est_children_card": 125285012,
                                                "above_udf_filter": False,
                                                "is_udf_filter": False
                                            },
                                            "children": [
                                                {
                                                    "plan_parameters": {
                                                        "act_card": 95163,
                                                        "act_time": 657.297,
                                                        "act_loops": 3,
                                                        "est_card": 40207,
                                                        "est_width": 11,
                                                        "pg_cost": 34369.98,
                                                        "est_loops": 1,
                                                        "table_name": "probehnuto",
                                                        "table_id": 2,
                                                        "filter": {
                                                            "table_name": "probehnuto",
                                                            "col_name": "month_year_datum_transakce",
                                                            "col_id": 8,
                                                            "operator": "=",
                                                            "literal": "'2015-02-01'",
                                                            "is_index_cond": False,
                                                            "filter_complexity": 0
                                                        },
                                                        "op_name": "Seq Scan",
                                                        "output_columns": [
                                                            {
                                                                "aggregation": "None",
                                                                "columns": [
                                                                    {
                                                                        "col_id": 7,
                                                                        "child_ref": None,
                                                                        "table_name": "probehnuto",
                                                                        "col_name": "client_id"
                                                                    }
                                                                ],
                                                                "udf_name": None,
                                                                "udf_output": "False",
                                                                "child_ref": False
                                                            },
                                                            {
                                                                "aggregation": "None",
                                                                "columns": [
                                                                    {
                                                                        "col_id": 8,
                                                                        "child_ref": None,
                                                                        "table_name": "probehnuto",
                                                                        "col_name": "month_year_datum_transakce"
                                                                    }
                                                                ],
                                                                "udf_name": None,
                                                                "udf_output": "False",
                                                                "child_ref": False
                                                            },
                                                            {
                                                                "aggregation": "None",
                                                                "columns": [
                                                                    {
                                                                        "col_id": 10,
                                                                        "child_ref": None,
                                                                        "table_name": "probehnuto",
                                                                        "col_name": "sluzba"
                                                                    }
                                                                ],
                                                                "udf_name": None,
                                                                "udf_output": "False",
                                                                "child_ref": False
                                                            },
                                                            {
                                                                "aggregation": "None",
                                                                "columns": [
                                                                    {
                                                                        "col_id": 9,
                                                                        "child_ref": None,
                                                                        "table_name": "probehnuto",
                                                                        "col_name": "kc_proklikano"
                                                                    }
                                                                ],
                                                                "udf_name": None,
                                                                "udf_output": "False",
                                                                "child_ref": False
                                                            }
                                                        ],
                                                        "act_children_card": 1,
                                                        "est_children_card": 1,
                                                        "above_udf_filter": False,
                                                        "is_udf_filter": False
                                                    },
                                                    "children": []
                                                },
                                                {
                                                    "plan_parameters": {
                                                        "act_card": 11592,
                                                        "act_time": 1067.835,
                                                        "act_loops": 3,
                                                        "est_card": 3116,
                                                        "est_width": 8,
                                                        "pg_cost": 24591.9,
                                                        "est_loops": 1,
                                                        "op_name": "Hash",
                                                        "output_columns": [
                                                            {
                                                                "aggregation": "None",
                                                                "columns": [
                                                                    {
                                                                        "col_id": 0,
                                                                        "child_ref": None,
                                                                        "table_name": "client",
                                                                        "col_name": "client_id"
                                                                    }
                                                                ],
                                                                "udf_name": None,
                                                                "udf_output": "False",
                                                                "child_ref": False
                                                            },
                                                            {
                                                                "aggregation": "None",
                                                                "columns": [
                                                                    {
                                                                        "col_id": 3,
                                                                        "child_ref": None,
                                                                        "table_name": "dobito",
                                                                        "col_name": "client_id"
                                                                    }
                                                                ],
                                                                "udf_name": None,
                                                                "udf_output": "False",
                                                                "child_ref": False
                                                            }
                                                        ],
                                                        "act_children_card": 11592,
                                                        "est_children_card": 3116,
                                                        "above_udf_filter": False,
                                                        "is_udf_filter": False
                                                    },
                                                    "children": [
                                                        {
                                                            "plan_parameters": {
                                                                "act_card": 11592,
                                                                "act_time": 1062.789,
                                                                "act_loops": 3,
                                                                "est_card": 3116,
                                                                "est_width": 8,
                                                                "pg_cost": 24591.9,
                                                                "est_loops": 1,
                                                                "join": {
                                                                    "table_name1": "dobito",
                                                                    "column_name1": "client_id",
                                                                    "col_id1": 3,
                                                                    "operator": "=",
                                                                    "table_name2": "client",
                                                                    "column_name2": "client_id",
                                                                    "col_id2": 0,
                                                                    "is_index_cond": False
                                                                },
                                                                "op_name": "Hash Join",
                                                                "output_columns": [
                                                                    {
                                                                        "aggregation": "None",
                                                                        "columns": [
                                                                            {
                                                                                "col_id": 0,
                                                                                "child_ref": None,
                                                                                "table_name": "client",
                                                                                "col_name": "client_id"
                                                                            }
                                                                        ],
                                                                        "udf_name": None,
                                                                        "udf_output": "False",
                                                                        "child_ref": False
                                                                    },
                                                                    {
                                                                        "aggregation": "None",
                                                                        "columns": [
                                                                            {
                                                                                "col_id": 3,
                                                                                "child_ref": None,
                                                                                "table_name": "dobito",
                                                                                "col_name": "client_id"
                                                                            }
                                                                        ],
                                                                        "udf_name": None,
                                                                        "udf_output": "False",
                                                                        "child_ref": False
                                                                    }
                                                                ],
                                                                "act_children_card": 4250725128,
                                                                "est_children_card": 457889796,
                                                                "above_udf_filter": False,
                                                                "is_udf_filter": False
                                                            },
                                                            "children": [
                                                                {
                                                                    "plan_parameters": {
                                                                        "act_card": 3326076,
                                                                        "act_time": 564.732,
                                                                        "act_loops": 3,
                                                                        "est_card": 1108692,
                                                                        "est_width": 4,
                                                                        "pg_cost": 18451.92,
                                                                        "est_loops": 1,
                                                                        "table_name": "dobito",
                                                                        "table_id": 1,
                                                                        "op_name": "Seq Scan",
                                                                        "output_columns": [
                                                                            {
                                                                                "aggregation": "None",
                                                                                "columns": [
                                                                                    {
                                                                                        "col_id": 3,
                                                                                        "child_ref": None,
                                                                                        "table_name": "dobito",
                                                                                        "col_name": "client_id"
                                                                                    }
                                                                                ],
                                                                                "udf_name": None,
                                                                                "udf_output": "False",
                                                                                "child_ref": False
                                                                            },
                                                                            {
                                                                                "aggregation": "None",
                                                                                "columns": [
                                                                                    {
                                                                                        "col_id": 4,
                                                                                        "child_ref": None,
                                                                                        "table_name": "dobito",
                                                                                        "col_name": "month_year_datum_transakce"
                                                                                    }
                                                                                ],
                                                                                "udf_name": None,
                                                                                "udf_output": "False",
                                                                                "child_ref": False
                                                                            },
                                                                            {
                                                                                "aggregation": "None",
                                                                                "columns": [
                                                                                    {
                                                                                        "col_id": 5,
                                                                                        "child_ref": None,
                                                                                        "table_name": "dobito",
                                                                                        "col_name": "sluzba"
                                                                                    }
                                                                                ],
                                                                                "udf_name": None,
                                                                                "udf_output": "False",
                                                                                "child_ref": False
                                                                            },
                                                                            {
                                                                                "aggregation": "None",
                                                                                "columns": [
                                                                                    {
                                                                                        "col_id": 6,
                                                                                        "child_ref": None,
                                                                                        "table_name": "dobito",
                                                                                        "col_name": "kc_dobito"
                                                                                    }
                                                                                ],
                                                                                "udf_name": None,
                                                                                "udf_output": "False",
                                                                                "child_ref": False
                                                                            }
                                                                        ],
                                                                        "act_children_card": 1,
                                                                        "est_children_card": 1,
                                                                        "above_udf_filter": False,
                                                                        "is_udf_filter": False
                                                                    },
                                                                    "children": []
                                                                },
                                                                {
                                                                    "plan_parameters": {
                                                                        "act_card": 1278,
                                                                        "act_time": 78.054,
                                                                        "act_loops": 3,
                                                                        "est_card": 413,
                                                                        "est_width": 4,
                                                                        "pg_cost": 3224.41,
                                                                        "est_loops": 1,
                                                                        "op_name": "Hash",
                                                                        "output_columns": [
                                                                            {
                                                                                "aggregation": "None",
                                                                                "columns": [
                                                                                    {
                                                                                        "col_id": 0,
                                                                                        "child_ref": None,
                                                                                        "table_name": "client",
                                                                                        "col_name": "client_id"
                                                                                    }
                                                                                ],
                                                                                "udf_name": None,
                                                                                "udf_output": "False",
                                                                                "child_ref": False
                                                                            }
                                                                        ],
                                                                        "act_children_card": 1278,
                                                                        "est_children_card": 413,
                                                                        "above_udf_filter": False,
                                                                        "is_udf_filter": False
                                                                    },
                                                                    "children": [
                                                                        {
                                                                            "plan_parameters": {
                                                                                "act_card": 1278,
                                                                                "act_time": 77.541,
                                                                                "act_loops": 3,
                                                                                "est_card": 413,
                                                                                "est_width": 4,
                                                                                "pg_cost": 3224.41,
                                                                                "est_loops": 1,
                                                                                "table_name": "client",
                                                                                "table_id": 0,
                                                                                "filter": {
                                                                                    "operator": "AND",
                                                                                    "is_index_cond": False,
                                                                                    "children": [
                                                                                        {
                                                                                            "table_name": "client",
                                                                                            "col_name": "kraj",
                                                                                            "col_id": 1,
                                                                                            "operator": "=",
                                                                                            "literal": "'St\u0159edo\u010desk\u00fd kraj'",
                                                                                            "is_index_cond": False,
                                                                                            "filter_complexity": 0
                                                                                        },
                                                                                        {
                                                                                            "table_name": "client",
                                                                                            "col_name": "obor",
                                                                                            "col_id": 2,
                                                                                            "operator": "=",
                                                                                            "literal": "'Vilma'",
                                                                                            "is_index_cond": False,
                                                                                            "filter_complexity": 0
                                                                                        }
                                                                                    ]
                                                                                },
                                                                                "op_name": "Seq Scan",
                                                                                "output_columns": [
                                                                                    {
                                                                                        "aggregation": "None",
                                                                                        "columns": [
                                                                                            {
                                                                                                "col_id": 0,
                                                                                                "child_ref": None,
                                                                                                "table_name": "client",
                                                                                                "col_name": "client_id"
                                                                                            }
                                                                                        ],
                                                                                        "udf_name": None,
                                                                                        "udf_output": "False",
                                                                                        "child_ref": False
                                                                                    }
                                                                                ],
                                                                                "act_children_card": 1,
                                                                                "est_children_card": 1,
                                                                                "above_udf_filter": False,
                                                                                "is_udf_filter": False
                                                                            },
                                                                            "children": []
                                                                        }
                                                                    ]
                                                                }
                                                            ]
                                                        }
                                                    ]
                                                }
                                            ]
                                        },
                                        {
                                            "plan_parameters": {
                                                "act_card": 48276,
                                                "act_time": 26.82,
                                                "act_loops": 5364,
                                                "est_card": 11076,
                                                "est_width": 4,
                                                "pg_cost": 0.96,
                                                "est_loops": 852,
                                                "table_name": "probehnuto_mimo_penezenku",
                                                "table_id": 3,
                                                "filter": {
                                                    "operator": "AND",
                                                    "is_index_cond": False,
                                                    "children": [
                                                        {
                                                            "table_name": "probehnuto_mimo_penezenku",
                                                            "col_name": "Month_Year",
                                                            "col_id": 12,
                                                            "operator": "!=",
                                                            "literal": "'2013-10-01_0'",
                                                            "is_index_cond": False,
                                                            "filter_complexity": 0
                                                        },
                                                        {
                                                            "table_name": "probehnuto_mimo_penezenku",
                                                            "col_name": "client_id",
                                                            "col_id": 11,
                                                            "operator": ">=",
                                                            "literal": "97288",
                                                            "is_index_cond": True,
                                                            "filter_complexity": 0
                                                        }
                                                    ]
                                                },
                                                "op_name": "Index Only Scan",
                                                "output_columns": [

                                                ],
                                                "act_children_card": 1,
                                                "est_children_card": 1,
                                                "above_udf_filter": False,
                                                "is_udf_filter": False
                                            },
                                            "children": []
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }]}

        expected_plan = {
            "plan_parameters": {
                "act_card": 1,
                "act_time": 617.294,
                "act_loops": 1,
                "est_card": 1,
                "est_width": 32,
                "pg_cost": 61534.5,
                "est_loops": 1,
                "op_name": "Finalize Aggregate",
                "output_columns": [
                    {
                        "aggregation": "SUM",
                        "columns": [
                            {
                                "col_id": 3,
                                "child_ref": None,
                                "table_name": "dobito",
                                "col_name": "client_id"
                            },
                            {
                                "col_id": 9,
                                "child_ref": None,
                                "table_name": "probehnuto",
                                "col_name": "kc_proklikano"
                            }
                        ],
                        "udf_name": None,
                        "udf_output": "False",
                        "child_ref": False
                    }
                ],
                "act_children_card": 3,
                "est_children_card": 2,
                "above_udf_filter": False,
                "is_udf_filter": False
            },
            "children": [
                {
                    "plan_parameters": {
                        "act_card": 3,
                        "act_time": 617.269,
                        "act_loops": 1,
                        "est_card": 2,
                        "est_width": 32,
                        "pg_cost": 61534.48,
                        "est_loops": 1,
                        "op_name": "Gather",
                        "output_columns": [
                            {
                                "aggregation": "SUM",
                                "columns": [
                                    {
                                        "col_id": 3,
                                        "child_ref": None,
                                        "table_name": "dobito",
                                        "col_name": "client_id"
                                    },
                                    {
                                        "col_id": 9,
                                        "child_ref": None,
                                        "table_name": "probehnuto",
                                        "col_name": "kc_proklikano"
                                    }
                                ],
                                "udf_name": None,
                                "udf_output": "False",
                                "child_ref": False
                            }
                        ],
                        "act_children_card": 3,
                        "est_children_card": 1,
                        "above_udf_filter": False,
                        "is_udf_filter": False
                    },
                    "children": [
                        {
                            "plan_parameters": {
                                "act_card": 3,
                                "act_time": 1793.181,
                                "act_loops": 3,
                                "est_card": 1,
                                "est_width": 32,
                                "pg_cost": 60534.28,
                                "est_loops": 1,
                                "op_name": "Partial Aggregate",
                                "output_columns": [
                                    {
                                        "aggregation": "SUM",
                                        "columns": [
                                            {
                                                "col_id": 3,
                                                "child_ref": None,
                                                "table_name": "dobito",
                                                "col_name": "client_id"
                                            },
                                            {
                                                "col_id": 9,
                                                "child_ref": None,
                                                "table_name": "probehnuto",
                                                "col_name": "kc_proklikano"
                                            }
                                        ],
                                        "udf_name": None,
                                        "udf_output": "False",
                                        "child_ref": False
                                    }
                                ],
                                "act_children_card": 47880,
                                "est_children_card": 8629,
                                "above_udf_filter": False,
                                "is_udf_filter": False
                            },
                            "children": [
                                {
                                    "plan_parameters": {
                                        "act_card": 47880,
                                        "act_time": 1774.071,
                                        "act_loops": 3,
                                        "est_card": 8629,
                                        "est_width": 11,
                                        "pg_cost": 60469.55,
                                        "est_loops": 1,
                                        "join": {
                                            "table_name1": "client",
                                            "column_name1": "client_id",
                                            "col_id1": 0,
                                            "operator": "=",
                                            "table_name2": "probehnuto_mimo_penezenku",
                                            "column_name2": "client_id",
                                            "col_id2": 11,
                                            "is_index_cond": False
                                        },
                                        "op_name": "Nested Loop",
                                        "output_columns": [
                                            {
                                                "aggregation": "None",
                                                "columns": [
                                                    {
                                                        "col_id": 3,
                                                        "child_ref": None,
                                                        "table_name": "dobito",
                                                        "col_name": "client_id"
                                                    }
                                                ],
                                                "udf_name": None,
                                                "udf_output": "False",
                                                "child_ref": False
                                            },
                                            {
                                                "aggregation": "None",
                                                "columns": [
                                                    {
                                                        "col_id": 9,
                                                        "child_ref": None,
                                                        "table_name": "probehnuto",
                                                        "col_name": "kc_proklikano"
                                                    }
                                                ],
                                                "udf_name": None,
                                                "udf_output": "False",
                                                "child_ref": False
                                            }
                                        ],
                                        "act_children_card": 258952464,
                                        "est_children_card": 9436752,
                                        "above_udf_filter": False,
                                        "is_udf_filter": False
                                    },
                                    "children": [
                                        {
                                            "plan_parameters": {
                                                "act_card": 5364,
                                                "act_time": 1740.372,
                                                "act_loops": 3,
                                                "est_card": 852,
                                                "est_width": 19,
                                                "pg_cost": 59511.94,
                                                "est_loops": 1,
                                                "join": {
                                                    "table_name1": "probehnuto",
                                                    "column_name1": "client_id",
                                                    "col_id1": 7,
                                                    "operator": "=",
                                                    "table_name2": "client",
                                                    "column_name2": "client_id",
                                                    "col_id2": 0,
                                                    "is_index_cond": False
                                                },
                                                "op_name": "Hash Join",
                                                "output_columns": [
                                                    {
                                                        "aggregation": "None",
                                                        "columns": [
                                                            {
                                                                "col_id": 0,
                                                                "child_ref": None,
                                                                "table_name": "client",
                                                                "col_name": "client_id"
                                                            }
                                                        ],
                                                        "udf_name": None,
                                                        "udf_output": "False",
                                                        "child_ref": False
                                                    },
                                                    {
                                                        "aggregation": "None",
                                                        "columns": [
                                                            {
                                                                "col_id": 3,
                                                                "child_ref": None,
                                                                "table_name": "dobito",
                                                                "col_name": "client_id"
                                                            }
                                                        ],
                                                        "udf_name": None,
                                                        "udf_output": "False",
                                                        "child_ref": False
                                                    },
                                                    {
                                                        "aggregation": "None",
                                                        "columns": [
                                                            {
                                                                "col_id": 9,
                                                                "child_ref": None,
                                                                "table_name": "probehnuto",
                                                                "col_name": "kc_proklikano"
                                                            }
                                                        ],
                                                        "udf_name": None,
                                                        "udf_output": "False",
                                                        "child_ref": False
                                                    }
                                                ],
                                                "act_children_card": 1103129496,
                                                "est_children_card": 125285012,
                                                "above_udf_filter": False,
                                                "is_udf_filter": False
                                            },
                                            "children": [
                                                {
                                                    "plan_parameters": {
                                                        "act_card": 95163,
                                                        "act_time": 657.297,
                                                        "act_loops": 3,
                                                        "est_card": 40207,
                                                        "est_width": 11,
                                                        "pg_cost": 34369.98,
                                                        "est_loops": 1,
                                                        "table_name": "probehnuto",
                                                        "table_id": 2,
                                                        "filter": {
                                                            "table_name": "probehnuto",
                                                            "col_name": "month_year_datum_transakce",
                                                            "col_id": 8,
                                                            "operator": "=",
                                                            "literal": "'2015-02-01'",
                                                            "is_index_cond": False,
                                                            "filter_complexity": 0
                                                        },
                                                        "op_name": "Seq Scan",
                                                        "output_columns": [
                                                            {
                                                                "aggregation": "None",
                                                                "columns": [
                                                                    {
                                                                        "col_id": 7,
                                                                        "child_ref": None,
                                                                        "table_name": "probehnuto",
                                                                        "col_name": "client_id"
                                                                    }
                                                                ],
                                                                "udf_name": None,
                                                                "udf_output": "False",
                                                                "child_ref": False
                                                            },
                                                            {
                                                                "aggregation": "None",
                                                                "columns": [
                                                                    {
                                                                        "col_id": 9,
                                                                        "child_ref": None,
                                                                        "table_name": "probehnuto",
                                                                        "col_name": "kc_proklikano"
                                                                    }
                                                                ],
                                                                "udf_name": None,
                                                                "udf_output": "False",
                                                                "child_ref": False
                                                            }
                                                        ],
                                                        "act_children_card": 1,
                                                        "est_children_card": 1,
                                                        "above_udf_filter": False,
                                                        "is_udf_filter": False
                                                    },
                                                    "children": []
                                                },
                                                {
                                                    "plan_parameters": {
                                                        "act_card": 11592,
                                                        "act_time": 1067.835,
                                                        "act_loops": 3,
                                                        "est_card": 3116,
                                                        "est_width": 8,
                                                        "pg_cost": 24591.9,
                                                        "est_loops": 1,
                                                        "op_name": "Hash",
                                                        "output_columns": [
                                                            {
                                                                "aggregation": "None",
                                                                "columns": [
                                                                    {
                                                                        "col_id": 0,
                                                                        "child_ref": None,
                                                                        "table_name": "client",
                                                                        "col_name": "client_id"
                                                                    }
                                                                ],
                                                                "udf_name": None,
                                                                "udf_output": "False",
                                                                "child_ref": False
                                                            },
                                                            {
                                                                "aggregation": "None",
                                                                "columns": [
                                                                    {
                                                                        "col_id": 3,
                                                                        "child_ref": None,
                                                                        "table_name": "dobito",
                                                                        "col_name": "client_id"
                                                                    }
                                                                ],
                                                                "udf_name": None,
                                                                "udf_output": "False",
                                                                "child_ref": False
                                                            }
                                                        ],
                                                        "act_children_card": 11592,
                                                        "est_children_card": 3116,
                                                        "above_udf_filter": False,
                                                        "is_udf_filter": False
                                                    },
                                                    "children": [
                                                        {
                                                            "plan_parameters": {
                                                                "act_card": 11592,
                                                                "act_time": 1062.789,
                                                                "act_loops": 3,
                                                                "est_card": 3116,
                                                                "est_width": 8,
                                                                "pg_cost": 24591.9,
                                                                "est_loops": 1,
                                                                "join": {
                                                                    "table_name1": "dobito",
                                                                    "column_name1": "client_id",
                                                                    "col_id1": 3,
                                                                    "operator": "=",
                                                                    "table_name2": "client",
                                                                    "column_name2": "client_id",
                                                                    "col_id2": 0,
                                                                    "is_index_cond": False
                                                                },
                                                                "op_name": "Hash Join",
                                                                "output_columns": [
                                                                    {
                                                                        "aggregation": "None",
                                                                        "columns": [
                                                                            {
                                                                                "col_id": 0,
                                                                                "child_ref": None,
                                                                                "table_name": "client",
                                                                                "col_name": "client_id"
                                                                            }
                                                                        ],
                                                                        "udf_name": None,
                                                                        "udf_output": "False",
                                                                        "child_ref": False
                                                                    },
                                                                    {
                                                                        "aggregation": "None",
                                                                        "columns": [
                                                                            {
                                                                                "col_id": 3,
                                                                                "child_ref": None,
                                                                                "table_name": "dobito",
                                                                                "col_name": "client_id"
                                                                            }
                                                                        ],
                                                                        "udf_name": None,
                                                                        "udf_output": "False",
                                                                        "child_ref": False
                                                                    }
                                                                ],
                                                                "act_children_card": 4250725128,
                                                                "est_children_card": 457889796,
                                                                "above_udf_filter": False,
                                                                "is_udf_filter": False
                                                            },
                                                            "children": [
                                                                {
                                                                    "plan_parameters": {
                                                                        "act_card": 3326076,
                                                                        "act_time": 564.732,
                                                                        "act_loops": 3,
                                                                        "est_card": 1108692,
                                                                        "est_width": 4,
                                                                        "pg_cost": 18451.92,
                                                                        "est_loops": 1,
                                                                        "table_name": "dobito",
                                                                        "table_id": 1,
                                                                        "op_name": "Seq Scan",
                                                                        "output_columns": [
                                                                            {
                                                                                "aggregation": "None",
                                                                                "columns": [
                                                                                    {
                                                                                        "col_id": 3,
                                                                                        "child_ref": None,
                                                                                        "table_name": "dobito",
                                                                                        "col_name": "client_id"
                                                                                    }
                                                                                ],
                                                                                "udf_name": None,
                                                                                "udf_output": "False",
                                                                                "child_ref": False
                                                                            },
                                                                        ],
                                                                        "act_children_card": 1,
                                                                        "est_children_card": 1,
                                                                        "above_udf_filter": False,
                                                                        "is_udf_filter": False
                                                                    },
                                                                    "children": []
                                                                },
                                                                {
                                                                    "plan_parameters": {
                                                                        "act_card": 1278,
                                                                        "act_time": 78.054,
                                                                        "act_loops": 3,
                                                                        "est_card": 413,
                                                                        "est_width": 4,
                                                                        "pg_cost": 3224.41,
                                                                        "est_loops": 1,
                                                                        "op_name": "Hash",
                                                                        "output_columns": [
                                                                            {
                                                                                "aggregation": "None",
                                                                                "columns": [
                                                                                    {
                                                                                        "col_id": 0,
                                                                                        "child_ref": None,
                                                                                        "table_name": "client",
                                                                                        "col_name": "client_id"
                                                                                    }
                                                                                ],
                                                                                "udf_name": None,
                                                                                "udf_output": "False",
                                                                                "child_ref": False
                                                                            }
                                                                        ],
                                                                        "act_children_card": 1278,
                                                                        "est_children_card": 413,
                                                                        "above_udf_filter": False,
                                                                        "is_udf_filter": False
                                                                    },
                                                                    "children": [
                                                                        {
                                                                            "plan_parameters": {
                                                                                "act_card": 1278,
                                                                                "act_time": 77.541,
                                                                                "act_loops": 3,
                                                                                "est_card": 413,
                                                                                "est_width": 4,
                                                                                "pg_cost": 3224.41,
                                                                                "est_loops": 1,
                                                                                "table_name": "client",
                                                                                "table_id": 0,
                                                                                "filter": {
                                                                                    "operator": "AND",
                                                                                    "is_index_cond": False,
                                                                                    "children": [
                                                                                        {
                                                                                            "table_name": "client",
                                                                                            "col_name": "kraj",
                                                                                            "col_id": 1,
                                                                                            "operator": "=",
                                                                                            "literal": "'St\u0159edo\u010desk\u00fd kraj'",
                                                                                            "is_index_cond": False,
                                                                                            "filter_complexity": 0
                                                                                        },
                                                                                        {
                                                                                            "table_name": "client",
                                                                                            "col_name": "obor",
                                                                                            "col_id": 2,
                                                                                            "operator": "=",
                                                                                            "literal": "'Vilma'",
                                                                                            "is_index_cond": False,
                                                                                            "filter_complexity": 0
                                                                                        }
                                                                                    ]
                                                                                },
                                                                                "op_name": "Seq Scan",
                                                                                "output_columns": [
                                                                                    {
                                                                                        "aggregation": "None",
                                                                                        "columns": [
                                                                                            {
                                                                                                "col_id": 0,
                                                                                                "child_ref": None,
                                                                                                "table_name": "client",
                                                                                                "col_name": "client_id"
                                                                                            }
                                                                                        ],
                                                                                        "udf_name": None,
                                                                                        "udf_output": "False",
                                                                                        "child_ref": False
                                                                                    }
                                                                                ],
                                                                                "act_children_card": 1,
                                                                                "est_children_card": 1,
                                                                                "above_udf_filter": False,
                                                                                "is_udf_filter": False
                                                                            },
                                                                            "children": []
                                                                        }
                                                                    ]
                                                                }
                                                            ]
                                                        }
                                                    ]
                                                }
                                            ]
                                        },
                                        {
                                            "plan_parameters": {
                                                "act_card": 48276,
                                                "act_time": 26.82,
                                                "act_loops": 5364,
                                                "est_card": 11076,
                                                "est_width": 4,
                                                "pg_cost": 0.96,
                                                "est_loops": 852,
                                                "table_name": "probehnuto_mimo_penezenku",
                                                "table_id": 3,
                                                "filter": {
                                                    "operator": "AND",
                                                    "is_index_cond": False,
                                                    "children": [
                                                        {
                                                            "table_name": "probehnuto_mimo_penezenku",
                                                            "col_name": "Month_Year",
                                                            "col_id": 12,
                                                            "operator": "!=",
                                                            "literal": "'2013-10-01_0'",
                                                            "is_index_cond": False,
                                                            "filter_complexity": 0
                                                        },
                                                        {
                                                            "table_name": "probehnuto_mimo_penezenku",
                                                            "col_name": "client_id",
                                                            "col_id": 11,
                                                            "operator": ">=",
                                                            "literal": "97288",
                                                            "is_index_cond": True,
                                                            "filter_complexity": 0
                                                        }
                                                    ]
                                                },
                                                "op_name": "Index Only Scan",
                                                "output_columns": [
                                                    {
                                                        "aggregation": "None",
                                                        "columns": [
                                                            {
                                                                "col_id": 11,
                                                                "child_ref": None,
                                                                "table_name": "probehnuto_mimo_penezenku",
                                                                "col_name": "client_id"
                                                            }
                                                        ],
                                                        "udf_name": None,
                                                        "udf_output": "False",
                                                        "child_ref": False
                                                    },
                                                ],
                                                "act_children_card": 1,
                                                "est_children_card": 1,
                                                "above_udf_filter": False,
                                                "is_udf_filter": False
                                            },
                                            "children": []
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }]}

        # set maxdiff to none
        self.maxDiff = None

        output = apply_projection_pushdown(plan, database_statistics=dataset_stats)[0]
        self.assertEqual(expected_plan, output)

        plan2 = {"plan_parameters": {
            "act_card": 1,
            "act_time": 617.294,
            "act_loops": 1,
            "est_card": 1,
            "est_width": 32,
            "pg_cost": 61534.5,
            "est_loops": 1,
            "op_name": "Finalize Aggregate",
            "output_columns": [
                {
                    "aggregation": "COUNT",
                    "columns": [], }],
            "act_children_card": 3,
            "est_children_card": 2,
            "above_udf_filter": False,
            "is_udf_filter": False
        },
            "children": [
                {
                    "plan_parameters": {
                        "act_card": 95163,
                        "act_time": 657.297,
                        "act_loops": 3,
                        "est_card": 40207,
                        "est_width": 11,
                        "pg_cost": 34369.98,
                        "est_loops": 1,
                        "table_name": "probehnuto",
                        "table_id": 2,
                        "filter": {
                            "table_name": "probehnuto",
                            "col_name": "month_year_datum_transakce",
                            "col_id": 8,
                            "operator": "=",
                            "literal": "'2015-02-01'",
                            "is_index_cond": False,
                            "filter_complexity": 0
                        },
                        "op_name": "Seq Scan",
                        "output_columns": [
                            {
                                "aggregation": "None",
                                "columns": [
                                    {
                                        "col_id": 7,
                                        "child_ref": None,
                                        "table_name": "probehnuto",
                                        "col_name": "client_id"
                                    }
                                ],
                                "udf_name": None,
                                "udf_output": "False",
                                "child_ref": False
                            },
                            {
                                "aggregation": "None",
                                "columns": [
                                    {
                                        "col_id": 9,
                                        "child_ref": None,
                                        "table_name": "probehnuto",
                                        "col_name": "kc_proklikano"
                                    }
                                ],
                                "udf_name": None,
                                "udf_output": "False",
                                "child_ref": False
                            }
                        ],
                        "act_children_card": 1,
                        "est_children_card": 1,
                        "above_udf_filter": False,
                        "is_udf_filter": False
                    },
                    "children": []
                }
            ]}

        expected_plan2 = {"plan_parameters": {
            "act_card": 1,
            "act_time": 617.294,
            "act_loops": 1,
            "est_card": 1,
            "est_width": 32,
            "pg_cost": 61534.5,
            "est_loops": 1,
            "op_name": "Finalize Aggregate",
            "output_columns": [
                {
                    "aggregation": "COUNT",
                    "columns": [], }],
            "act_children_card": 3,
            "est_children_card": 2,
            "above_udf_filter": False,
            "is_udf_filter": False
        },
            "children": [
                {
                    "plan_parameters": {
                        "act_card": 95163,
                        "act_time": 657.297,
                        "act_loops": 3,
                        "est_card": 40207,
                        "est_width": 11,
                        "pg_cost": 34369.98,
                        "est_loops": 1,
                        "table_name": "probehnuto",
                        "table_id": 2,
                        "filter": {
                            "table_name": "probehnuto",
                            "col_name": "month_year_datum_transakce",
                            "col_id": 8,
                            "operator": "=",
                            "literal": "'2015-02-01'",
                            "is_index_cond": False,
                            "filter_complexity": 0
                        },
                        "op_name": "Seq Scan",
                        "output_columns": [
                            {
                                "aggregation": "None",
                                "columns": [
                                    {
                                        "col_id": 7,
                                        "child_ref": None,
                                        "table_name": "probehnuto",
                                        "col_name": "client_id"
                                    }
                                ],
                                "udf_name": None,
                                "udf_output": "False",
                                "child_ref": False
                            }
                        ],
                        "act_children_card": 1,
                        "est_children_card": 1,
                        "above_udf_filter": False,
                        "is_udf_filter": False
                    },
                    "children": []
                }
            ]}

        self.assertEqual(expected_plan2, apply_projection_pushdown(plan2, database_statistics=dataset_stats)[0])

        plan3 = {
            "plan_parameters": {
                "act_card": 36244344,
                "act_time": 4112.657,
                "act_loops": 1,
                "est_card": 36244344,
                "est_width": 42,
                "pg_cost": 705740.3,
                "est_loops": 1,
                "table_name": "cast_info",
                "table_id": 2,
                "filter": {
                    "table_name": "cast_info",
                    "col_name": "movie_id",
                    "col_id": 87,
                    "operator": ">=",
                    "literal": "1",
                    "is_index_cond": False,
                    "filter_complexity": 0,
                },
                "op_name": "Seq Scan",
                "output_columns": [
                    {
                        "aggregation": "None",
                        "columns": [
                            {
                                "col_id": 85,
                                "child_ref": None,
                                "table_name": "cast_info",
                                "col_name": "id",
                            }
                        ],
                        "udf_name": None,
                        "udf_output": "False",
                        "child_ref": False,
                    },
                    {
                        "aggregation": "None",
                        "columns": [
                            {
                                "col_id": 86,
                                "child_ref": None,
                                "table_name": "cast_info",
                                "col_name": "person_id",
                            }
                        ],
                        "udf_name": None,
                        "udf_output": "False",
                        "child_ref": False,
                    },
                    {
                        "aggregation": "None",
                        "columns": [
                            {
                                "col_id": 87,
                                "child_ref": None,
                                "table_name": "cast_info",
                                "col_name": "movie_id",
                            }
                        ],
                        "udf_name": None,
                        "udf_output": "False",
                        "child_ref": False,
                    },
                    {
                        "aggregation": "None",
                        "columns": [
                            {
                                "col_id": 63,
                                "child_ref": None,
                                "table_name": "cast_info",
                                "col_name": "person_role_id",
                            }
                        ],
                        "udf_name": None,
                        "udf_output": "False",
                        "child_ref": False,
                    },
                    {
                        "aggregation": "None",
                        "columns": [
                            {
                                "col_id": 64,
                                "child_ref": None,
                                "table_name": "cast_info",
                                "col_name": "note",
                            }
                        ],
                        "udf_name": None,
                        "udf_output": "False",
                        "child_ref": False,
                    },
                    {
                        "aggregation": "None",
                        "columns": [
                            {
                                "col_id": 88,
                                "child_ref": None,
                                "table_name": "cast_info",
                                "col_name": "nr_order",
                            }
                        ],
                        "udf_name": None,
                        "udf_output": "False",
                        "child_ref": False,
                    },
                    {
                        "aggregation": "None",
                        "columns": [
                            {
                                "col_id": 65,
                                "child_ref": None,
                                "table_name": "cast_info",
                                "col_name": "role_id",
                            }
                        ],
                        "udf_name": None,
                        "udf_output": "False",
                        "child_ref": False,
                    },
                ],
                "act_children_card": 1,
                "est_children_card": 1,
                "above_udf_filter": False,
                "is_udf_filter": False,
            },
            "children": [],
            "sql": "SELECT * FROM cast_info WHERE movie_id >= 1;",
            "dataset": "imdb",
            "plan_runtime_ms": 5586.937,
            "num_tables": 1,
            "num_filters": 1,
            "hint": "/*+SeqScan(cast_info)*/",
            "invalid_hint": False,
            "zero_card": False,
            "timeout": False,
            "timeout_threshold": 180000,
            "database_id": 0,
        }
        self.assertEqual(plan3, apply_projection_pushdown(plan3, database_statistics=dataset_stats)[0])

        plan4 = {
            "plan_parameters": {
                "act_card": 1102414,
                "act_time": 478.194,
                "act_loops": 1,
                "est_card": 1092989,
                "est_width": 94,
                "pg_cost": 67601.9,
                "est_loops": 1,
                "table_name": "title",
                "table_id": 22,
                "filter": {
                    "table_name": "title",
                    "col_name": "episode_nr",
                    "col_id": 34,
                    "operator": ">=",
                    "literal": "1",
                    "is_index_cond": False,
                    "filter_complexity": 0,
                },
                "op_name": "Seq Scan",
                "output_columns": [],
                "act_children_card": 1,
                "est_children_card": 1,
                "above_udf_filter": False,
                "is_udf_filter": False,
            },
            "children": [],
            "sql": "SELECT * FROM title WHERE episode_nr >= 1;",
            "dataset": "imdb",
            "plan_runtime_ms": 523.311,
            "num_tables": 1,
            "num_filters": 1,
            "hint": "/*+SeqScan(title)*/",
            "invalid_hint": False,
            "zero_card": False,
            "timeout": False,
            "timeout_threshold": 180000,
            "database_id": 0,
        }

        expected_plan4 = {
            "plan_parameters": {
                "act_card": 1102414,
                "act_time": 478.194,
                "act_loops": 1,
                "est_card": 1092989,
                "est_width": 94,
                "pg_cost": 67601.9,
                "est_loops": 1,
                "table_name": "title",
                "table_id": 22,
                "filter": {
                    "table_name": "title",
                    "col_name": "episode_nr",
                    "col_id": 34,
                    "operator": ">=",
                    "literal": "1",
                    "is_index_cond": False,
                    "filter_complexity": 0,
                },
                "op_name": "Seq Scan",
                "output_columns": [
                    {
                        "aggregation": "None",
                        "columns": [
                            {
                                "col_id": 1,
                                "child_ref": None,
                                "table_name": "title",
                                'table_alias': None,
                                "col_name": "id",
                            }
                        ],
                        "udf_name": None,
                        "udf_output": "False",
                        "child_ref": False,
                    }
                ],
                "act_children_card": 1,
                "est_children_card": 1,
                "above_udf_filter": False,
                "is_udf_filter": False,
            },
            "children": [],
            "sql": "SELECT * FROM title WHERE episode_nr >= 1;",
            "dataset": "imdb",
            "plan_runtime_ms": 523.311,
            "num_tables": 1,
            "num_filters": 1,
            "hint": "/*+SeqScan(title)*/",
            "invalid_hint": False,
            "zero_card": False,
            "timeout": False,
            "timeout_threshold": 180000,
            "database_id": 0,
        }
        self.assertEqual(expected_plan4, apply_projection_pushdown(plan4, database_statistics=dataset_stats)[0])


if __name__ == '__main__':
    unittest.main()
