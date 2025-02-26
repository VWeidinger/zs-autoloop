import unittest
from pathlib import Path

from dotenv import load_dotenv
from ldb_schema.paths import LocalPaths

from classes.classes import T3ModelConfig
from models.t3.t3_dataloader import gen_t3_dataset, rewrite_children, prune_nodes_to_ignore, extract_pipeline_infos, \
    annotate_op_id


class MyTestCase(unittest.TestCase):

    def setUp(self):
        super().setUp()
        load_dotenv()

    def test_create_t3_dataloader(self):
        sample_wl_file = Path(f'{LocalPaths().Data.parsed_plans.as_posix()}/imdb_scaled1/workload_200k_s1_reduced.json')
        sample_wl_file = Path(f'{LocalPaths().Data.parsed_plans.as_posix()}/imdb_scaled1/access_path_selection.json')
        queries, _, labels, _, sqls, hints, selectivities, join_ops, scan_ops = gen_t3_dataset([sample_wl_file],
                                                                                               statistics_file=None,
                                                                                               cap_training_samples=None,
                                                                                               featurization=None,
                                                                                               limit_plans_per_file=100,
                                                                                               limit_plans_overall=None,
                                                                                               val_ratio=0,
                                                                                               return_dataset=False,
                                                                                               model_config=T3ModelConfig())

        self.assertTrue(None not in scan_ops)

    def test_extract_pipeline_infos(self):
        plan = {
            "plan_parameters": {
                "act_time": 3977.943,
                "op_name": "Finalize Aggregate",
            },
            "children": [

                {
                    "plan_parameters": {
                        "act_time": 3910.056,
                        "op_name": "Hash Join",
                    },
                    "children": [
                        {
                            "plan_parameters": {
                                "act_time": 0.024,
                                "table_name": "role_type",
                                "table_id": 21,
                                "op_name": "Seq Scan",
                            },
                            "children": []
                        },
                        {"plan_parameters": {
                            "op_name": "Hash",
                            "act_time": 3910.032,
                        }, "children": [
                            {
                                "plan_parameters": {
                                    "act_time": 3909.838,
                                    "op_name": "Hash Join",
                                },
                                "children": [
                                    {
                                        "plan_parameters": {
                                            "act_time": 3909.728,
                                            "op_name": "Hash Join",
                                        },
                                        "children": [
                                            {
                                                "plan_parameters": {
                                                    "act_time": 1986.198,
                                                    "table_name": "cast_info",
                                                    "table_id": 2,
                                                    "op_name": "Seq Scan",
                                                },
                                                "children": []
                                            },
                                            {
                                                "plan_parameters": {
                                                    "act_time": 370.144,
                                                    "op_name": "Hash",
                                                },
                                                "children": [
                                                    {
                                                        "plan_parameters": {
                                                            "act_time": 370.124,
                                                            "op_name": "Smth",
                                                        },
                                                        "children": [
                                                            {
                                                                "plan_parameters": {
                                                                    "act_time": 370.079,
                                                                    "op_name": "Seq Scan",
                                                                    "table_name": "title",
                                                                },
                                                                "children": []
                                                            }
                                                        ]
                                                    }

                                                ]
                                            }
                                        ]
                                    },
                                    {
                                        "plan_parameters": {
                                            "act_time": 0.019,
                                            "op_name": "Hash",
                                        },
                                        "children": [
                                            {
                                                "plan_parameters": {
                                                    "act_time": 0.015,
                                                    "table_name": "kind_type",
                                                    "table_id": 10,
                                                    "op_name": "Seq Scan",
                                                },
                                                "children": []
                                            }
                                        ]
                                    }
                                ]
                            }, ]
                        },

                    ]
                }
            ],
            "sql": "SELECT COUNT(*) as agg_0 FROM \"role_type\" LEFT OUTER JOIN \"cast_info\" ON \"role_type\".\"id\" = \"cast_info\".\"role_id\" LEFT OUTER JOIN \"title\" ON \"cast_info\".\"movie_id\" = \"title\".\"id\" LEFT OUTER JOIN \"kind_type\" ON \"title\".\"kind_id\" = \"kind_type\".\"id\"  WHERE \"role_type\".\"role\" != 'producer' AND \"title\".\"imdb_index\" = 'VIII' AND \"kind_type\".\"id\" >= 6 AND \"kind_type\".\"kind\" != 'video movie' AND \"role_type\".\"id\" != 8;",
            "dataset": "imdb",
            "plan_runtime_ms": 3978.068,
            "num_tables": 4,
            "num_filters": 5,
            "hint": None,
            "invalid_hint": False,
            "zero_card": False,
            "timeout": False,
            "timeout_threshold": 30000
        }

        rewrite_children(plan)

        plan = prune_nodes_to_ignore(plan)
        annotate_op_id(plan)

        pipelines = []
        extract_pipeline_infos(plan, pipelines=pipelines)

        expected_pipelines = [{
            "start": 257,
            "stop": 259,
            "duration": 2,
            "parallelism": "multi-threaded",
            "operators": [
                0
            ]
        },
            {
                "start": 79,
                "stop": 254,
                "duration": 175,
                "parallelism": "multi-threaded",
                "operators": [
                    0,
                    1,
                    8
                ]
            },
            {
                "start": 15,
                "stop": 76,
                "duration": 61,
                "parallelism": "multi-threaded",
                "operators": [
                    1,
                    2,
                    4,
                    7
                ]
            },
            {
                "start": 3,
                "stop": 9,
                "duration": 6,
                "parallelism": "multi-threaded",
                "operators": [
                    2,
                    3
                ]
            },
            {
                "start": 10,
                "stop": 15,
                "duration": 5,
                "parallelism": "multi-threaded",
                "operators": [
                    4,
                    5,
                    6
                ]
            }]
        self.assertEqual(len(expected_pipelines), len(pipelines))
        for p1, p2 in zip(expected_pipelines, pipelines):
            self.assertEqual(p1['operators'], p2['operators'])



if __name__ == '__main__':
    unittest.main()
