import unittest
from pathlib import Path

from dotenv import load_dotenv
from ldb_schema.paths import LocalPaths

from ldb_models.classes import T3ModelConfig
from ldb_models.models.t3.t3_dataloader import (
    gen_t3_dataset,
    rewrite_children,
    extract_pipeline_infos,
    annotate_op_id,
)


class MyTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        load_dotenv()

    def test_create_t3_dataloader(self):
        sample_wl_file = Path(
            f"{LocalPaths().Data.parsed_plans.as_posix()}/imdb_scaled1/access_path_selection.json"
        )
        queries, _, labels, _, sqls, hints, selectivities, scan_ops, join_ops = (
            gen_t3_dataset(
                [sample_wl_file],
                statistics_file=None,
                cap_training_samples=None,
                featurization=None,
                limit_plans_per_file=1000,
                limit_plans_overall=None,
                val_ratio=0,
                return_dataset=True,
                model_config=T3ModelConfig(),
            )
        )
        self.assertTrue(None not in scan_ops)

        sample_wl_file = Path(
            f"{LocalPaths().Data.parsed_plans.as_posix()}/imdb_scaled1/workload_200k_s1_reduced.json"
        )
        queries, _, labels, _, sqls, hints, selectivities, scan_ops, join_ops = (
            gen_t3_dataset(
                [sample_wl_file],
                statistics_file=None,
                cap_training_samples=None,
                featurization=None,
                limit_plans_per_file=1000,
                limit_plans_overall=None,
                val_ratio=0,
                return_dataset=False,
                model_config=T3ModelConfig(),
            )
        )

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
                            "children": [],
                        },
                        {
                            "plan_parameters": {
                                "op_name": "Hash",
                                "act_time": 3910.032,
                            },
                            "children": [
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
                                                    "children": [],
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
                                                                    "children": [],
                                                                }
                                                            ],
                                                        }
                                                    ],
                                                },
                                            ],
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
                                                    "children": [],
                                                }
                                            ],
                                        },
                                    ],
                                },
                            ],
                        },
                    ],
                }
            ],
            "sql": 'SELECT COUNT(*) as agg_0 FROM "role_type" LEFT OUTER JOIN "cast_info" ON "role_type"."id" = "cast_info"."role_id" LEFT OUTER JOIN "title" ON "cast_info"."movie_id" = "title"."id" LEFT OUTER JOIN "kind_type" ON "title"."kind_id" = "kind_type"."id"  WHERE "role_type"."role" != \'producer\' AND "title"."imdb_index" = \'VIII\' AND "kind_type"."id" >= 6 AND "kind_type"."kind" != \'video movie\' AND "role_type"."id" != 8;',
            "dataset": "imdb",
            "plan_runtime_ms": 3978.068,
            "num_tables": 4,
            "num_filters": 5,
            "hint": None,
            "invalid_hint": False,
            "zero_card": False,
            "timeout": False,
            "timeout_threshold": 30000,
        }

        rewrite_children(plan)

        annotate_op_id(plan)

        pipelines = []
        extract_pipeline_infos(plan, pipelines=pipelines)

        expected_pipelines = [
            {"operators": [0], "duration": 0.125, "start": 3977.943, "stop": 3978.068},
            {
                "operators": [0, 1, 8],
                "duration": 67.911,
                "start": 3910.032,
                "stop": 3977.943,
            },
            {
                "operators": [1, 2, 4, 7],
                "duration": 3539.869,
                "start": 370.163,
                "stop": 3910.032,
            },
            {"operators": [2, 3], "duration": 0.019, "start": 370.144, "stop": 370.163},
            {"operators": [4, 5, 6], "duration": 370.144, "start": 0, "stop": 370.144},
        ]
        self.assertEqual(len(expected_pipelines), len(pipelines))
        for p1, p2 in zip(expected_pipelines, pipelines):
            self.assertEqual(p1["operators"], p2["operators"])
            self.assertAlmostEqual(p1["duration"], p2["duration"])
            self.assertEqual(p1["start"], p2["start"])
            self.assertEqual(p1["stop"], p2["stop"])

        plan = {
            "plan_parameters": {
                "act_time": 3977.943,
                "op_name": "Finalize Aggregate",
            },
            "children": [
                {
                    "plan_parameters": {
                        "act_time": 3910.056,
                        "op_name": "Index Nested Loop",
                    },
                    "children": [
                        {
                            "plan_parameters": {
                                "act_time": 370.079,
                                "op_name": "Seq Scan",
                                "table_name": "title",
                            },
                            "children": [],
                        },
                        {
                            "plan_parameters": {
                                "act_time": 0.024,
                                "table_name": "role_type",
                                "table_id": 21,
                                "op_name": "Index Scan",
                            },
                            "children": [],
                        },
                    ],
                }
            ],
            "dataset": "imdb",
            "plan_runtime_ms": 3978.068,
        }

        rewrite_children(plan)

        annotate_op_id(plan)

        pipelines = []
        extract_pipeline_infos(plan, pipelines=pipelines)

        expected_pipelines = [
            {"operators": [0], "duration": 0.125, "start": 3977.943, "stop": 3978.068},
            {
                "operators": [0, 1, 2],
                "duration": 3977.943,
                "start": 0,
                "stop": 3977.943,
            },
        ]
        self.assertEqual(len(expected_pipelines), len(pipelines))
        for p1, p2 in zip(expected_pipelines, pipelines):
            self.assertEqual(p1["operators"], p2["operators"])
            self.assertAlmostEqual(p1["duration"], p2["duration"])
            self.assertEqual(p1["start"], p2["start"])
            self.assertEqual(p1["stop"], p2["stop"])

        plan = {
            "plan_parameters": {
                "act_time": 3977.943,
                "op_name": "Finalize Aggregate",
            },
            "children": [
                {
                    "plan_parameters": {
                        "act_time": 3910.056,
                        "op_name": "Nested Loop",
                    },
                    "children": [
                        {
                            "plan_parameters": {
                                "act_time": 370.079,
                                "op_name": "Seq Scan",
                                "table_name": "title",
                            },
                            "children": [],
                        },
                        {
                            "plan_parameters": {
                                "act_time": 0.024,
                                "table_name": "role_type",
                                "table_id": 21,
                                "op_name": "Index Scan",
                            },
                            "children": [],
                        },
                    ],
                }
            ],
            "dataset": "imdb",
            "plan_runtime_ms": 3978.068,
        }

        rewrite_children(plan)

        annotate_op_id(plan)

        pipelines = []
        extract_pipeline_infos(plan, pipelines=pipelines)

        expected_pipelines = [
            {"operators": [0], "duration": 0.125, "start": 3977.943, "stop": 3978.068},
            {
                "operators": [0, 1, 3],
                "duration": 3977.919,
                "start": 0.024,
                "stop": 3977.943,
            },
            {
                "operators": [1, 2],
                "duration": 0.024,
                "start": 0,
                "stop": 0.024,
            },
        ]
        self.assertEqual(len(expected_pipelines), len(pipelines))
        for p1, p2 in zip(expected_pipelines, pipelines):
            self.assertEqual(p1["operators"], p2["operators"])
            self.assertAlmostEqual(p1["duration"], p2["duration"])
            self.assertEqual(p1["start"], p2["start"])
            self.assertEqual(p1["stop"], p2["stop"])


if __name__ == "__main__":
    unittest.main()
