import unittest

from ldb_schema.load_schema import load_json
from models.neo.neo_dataloader import (
    extract_predicate_encoding,
    init_table_ids,
    extract_join_recursively,
    extract_plan_encoding_recursively,
    make_indexes,
    _featurize_tree,
)
import re


def init():
    data = load_json("test_plan.json")
    plan = data["parsed_plans"][0]
    path = "/Users/rheinrich/Projekte/ldb_models/data/feature_statistics/parsed_plans_feature_statistics_imdb.json"
    feature_stats = load_json(path)
    column_stats = data["database_stats"]["column_stats"]
    table_stats = data["database_stats"]["table_stats"]
    table_stats = [x for x in table_stats if not re.match(r"^(pg_|sql_)", x["relname"])]
    return plan, feature_stats, column_stats, table_stats


class TestNeoFeaturization(unittest.TestCase):
    def test_predicate_encoding(self):
        plan, feature_stats, column_stats, _ = init()
        tables_cols = [
            ("movie_keyword", "movie_id"),
            ("movie_info", "movie_id"),
            ("movie_info", "info_type_id"),
            ("title", "series_years"),
            ("keyword", "id"),
        ]

        predicate_encoding = extract_predicate_encoding(
            plan=plan, column_statistics=column_stats, feature_statistics=feature_stats
        )

        assert len(predicate_encoding) == len(column_stats), (
            "Predicate encoding length mismatch"
        )
        assert sum(predicate_encoding) == int(plan["num_filters"]), (
            "Predicate encoding length mismatch"
        )
        result = [
            (column["tablename"], column["attname"])
            for index, column in enumerate(column_stats)
            if predicate_encoding[index] == 1
        ]

        tables_cols.sort(key=lambda x: x[0])
        result.sort(key=lambda x: x[0])
        self.assertEqual(
            result, tables_cols, "Predicate encoding does not match expected values"
        )

    def test_join_graph(self):
        plan, feature_stats, column_stats, table_stats = init()
        joins_infos = extract_join_recursively(
            sql=plan["sql"],
            hint=plan["hint"],
            plan=plan,
            table_statistics=table_stats,
            join_tables=[],
        )
        tables = []
        for join_info in joins_infos:
            tables.append(join_info["left_table"])
            tables.append(join_info["right_table"])
        self.assertEqual(
            sorted(list(set(tables))),
            sorted(["keyword", "movie_keyword", "movie_info", "title"]),
        )

    def test_plan_encoding(self):
        plan, feature_stats, column_stats, table_stats = init()
        extract_plan_encoding_recursively(
            plan=plan,
            table_statistics=table_stats,
            plan_encodings=[],
            depth=0,
            table_ids=init_table_ids(table_stats),
            max_depth=25,
        )

        indexes = make_indexes(plan=plan)

        tree = _featurize_tree(plan=plan, maximum_depth=25)
        print(tree)
