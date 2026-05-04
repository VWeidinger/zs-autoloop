import argparse
import collections
import json
import os

import pandas as pd
from ldb_schema.load_schema import (
    get_json_schema,
    load_column_statistics,
    load_json,
    load_string_statistics,
)
from tqdm import tqdm


def get_col_stats(column_stats: list, column_name: str):
    for col_stat in column_stats:
        if col_stat["attname"] == column_name:
            return col_stat


def dumper(obj):
    try:
        return obj.toJSON()
    except:  # noqa: E722
        return obj.__dict__


def create_sentences(
    dataset: str,
    workload_path: str,
    workloads: list[str],
    data_dir: str,
    target_path: str,
    dataset_statistics: str,
):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    runs = []
    # per column find regex queries
    for workload in workloads:
        source_path = os.path.join(workload_path, f"{workload}.json")
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Could not find workload file {source_path}")
        runs.append(load_json(source_path))
    col_stats = runs[0]["database_stats"]["column_stats"]
    regex_patterns = collections.defaultdict(set)

    for run in runs:
        for p in run["parsed_plans"]:
            augment_regex_patterns(p, regex_patterns, col_stats)

    # load statistics
    schema = get_json_schema(dataset)
    column_statistics = load_column_statistics(
        dataset, dataset_statistics_dir=dataset_statistics, namespace=False
    )
    string_statistics = load_string_statistics(
        dataset,
        scale_factor="1",
        dataset_statistics_dir=dataset_statistics,
        namespace=False,
    )

    # read individual table csvs and derive sentences
    sentences = []
    for table, columns in schema["table_col_info"].items():
        table_dir = os.path.join(data_dir, f"{table}.csv")
        assert os.path.exists(data_dir), f"Could not find table csv {table_dir}"
        print(f"Generating sentences for word embeddings for {table_dir}")

        # find string columns
        sentence_columns, id_columns = find_sentence_columns(
            column_statistics, schema, string_statistics, table
        )

        df_sample = pd.read_csv(table_dir)
        if len(df_sample) > 100000:
            df_sample = df_sample.sample(random_state=0, n=100000)

        for _, row in tqdm(df_sample.iterrows(), total=df_sample.shape[0]):
            sentence = []
            sentence_reg = []

            # base sentences
            for c in df_sample.columns:
                if c not in sentence_columns:
                    continue
                val = str(row[c])
                sentence.append(f"{c}_{val}")
                if c in id_columns:
                    sentence_reg.append(f"{c}_{val}")
                if " " in val:
                    vals = val.split(" ")
                    sentence += [f"{c}_{v}" for v in vals]
            if len(sentence) > 0:
                sentences.append(sentence)

            # regex query sentence (sentence_reg)
            for c in df_sample.columns:
                val = str(row[c])
                for regex_p in regex_patterns[(table, c)]:
                    # check whether it applies
                    if all([str_part in val for str_part in regex_p]):
                        sentence_reg += [f"{c}_{str_part}" for str_part in regex_p]
            if len(sentence_reg) > 0:
                sentences.append(sentence_reg)

    with open(target_path, "w") as outfile:
        json.dump(sentences, outfile, default=dumper)
    os.chmod(target_path, 0o777)


def augment_regex_pattern_per_filter(filter: dict, regex_patterns, col_stats):
    if filter["operator"] in {"LIKE", "NOT LIKE"}:
        col_stats = get_col_stats(col_stats, filter["col_name"])
        col_name = col_stats["attname"]
        table_name = col_stats["tablename"]
        reg_parts = [
            str(fp.strip())
            for fp in filter["literal"].split("%")
            if len(fp.strip()) > 0
        ]
        if len(reg_parts) > 0:
            regex_patterns[(table_name, col_name)].add(tuple(reg_parts))

    if "children" in filter.keys():
        for c in filter["children"]:
            augment_regex_pattern_per_filter(
                filter=c, regex_patterns=regex_patterns, col_stats=col_stats
            )


def augment_regex_patterns(plan: dict, regex_patterns, col_stats: list):
    if "filter" in plan["plan_parameters"].keys():
        filter_column = plan["plan_parameters"]["filter"]
        if filter_column is not None:
            augment_regex_pattern_per_filter(
                filter=filter_column, regex_patterns=regex_patterns, col_stats=col_stats
            )
    if "children" in plan.keys():
        for c in plan["children"]:
            augment_regex_patterns(
                plan=c, regex_patterns=regex_patterns, col_stats=col_stats
            )


def find_sentence_columns(
    column_statistics: dict, schema: dict, string_statistics: dict, t: str
):
    string_columns = {
        c for c, v in column_statistics[t].items() if v["data_type"] == "categorical"
    }
    string_columns.update([c for c, v in string_statistics[t].items()])
    # string_columns = dict()
    # for column_name, column in column_statistics[t].items():
    #    if column['data_type'] == 'categorical':
    #        string_columns[column_name] = column

    # string_columns.update([c for c, v in string_statistics[t].items()])
    id_columns = set()
    for t_l, c_l, t_r, c_r in schema["relationships"]:
        if not (isinstance(c_l, list) or isinstance(c_l, tuple)):
            c_l = [c_l]
            c_r = [c_r]
        if t_l == t:
            id_columns.update(c_l)
        if t_r == t:
            id_columns.update(c_r)
    sentence_columns = string_columns.union(id_columns)
    return sentence_columns, id_columns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create sentences for word embeddings")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset"
    )
    parser.add_argument(
        "--workloads",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the plan files",
    )
    parser.add_argument(
        "--workload_path", type=str, required=True, help="Paths to the workload files"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the data files",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        required=True,
        help="Target path to save the sentences",
    )
    parser.add_argument(
        "--statistics_path", type=str, required=True, help="Path to the plan files"
    )
    args = parser.parse_args()
    create_sentences(
        args.dataset,
        args.workload_path,
        args.workloads,
        args.data_dir,
        args.target_path,
        args.statistics_path,
    )
