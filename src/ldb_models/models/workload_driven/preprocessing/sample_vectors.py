import argparse
import concurrent.futures
import json
import os
from typing import Any, Dict

import pandas as pd
from ldb_schema.load_schema import get_json_schema, load_json
from tqdm import tqdm

from ldb_models.models.workload_driven.preprocessing.sentence_creation import dumper


def get_col_stats(column_stats: list, column_name: str):
    for col_stat in column_stats:
        if col_stat["attname"] == column_name:
            return col_stat


def _parse_based_on_dtype(literal: str, dtype: str) -> str | int | float:
    if dtype == "integer":
        return int(float(literal))
    elif dtype in ["float", "numeric", "double", "double precision"]:
        return float(literal)
    elif dtype in [
        "character varying",
        "varchar",
        "time without time zone",
        "character",
        "char",
    ]:
        return str(literal)
    else:
        raise ValueError(f"Unsupported datatype: {dtype}")


def construct_filter_sample(
    table_samples: dict, column_stats: list, filter_column: dict
):
    sample_vec = None
    if filter_column["operator"] in {
        "=",
        ">=",
        "<=",
        "!=",
        "NOT LIKE",
        "LIKE",
        "IN",
        "IS NOT NULL",
        "IS NULL",
    }:
        col_stats = get_col_stats(column_stats, filter_column["col_name"])
        col_name = col_stats["attname"]
        table_name = col_stats["tablename"]
        datatype = col_stats["data_type"]
        literal = filter_column["literal"]
        base_sample = table_samples[table_name][col_name]

        # Cast literal to the correct type
        if literal:
            literal = _parse_based_on_dtype(literal, datatype)

        # Compute Sample Vectors
        if filter_column["operator"] == "=":
            sample_vec = base_sample == literal
        elif filter_column["operator"] == ">=":
            sample_vec = base_sample >= literal
        elif filter_column["operator"] == "<=":
            sample_vec = base_sample <= literal
        elif filter_column["operator"] == "!=":
            sample_vec = base_sample != literal
        elif filter_column["operator"] in {"NOT LIKE", "LIKE"}:
            # replace wildcards to obtain valid regex
            regex = literal
            for esc_char in [".", "(", ")", "?"]:
                regex = regex.replace(esc_char, f"\{esc_char}")
            regex = regex.replace("%", ".*?")
            regex = f"^{regex}$"
            # sample_vec = base_sample.str.contains(regex)
            sample_vec = base_sample.astype(str).str.contains(
                regex, regex=True, na=False
            )
            if filter_column["operator"] == "NOT LIKE":
                sample_vec[sample_vec.isna()] = False
                sample_vec = ~sample_vec
        elif filter_column["operator"] == "IN":
            literal = literal.split(",")
            casted_literals = []
            for lit in literal:
                lit = _parse_based_on_dtype(lit, datatype)
                casted_literals.append(lit)
            sample_vec = base_sample.isin(casted_literals)
        elif filter_column["operator"] == "IS NOT NULL":
            sample_vec = ~base_sample.isna()
        elif filter_column["operator"] == "IS NULL":
            sample_vec = base_sample.isna()

        # handle nans
        sample_vec[sample_vec.isna()] = False

    elif filter_column["operator"] in {"AND", "OR"}:
        children_vec = [
            construct_filter_sample(
                table_samples=table_samples, column_stats=column_stats, filter_column=c
            )
            for c in filter_column["children"]
        ]
        sample_vec = children_vec[0]
        for c_vec in children_vec[1:]:
            if filter_column["operator"] == "AND":
                sample_vec &= c_vec
            elif filter_column["operator"] == "OR":
                sample_vec ^= c_vec
            else:
                raise NotImplementedError
    else:
        print(filter_column["operator"])
        raise NotImplementedError

    assert sample_vec is not None
    return sample_vec.astype(bool)


def augment_sample(table_samples: dict, col_stats: list, p_node: dict):
    if "filter" in p_node["plan_parameters"]:
        filter_column = p_node["plan_parameters"]["filter"]
        sample_vec = construct_filter_sample(table_samples, col_stats, filter_column)
        # convert to binary
        sample_vec = [1 if s_i else 0 for s_i in sample_vec]
        p_node["plan_parameters"]["sample_vec"] = sample_vec

    if "children" in p_node.keys():
        for c in p_node["children"]:
            augment_sample(table_samples, col_stats, c)


def get_sample_for_table(
    table: str, columns: Dict[str, Any], data_dir: str, no_samples: int
):
    table_dir = os.path.join(data_dir, f"{table}.csv")
    assert os.path.exists(data_dir), f"Could not find table csv {table_dir}"

    dtype_mapping = {}
    for column, data in columns.items():
        if data["type"] == "integer":
            dtype_mapping[column] = "Int64"
        elif data["type"] == "float":
            dtype_mapping[column] = float
        elif data["type"].startswith("varchar"):
            dtype_mapping[column] = str

    df_sample = pd.read_csv(table_dir, na_values=["<!NULL-?>"], dtype=dtype_mapping)
    if len(df_sample) > no_samples:
        df_sample = df_sample.sample(random_state=0, n=no_samples)

    return df_sample


def augment_sample_vectors(
    dataset: str,
    data_dir: str,
    plan_path: str,
    target_path: str,
    workloads: list[str],
    no_samples: int = 1000,
):
    # Check first that all workload paths exist
    print("Checking existence of all workloads", flush=True)
    filtered_workloads = []
    for workload in workloads:
        source_path = os.path.join(plan_path, f"{workload}.json")
        if os.path.exists(source_path):
            filtered_workloads.append(workload)
        else:
            print(f"Could not find workload file {source_path}, skipping {workload}")
    workloads = filtered_workloads

    # create output directory
    os.makedirs(target_path, exist_ok=True)

    # load first run to get column statistics
    print("Load first run to retrieve column stats", flush=True)
    source_path = os.path.join(plan_path, f"{workloads[0]}.json")
    run = load_json(source_path)
    col_stats = run["database_stats"]["column_stats"]

    # create sample per table
    print("Retrieve schema of dataset...")
    schema = get_json_schema(dataset)

    # read individual table csvs and derive statistics
    table_samples = dict()

    parallelize = True
    if parallelize:
        print("Starting parallel processing for table samples")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_table = {
                executor.submit(
                    get_sample_for_table, table, columns, data_dir, no_samples
                ): table
                for table, columns in schema["table_col_info"].items()
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_table),
                total=len(future_to_table),
                desc="Creating samples for each table",
            ):
                table = future_to_table[future]
                try:
                    data = future.result()
                    table_samples[table] = data
                except Exception as exc:
                    print(f"{table} generated an exception: {exc}")

    else:
        for table, columns in tqdm(
            schema["table_col_info"].items(), desc="Creating samples for each table"
        ):
            df = get_sample_for_table(table, columns)
            table_samples[table] = df

    for workload in workloads:
        source_path = os.path.join(plan_path, f"{workload}.json")
        out_path = os.path.join(target_path, f"{workload}.json")
        if os.path.exists(out_path):
            print(f"{out_path} already exists, skipping augmentation")
            continue

        run = load_json(source_path)
        print(f"Augment Sample Vectors for {source_path}")

        for p in tqdm(run["parsed_plans"]):
            augment_sample(table_samples=table_samples, col_stats=col_stats, p_node=p)

        print(f"Writing augmented sample vectors to {out_path}")
        with open(out_path, "w") as outfile:
            json.dump(run, outfile, default=dumper)
        os.chmod(target_path, 0o777)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment sample vectors")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--workloads", nargs="+", required=True, help="Workload Names")
    parser.add_argument(
        "--plan_path", type=str, required=True, help="Path to the plan file"
    )
    parser.add_argument(
        "--target_path", type=str, required=True, help="Path to the target file"
    )
    args = parser.parse_args()

    augment_sample_vectors(
        args.dataset, args.data_dir, args.plan_path, args.target_path, args.workloads
    )
