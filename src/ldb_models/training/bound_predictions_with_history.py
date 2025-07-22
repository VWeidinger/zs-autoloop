import concurrent.futures
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd
from ldb_schema.utils.utils import t_print
from tabulate import tabulate
from tqdm import tqdm

from training.dataset.plan_dataset import PlanDataset


def extract_plan_metadata(plan):
    # num joins
    # base table sizes
    # sum intermed card

    num_joins = 0
    base_table_sizes = []
    sum_intermed_card = 0

    op_name = plan["plan_parameters"]["op_name"]

    if op_name in ["Hash Join", "Nested Loop", "Index Nested Loop", "Merge Join"]:
        num_joins += 1
        sum_intermed_card += plan["plan_parameters"]["est_card"]
    elif op_name in ["Seq Scan", "Index Scan", "Index Only Scan"]:
        base_table_sizes.append(plan["plan_parameters"]["est_card"])
    elif op_name in ["Aggregate", "Simple Aggregate", "Hash", "Materialize", "Sort"]:
        pass
    else:
        raise ValueError(f"Unexpected op_name: {op_name}")

    # call on children
    for child in plan["children"]:
        child_num_joins, child_base_table_sizes, child_sum_intermed_card = (
            extract_plan_metadata(child)
        )
        num_joins += child_num_joins
        base_table_sizes.extend(child_base_table_sizes)
        sum_intermed_card += child_sum_intermed_card

    return num_joins, base_table_sizes, sum_intermed_card


def extract_plan_metadata_for_dataset(dataset, include_timeout=False):
    plans = dataset.plans

    stats_list = []

    for plan in plans:
        if plan["timeout"] and not include_timeout:
            continue

        num_joins, base_table_sizes, sum_intermed_card = extract_plan_metadata(plan)

        runtime = plan["plan_runtime_ms"]

        pg_cost = plan["plan_parameters"]["pg_cost"]
        stats_list.append(
            {
                "num_joins": num_joins,
                "base_table_sizes": base_table_sizes,
                "largest_table_size": max(base_table_sizes),
                "snd_largest_table_size": sorted(base_table_sizes)[-2]
                if len(base_table_sizes) > 1
                else 0,
                "trd_largest_table_size": sorted(base_table_sizes)[-3]
                if len(base_table_sizes) > 2
                else 0,
                "tables_larger_10k": sum(
                    [1 for size in base_table_sizes if size > 10000]
                ),
                "tables_larger_100k": sum(
                    [1 for size in base_table_sizes if size > 100000]
                ),
                "tables_larger_1m": sum(
                    [1 for size in base_table_sizes if size > 1000000]
                ),
                "sum_intermed_card": sum_intermed_card,
                "runtime": runtime,
                "pg_cost": pg_cost,
                "scaled_pg_cost": pg_cost * 0.00018233621202897073 + 4261.925798066092,
                "dataset": "unknown",  # plan['dataset']
            }
        )

    # convert to pandas dataframe
    stats_df = pd.DataFrame(
        stats_list,
        columns=[
            "num_joins",
            "base_table_sizes",
            "largest_table_size",
            "snd_largest_table_size",
            "trd_largest_table_size",
            "tables_larger_10k",
            "tables_larger_100k",
            "tables_larger_1m",
            "sum_intermed_card",
            "runtime",
            "pg_cost",
            "scaled_pg_cost",
            "dataset",
        ],
    )

    return stats_df


class PredictionBounder:
    def __init__(self, history_dataset: PlanDataset):
        self.history_dataset = history_dataset
        stats_df = extract_plan_metadata_for_dataset(
            self.history_dataset, include_timeout=False
        )
        self.stats_df = stats_df

        # create duckdb connection
        self.con = duckdb.connect()
        self.con.execute("CREATE TABLE plans AS SELECT * FROM stats_df")

    def bound_prediction(self, plan, prediction: float):
        related_plans = self.bound_prediction_duckdb(plan, prediction)

        stats = {"num_related": len(related_plans)}

        # extract min / max / mean / std of runtimes in related_plans
        if len(related_plans) < 3:
            return prediction, stats

        related_runtimes = related_plans["runtime"]
        min_runtime = related_runtimes.min()
        max_runtime = related_runtimes.max()
        mean_runtime = related_runtimes.mean()

        if len(related_runtimes) == 1:
            # return avg between prediction and related runtime
            return (prediction + mean_runtime) / 2, stats
        else:
            # return prediction bounded by min and max runtime
            return max(min(prediction, max_runtime), min_runtime), stats

    def bound_prediction_duckdb(self, plan, prediction: float):
        # extract metadata from plan
        num_joins, base_table_sizes, sum_intermed_card = extract_plan_metadata(plan)
        plan_largest = max(base_table_sizes)
        plan_2nd_largest = (
            sorted(base_table_sizes)[-2] if len(base_table_sizes) > 1 else 0
        )
        plan_3rd_largest = (
            sorted(base_table_sizes)[-3] if len(base_table_sizes) > 2 else 0
        )

        # query for related plans
        query = f"""
        SELECT runtime FROM plans
        WHERE tables_larger_1m = {len([size for size in base_table_sizes if size > 1000000])}
        AND tables_larger_100k = {len([size for size in base_table_sizes if size > 100000])}
        AND tables_larger_10k = {len([size for size in base_table_sizes if size > 10000])}
        AND largest_table_size BETWEEN {0.8 * plan_largest} AND {1.2 * plan_largest}
        AND pg_cost BETWEEN {0.5 * plan["plan_parameters"]["pg_cost"]} AND {1.5 * plan["plan_parameters"]["pg_cost"]}
        """

        if plan_2nd_largest > 100000:
            query += f"AND snd_largest_table_size BETWEEN {0.9 * plan_2nd_largest} AND {1.1 * plan_2nd_largest}\n"
        # else:
        #     query +=  f"AND snd_largest_table_size <= 100000\n"

        if plan_3rd_largest > 100000:
            query += f"AND trd_largest_table_size BETWEEN {0.9 * plan_3rd_largest} AND {1.1 * plan_3rd_largest}\n"
        # else:
        #     query += f"AND trd_largest_table_size <= 100000\n"

        related_plans = self.con.execute(query).fetchdf()

        return related_plans

    def bound_prediction_pandas(self, plan, prediction: float):
        # extract metadata from plan
        num_joins, base_table_sizes, sum_intermed_card = extract_plan_metadata(plan)

        # sort out plans with different table sizes
        related_plans = self.stats_df[
            self.stats_df["tables_larger_1m"]
            == len([size for size in base_table_sizes if size > 1000000])
        ]
        related_plans = related_plans[
            related_plans["tables_larger_100k"]
            == len([size for size in base_table_sizes if size > 100000])
        ]
        related_plans = related_plans[
            related_plans["tables_larger_10k"]
            == len([size for size in base_table_sizes if size > 10000])
        ]

        # filter out plans where the largest / 2nd largest / 3rd largest table size are differing by more than 10%
        plan_largest = max(base_table_sizes)
        plan_2nd_largest = (
            sorted(base_table_sizes)[-2] if len(base_table_sizes) > 1 else 0
        )
        plan_3rd_largest = (
            sorted(base_table_sizes)[-3] if len(base_table_sizes) > 2 else 0
        )
        related_plans = related_plans[
            related_plans["largest_table_size"].between(
                0.9 * plan_largest, 1.1 * plan_largest
            )
        ]
        if plan_2nd_largest > 100000:
            related_plans = related_plans[
                related_plans["snd_largest_table_size"].between(
                    0.9 * plan_2nd_largest, 1.1 * plan_2nd_largest
                )
            ]
        if plan_3rd_largest > 100000:
            related_plans = related_plans[
                related_plans["trd_largest_table_size"].between(
                    0.9 * plan_3rd_largest, 1.1 * plan_3rd_largest
                )
            ]

        if len(related_plans) > 50:
            # reduce by pg_cost
            related_plans = related_plans[
                related_plans["pg_cost"].between(
                    0.7 * plan["plan_parameters"]["pg_cost"],
                    1.3 * plan["plan_parameters"]["pg_cost"],
                )
            ]

        return related_plans


def bound_predictions_helper(
    target_dataset: PlanDataset, predictions, prediction_bounder: PredictionBounder
):
    # make sure the dataset fits
    assert len(target_dataset.plans) == len(predictions)

    bounded_predictions = []

    combined_stats = defaultdict(list)

    parallelize = False
    if not parallelize:
        for plan, prediction in tqdm(
            zip(target_dataset.plans, predictions),
            desc="Bounding predictions",
            total=len(predictions),
        ):
            bounded_prediction, stats = prediction_bounder.bound_prediction(
                plan, prediction
            )
            bounded_predictions.append(bounded_prediction)

            for key, value in stats.items():
                combined_stats[key].append(value)
    else:
        bounded_predictions_dict = dict()
        t_print(f"Bounding predictions for {len(predictions)} plans")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_plan = {
                executor.submit(
                    prediction_bounder.bound_prediction, plan, prediction
                ): id
                for plan, prediction, id in zip(
                    target_dataset.plans, predictions, range(len(predictions))
                )
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_plan),
                desc="Bounding predictions",
                total=len(predictions),
            ):
                id = future_to_plan[future]
                bounded_prediction, stats = future.result()
                bounded_predictions_dict[id] = bounded_prediction

                for key, value in stats.items():
                    combined_stats[key].append(value)

        assert len(bounded_predictions_dict) == len(predictions)
        for i in range(len(predictions)):
            bounded_predictions.append(bounded_predictions_dict[i])

    tabulate_rows = []
    for key, value in combined_stats.items():
        tabulate_rows.append(
            [
                key,
                min(value),
                max(value),
                np.percentile(value, 5),
                np.percentile(value, 50),
                np.percentile(value, 95),
                np.mean(value),
            ]
        )

    print(
        tabulate(
            tabulate_rows, headers=["key", "min", "max", "p05", "p50", "p95", "avg"]
        )
    )

    return bounded_predictions
