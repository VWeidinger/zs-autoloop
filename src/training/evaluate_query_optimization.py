import numpy as np
import pandas as pd
from training.training.metrics import Metric, SpearmanRank, SelectedRuntime, SurpassedPlans, QError, MaxOverestimation, \
    MaxUnderestimation, PickRate, BalancedAccuracy

def evaluate_join_order(test_stats: dict):
    """ This method evaluates a given workload of queries, how well the join order was predicted."""
    predictions = test_stats["test_preds"]
    prediction_dataframe = pd.DataFrame(predictions)
    prediction_dataframe["hint"] = prediction_dataframe["hint"].fillna("")

    metrics = {
        "spearman_corr": SpearmanRank(),
        "selected_runtime": SelectedRuntime(),
        "surpassed_plans": SurpassedPlans(),
        "q_error": QError(),
        "max_overestimation": MaxOverestimation(),
        "max_underestimation": MaxUnderestimation()}

    aggregated_results = {key: [] for key in metrics}

    # Group dataframe by SQL and iterate over each group
    for sql, group_df in prediction_dataframe.groupby("sql"):
        preds = group_df["prediction"]
        labels = group_df["label"]

        # print(f"Found {len(group_df)} predictions for SQL query in {sql}")

        if len(group_df) == 1 or pd.isna(SpearmanRank().evaluate_metric(preds=preds, labels=labels)):
            print(f'Join order evaluation: Skipping SQL query in {sql} as Spearman correlation is invalid/NaN')
            continue

        for key, metric in metrics.items():
            aggregated_results[key].append(metric.evaluate_metric(preds=preds, labels=labels))

    print("Join order evaluation results:")
    print(f"Average Spearman correlation: {np.mean(aggregated_results['spearman_corr'])}")
    print(f"Total Selected runtime: {np.sum(aggregated_results['selected_runtime'])}")
    print(f"Average Ratio Surpassed plans: {np.mean(aggregated_results['surpassed_plans'])}")
    print(f"Average QError over plan alternatives: {np.mean(aggregated_results['q_error'])}")
    print(f"Average Max Overestimation: {np.mean(aggregated_results['max_overestimation'])}")
    print(f"Average Max Underestimation: {np.mean(aggregated_results['max_underestimation'])}")
    return


def evaluate_physical_plan_selection(test_stats: dict):
    """ This method evaluates a given workload of queries, how well physical operators were selected.
    ToDo: Test on real dataset"""

    predictions = test_stats["test_preds"]
    prediction_dataframe = pd.DataFrame(predictions)
    prediction_dataframe["hint"] = prediction_dataframe["hint"].fillna("")

    metrics = {"selected_runtime": SelectedRuntime(),
               "pick_rate": PickRate()}
    aggregated_results = {key: [] for key in metrics}

    # Group dataframe by SQL and iterate over each group
    for sql, group_df in prediction_dataframe.groupby("sql"):
        preds = group_df["prediction"]
        labels = group_df["label"]

        assert len(group_df) == 3, "Physical operator selection requires three different query plans per SQL statement"

        for key, metric in metrics.items():
            aggregated_results[key].append(metric.evaluate_metric(preds=preds, labels=labels))

    print("Physical operator evaluation results:")
    print(f"Total Selected runtime: {np.sum(aggregated_results['selected_runtime'])}")
    print(f"Average Pick Rate: {np.mean(aggregated_results['pick_rate'])}")
    return


def evaluate_access_path_selection(test_stats: dict):
    """ This method evaluates a given workload of queries, how well access paths were selected.
    ToDo: Test on real dataset"""
    predictions = test_stats["test_preds"]
    prediction_dataframe = pd.DataFrame(predictions)
    prediction_dataframe["hint"] = prediction_dataframe["hint"].fillna("")

    metrics = {"selected_runtime": SelectedRuntime()}  # ToDO: Implement Balanced accuracy, which is more complex
    aggregated_results = {key: [] for key in metrics}

    # Group dataframe by SQL and iterate over each group
    for sql, group_df in prediction_dataframe.groupby("sql"):
        preds = group_df["prediction"]
        labels = group_df["label"]

        assert len(group_df) == 2, "Access path selection requires exactly "

        for key, metric in metrics.items():
            aggregated_results[key].append(metric.evaluate_metric(preds=preds, labels=labels))

    print("Access path evaluation results:")
    print(f"Total Selected runtime: {np.sum(aggregated_results['selected_runtime'])}")
    return