from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from dgl.dataloading import DataLoader
from torch import nn
from wandb.apis.public import Run

from classes.classes import ModelConfig
from classes.workload_runs import WorkloadRuns
from training.bound_predictions_with_history import PredictionBounder
from training.training.checkpoint import save_csv
from training.training.metrics import Metric, SpearmanRank, SelectedRuntime, SurpassedPlans, QError, MaxOverestimation, \
    MaxUnderestimation
from training.training.train import validate_model
from training.training.utils import find_early_stopping_metric


def evaluate_join_order(test_stats: dict):
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


def predict_model(mode: str,
                  config: ModelConfig,
                  test_loaders: List[DataLoader],
                  workload_runs: WorkloadRuns,
                  model_dir: Path,
                  metrics: List[Metric],
                  model: nn.Module,
                  epoch: int,
                  run: Run,
                  file_suffix: str,
                  bound_predictions: bool = False,
                  prediction_bounder: Optional[PredictionBounder] = None):
    if test_loaders is not None:
        if not (model_dir is None or config.name.NAME is None):
            for test_path, test_loader in zip(workload_runs.target_test_csv_paths, test_loaders):
                print(f"Starting validation for {test_path}")
                test_stats = dict()

                # In case of retraining, do not load the totally best model but the latest one
                if mode != "retrain":
                    early_stop_m = find_early_stopping_metric(metrics)
                    model.load_state_dict(early_stop_m.best_model)

                validate_model(config=config,
                               val_loader=test_loader,
                               model=model,
                               epoch=epoch,
                               epoch_stats=test_stats,
                               metrics=metrics,
                               custom_batch_to=config.batch_to_func,
                               log_all_queries=True,
                               run=run,
                               model_dir=model_dir,
                               target_path=test_path,
                               bound_predictions=bound_predictions,
                               prediction_bounder=prediction_bounder)

                save_csv([test_stats], str(test_path) + f"_test_stats{file_suffix}.csv")

                if "join_order" in str(test_path):
                    evaluate_join_order(test_stats=test_stats)
        else:
            print("Skipping saving the test stats")
