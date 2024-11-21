from pathlib import Path
from typing import List, Optional

from dgl.dataloading import DataLoader
from torch import nn
from wandb.apis.public import Run

from classes.classes import ModelConfig
from classes.workload_runs import WorkloadRuns
from training.bound_predictions_with_history import PredictionBounder
from training.evaluate_query_optimization import evaluate_join_order, evaluate_physical_plan_selection, \
    evaluate_access_path_selection
from training.training.checkpoint import save_csv
from training.training.metrics import Metric
from training.training.train import validate_model
from training.training.utils import find_early_stopping_metric


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

                _, predictions_dict = validate_model(config=config,
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
                    evaluate_join_order(predictions=predictions_dict)
                if "physical_operator" in str(test_path):
                    evaluate_physical_plan_selection(predictions=predictions_dict, with_idxs='_w_idx' in str(test_path))
                if "access_path_selection" in str(test_path):
                    evaluate_access_path_selection(predictions=predictions_dict)
        else:
            print("Skipping saving the test stats")
