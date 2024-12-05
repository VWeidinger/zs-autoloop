from pathlib import Path
from typing import List, Optional

import wandb
from dgl.dataloading import DataLoader
from torch import nn
from wandb.apis.public import Run

from classes.classes import ModelConfig
from training.bound_predictions_with_history import PredictionBounder
from training.evaluate_query_optimization import evaluate_join_order, evaluate_physical_plan_selection, \
    evaluate_access_path_selection
from training.training.checkpoint import save_csv
from training.training.metrics import Metric
from training.training.utils import find_early_stopping_metric, prefix_dict
from training.training.validate import validate_model


def test_model(config: ModelConfig,
               test_loaders: List[DataLoader],
               test_csv_paths: List[Path],
               model_dir: Path,
               metrics: List[Metric],
               model: nn.Module,
               epoch: int,
               run: Run,
               file_suffix: str,
               bound_predictions: bool = False,
               prediction_bounder: Optional[PredictionBounder] = None,
               load_best_checkpoint: bool = True,
               log_prefix: str = 'test',
               is_per_epoch: bool = False):
    if test_loaders is not None:
        if not (model_dir is None or config.name.NAME is None):
            for test_path, test_loader in zip(test_csv_paths, test_loaders):
                print(f"Starting validation for {test_path}")
                test_stats = dict()

                # In case of retraining, do not load the totally best model but the latest one
                if load_best_checkpoint:
                    early_stop_m = find_early_stopping_metric(metrics)
                    model.load_state_dict(early_stop_m.best_model)

                _, predictions_dict = validate_model(
                    val_loader=test_loader,
                    model=model,
                    epoch=epoch,
                    epoch_stats=test_stats,
                    metrics=metrics,
                    custom_batch_to=config.batch_to_func,
                    log_all_queries=False,
                    target_path=test_path,
                    bound_predictions=bound_predictions,
                    prediction_bounder=prediction_bounder,
                    update_metrics=False,
                    log_prefix=log_prefix
                )

                if run:
                    # extract filename from path
                    filename = test_path.stem
                    # remove the number annotation
                    filename = '_'.join(filename.split('_')[:-1])

                    if is_per_epoch:
                        wandb.log(prefix_dict(test_stats, f'{filename}.json/'), step=epoch)
                    else:
                        wandb.run.summary.update(prefix_dict(test_stats, f'{filename}.json/'))

                save_csv([test_stats], str(test_path) + f"_test_stats{file_suffix}.csv")

                if "join_order" in str(test_path):
                    evaluate_join_order(predictions=predictions_dict, is_per_epoch=is_per_epoch, epoch=epoch, run=run)
                if "physical_operator" in str(test_path):
                    evaluate_physical_plan_selection(predictions=predictions_dict, with_idxs='_w_idx' in str(test_path),
                                                     is_per_epoch=is_per_epoch, epoch=epoch, run=run)
                if "access_path_selection" in str(test_path):
                    evaluate_access_path_selection(predictions=predictions_dict, is_per_epoch=is_per_epoch, epoch=epoch, run=run)
        else:
            print("Skipping saving the test stats")
