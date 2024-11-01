import time
from pathlib import Path
from typing import List

import torch
import wandb
from dgl.dataloading import DataLoader
from torch import nn
from torch.optim import Optimizer
from wandb.apis.public import Run

from classes.classes import ModelConfig
from training.bound_predictions_with_history import PredictionBounder
from training.training.checkpoint import save_checkpoint
from training.training.metrics import Metric
from training.training.train import validate_model, train_epoch
from training.training.train_w_rank import RankModel, train_epoch_rank, validate_rank


def train_model(csv_stats: List,
                model_dir: Path,
                config: ModelConfig,
                train_loader: DataLoader,
                epochs_wo_improvement: int,
                epoch_offset: int,  # this is only for reporting purposes
                epoch: int,
                val_loader: DataLoader,
                model: nn.Module,
                optimizer: Optimizer,
                metrics: List[Metric],
                rank_metric: Metric,
                run: Run,
                pt_profiler: bool = False,
                train_w_rank_model: bool = False,
                bound_predictions: bool = False,
                prediction_bounder:PredictionBounder = None):
    if train_w_rank_model:
        # wrap the model with the rank model
        ranking_wrapper = RankModel(model)
    else:
        ranking_wrapper = None

    prof = None
    if pt_profiler:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1,
                                             warmup=1,
                                             active=3,
                                             repeat=1),
            # on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_output_dir),
            with_stack=True, record_shapes=True)
        prof.start()

    target_epochs = config.epochs if not train_w_rank_model else 20

    while epoch < target_epochs:
        epoch_stats = dict()
        epoch_stats.update(epoch=epoch + epoch_offset, train_w_rank=train_w_rank_model)
        epoch_start_time: float = time.perf_counter()

        any_best_metric = False
        if train_w_rank_model:
            train_epoch_rank(
                epoch_stats=epoch_stats,
                train_loader=train_loader,
                model=ranking_wrapper,
                optimizer=optimizer,
                max_epoch_tuples=config.max_epoch_tuples,
                custom_batch_to=config.batch_to_func,
                profiler=prof
            )

            # perform rank validation
            any_best_metric = validate_rank(
                val_loader=val_loader,
                model=ranking_wrapper,
                epoch_stats=epoch_stats,
                rank_metric=rank_metric,
                custom_batch_to=config.batch_to_func
            )
        else:
            train_epoch(epoch_stats=epoch_stats,
                        train_loader=train_loader,
                        model=model,
                        optimizer=optimizer,
                        max_epoch_tuples=config.max_epoch_tuples,
                        custom_batch_to=config.batch_to_func,
                        profiler=prof)

        any_best_metric = any_best_metric or validate_model(
            config=config,
            val_loader=val_loader,
            model=model,
            epoch=epoch,
            epoch_stats=epoch_stats,
            metrics=metrics,
            max_epoch_tuples=config.max_epoch_tuples,
            custom_batch_to=config.batch_to_func,
            log_all_queries=False,
            target_path=None,
            run=None,
            bound_predictions=bound_predictions,
            prediction_bounder=prediction_bounder)

        epoch_stats.update(epoch_time=time.perf_counter() - epoch_start_time)

        if run:
            wandb.log(epoch_stats)

        # See if we can already stop the training
        stop_early = False
        if not any_best_metric:
            epochs_wo_improvement += 1
            if config.early_stopping_patience is not None and epochs_wo_improvement > config.early_stopping_patience:
                stop_early = True
        else:
            epochs_wo_improvement = 0

        # also set finished to true if this is the last epoch
        if epoch == config.epochs - 1:
            stop_early = True

        epoch_stats.update(stop_early=stop_early)
        print(f"--> Epochs without improvement: {epochs_wo_improvement}/{config.early_stopping_patience}")

        # save stats to file
        csv_stats.append(epoch_stats)

        # save current state of training allowing us to resume if this is stopped
        save_checkpoint(epochs_wo_improvement=epochs_wo_improvement,
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        target_path=model_dir,
                        config=config,
                        metrics=metrics,
                        csv_stats=csv_stats,
                        finished=stop_early,
                        rank_model=train_w_rank_model)
        epoch += 1
        if stop_early:
            print(f"Early stopping kicked in due to no improvement in {config.early_stopping_patience} epochs")
            break

    if prof:
        prof.stop()
        profile_output_dir = str(model_dir / f'{config.name.NAME}_{config.seed}')
        print(f'Profile with pytorch profiler to logdir {profile_output_dir}')
        prof.export_chrome_trace(profile_output_dir + '.prof')
