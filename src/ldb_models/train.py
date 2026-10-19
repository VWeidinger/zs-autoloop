import time
from pathlib import Path
from typing import List

import torch
from ldb_models.classes import ModelConfig
from dgl.dataloading import DataLoader
from torch import nn
from torch.optim import Optimizer
from ldb_models.training.bound_predictions_with_history import PredictionBounder
from ldb_models.training.scheduler import step_scheduler
from ldb_models.training.training.checkpoint import save_checkpoint
from ldb_models.training.training.metrics import Metric
from ldb_models.training.training.test import test_model
from ldb_models.training.training.train_epoch import train_epoch
from ldb_models.training.training.validate import validate_model
from wandb.apis.public import Run


def train_model(
    csv_stats: List,
    model_dir: Path,
    config: ModelConfig,
    train_loader: DataLoader,
    test_loaders: List[DataLoader],
    test_csv_paths: List[Path],
    epochs_wo_improvement: int,
    epoch_offset: int,  # this is only for reporting purposes
    epoch: int,
    target_epochs: int,
    val_loader: DataLoader,
    model: nn.Module,
    optimizer: Optimizer,
    metrics: List[Metric],
    wandb_run: Run,
    pt_profiler: bool = False,
    train_w_rank_model: bool = False,
    bound_predictions: bool = False,
    prediction_bounder: PredictionBounder = None,
    quick_run: bool = False,
    scheduler=None,
    training_info: dict = None,
):
    prof = None
    if pt_profiler:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            # on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_output_dir),
            with_stack=True,
            record_shapes=True,
        )
        prof.start()

    while epoch < target_epochs:
        print(f"Epoch {epoch + epoch_offset + 1}/{target_epochs + epoch_offset}")
        epoch_stats = dict()
        epoch_stats.update(epoch=epoch + epoch_offset, train_w_rank=train_w_rank_model)
        epoch_start_time: float = time.perf_counter()

        wandb_plots_dict = dict() if wandb_run is not None else None

        any_best_metric = False
        train_epoch(
            epoch_stats=epoch_stats,
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            max_epoch_tuples=config.max_epoch_tuples,
            custom_batch_to=config.batch_to_func,
            model_config=config,
            profiler=prof,
        )

        # validate model
        new_best_metric_seen, _ = validate_model(
            val_loader=val_loader,
            model=model,
            epoch=epoch,
            epoch_stats=epoch_stats,
            metrics=metrics,
            max_epoch_tuples=config.max_epoch_tuples,
            custom_batch_to=config.batch_to_func,
            log_all_queries=False,
            target_path=None,
            bound_predictions=bound_predictions,
            prediction_bounder=prediction_bounder,
            log_prefix="epoch_val",
            model_config=config,
            wandb_plots_dict=wandb_plots_dict,
        )
        any_best_metric = any_best_metric or new_best_metric_seen

        # validate on test sets with test evaluations
        assert len(test_loaders) == len(test_csv_paths), (
            f"Mismatch in test loaders and csv paths lengths: {len(test_loaders)} vs {len(test_csv_paths)}"
        )
        test_model(
            config=config,
            test_csv_paths=test_csv_paths,
            test_loaders=test_loaders,
            model_dir=model_dir,
            epoch=epoch,
            model=model,
            metrics=metrics,
            wandb_run=wandb_run,
            file_suffix=f"_epoch{epoch}",
            bound_predictions=bound_predictions,
            prediction_bounder=prediction_bounder,
            load_best_checkpoint=False,
            log_prefix="epoch_test",
            is_per_epoch=True,
        )

        # See if we can already stop the training
        stop_early = False
        if not any_best_metric:
            epochs_wo_improvement += 1
            if (
                config.early_stopping_patience is not None
                and epochs_wo_improvement > config.early_stopping_patience
            ):
                stop_early = True
        else:
            epochs_wo_improvement = 0

        # also set finished to true if this is the last epoch
        if epoch == target_epochs - 1:
            stop_early = True

        epoch_stats.update(stop_early=stop_early)
        print(
            f"--> Epochs without improvement: {epochs_wo_improvement}/{config.early_stopping_patience}"
        )

        # save stats to file
        csv_stats.append(epoch_stats)

        if not quick_run:  # in case of quick-run we do not store artifacts
            # save current state of training allowing us to resume if this is stopped
            save_checkpoint(
                epochs_wo_improvement=epochs_wo_improvement,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                target_path=model_dir,
                config=config,
                metrics=metrics,
                csv_stats=csv_stats,
                finished=stop_early,
                rank_model=train_w_rank_model,
                training_info=training_info,
            )

        epoch_stats.update(epoch_time=time.perf_counter() - epoch_start_time)

        if wandb_run:
            wandb_run.log(epoch_stats, step=epoch)
            # wandb_run.log(wandb_plots_dict, step=epoch) # do not log plots per epoch

        if (
            hasattr(config, "online_hard_example_mining")
            and config.online_hard_example_mining
        ):
            if epoch % config.ohem_resort_after_num_epochs == 0:
                print("Resorting OHEM dataset...")
                train_loader.dataset.update_difficulty()
                val_loader.dataset.update_difficulty()

        epoch += 1
        if stop_early:
            print(
                f"Early stopping kicked in due to no improvement in {config.early_stopping_patience} epochs"
            )
            break

        # step the scheduler
        step_scheduler(
            scheduler=scheduler,
            epoch=epoch,
            avg_loss=epoch_stats.get("epoch_val_loss", None),
        )

        # Get the learning rate used for this epoch
        learning_rate = (
            scheduler.get_last_lr()[0]
            if hasattr(scheduler, "get_last_lr")
            else scheduler.get_lr()[0]
        )
        if wandb_run:
            # Read-only scheduler access used purely for W&B logging.
            # Scheduler stepping/state updates are handled above by step_scheduler().
            wandb_run.log({"learning_rate": learning_rate}, step=epoch)

    if prof:
        prof.stop()
        profile_output_dir = str(
            model_dir / f"{config.name.NAME}_{config.seed}v{config.version}"
        )
        print(f"Profile with pytorch profiler to logdir {profile_output_dir}")
        prof.export_chrome_trace(profile_output_dir + ".prof")
