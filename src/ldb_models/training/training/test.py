from pathlib import Path
from typing import List, Optional

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
               wandb_run: Run,
               file_suffix: str,
               bound_predictions: bool = False,
               prediction_bounder: Optional[PredictionBounder] = None,
               load_best_checkpoint: bool = True,
               log_prefix: str = 'test',
               is_per_epoch: bool = False,
               quick_run: bool = False):
    # initialize variables for aggregated reporting across all experiments
    job_total_selected_runtime = -1
    job_total_slowdown = None
    joblight_total_selected_runtime = -1
    joblight_total_slowdown = None
    physop_total_selected_runtime = -1
    physop_total_slowdown = None
    physopidx_total_selected_runtime = -1
    physopidx_total_slowdown = None
    accesspath_total_selected_runtime = -1
    accesspath_total_slowdown = None

    if test_loaders is not None:
        if not (model_dir is None or config.name.NAME is None):

            # In case of retraining, do not load the totally best model but the latest one
            if load_best_checkpoint:
                early_stop_m = find_early_stopping_metric(metrics)
                model.load_state_dict(early_stop_m.best_model)

            for test_path, test_loader in zip(test_csv_paths, test_loaders):
                print(f"Starting validation for {test_path}")
                test_stats = dict()
                wandb_plots_dict = dict() if wandb_run is not None else None

                _, predictions_dict = validate_model(
                    val_loader=test_loader,
                    model=model,
                    epoch=epoch,
                    epoch_stats=test_stats,
                    wandb_plots_dict=wandb_plots_dict,
                    metrics=metrics,
                    custom_batch_to=config.batch_to_func,
                    log_all_queries=True,
                    target_path=test_path,
                    bound_predictions=bound_predictions,
                    prediction_bounder=prediction_bounder,
                    update_metrics=False,  # do not update best seen metrics - this is determined by val set
                    store_model_snapshot=False,  # do not update best model snapshot - this is determined by val set
                    log_prefix=log_prefix,
                    show_best_seen=False,  # do not show - it was computed during val i.e. not relevant
                    model_config=config,
                    quick_run=quick_run
                )

                # extract filename from path
                filename = test_path.stem
                # remove _seed from name (this is different from the _s0 suffix)
                filename = '_'.join(filename.split('_')[:-1])

                if wandb_run is not None:
                    prefixed = prefix_dict(test_stats, f'{filename}.json/')
                    if is_per_epoch:
                        # prefixed_plots = prefix_dict(wandb_plots_dict, f'{filename}.json/') # do not log plots per epoch
                        wandb_run.log(prefixed, step=epoch)
                        # wandb_run.log(prefixed_plots, step=epoch)
                    else:
                        prefixed_plots = prefix_dict(wandb_plots_dict, f'{filename}.json/')
                        wandb_run.summary.update(prefixed)
                        wandb_run.log(prefixed_plots)

                if not is_per_epoch:
                    save_csv([test_stats], str(test_path) + f"_test_stats{file_suffix}.csv")

                if "join_order" in filename:
                    if filename == 'join_order':
                        wl_name = 'job_light'  # job_light
                    elif filename.startswith('join_order_job'):  # could also be 'join_order_job_all_valid'
                        wl_name = 'job'
                    else:
                        raise ValueError(f"Unknown workload name: {filename}")
                    rt, slowdown = evaluate_join_order(predictions=predictions_dict, is_per_epoch=is_per_epoch,
                                                       epoch=epoch,
                                        wandb_run=wandb_run, wl_name=wl_name)

                    if wl_name == 'job':
                        job_total_selected_runtime = rt
                        job_total_slowdown = slowdown
                    elif wl_name == 'job_light':
                        joblight_total_selected_runtime = rt
                        joblight_total_slowdown = slowdown
                    else:
                        raise ValueError(f"Unknown workload name: {wl_name}")


                if "physical_operator" in str(test_path):
                    with_idxs = '_w_idx' in str(test_path)
                    rt, slowdown = evaluate_physical_plan_selection(predictions=predictions_dict, with_idxs=with_idxs,
                                                                    is_per_epoch=is_per_epoch, epoch=epoch,
                                                                    wandb_run=wandb_run)
                    if with_idxs:
                        physopidx_total_selected_runtime = rt
                        physopidx_total_slowdown = slowdown
                    else:
                        physop_total_selected_runtime = rt
                        physop_total_slowdown = slowdown

                if "access_path_selection" in str(test_path):
                    if 'with_aggregate' in str(test_path):
                        ap_w_aggregate = True
                        wl_name = 'w_aggregate'
                    else:
                        ap_w_aggregate = False
                        wl_name = ''

                    res = evaluate_access_path_selection(
                        predictions=predictions_dict, is_per_epoch=is_per_epoch, epoch=epoch,
                        wandb_run=wandb_run, wl_name=wl_name)

                    if not ap_w_aggregate:
                        accesspath_total_selected_runtime, accesspath_total_slowdown = res
        else:
            print("Skipping saving the test stats")

    # aggregate metrics
    if job_total_selected_runtime != -1 and joblight_total_selected_runtime != -1 and physop_total_selected_runtime != -1 and physopidx_total_selected_runtime != -1 and accesspath_total_selected_runtime != -1:
        # if not all 5 workloads are present, do not aggregate
        assert job_total_selected_runtime is not None
        assert joblight_total_selected_runtime is not None
        assert physop_total_selected_runtime is not None
        assert physopidx_total_selected_runtime is not None
        assert accesspath_total_selected_runtime is not None

        total_selected_runtime = job_total_selected_runtime + joblight_total_selected_runtime + physop_total_selected_runtime + physopidx_total_selected_runtime + accesspath_total_selected_runtime
        avg_total_slowdown = (
                                         job_total_slowdown + joblight_total_slowdown + physop_total_slowdown + physopidx_total_slowdown + accesspath_total_slowdown) / 5

        if wandb_run is not None:
            if is_per_epoch:
                wandb_run.log({
                    'qo_total/selected_runtime': total_selected_runtime,
                    'qo_total/avg_slowdown': avg_total_slowdown
                }, step=epoch)
            else:
                wandb_run.summary.update({
                    'qo_total/selected_runtime': total_selected_runtime,
                    'qo_total/avg_slowdown': avg_total_slowdown
                })
