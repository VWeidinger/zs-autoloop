import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import wandb
from classes import ModelConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.bound_predictions_with_history import (
    PredictionBounder,
    bound_predictions_helper,
)
from training.training.checkpoint import save_csv
from training.training.metrics import Metric, QError
from training.training.outputs import LossOutput, ModelOutput


def validate_model(
    val_loader: DataLoader,
    model: nn.Module,
    target_path: Optional[Path],
    epoch: int = 0,
    epoch_stats: dict = None,
    metrics: List[Metric] = None,
    max_epoch_tuples: int = None,
    custom_batch_to: Callable = None,
    log_all_queries: bool = False,
    bound_predictions: bool = False,
    prediction_bounder: Optional[PredictionBounder] = None,
    log_prefix: str = "val",
    update_metrics: bool = True,
    store_model_snapshot: bool = True,
    show_best_seen: bool = True,
    model_config: ModelConfig = None,
    quick_run: bool = False,
    wandb_plots_dict: Dict = None,
) -> Tuple[bool, Dict]:
    model.eval()

    with torch.autograd.no_grad():
        losses = {
            "val_loss": 0,
            "plan_loss": 0,
            "subplan_loss": 0,
            "monotonicity_loss": 0,
            "card_loss": 0,
        }

        # Collect as tensors for efficiency
        labels = []
        preds = []
        probs = []
        query_stats_dict = defaultdict(list)

        subplan_preds_list, subplan_pred_card_list = [], []

        # evaluate test set using model
        test_start_t = time.perf_counter()
        val_num_tuples = 0
        len_val_loader = len(val_loader)
        # avoid ZeroDivisionError when dataset is small
        debug_denom = max(1, len_val_loader // 2)
        subplan_estimation = (
            hasattr(model_config, "subplan_estimation")
            and model_config.subplan_estimation
        )
        cardinality_estimation = (
            hasattr(model_config, "cardinality_estimation")
            and model_config.cardinality_estimation
        )
        for batch_idx, batch in enumerate(tqdm(val_loader)):
            debug = batch_idx % debug_denom == 0
            if (
                max_epoch_tuples is not None
                and batch_idx * model_config.batch_size > max_epoch_tuples
            ):
                break

            val_num_tuples += model_config.batch_size
            # prepare batch
            input_model, label, query_stats = custom_batch_to(
                batch, model.device, model.label_norm
            )
            # forward pass
            output: ModelOutput = model(
                input_model,
                plan_dict=query_stats.get("plan_dict", None),
                debug=debug,
                plan_labels=label,
                subplan_labels=query_stats.get("subplan_labels", None),
                card_labels=query_stats.get("subplan_card_labels", None),
                num_samples=model_config.num_samples
                if hasattr(model_config, "num_samples")
                else None,
            )
            # compute loss
            loss_output: LossOutput = model.compute_loss(
                output=output, query_stats=query_stats, labels=label
            )
            # accumulate losses
            losses["val_loss"] += loss_output.loss.detach().cpu().item()
            losses["plan_loss"] += loss_output.plan_loss.detach().cpu().item()
            if loss_output.monotonicity_loss is not None:
                losses["monotonicity_loss"] += (
                    loss_output.monotonicity_loss.detach().cpu().item()
                )
            # inverse transform the predictions and labels
            curr_pred = output.plan_predictions.cpu().numpy()
            curr_label = label.cpu().numpy()
            if model.label_norm is not None:
                curr_pred = model.label_norm.inverse_transform(curr_pred)
                curr_label = model.label_norm.inverse_transform(
                    curr_label.reshape(-1, 1)
                ).reshape(-1)
            preds.append(curr_pred.reshape(-1))
            labels.append(curr_label.reshape(-1))
            # add all info to query stats dict
            for key, value in query_stats.items():
                if key in ["subplan_est_card_labels", "subplan_card_labels"]:
                    # keep as list of lists
                    query_stats_dict[key].extend(
                        [v.cpu().numpy().tolist() for v in value]
                    )
                else:
                    query_stats_dict[key].extend(value)
            # collect subplan predictions if available
            if subplan_estimation:
                losses["subplan_loss"] += loss_output.subplan_loss.detach().cpu().item()
                subplan_preds_list.extend(
                    [
                        t.cpu().numpy().tolist()
                        for t in torch.split(
                            output.subplan_predictions, output.subplan_lengths.tolist()
                        )
                    ]
                )
            if cardinality_estimation:
                losses["card_loss"] += loss_output.card_loss.detach().cpu().item()
                subplan_pred_card_list.extend(
                    [
                        t.cpu().numpy().tolist()
                        for t in torch.split(
                            output.subplan_card_predictions,
                            output.subplan_lengths.tolist(),
                        )
                    ]
                )

        labels = np.concatenate(labels, axis=0)
        preds = np.concatenate(preds, axis=0)

        if len(subplan_preds_list) > 0:
            query_stats_dict["subplan_predictions"] = subplan_preds_list
        if len(subplan_pred_card_list) > 0:
            query_stats_dict["subplan_card_predictions"] = subplan_pred_card_list

        if epoch_stats is not None:
            epoch_stats[f"{log_prefix}_time"] = time.perf_counter() - test_start_t
            epoch_stats[f"{log_prefix}_num_tuples"] = val_num_tuples
            length = len(val_loader)
            for key, value in losses.items():
                if value > 0:
                    epoch_stats[f"{log_prefix}_{key}"] = value / length

            if (
                "subplan_card_labels" in query_stats_dict
                and "subplan_est_card_labels" in query_stats_dict
            ):
                est_metrics = compute_card_qerror_metrics(
                    actual=query_stats_dict["subplan_card_labels"],
                    compared=query_stats_dict["subplan_est_card_labels"],
                    log_scale=model_config.log_transform_card_labels
                    if hasattr(model_config, "log_transform_card_labels")
                    else False,
                )
                for k, v in est_metrics.items():
                    epoch_stats[f"{log_prefix}_est_card_qerror_{k}"] = v
                if cardinality_estimation:
                    flat_pred_cards = [
                        item for sublist in subplan_pred_card_list for item in sublist
                    ]
                    pred_metrics = compute_card_qerror_metrics(
                        actual=query_stats_dict["subplan_card_labels"],
                        compared=flat_pred_cards,
                        log_scale=model_config.log_transform_card_labels
                        if hasattr(model_config, "log_transform_card_labels")
                        else False,
                    )
                    for k, v in pred_metrics.items():
                        epoch_stats[f"{log_prefix}_pred_card_qerror_{k}"] = v

        epoch_stats.update(val_std=np.std(labels))

        if bound_predictions:
            bounded_preds = bound_predictions_helper(
                target_dataset=val_loader.dataset,
                predictions=preds,
                prediction_bounder=prediction_bounder,
            )
            bound_list = [True, False]
        else:
            bound_list = [False]

        # save best model for every metric
        any_best_metric = False
        metrics_dict = dict()
        if metrics is not None:
            for bound in bound_list:
                for metric in metrics:
                    best_seen = metric.evaluate(
                        metrics_dict=metrics_dict,
                        model=model if store_model_snapshot else None,
                        labels=labels,
                        preds=preds if not bound else bounded_preds,
                        probs=probs,
                        logging_suffix="bounded" if bound else None,
                        update_best_seen=not bound and update_metrics,
                        show_best_seen=show_best_seen,
                    )

                    if (
                        best_seen
                        and metric.early_stopping_metric
                        and not bound
                        and update_metrics
                    ):
                        any_best_metric = True
                        print(f"New best model for {metric.metric_name}")

        # add metrics dict to epoch stats
        for key, value in metrics_dict.items():
            epoch_stats[f"{log_prefix}_{key}"] = value

        # drop the query stats that are not available for all queries (plan_dict, subplan_labels))
        drop_keys = [
            "plan_dict",
            "subplan_labels",
            "subplan_card_labels",
            "subplan_est_card_labels",
        ]
        for key in drop_keys:
            if key in query_stats_dict:
                query_stats_dict.pop(key)

        predictions = to_rows(
            dict(
                label=[float(f) for f in labels],
                prediction=[float(f) for f in preds],
                **query_stats_dict,
            )
        )

        if wandb_plots_dict is not None:
            # create a q-error over query stats
            predictions_df = pd.DataFrame(predictions)

            columns = ["num_tables", "join_op", "scan_op"]

            for plan_statistic_name in columns:
                preds_grouped = predictions_df.groupby(plan_statistic_name)

                # assemble metrics
                qerror = QError()
                qerror95 = QError(95)

                plot_data = []
                for val, group in preds_grouped:
                    qerror_median = qerror.evaluate_metric(
                        group["label"], group["prediction"]
                    )
                    qerror_95 = qerror95.evaluate_metric(
                        group["label"], group["prediction"]
                    )
                    qerror_max = group["qerror"].max()

                    plot_data.append([val, qerror_median, qerror_95, qerror_max])

                # assemble wandb table
                table = wandb.Table(
                    data=plot_data,
                    columns=[
                        plan_statistic_name,
                        "qerror_median",
                        "qerror_95",
                        "qerror_max",
                    ],
                )
                wandb_plots_dict[f"{log_prefix}_qerror_over_{plan_statistic_name}"] = (
                    wandb.plot.bar(
                        table,
                        plan_statistic_name,
                        "qerror_median",
                        title=f"QError over {plan_statistic_name}",
                    )
                )
                wandb_plots_dict[
                    f"{log_prefix}_qerror_over_{plan_statistic_name}_95"
                ] = wandb.plot.bar(
                    table,
                    plan_statistic_name,
                    "qerror_95",
                    title=f"QError 95 over {plan_statistic_name}",
                )
                wandb_plots_dict[
                    f"{log_prefix}_qerror_over_{plan_statistic_name}_max"
                ] = wandb.plot.bar(
                    table,
                    plan_statistic_name,
                    "qerror_max",
                    title=f"QError max over {plan_statistic_name}",
                )

        if log_all_queries:
            if bound_predictions:
                for d, pred in zip(predictions, bounded_preds):
                    d["bounded_pred"] = pred

            if not quick_run:  # do not store artifacts in quick run
                save_csv(
                    predictions, target_csv_path=str(target_path) + "_test_pred.csv"
                )

    return any_best_metric, predictions


def to_rows(data: dict) -> list:
    """
    Converts a dict of lists (where each key is a column and each value is a list of values)
    into a list of dicts (rows), adding qerror for each row if 'label' and 'prediction' are present.
    """
    # Ensure all lists are of the same length
    lengths = {k: len(v) for k, v in data.items()}
    if len(set(lengths.values())) > 1:
        raise AssertionError(f"Input lists have different lengths: {lengths}")

    num_rows = list(lengths.values())[0] if lengths else 0

    rows = []
    keys = list(data.keys())
    for i in range(num_rows):
        entry = {k: data[k][i] for k in keys}
        if "label" in entry and "prediction" in entry:
            entry["qerror"] = QError().evaluate_metric(
                entry["label"], entry["prediction"]
            )

        rows.append(entry)
    return rows


def compute_card_qerror_metrics(actual: List, compared: List, log_scale=False) -> dict:
    """
    Compute richer q-error metrics when values are on a log scale.

    Parameters
    - actual: array-like or torch.Tensor of ground-truth values on the log(scale) (e.g., log1p(card))
    - compared: array-like or torch.Tensor of predicted/estimated values on the same log scale

    q-error for each item is computed as: np.expm1(np.maximum(actual - compared, compared - actual))
    (equivalently: np.expm1(np.abs(actual - compared))). This converts the log-difference
    back into a multiplicative q-error on the original scale.

    Returns a dict with keys:
    - "mean": mean q-error
    - "median": median q-error
    - "p50": 50th percentile (same as median)
    - "p95": 95th percentile
    - "max": max q-error
    """

    actual = np.array(actual).reshape(-1)
    compared = np.array(compared).reshape(-1)

    if actual.size != compared.size:
        raise AssertionError(
            f"actual and compared must have the same length: {actual.size} != {compared.size}"
        )
    if log_scale:
        # compute q-errors from log-scale differences
        log_abs_diff = np.abs(actual - compared)
        qerrors = np.exp(log_abs_diff)
    else:
        # compute standard q-errors
        qerrors = np.maximum(
            actual / np.clip(compared, a_min=1e-10, a_max=None),
            compared / np.clip(actual, a_min=1e-10, a_max=None),
        )
    # remove NaNs/Infs just in case
    mask = np.isfinite(qerrors)
    qerrors = qerrors[mask]

    metrics = {
        "mean": float(np.mean(qerrors)),
        "median": float(np.median(qerrors)),
        "p50": float(np.percentile(qerrors, 50)),
        "p95": float(np.percentile(qerrors, 95)),
        "max": float(np.max(qerrors)),
    }

    return metrics
