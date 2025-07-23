import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from classes.classes import ModelConfig
from models.zeroshot.specific_models.postgres_zero_shot_vae import (
    PostgresZeroShotModelVAE,
)
from models.zeroshot.subplan_runtime_helper import (
    extract_smaller_larger_preds_tensor_for_subplan_graphs,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.bound_predictions_with_history import (
    PredictionBounder,
    bound_predictions_helper,
)
from training.losses import LessThanLoss
from training.training.checkpoint import save_csv
from training.training.metrics import Metric, QError, RankAccuracy

import wandb


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
        val_loss = torch.tensor([0.0], device=model.device)
        monotonicity_loss = torch.tensor([0.0], device=model.device)

        # Collect as tensors for efficiency
        labels = []
        preds = []
        probs = []
        sample_idxs = []
        all_sql = []
        all_hints = []
        all_selectivity = []
        all_scan_op = []
        all_join_op = []
        all_num_tables = []
        all_plan_depths = []
        all_col_correlations = []

        if (
            hasattr(model_config, "subplan_monotonicity_loss")
            and model_config.subplan_monotonicity_loss
        ):
            perform_subplan_monotonicity_loss = True
            subplan_monotonicity_loss = LessThanLoss()
        else:
            perform_subplan_monotonicity_loss = False
            subplan_monotonicity_loss = None

        # evaluate test set using model
        test_start_t = time.perf_counter()
        val_num_tuples = 0
        for batch_idx, batch in enumerate(tqdm(val_loader)):
            if (
                max_epoch_tuples is not None
                and batch_idx * model_config.batch_size > max_epoch_tuples
            ):
                break

            val_num_tuples += model_config.batch_size

            # Access path selection eval: selectivity + scan_op
            # Physicl op selection eval: join_op
            (
                input_model,
                label,
                sample_idxs_batch,
                sql,
                hints,
                selectivity,
                scan_op,
                join_op,
                num_tables,
                plan_depths,
                col_correlations,
            ) = custom_batch_to(batch, model.device, model.label_norm)

            sample_idxs += sample_idxs_batch
            if isinstance(model, PostgresZeroShotModelVAE):
                output = model.predict(input_model, model_config.num_samples)
            else:
                output = model(input_model)

            if perform_subplan_monotonicity_loss:
                assert len(output) == 3
                output, subplan_graphs, subplan_predictions_dict = output

                # extract smaller and larger tensors - input for our monotonicity loss
                smaller_tensor, larger_tensor = (
                    extract_smaller_larger_preds_tensor_for_subplan_graphs(
                        subplan_graphs, subplan_predictions_dict
                    )
                )

                # Loss computation
                loss = subplan_monotonicity_loss(smaller_tensor, larger_tensor).cpu()

                # sum up
                monotonicity_loss += loss
                val_loss += loss

            # sum up mean batch losses
            val_loss += model.loss_fxn(output, label).cpu()

            # inverse transform the predictions and labels
            curr_pred = output.cpu().numpy()
            curr_label = label.cpu().numpy()

            if model.label_norm is not None:
                curr_pred = model.label_norm.inverse_transform(curr_pred)
                curr_label = model.label_norm.inverse_transform(
                    curr_label.reshape(-1, 1)
                ).reshape(-1)

            preds.append(curr_pred.reshape(-1))
            labels.append(curr_label.reshape(-1))

            # collect hints and sql statements and ...
            all_hints.extend(hints)
            all_sql.extend(sql)
            all_selectivity.extend(selectivity)
            all_scan_op.extend(scan_op)
            all_join_op.extend(join_op)
            all_num_tables.extend(num_tables)
            all_plan_depths.extend(plan_depths)
            all_col_correlations.extend(col_correlations)

        labels = np.concatenate(labels, axis=0)
        preds = np.concatenate(preds, axis=0)

        if epoch_stats is not None:
            epoch_stats[f"{log_prefix}_time"] = time.perf_counter() - test_start_t
            epoch_stats[f"{log_prefix}_num_tuples"] = val_num_tuples

            val_loss = (val_loss.cpu() / len(val_loader)).item()
            print(f"{log_prefix}_loss epoch {epoch}: {val_loss}")
            epoch_stats[f"{log_prefix}_loss"] = val_loss

            if perform_subplan_monotonicity_loss:
                monotonicity_loss = (monotonicity_loss.cpu() / len(val_loader)).item()
                print(
                    f"{log_prefix}_monotonicity_loss epoch {epoch}: {monotonicity_loss}"
                )
                epoch_stats[f"{log_prefix}_monotonicity_loss"] = monotonicity_loss

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
                    if isinstance(metric, RankAccuracy):
                        # skip - has to be handled separately
                        continue

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

        predictions = to_rows(
            indexes=[int(idx) for idx in sample_idxs],
            labels=[float(f) for f in labels],
            predictions=[float(f) for f in preds],
            sqls=all_sql,
            hints=all_hints,
            selectivitys=all_selectivity,
            scan_ops=all_scan_op,
            join_ops=all_join_op,
            num_tables=all_num_tables,
            plan_depths=all_plan_depths,
            col_correlations=all_col_correlations,
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


def to_rows(
    indexes,
    labels,
    predictions,
    sqls,
    hints,
    selectivitys,
    scan_ops,
    join_ops,
    num_tables,
    plan_depths,
    col_correlations,
):
    """Write down test-set predictions to file and to wandb"""
    assert (
        len(indexes)
        == len(labels)
        == len(predictions)
        == len(sqls)
        == len(hints)
        == len(selectivitys)
        == len(scan_ops)
        == len(join_ops)
        == len(num_tables)
        == len(plan_depths)
        == len(col_correlations)
    ), (
        f"len(indexes): {len(indexes)} len(labels): {len(labels)} len(predictions): {len(predictions)} len(sqls): {len(sqls)} len(hints): {len(hints)} len(selectivitys): {len(selectivitys)} len(scan_ops): {len(scan_ops)} len(join_ops): {len(join_ops)} len(num_tables): {len(num_tables)} len(plan_depths): {len(plan_depths)} len(col_correlations): {len(col_correlations)}"
    )

    rows = []
    for (
        index,
        label,
        prediction,
        sql,
        hint,
        selectivity,
        scan_op,
        join_op,
        num_t,
        depth,
        col_correlation,
    ) in zip(
        indexes,
        labels,
        predictions,
        sqls,
        hints,
        selectivitys,
        scan_ops,
        join_ops,
        num_tables,
        plan_depths,
        col_correlations,
    ):
        entry = dict(
            query_index=index,
            label=label,
            prediction=prediction,
            sql=sql,
            hint=hint,
            selectivity=selectivity,
            scan_op=scan_op,
            join_op=join_op,
            num_tables=num_t,
            plan_depth=depth,
            col_correlation=col_correlation,
        )
        entry.update(qerror=QError().evaluate_metric(label, prediction))
        rows.append(entry)
    return rows
