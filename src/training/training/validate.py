import time
from pathlib import Path
from typing import Optional, List, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.batch_to_funcs import batch_to
from training.bound_predictions_with_history import PredictionBounder, bound_predictions_helper
from training.training.checkpoint import save_csv
from training.training.metrics import Metric, RankAccuracy, QError


def validate_model(val_loader: DataLoader,
                   model: nn.Module,
                   target_path: Optional[Path],
                   epoch: int = 0,
                   epoch_stats: dict = None,
                   metrics: List[Metric] = None,
                   max_epoch_tuples: int = None,
                   custom_batch_to: Callable = batch_to,
                   log_all_queries: bool = False,
                   bound_predictions: bool = False,
                   prediction_bounder: Optional[PredictionBounder] = None,
                   log_prefix: str = 'val',
                   update_metrics: bool = True) -> bool:
    model.eval()

    with torch.autograd.no_grad():
        val_loss = torch.Tensor([0])
        labels = []
        preds = []
        probs = []
        sample_idxs = []
        all_sql = []
        all_hints = []
        all_selectivity = []
        all_scan_op = []
        all_join_op = []

        # evaluate test set using model
        test_start_t = time.perf_counter()
        val_num_tuples = 0
        for batch_idx, batch in enumerate(tqdm(val_loader)):
            if max_epoch_tuples is not None and batch_idx * val_loader.batch_size > max_epoch_tuples:
                break

            val_num_tuples += val_loader.batch_size

            # Access path selection eval: selectivity + scan_op
            # Physicl op selection eval: join_op
            input_model, label, sample_idxs_batch, sql, hints, selectivity, scan_op, join_op = custom_batch_to(batch,
                                                                                                               model.device,
                                                                                                               model.label_norm)

            sample_idxs += sample_idxs_batch
            output = model(input_model)

            # sum up mean batch losses
            val_loss += model.loss_fxn(output, label).cpu()

            # inverse transform the predictions and labels
            curr_pred = output.cpu().numpy()
            curr_label = label.cpu().numpy()
            if model.label_norm is not None:
                curr_pred = model.label_norm.inverse_transform(curr_pred)
                curr_label = model.label_norm.inverse_transform(curr_label.reshape(-1, 1))
                curr_label = curr_label.reshape(-1)

            preds.append(curr_pred.reshape(-1))
            labels.append(curr_label.reshape(-1))

            # Collect hints and sql statements
            all_hints.extend(hints)
            all_sql.extend(sql)

            # Collect selectivity, scan_op, join_op
            all_selectivity.extend(selectivity)
            all_scan_op.extend(scan_op)
            all_join_op.extend(join_op)

        if epoch_stats is not None:
            epoch_stats[f'{log_prefix}_time'] = time.perf_counter() - test_start_t
            epoch_stats[f'{log_prefix}_num_tuples'] = val_num_tuples

            val_loss = (val_loss.cpu() / len(val_loader)).item()
            print(f'{log_prefix}_loss epoch {epoch}: {val_loss}')
            epoch_stats[f'{log_prefix}_loss'] = val_loss

        labels = np.concatenate(labels, axis=0)
        preds = np.concatenate(preds, axis=0)

        epoch_stats.update(val_std=np.std(labels))

        if bound_predictions:
            bounded_preds = bound_predictions_helper(target_dataset=val_loader.dataset, predictions=preds,
                                                     prediction_bounder=prediction_bounder)
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

                    best_seen = metric.evaluate(metrics_dict=metrics_dict,
                                                model=model,
                                                labels=labels,
                                                preds=preds if not bound else bounded_preds,
                                                probs=probs,
                                                logging_suffix='bounded' if bound else None,
                                                update_best_seen=not bound and update_metrics)

                    if best_seen and metric.early_stopping_metric and not bound and update_metrics:
                        any_best_metric = True
                        print(f"New best model for {metric.metric_name}")

        # add metrics dict to epoch stats
        for key, value in metrics_dict.items():
            epoch_stats[f'{log_prefix}_{key}'] = value

        predictions = to_rows(indexes=sample_idxs,
                              labels=[float(f) for f in labels],
                              predictions=[float(f) for f in preds],
                              sqls=all_sql,
                              hints=all_hints,
                              selectivitys=all_selectivity,
                              scan_ops=all_scan_op,
                              join_ops=all_join_op)
        if log_all_queries:
            if bound_predictions:
                for d, pred in zip(predictions, bounded_preds):
                    d['bounded_pred'] = pred

            save_csv(predictions, target_csv_path=str(target_path) + "_test_pred.csv")

            # if run: # too large for wandb
            #     wandb.run.summary[f"test_preds"] = predictions

            # Add rows to epoch_stats to use them in other evaluation tasks later
            epoch_stats[f'{log_prefix}_predictions'] = predictions

    return any_best_metric, predictions


def to_rows(indexes, labels, predictions, sqls, hints, selectivitys, scan_ops, join_ops):
    """ Write down test-set predictions to file and to wandb"""
    rows = []
    for (index, label, prediction, sql, hint, selectivity, scan_op, join_op) in zip(indexes, labels, predictions, sqls,
                                                                                    hints, selectivitys, scan_ops,
                                                                                    join_ops):
        entry = dict(query_index=index, label=label, prediction=prediction, sql=sql, hint=hint, selectivity=selectivity,
                     scan_op=scan_op, join_op=join_op)
        entry.update(qerror=QError().evaluate_metric(label, prediction))
        rows.append(entry)
    return rows
