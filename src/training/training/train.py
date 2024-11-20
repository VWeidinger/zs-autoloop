import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
import wandb
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.apis.public import Run

from classes.classes import ModelConfig
from models.qppnet.qppnet_model import QPPNet
from models.query_former.model import QueryFormer
from models.workload_driven.model.e2e_model import E2EModel
from models.workload_driven.model.mscn_model import MSCNModel
from models.zeroshot.specific_models.postgres_zero_shot import PostgresZeroShotModel
from training.batch_to_funcs import batch_to
from training.bound_predictions_with_history import bound_predictions_helper, PredictionBounder
from training.training.checkpoint import save_csv
from training.training.metrics import QError, Metric, RankAccuracy


def train_epoch(epoch_stats: dict, train_loader: DataLoader, model: nn.Module,
                optimizer: Optimizer, max_epoch_tuples: int, custom_batch_to=batch_to,
                profiler: Optional[torch.profiler.profile] = None):
    model.train()

    # run remaining batches
    train_start_t = time.perf_counter()
    losses = []
    errs = []
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        if max_epoch_tuples is not None and batch_idx * train_loader.batch_size > max_epoch_tuples:
            break
        input_model, label, sample_idxs, _, _ = custom_batch_to(batch, model.device, model.label_norm)

        # Reset gradients first
        if isinstance(model, QPPNet):
            model.zero_grad()
        else:
            optimizer.zero_grad()

        # Forward pass
        output = model(input_model)

        # Loss computation
        loss = model.loss_fxn(output, label)
        if torch.isnan(loss):
            raise ValueError('Loss was NaN')

        # Loss backward step
        loss.backward()

        if isinstance(model, (PostgresZeroShotModel, MSCNModel, E2EModel, QPPNet, QueryFormer)):
            clipping_value = 1  # arbitrary value of your choosing
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)

        # Optimizer backward step
        if isinstance(model, QPPNet):
            model.backward()
        else:
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        output = output.detach().cpu().numpy().reshape(-1)
        label = label.detach().cpu().numpy().reshape(-1)
        errs = np.concatenate((errs, output - label))
        losses.append(loss)

        if profiler is not None:
            profiler.step()

    mean_loss = np.mean(losses)
    mean_rmse = np.sqrt(np.mean(np.square(errs)))
    # print(f"Train Loss: {mean_loss:.2f}")
    # print(f"Train RMSE: {mean_rmse:.2f}")
    epoch_stats.update(train_time=time.perf_counter() - train_start_t, mean_loss=mean_loss, mean_rmse=mean_rmse)


def validate_model(val_loader: DataLoader,
                   model: nn.Module,
                   config: ModelConfig,
                   target_path: Optional[Path],
                   epoch: int = 0,
                   model_dir: Path = None,
                   epoch_stats: dict = None,
                   metrics: List[Metric] = None,
                   max_epoch_tuples: int = None,
                   custom_batch_to: Callable = batch_to,
                   log_all_queries: bool = False,
                   run: Optional[Run] = None,
                   bound_predictions: bool = False,
                   prediction_bounder:Optional[PredictionBounder] = None) -> bool:
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
            epoch_stats.update(val_time=time.perf_counter() - test_start_t)
            epoch_stats.update(val_num_tuples=val_num_tuples)
            val_loss = (val_loss.cpu() / len(val_loader)).item()
            print(f'val_loss epoch {epoch}: {val_loss}')
            epoch_stats.update(val_loss=val_loss)

        labels = np.concatenate(labels, axis=0)
        preds = np.concatenate(preds, axis=0)

        epoch_stats.update(val_std=np.std(labels))

        if bound_predictions:
            bounded_preds = bound_predictions_helper(target_dataset=val_loader.dataset,predictions=preds, prediction_bounder=prediction_bounder)
            bound_list = [True, False]
        else:
            bound_list = [False]

        # save best model for every metric
        any_best_metric = False
        if metrics is not None:
            for bound in bound_list:
                for metric in metrics:
                    if isinstance(metric, RankAccuracy):
                        # skip - has to be handled separately
                        continue

                    best_seen = metric.evaluate(metrics_dict=epoch_stats,
                                                model=model,
                                                labels=labels,
                                                preds=preds if not bound else bounded_preds,
                                                probs=probs,
                                                logging_suffix='bounded' if bound else None,
                                                update_best_seen=not bound)

                    if best_seen and metric.early_stopping_metric and not bound:
                        any_best_metric = True
                        print(f"New best model for {metric.metric_name}")

        if run:
            for key, value in epoch_stats.items():
                wandb.run.summary[f'test_{key}'] = value

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
            epoch_stats.update(test_preds=predictions)

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


def optuna_intermediate_value(metrics):
    for m in metrics:
        if m.early_stopping_metric:
            assert isinstance(m, QError)
            return m.best_seen_value
    raise ValueError('Metric invalid')
