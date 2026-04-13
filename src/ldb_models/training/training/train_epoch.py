import time
from typing import Optional

import numpy as np
import torch
from classes import ModelConfig
from models.qppnet.qppnet_model import QPPNet
from schedulefree import AdamWScheduleFree
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.training.metrics import EpochMetrics
from training.training.outputs import LossOutput, ModelOutput
from training.training.utils import print_debug_table, q_error


def train_epoch(
    epoch_stats: dict,
    train_loader: DataLoader,
    model: nn.Module,
    optimizer: Optimizer,
    max_epoch_tuples: int,
    model_config: ModelConfig,
    custom_batch_to=None,
    profiler: Optional[torch.profiler.profile] = None,
):
    model.train()
    if isinstance(optimizer, AdamWScheduleFree):
        # set into train mode
        optimizer.train()
    # torch.autograd.set_detect_anomaly(True)
    train_start_t = time.perf_counter()

    # Generic metric collection
    epoch_metrics = EpochMetrics()
    errs = []

    pbar = tqdm(train_loader, desc="Training", dynamic_ncols=True)

    debug_denom = max(1, len(train_loader) // 4)
    for batch_idx, batch in enumerate(pbar):
        if (
            max_epoch_tuples is not None
            and batch_idx * model_config.batch_size > max_epoch_tuples
        ):
            break
        # debug flag for each quarter batcg
        debug = batch_idx % debug_denom == 0
        # prepare batch
        inputs, labels, query_stats = custom_batch_to(
            batch, model.device, model.label_norm
        )
        # Reset gradients first
        optimizer.zero_grad() if not isinstance(model, QPPNet) else model.zero_grad()
        # Forward pass
        output: ModelOutput = model(
            inputs,
            debug=debug,
            plan_labels=labels,
            **query_stats,
        )
        # Calculate loss
        loss_output: LossOutput = model.compute_loss(
            output=output, query_stats=query_stats, labels=labels, debug=debug
        )
        # check for nan loss
        if torch.isnan(loss_output.loss):
            raise ValueError("Loss was NaN")
        # Loss backward step
        loss_output.loss.backward()
        # Clip gradients and optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Step optimizer
        optimizer.step() if not isinstance(model, QPPNet) else model.backward()
        # print debug table
        if debug:
            print_debug_table(batch_idx, loss_output, output, labels, query_stats)

        # collect stats
        err = (output.plan_predictions - labels).detach().cpu().numpy().reshape(-1)
        errs = np.concatenate((errs, err))

        epoch_metrics.append("loss", loss_output.loss.item())

        # Collect detailed breakdown metrics
        breakdown = getattr(loss_output, "loss_breakdown", {})
        for k, v in breakdown.items():
            epoch_metrics.append(k, v)

        # Fallback/Legacy for main loss components if not in breakdown
        if loss_output.plan_loss is not None and "plan_loss" not in breakdown:
            epoch_metrics.append("plan_loss", loss_output.plan_loss.item())

        if loss_output.subplan_loss is not None and "subplan_loss" not in breakdown:
            epoch_metrics.append("subplan_loss", loss_output.subplan_loss.item())

        if loss_output.card_loss is not None:
            if "card_loss" not in breakdown:
                epoch_metrics.append("card_loss", loss_output.card_loss.item())

            # Card Q-Errors - keep specific logic
            epoch_metrics.append(
                "est_card_q_error",
                q_error(
                    query_stats["subplan_est_card_labels"],
                    query_stats["subplan_card_labels"],
                ),
            )
            epoch_metrics.append(
                "pred_card_q_error",
                q_error(
                    output.subplan_card_predictions, query_stats["subplan_card_labels"]
                ),
            )

        if loss_output.monotonicity_loss is not None:
            epoch_metrics.append(
                "monotonicity_loss", loss_output.monotonicity_loss.item()
            )

        # Update progress bar
        pbar_desc = f"Loss: {loss_output.loss.item():.2f}"
        if len(epoch_metrics.values.get("plan_loss", [])) > 0:
            pbar_desc += f", Plan: {epoch_metrics.values['plan_loss'][-1]:.2f}"
        if len(epoch_metrics.values.get("subplan_loss", [])) > 0:
            pbar_desc += f", Sub: {epoch_metrics.values['subplan_loss'][-1]:.2f}"
        pbar.set_description(pbar_desc)

        # step profiler
        if profiler is not None:
            profiler.step()

    # Update epoch statistics
    epoch_stats.update(
        train_time=time.perf_counter() - train_start_t,
        mean_train_loss=epoch_metrics.mean("loss"),
        mean_rmse=np.sqrt(np.mean(np.square(errs))),
        # Keep explicit legacy keys for dashboards and backward compatibility.
        mean_train_plan_loss=epoch_metrics.mean("plan_loss"),
        mean_train_subplan_loss=epoch_metrics.mean("subplan_loss"),
        mean_train_card_loss=epoch_metrics.mean("card_loss"),
        mean_train_monotonicity_loss=epoch_metrics.mean("monotonicity_loss"),
        mean_train_est_card_q_error=epoch_metrics.mean("est_card_q_error"),
        mean_train_pred_card_q_error=epoch_metrics.mean("pred_card_q_error"),
    )

    # Generic calculation of means for all collected metrics
    epoch_stats.update(epoch_metrics.means_with_prefix(prefix="mean_train_"))

    if isinstance(optimizer, AdamWScheduleFree):
        # deactivate train mode
        optimizer.eval()
