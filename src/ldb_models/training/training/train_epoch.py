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
from training.training.outputs import LossOutput, ModelOutput
from training.training.rank_utils import print_debug_table, q_error


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
    losses, plan_losses, subplan_losses, card_losses, monotonicity_losses = (
        [],
        [],
        [],
        [],
        [],
    )
    errs, est_card_q_error_list, pred_card_q_error_list = [], [], []

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
            plan_dict=query_stats.get("plan_dict", None),
            debug=debug,
            plan_labels=labels,
            subplan_labels=query_stats.get("subplan_labels", None),
            card_labels=query_stats.get("subplan_card_labels", None),
        )
        # Calculate loss
        loss_output: LossOutput = model.compute_loss(
            output=output, query_stats=query_stats, labels=labels
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
            print_debug_table(
                batch_idx, loss_output, output, labels, query_stats, model
            )
        # collect stats
        err = (output.plan_predictions - labels).detach().cpu().numpy().reshape(-1)
        errs = np.concatenate((errs, err))
        losses.append(loss_output.loss.item())
        pbar_desc = f"Loss: {loss_output.loss.item():.2f}"
        if loss_output.plan_loss is not None:
            plan_losses.append(loss_output.plan_loss.item())
            pbar_desc += f", Plan: {loss_output.plan_loss.item():.2f}"
        if loss_output.subplan_loss is not None:
            subplan_losses.append(loss_output.subplan_loss.item())
            pbar_desc += f", Subplan: {loss_output.subplan_loss.item():.2f}"
        if loss_output.monotonicity_loss is not None:
            monotonicity_losses.append(loss_output.monotonicity_loss.item())
            pbar_desc += f", Mono: {loss_output.monotonicity_loss.item():.2f}"
        if loss_output.card_loss is not None:
            card_losses.append(loss_output.card_loss.item())
            pbar_desc += f", Card: {loss_output.card_loss.item():.2f}"
            est_card_q_error_list.append(
                q_error(
                    query_stats["subplan_est_card_labels"],
                    query_stats["subplan_card_labels"],
                )
            )
            pred_card_q_error_list.append(
                q_error(
                    output.subplan_card_predictions, query_stats["subplan_card_labels"]
                )
            )
        pbar.set_description(pbar_desc)
        # step profiler
        if profiler is not None:
            profiler.step()
    # Update epoch statistics
    epoch_stats.update(
        train_time=time.perf_counter() - train_start_t,
        mean_train_loss=np.mean(losses),
        mean_rmse=np.sqrt(np.mean(np.square(errs))),
        mean_train_plan_loss=np.mean(plan_losses) if len(plan_losses) > 0 else None,
        mean_train_subplan_loss=np.mean(subplan_losses)
        if len(subplan_losses) > 0
        else None,
        mean_train_card_loss=np.mean(card_losses) if len(card_losses) > 0 else None,
        mean_train_monotonicity_loss=np.mean(monotonicity_losses)
        if len(monotonicity_losses) > 0
        else None,
        mean_train_est_card_q_error=np.mean(np.concatenate(est_card_q_error_list))
        if len(est_card_q_error_list) > 0
        else None,
        mean_train_pred_card_q_error=np.mean(np.concatenate(pred_card_q_error_list))
        if len(pred_card_q_error_list) > 0
        else None,
    )
    if isinstance(optimizer, AdamWScheduleFree):
        # deactivate train mode
        optimizer.eval()
