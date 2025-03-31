import time
from typing import Optional

import numpy as np
import torch
from schedulefree import AdamWScheduleFree
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from classes.classes import ModelConfig
from models.qppnet.qppnet_model import QPPNet
from models.query_former.model import QueryFormer
from models.workload_driven.model.e2e_model import E2EModel
from models.workload_driven.model.mscn_model import MSCNModel
from models.zeroshot.specific_models.postgres_zero_shot import PostgresZeroShotModel
from models.zeroshot.subplan_runtime_helper import extract_smaller_larger_preds_tensor_for_subplan_graphs
from training.batch_to_funcs import batch_to
from training.losses import LessThanLoss, RelativeLessThanLoss


def train_epoch(epoch_stats: dict, train_loader: DataLoader, model: nn.Module,
                optimizer: Optimizer, max_epoch_tuples: int, model_config: ModelConfig,
                custom_batch_to=batch_to, profiler: Optional[torch.profiler.profile] = None):
    model.train()
    if isinstance(optimizer, AdamWScheduleFree):
        # set into train mode
        optimizer.train()

    # run remaining batches
    train_start_t = time.perf_counter()
    losses = []
    monotonicity_losses = []
    errs = []

    if hasattr(model_config, 'subplan_monotonicity_loss') and model_config.subplan_monotonicity_loss:
        perform_subplan_monotonicity_loss = True

        # select loss function
        assert hasattr(model_config, 'subplan_monotonicity_loss_fn')
        if model_config.subplan_monotonicity_loss_fn == 'lt_loss':
            subplan_monotonicity_loss = LessThanLoss()
        elif model_config.subplan_monotonicity_loss_fn == 'relative_lt_loss':
            subplan_monotonicity_loss = RelativeLessThanLoss()
        else:
            raise ValueError(f"Unknown subplan monotonicity loss function: {model_config.subplan_monotonicity_loss_fn}")
    else:
        perform_subplan_monotonicity_loss = False
        subplan_monotonicity_loss = None

    for batch_idx, batch in enumerate(tqdm(train_loader)):
        if max_epoch_tuples is not None and batch_idx * train_loader.batch_size > max_epoch_tuples:
            break
        input_model, label, sample_idxs, _, _, _, _, _ = custom_batch_to(batch, model.device, model.label_norm)

        # Reset gradients first
        if isinstance(model, QPPNet):
            model.zero_grad()
        else:
            optimizer.zero_grad()

        # Forward pass
        output = model(input_model)

        if perform_subplan_monotonicity_loss:
            assert len(output) == 3
            output, subplan_graphs, subplan_predictions_tensor = output

            # extract smaller and larger tensors - input for our monotonicity loss
            smaller_tensor, larger_tensor = extract_smaller_larger_preds_tensor_for_subplan_graphs(subplan_graphs,
                                                                                                   subplan_predictions_tensor)

            # Loss computation
            monotonicity_loss = subplan_monotonicity_loss(smaller_tensor, larger_tensor)
            monotonicity_losses.append(monotonicity_loss.detach().cpu().numpy())

        else:
            monotonicity_loss = None

        # Loss computation
        loss = model.loss_fxn(output, label)

        if perform_subplan_monotonicity_loss:
            # add monotonicity loss to the total loss
            loss += monotonicity_loss

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

    if perform_subplan_monotonicity_loss:
        epoch_stats.update(monotonicity_loss=np.mean(monotonicity_losses))

    if isinstance(optimizer, AdamWScheduleFree):
        # deactivate train mode
        optimizer.eval()
