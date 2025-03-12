import time
from typing import Optional

import numpy as np
import torch
from schedulefree import AdamWScheduleFree
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.qppnet.qppnet_model import QPPNet
from models.query_former.model import QueryFormer
from models.workload_driven.model.e2e_model import E2EModel
from models.workload_driven.model.mscn_model import MSCNModel
from models.zeroshot.specific_models.postgres_zero_shot import PostgresZeroShotModel
from training.batch_to_funcs import batch_to


def train_epoch(epoch_stats: dict, train_loader: DataLoader, model: nn.Module,
                optimizer: Optimizer, max_epoch_tuples: int, custom_batch_to=batch_to,
                profiler: Optional[torch.profiler.profile] = None):
    model.train()
    if isinstance(optimizer, AdamWScheduleFree):
        # set into train mode
        optimizer.train()

    # run remaining batches
    train_start_t = time.perf_counter()
    losses = []
    errs = []
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

    if isinstance(optimizer, AdamWScheduleFree):
        # deactivate train mode
        optimizer.eval()
