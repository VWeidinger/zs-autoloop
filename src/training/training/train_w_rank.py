import time
from typing import Optional, Callable, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.qppnet.qppnet_model import QPPNet
from models.query_former.model import QueryFormer
from models.workload_driven.model.e2e_model import E2EModel
from models.workload_driven.model.mscn_model import MSCNModel
from models.zeroshot.specific_models.postgres_zero_shot import PostgresZeroShotModel
from training.batch_to_funcs import batch_to
from training.training.metrics import Metric


class RankModel(nn.Module):
    def __init__(self, model):
        super(RankModel, self).__init__()
        self.model = model
        self.device = model.device
        self.label_norm = model.label_norm

    def forward(self, x1, x2):
        # plan1 and plan2
        x1 = self.model(x1)
        x2 = self.model(x2)

        # combine x1 and x2
        x = torch.cat((x1, x2), dim=1)

        return x


def train_epoch_rank(epoch_stats: dict, train_loader: DataLoader, model: nn.Module,
                     optimizer: Optimizer, max_epoch_tuples: int, custom_batch_to=batch_to,
                     profiler: Optional[torch.profiler.profile] = None):
    model.train()

    train_start_t = time.perf_counter()
    losses = []
    ranking_correct = []

    # define loss function
    loss_fn = nn.CrossEntropyLoss()

    tuples_count = 0

    # batch statistics
    batch_size = train_loader.batch_size
    num_batches = len(train_loader)

    # perform ranking training between two consecutive batches (to avoid the need for nested loops / dataloader)
    last_batch = None

    for batch_idx, batch in enumerate(tqdm(train_loader)):
        # batch data
        input1, label1, sample_idxs1 = custom_batch_to(batch, model.device, model.label_norm)

        # skip the batch if it is not full
        if len(label1) != batch_size:
            continue

        # break if max_epoch_tuples is reached
        if max_epoch_tuples is not None and tuples_count > max_epoch_tuples:
            break

        # increment tuples count
        tuples_count += train_loader.batch_size

        if last_batch is not None:
            input2, label2, sample_idxs2 = last_batch

            # perform training
            # Reset gradients first
            if isinstance(model, QPPNet):
                model.zero_grad()
            else:
                optimizer.zero_grad()

            # Forward pass
            output = model(input1, input2)

            # concat labels n / n --> nx2
            label = torch.stack((label1, label2), dim=1)

            # apply softmax on labels
            label = torch.softmax(input=label, dim=1)

            # apply cross entropy loss
            loss = loss_fn(output, label)
            if torch.isnan(loss):
                raise ValueError('Loss was NaN')

            # Loss backward step
            loss.backward()

            if isinstance(model, (PostgresZeroShotModel, MSCNModel, E2EModel, QPPNet, QueryFormer)):
                clipping_value = 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)

            # Optimizer backward step
            if isinstance(model, QPPNet):
                model.backward()
            else:
                optimizer.step()

            loss = loss.detach().cpu().numpy()
            losses.append(loss)

            # compute ranking accuracy
            faster_real = torch.argmin(label, dim=1)
            faster_pred = torch.argmin(output, dim=1)
            ranking_correct.append(torch.eq(faster_real, faster_pred).detach().cpu().numpy())

            if profiler is not None:
                profiler.step()

        # update last batch information
        last_batch = input1, label1, sample_idxs1

    mean_loss = np.mean(losses)

    # collapse ranking correct list
    ranking_correct = np.concatenate(ranking_correct)
    ranking_accuracy = np.mean(ranking_correct)

    epoch_stats.update(train_time=time.perf_counter() - train_start_t, train_loss=mean_loss,
                       train_ranking_accuracy=ranking_accuracy)


def validate_rank(val_loader: DataLoader, model: nn.Module, epoch_stats: Dict, rank_metric: Metric,
                  custom_batch_to: Callable = batch_to):
    model.eval()

    with torch.autograd.no_grad():

        ranking_correct = []
        last_batch = None

        for batch_idx, batch in enumerate(tqdm(val_loader)):
            # batch data
            input1, label1, sample_idxs1 = custom_batch_to(batch, model.device, model.label_norm)

            # skip the batch if it is not full
            if len(label1) != val_loader.batch_size:
                continue

            if last_batch is not None:
                input2, label2, sample_idxs2 = last_batch

                # Forward pass
                output = model(input1, input2)

                # concat labels n / n --> nx2
                label = torch.stack((label1, label2), dim=1)

                # compute ranking accuracy
                faster_real = torch.argmin(label, dim=1)
                faster_pred = torch.argmin(output, dim=1)
                ranking_correct.append(torch.eq(faster_real, faster_pred).detach().cpu().numpy())

            # update last batch information
            last_batch = input1, label1, sample_idxs1

        # extract ranking accuracy
        ranking_correct = np.concatenate(ranking_correct)

        # save best model for metric
        any_best_metric=False
        assert isinstance(model,RankModel),f'Model instance: {model} is not of type RankModel'
        best_seen = rank_metric.evaluate(metrics_dict=epoch_stats,
                                                model=model.model,
                                                ranking_correct=ranking_correct)

        if best_seen and rank_metric.early_stopping_metric:
            any_best_metric = True
            print(f"New best model for {rank_metric.metric_name}")

    return any_best_metric
