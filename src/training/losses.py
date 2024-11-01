from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_soft_sort.pytorch_ops import soft_rank


def create_loss(loss_name: str, loss_kwargs: Dict):
    if loss_name == 'MSELoss':
        loss = MSELoss(**loss_kwargs)
    elif loss_name == 'QLoss':
        loss = QLoss(**loss_kwargs)
    elif loss_name == 'DaceLoss':
        loss = DaceLoss(**loss_kwargs)
    elif loss_name == 'QPPLoss':
        loss = QPPLoss(**loss_kwargs)
    elif loss_name == 'RankLoss':
        loss = RankLoss(**loss_kwargs)
    else:
        raise ValueError(f"Loss {loss_name} not supported")

    print(f'Using loss {loss_name} with kwargs {loss_kwargs}')
    learn_from_timeout = loss_kwargs.pop('learn_from_timeout', False)  # learn from timeout
    timeout_threshold = loss_kwargs.pop('timeout_threshold', None)

    if learn_from_timeout:
        print(f'Using timeout masking with threshold {timeout_threshold}')
        loss = LearnFromTimeoutLossWrapper(timeout_threshold, loss)

    return loss


class MSELoss(nn.Module):
    def __init__(self, model, weight=None, **kwargs):
        super().__init__()

    def forward(self, input, target):
        return F.mse_loss(input.view(-1), target.view(-1), reduction='mean')


class QLoss(nn.Module):
    """
    Regression loss that minimizes the q-error for each prediction
    """

    def __init__(self, min_val=1e-3, penalty_negative=1e5, **kwargs):
        self.min_val = min_val
        self.penalty_negative = penalty_negative
        super().__init__()

    def forward(self, input, target):
        input_zero_mask = input == 0
        target_zero_mask = target == 0

        if input_zero_mask.any():
            input = input + input_zero_mask * torch.full(input.shape, 0.0000001, device=input.device)
        if target_zero_mask.any():
            target = target + target_zero_mask * torch.full(input.shape, 0.0000001, device=target.device)

        q_error = torch.zeros((len(target), 1), device=target.device)

        # create mask for entries which should be penalized for negative/too small estimates
        penalty_mask = input < self.min_val
        inverse_penalty_mask = input >= self.min_val
        q_error_penalties = torch.mul(1 - input, penalty_mask) * self.penalty_negative

        # influence on loss for a negative estimate is >= penalty_negative constant
        q_error = torch.add(q_error, q_error_penalties)

        # calculate normal q error for other instances
        input_masked = torch.mul(input, inverse_penalty_mask)
        target_masked = torch.mul(target.reshape((-1, 1)), inverse_penalty_mask)

        q_error = torch.add(q_error, torch.max(torch.div(input_masked, target.reshape((-1, 1))),
                                               torch.div(target_masked, input)))

        loss = torch.mean(q_error)
        return loss


class DaceLoss(nn.Module):
    def __init__(self, model, **loss_kwargs):
        super().__init__()
        self.loss_masks = None
        self.preds = None
        self.real_run_times = None
        self.to(model.device)

    def forward(self, input, target):
        input = self.preds
        target = self.real_run_times
        # print(self.preds.device, self.real_run_times.device)
        loss = torch.max(input / target, target / input)
        loss = loss * self.loss_masks
        loss = torch.log(torch.where(loss > 1, loss, 1))
        loss = torch.sum(loss, dim=1)
        loss = torch.mean(loss)
        return loss


class QPPLoss(torch.nn.Module):
    def __init__(self, model, **loss_kwargs):
        super().__init__()
        self.operator_types = model.operator_types
        self.device = model.device
        self.zeros = torch.zeros(1)
        self.curr_losses = {operator: 0 for operator, _ in self.operator_types.items()}
        self.accumulated_loss = {operator: [] for operator, _ in self.operator_types.items()}
        self.total_loss = None
        self.total_losses = None

    def update_operator_loss(self, node_type: str, predicted_operator_time: torch.Tensor, label: torch.Tensor) -> None:
        loss = torch.sqrt(((predicted_operator_time - label) ** 2) + 1e-6)
        # This works also with Q-error, but original paper uses MSE
        # We apply a lower bound of loss and labels to avoid nans.
        # This additionally helps to preventing loss explosions.
        # predicted_operator_time = torch.max(predicted_operator_time, torch.tensor(0.001))
        # label = torch.max(label, torch.tensor(0.001))
        # loss = torch.max(predicted_operator_time / label, label / predicted_operator_time)
        # ---------------------------------------------------------------------------------------------

        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(f"Loss was {loss} "
                             f"for node type {node_type}, "
                             f"prediction: {predicted_operator_time}, "
                             f"label: {label}, "
                             f"q-Error: {loss}")

        # Gather loss in accumulated loss
        self.accumulated_loss[node_type].append(loss)

    def reset_accumulated_losses(self) -> None:
        self.accumulated_loss = {operator: [] for operator, _ in self.operator_types.items()}
        self.total_loss = torch.zeros(1, dtype=torch.double).to(self.device)

    def update_total_losses(self) -> None:
        for operator_type, loss in self.accumulated_loss.items():
            if loss and loss != []:
                operator_loss = torch.cat(loss).to(self.device)
                self.total_loss += torch.sum(operator_loss)

    def forward(self, input, target) -> torch.Tensor:
        out = torch.mean(self.total_loss)
        self.reset_accumulated_losses()
        return out


class LearnFromTimeoutLossWrapper(nn.Module):
    def __init__(self, timeout_threshold, loss):
        super().__init__()
        self.timeout_threshold = timeout_threshold
        self.loss = loss

    def forward(self, input, target):
        assert len(input.shape)==1 or input.shape[1]==1, f'Input shape is {input.shape}'
        if len(input.shape) == 2:
            assert input.shape[1] == 1, f'Target shape is {input.shape}'
            input = input.squeeze(1) # reduce nx1 vector to n vector

        # mask out elements where target is None (timeout) and input is larger than timeout_threshold
        timeout_mask = (target.isnan() | target.isinf()) & (input >= self.timeout_threshold)

        # set target to threshold where target is None (timeout)
        target[(target.isnan() | target.isinf())] = self.timeout_threshold

        # set the target to the input where the query timeouted and the model predicts a value above the timeout threshold
        target[timeout_mask] = input[timeout_mask]

        # calculate loss on the adjusted targets
        return self.loss(input, target)

def corrcoef(target, pred):
    pred_n = pred - pred.mean()
    target_n = target - target.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    return (pred_n * target_n).sum()

def spearman(
    target,
    pred,
    regularization="l2",
    regularization_strength=1.0,
):
    # fast_soft_sort uses 1-based indexing, divide by len to compute percentage of rank
    rank_pred = soft_rank(
        torch.reshape(pred,(1,-1)), # reshape to 2d tensor - performs ranking -1 dim
        regularization=regularization,
        regularization_strength=regularization_strength,
    )
    corr = corrcoef(target, rank_pred / rank_pred.shape[-1])
    return corr

class RankLoss(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, input, target):
        input = torch.reshape(input, (-1,))
        corr =  spearman(target, input) # invert so that the optimizer minimizes the loss
        corr = -1 * (corr + 1) # offset with 1 to make it positive, invert to minimize
        return corr



