from itertools import product
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_soft_sort.pytorch_ops import soft_rank

# ranking loss functions are adapted from https://github.com/allegro/allRank/tree/master


def create_loss(loss_name: str, loss_kwargs: Dict):
    if loss_name == "MSELoss":
        loss = MSELoss(**loss_kwargs)
    elif loss_name == "MSELogLoss":
        loss = MSELogLoss(**loss_kwargs)
    elif loss_name == "QLoss":
        loss = QLoss(**loss_kwargs)
    elif loss_name == "QLogSqLoss":
        loss = QLogSqLoss(**loss_kwargs)
    elif loss_name == "LogQLoss":
        loss = LogQLoss(**loss_kwargs)
    elif loss_name == "Log1pQLoss":
        loss = Log1pQLoss(**loss_kwargs)
    elif loss_name == "Log1pQLossScaled":
        loss = Log1pQLossScaled(**loss_kwargs)
    elif loss_name == "Log1pQLossPenalty":
        loss = Log1pQLossPenalty(**loss_kwargs)
    elif loss_name == "LogHuberQLoss":
        loss = LogHuberQLoss(**loss_kwargs)
    elif loss_name == "Log1pHuberQLoss":
        loss = Log1pHuberQLoss(**loss_kwargs)
    elif loss_name == "LogQLossPenalty":
        loss = LogQLossPenalty(**loss_kwargs)
    elif loss_name == "DaceLoss":
        loss = DaceLoss(**loss_kwargs)
    elif loss_name == "QPPLoss":
        loss = QPPLoss(**loss_kwargs)
    elif loss_name == "RankLoss":
        loss = RankLoss(**loss_kwargs)
    elif loss_name == "LambdaLoss":
        loss = LambdaLoss(**loss_kwargs)
    elif loss_name == "NDCGLoss":
        loss = NDCGLoss(**loss_kwargs)
    elif loss_name == "RankNetLoss":
        loss = RankNetLoss(**loss_kwargs)
    elif loss_name == "ListNetLoss":
        loss = ListNetLoss(**loss_kwargs)
    elif loss_name == "ListMLELoss":
        loss = ListMLELoss(**loss_kwargs)
    elif loss_name == "BCELoss":
        loss = BCELoss(**loss_kwargs)
    elif loss_name == "PairwiseLoss":
        loss = PairwiseLoss(**loss_kwargs)
    elif loss_name == "LearnToRankWithRegressionLoss":
        loss = LearnToRankWithRegressionLoss(**loss_kwargs)
    else:
        raise ValueError(f"Loss {loss_name} not supported")

    print(f"Using loss {loss_name} with kwargs {loss_kwargs}")
    learn_from_timeout = loss_kwargs.pop(
        "learn_from_timeout", False
    )  # learn from timeout
    timeout_threshold = loss_kwargs.pop("timeout_threshold", None)

    if learn_from_timeout:
        print(f"Using timeout masking with threshold {timeout_threshold}")
        loss = LearnFromTimeoutLossWrapper(timeout_threshold, loss)
    return loss


class BCELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, input, target):
        return self.loss(input, target)


class MSELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, input, target):
        return F.mse_loss(input.view(-1), target.view(-1), reduction="mean")


class MSELogLoss(nn.Module):
    """
    Computes the Mean Squared Error in log space.
    It uses log1p (log(1+x)) for numerical stability with zero-cost values.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # Ensure predictions are non-negative before log transform
        # Using relu is a simple way to enforce this.
        input_log = torch.log1p(F.relu(input))
        target_log = torch.log1p(target)

        return self.mse(input_log, target_log)


class QLoss(nn.Module):
    """
    Regression loss that minimizes the q-error for each prediction
    """

    def __init__(self, min_val=1e-3, penalty_negative=1e5, **kwargs):
        self.min_val = min_val
        self.penalty_negative = penalty_negative
        super().__init__()

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, return_aggregated: bool = True
    ):
        assert isinstance(input, torch.Tensor), (
            f"Input is not a tensor, but {type(input)}"
        )
        assert isinstance(target, torch.Tensor), (
            f"Target is not a tensor, but {type(target)}"
        )

        assert len(input.shape) == 2, f"Input is not a tensor, but {len(input.shape)}"
        assert input.shape[1] == 1, f"Expected 2nd dim to be 1, but got {input.shape}"

        input_zero_mask = input == 0
        target_zero_mask = target == 0

        if input_zero_mask.any():
            input = input + input_zero_mask * torch.full(
                input.shape, 0.0000001, device=input.device
            )
        if target_zero_mask.any():
            target = target + target_zero_mask * torch.full(
                target.shape, 0.0000001, device=target.device
            )

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

        q_error = torch.add(
            q_error,
            torch.max(
                torch.div(input_masked, target.reshape((-1, 1))),
                torch.div(target_masked, input),
            ),
        )

        if return_aggregated:
            loss = torch.mean(q_error)
            return loss
        else:
            return q_error


class QLogSqLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.qloss = QLoss(**kwargs)

    def forward(self, input, target, return_aggregated: bool = True):
        return (
            torch.log(self.qloss(input, target, return_aggregated=return_aggregated))
            ** 2
        )


class LogQLossPenalty(nn.Module):
    # computes absolute difference in log space
    # symmetric like q-error
    # but smoother gradient near 1
    # penalizes deviations less aggressively for large errors compared to q-error

    def __init__(self, min_val=1e-6, **kwargs):
        super().__init__()
        self.min_val = min_val

    def forward(self, input, target, return_aggregated: bool = True):
        pred = torch.clamp(input, min=self.min_val)
        label = torch.clamp(target, min=self.min_val)

        # Calculate the log-based q-error
        loss = torch.abs(torch.log(pred) - torch.log(label))
        penalty = torch.relu(-input)  # penalizes negatives only

        total_loss = loss + 0.001 * penalty

        if return_aggregated:
            return torch.mean(total_loss)
        else:
            return total_loss


class LogQLoss(nn.Module):
    # computes absolute difference in log space
    # symmetric like q-error
    # but smoother gradient near 1
    # penalizes deviations less aggressively for large errors compared to q-error

    def __init__(self, min_val=1e-6):
        super().__init__()
        self.min_val = min_val

    def forward(self, input, target, return_aggregated: bool = True):
        pred = torch.clamp(input, min=self.min_val)
        label = torch.clamp(target, min=self.min_val)

        # Calculate the log-based q-error
        loss = torch.abs(torch.log(pred) - torch.log(label))

        if return_aggregated:
            return torch.mean(loss)
        else:
            return loss


class Log1pQLoss(nn.Module):
    def __init__(self, min_val=1e-6, **kwargs):
        super().__init__()
        self.min_val = min_val

    def forward(self, input, target, return_aggregated: bool = True):
        # Calculate the log-based q-error
        loss = torch.abs(torch.log1p(input) - torch.log1p(target))

        if return_aggregated:
            return torch.mean(loss)
        else:
            return loss


class Log1pQLossScaled(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, input, target, return_aggregated: bool = True):
        # Calculate the log-based q-error
        # Scale back from log1p space
        loss = torch.exp(torch.abs(torch.log1p(input) - torch.log1p(target))) - 1

        if return_aggregated:
            return torch.mean(loss)
        else:
            return loss


class Log1pQLossPenalty(nn.Module):
    def __init__(self, min_val=1e-6, **kwargs):
        super().__init__()
        self.min_val = min_val

    def forward(self, input, target, return_aggregated: bool = True):
        pred = torch.clamp(input, min=self.min_val)
        label = torch.clamp(target, min=self.min_val)

        # Calculate the log-based q-error
        loss = torch.abs(torch.log1p(pred) - torch.log1p(label))

        # penalize less than min_val predictions
        penalty_mask = input < self.min_val
        penalty = torch.mul(1 - input, penalty_mask) * 0.001
        total_loss = loss + penalty

        if return_aggregated:
            return torch.mean(total_loss)
        else:
            return total_loss


class LogHuberQLoss(nn.Module):
    # computes absolute difference in log space
    # symmetric like q-error
    # but smoother gradient near 1
    # penalizes deviations less aggressively for large errors compared to q-error

    def __init__(self, min_val=1e-7, **kwargs):
        super().__init__()
        self.min_val = min_val

    def forward(self, input, target, return_aggregated: bool = True):
        pred = torch.clamp(input, min=self.min_val)
        label = torch.clamp(target, min=self.min_val)

        # Calculate the log-based q-error
        log_diff = torch.abs(torch.log(pred) - torch.log(label))

        # huber loss
        loss = torch.where(
            torch.abs(log_diff) < 1, 0.5 * log_diff**2, torch.abs(log_diff) - 0.5
        )

        if return_aggregated:
            return torch.mean(loss)
        else:
            return loss


class Log1pHuberQLoss(nn.Module):
    def __init__(self, min_val=1e-7, **kwargs):
        super().__init__()
        self.min_val = min_val

    def forward(self, input, target, return_aggregated: bool = True):
        pred = torch.clamp(input, min=self.min_val)
        label = torch.clamp(target, min=self.min_val)

        # Calculate the log-based q-error
        log_diff = torch.abs(torch.log1p(pred) - torch.log1p(label))

        # huber loss
        loss = torch.where(
            torch.abs(log_diff) < 1, 0.5 * log_diff**2, torch.abs(log_diff) - 0.5
        )

        if return_aggregated:
            return torch.mean(loss)
        else:
            return loss


class DaceLoss(nn.Module):
    def __init__(self, **loss_kwargs):
        super().__init__()
        self.loss_masks = None
        self.preds = None
        self.real_run_times = None

    def forward(self):
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
    def __init__(self, operator_types: dict, device: str, **loss_kwargs):
        super().__init__()
        self.operator_types = operator_types
        self.device = device
        self.zeros = torch.zeros(1)
        self.curr_losses = {operator: 0 for operator, _ in self.operator_types.items()}
        self.accumulated_loss = {
            operator: [] for operator, _ in self.operator_types.items()
        }
        self.total_loss = None
        self.total_losses = None

    def update_operator_loss(
        self, node_type: str, predicted_operator_time: torch.Tensor, label: torch.Tensor
    ) -> None:
        loss = torch.sqrt(((predicted_operator_time - label) ** 2) + 1e-6)
        # This works also with Q-error, but original paper uses MSE
        # We apply a lower bound of loss and labels to avoid nans.
        # This additionally helps to preventing loss explosions.
        # predicted_operator_time = torch.max(predicted_operator_time, torch.tensor(0.001))
        # label = torch.max(label, torch.tensor(0.001))
        # loss = torch.max(predicted_operator_time / label, label / predicted_operator_time)
        # ---------------------------------------------------------------------------------------------

        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(
                f"Loss was {loss} "
                f"for node type {node_type}, "
                f"prediction: {predicted_operator_time}, "
                f"label: {label}, "
                f"q-Error: {loss}"
            )

        # Gather loss in accumulated loss
        self.accumulated_loss[node_type].append(loss)

    def reset_accumulated_losses(self) -> None:
        self.accumulated_loss = {
            operator: [] for operator, _ in self.operator_types.items()
        }
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
        assert len(input.shape) == 1 or input.shape[1] == 1, (
            f"Input shape is {input.shape}"
        )
        if len(input.shape) == 2:
            assert input.shape[1] == 1, f"Target shape is {input.shape}"
            input = input.squeeze(1)  # reduce nx1 vector to n vector

        # mask out elements where target is None (timeout) and input is larger than timeout_threshold
        timeout_mask = (target.isnan() | target.isinf()) & (
            input >= self.timeout_threshold
        )

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
        torch.reshape(pred, (1, -1)),  # reshape to 2d tensor - performs ranking -1 dim
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
        corr = spearman(
            target, input
        )  # invert so that the optimizer minimizes the loss
        corr = -1 * (corr + 1)  # offset with 1 to make it positive, invert to minimize
        return corr


class LessThanLoss(nn.Module):
    # This loss is used to check if the model predicts a value smaller than the target
    # input1 should be smaller than input2 - in this case 0 will be returned
    # otherwise the loss will be the difference between input1 and input2
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, input1, input2):
        return F.relu(input1 - input2).mean()


class RelativeLessThanLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.qerror = QLoss(**kwargs)

    def forward(self, input1, input2):
        assert len(input1.shape) == 1, f"Input shape is {input1.shape}"
        assert len(input2.shape) == 1, f"Target shape is {input2.shape}"

        # reshape to -1,1
        input1 = torch.reshape(input1, (-1, 1))
        input2 = torch.reshape(input2, (-1, 1))

        # compute qerror
        qerror = self.qerror(input1, input2, return_aggregated=False)

        # check where input1 > input2 - only here we apply a loss
        mask = input1 > input2

        # initialize with 0
        error = torch.zeros_like(input1, device=input1.device)

        # replace with qerror where input1 was larger than input2
        error[mask] = qerror[mask]

        # aggregate results
        return error.mean()


class LambdaLoss(nn.Module):
    def __init__(
        self,
        eps=1e-10,
        padded_value_indicator=-1,
        weighing_scheme=None,
        k=None,
        sigma=1.0,
        mu=10.0,
        reduction="mean",
        reduction_log="binary",
        **kwargs,
    ):
        """
        :param eps: epsilon value, used for numerical stability
        :param padded_value_indicator: an indicator of the target index containing a padded item, e.g. -1
        :param weighing_scheme: a string corresponding to a name of one of the weighing schemes
        :param k: rank at which the loss is truncated
        :param sigma: score difference weight used in the sigmoid function
        :param mu: optional weight used in NDCGLoss2++ weighing scheme
        :param reduction: losses reduction method, could be either a sum or a mean
        :param reduction_log: logarithm variant used prior to masking and loss reduction, either binary or natural
        """
        super().__init__()
        self.eps = eps
        self.padded_value_indicator = padded_value_indicator
        self.weighing_scheme = weighing_scheme
        self.k = k
        self.sigma = sigma
        self.mu = mu
        self.reduction = reduction
        self.reduction_log = reduction_log

    def forward(self, input, target):
        """
        lambda_loss framework for LTR losses implementations, introduced in "The lambda_loss Framework for Ranking Metric Optimization".
        Contains implementations of different weighing schemes corresponding to e.g. LambdaRank or RankNet.
        :param input: predictions from the model, shape [batch_size, slate_length]
        :param target: ground truth labels, shape [batch_size, slate_length]
        :return: loss value, a torch.Tensor
        """
        device = input.device
        input = input.clone()
        target = target.clone()

        padded_mask = target == self.padded_value_indicator
        input[padded_mask] = float("-inf")
        target[padded_mask] = float("-inf")

        # Here we sort the true and predicted relevancy scores.
        input_sorted, indices_pred = input.sort(descending=True, dim=-1)
        target_sorted, _ = target.sort(descending=True, dim=-1)

        # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
        true_sorted_binputs = torch.gather(target, dim=1, index=indices_pred)
        true_diffs = true_sorted_binputs[:, :, None] - true_sorted_binputs[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        if self.weighing_scheme != "ndcgLoss1_scheme":
            padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

        ndcg_at_k_mask = torch.zeros(
            (input.shape[1], input.shape[1]), dtype=torch.bool, device=device
        )
        ndcg_at_k_mask[: self.k, : self.k] = 1

        # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
        true_sorted_binputs.clamp_(min=0.0)
        target_sorted.clamp_(min=0.0)

        # Here we find the gains, discounts and ideal DCGs per slate.
        pos_idxs = torch.arange(1, input.shape[1] + 1).to(device)
        D = torch.log2(1.0 + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(
            ((torch.pow(2, target_sorted) - 1) / D)[:, : self.k], dim=-1
        ).clamp(min=self.eps)
        G = (torch.pow(2, true_sorted_binputs) - 1) / maxDCGs[:, None]

        # Here we apply appropriate weighing scheme - ndcgLoss1, ndcgLoss2, ndcgLoss2++ or no weights (=1.0)
        if self.weighing_scheme is None:
            weights = 1.0
        else:
            weights = globals()[self.weighing_scheme](
                G, D, self.mu, true_sorted_binputs
            )  # type: ignore

        # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
        scores_diffs = (input_sorted[:, :, None] - input_sorted[:, None, :]).clamp(
            min=-1e8, max=1e8
        )
        scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.0)
        weighted_probas = (
            torch.sigmoid(self.sigma * scores_diffs).clamp(min=self.eps) ** weights
        ).clamp(min=self.eps)
        if self.reduction_log == "natural":
            losses = torch.log(weighted_probas)
        elif self.reduction_log == "binary":
            losses = torch.log2(weighted_probas)
        else:
            raise ValueError("Reduction logarithm base can be either natural or binary")

        if self.reduction == "sum":
            loss = -torch.sum(losses[padded_pairs_mask & ndcg_at_k_mask])
        elif self.reduction == "mean":
            loss = -torch.mean(losses[padded_pairs_mask & ndcg_at_k_mask])
        else:
            raise ValueError("Reduction method can be either sum or mean")

        return loss


def ndcgLoss1_scheme(G, D, *args):
    return (G / D)[:, :, None]


def ndcgLoss2_scheme(G, D, *args):
    pos_idxs = torch.arange(1, G.shape[1] + 1, device=G.device)
    delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])
    deltas = torch.abs(
        torch.pow(torch.abs(D[0, delta_idxs - 1]), -1.0)
        - torch.pow(torch.abs(D[0, delta_idxs]), -1.0)
    )
    deltas.diagonal().zero_()

    return deltas[None, :, :] * torch.abs(G[:, :, None] - G[:, None, :])


def lambdaRank_scheme(G, D, *args):
    return torch.abs(
        torch.pow(D[:, :, None], -1.0) - torch.pow(D[:, None, :], -1.0)
    ) * torch.abs(G[:, :, None] - G[:, None, :])


def ndcgLoss2PP_scheme(G, D, *args):
    return args[0] * ndcgLoss2_scheme(G, D) + lambdaRank_scheme(G, D)


def rankNet_scheme(G, D, *args):
    return 1.0


def rankNetWeightedByGTDiff_scheme(G, D, *args):
    return torch.abs(args[1][:, :, None] - args[1][:, None, :])


def rankNetWeightedByGTDiffPowed_scheme(G, D, *args):
    return torch.abs(
        torch.pow(args[1][:, :, None], 2) - torch.pow(args[1][:, None, :], 2)
    )


class NDCGLoss(nn.Module):
    def __init__(self, k=None):
        self.k = k
        super().__init__()

    def forward(self, input, target):
        """
        Compute NDCG loss.
        Args:
            input (torch.Tensor): Predicted scores for the group.
            target (torch.Tensor): True relevance scores for the group.
            k (int): Cutoff for NDCG calculation (None means full list).
        """
        device = input.device
        _, sorted_indices = torch.sort(target, descending=True)
        ideal_gains = torch.pow(2, target[sorted_indices]) - 1
        ideal_discounts = torch.log2(
            torch.arange(2, 2 + len(target), device=device).float()
        )
        ideal_dcg = torch.sum(ideal_gains / ideal_discounts)

        _, sorted_pred_indices = torch.sort(input, descending=True)
        gains = torch.pow(2, target[sorted_pred_indices]) - 1
        discounts = torch.log2(torch.arange(2, 2 + len(target), device=device).float())
        dcg = torch.sum(gains / discounts)

        ndcg = dcg / (ideal_dcg + 1e-10)
        return 1 - ndcg


class RankNetLoss(nn.Module):
    def __init__(
        self,
        padded_value_indicator=-1,
        weight_by_diff=False,
        weight_by_diff_powed=False,
    ):
        self.padded_value_indicator = padded_value_indicator
        self.weight_by_diff = weight_by_diff
        self.weight_by_diff_powed = weight_by_diff_powed
        super().__init__()

    def forward(self, input, target):
        """
        RankNet loss introduced in "Learning to Rank using Gradient Descent".
        :param input: predictions from the model, shape [batch_size, slate_length]
        :param target: ground truth labels, shape [batch_size, slate_length]
        :param weight_by_diff: flag indicating whether to weight the score differences by ground truth differences.
        :param weight_by_diff_powed: flag indicating whether to weight the score differences by the squared ground truth differences.
        :return: loss value, a torch.Tensor
        """
        input = input.clone()
        target = target.clone()

        mask = target == self.padded_value_indicator
        input[mask] = float("-inf")
        target[mask] = float("-inf")

        # here we generate every pair of indices from the range of document length in the batch
        document_pairs_candidates = list(product(range(target.shape[1]), repeat=2))

        pairs_true = target[:, document_pairs_candidates]
        selected_pred = input[:, document_pairs_candidates]

        # here we calculate the relative true relevance of every candidate pair
        true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
        pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

        # here we filter just the pairs that are 'positive' and did not involve a padded instance
        # we can do that since in the candidate pairs we had symetric pairs so we can stick with
        # positive ones for a simpler loss function formulation
        the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

        pred_diffs = pred_diffs[the_mask]

        weight = None
        if self.weight_by_diff:
            abs_diff = torch.abs(true_diffs)
            weight = abs_diff[the_mask]
        elif self.weight_by_diff_powed:
            true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(
                pairs_true[:, :, 1], 2
            )
            abs_diff = torch.abs(true_pow_diffs)
            weight = abs_diff[the_mask]

        # here we 'binarize' true relevancy diffs since for a pairwise loss we just need to know
        # whether one document is better than the other and not about the actual difference in
        # their relevancy levels
        true_diffs = (true_diffs > 0).type(torch.float32)
        true_diffs = true_diffs[the_mask]

        return torch.nn.BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)


class ListNetLoss(nn.Module):
    def __init__(self, eps=1e-10):
        self.eps = eps
        super().__init__()

    def forward(self, input, target):
        """
        Compute ListNet loss.
        Args:
            input (torch.Tensor): Predicted scores for the group.
            target (torch.Tensor): True relevance scores for the group.
        """
        pred_probs = torch.softmax(input, dim=0)
        true_probs = torch.softmax(target, dim=0)
        return -torch.sum(true_probs * torch.log(pred_probs + self.eps))


class ListMLELoss(nn.Module):
    def __init__(self, eps=1e-10):
        self.eps = eps
        super().__init__()

    def forward(self, input, target):
        """
        ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
        :param input: predictions from the model, shape [batch_size, slate_length]
        :param target: ground truth labels, shape [batch_size, slate_length]
        :param eps: epsilon value, used for numerical stability
        :param padded_value_indicator: an indicator of the target index containing a padded item, e.g. -1
        :return: loss value, a torch.Tensor
        """
        # shuffle for randomised tie resolution
        random_indices = torch.randperm(input.shape[-1])
        input_shuffled = input[:, random_indices]
        target_shuffled = target[:, random_indices]

        target_sorted, indices = target_shuffled.sort(descending=True, dim=-1)

        mask = target_sorted == -1

        preds_sorted_btarget = torch.gather(input_shuffled, dim=1, index=indices)
        preds_sorted_btarget[mask] = float("-inf")

        max_pred_values, _ = preds_sorted_btarget.max(dim=1, keepdim=True)

        preds_sorted_btarget_minus_max = preds_sorted_btarget - max_pred_values

        cumsums = torch.cumsum(
            preds_sorted_btarget_minus_max.exp().flip(dims=[1]), dim=1
        ).flip(dims=[1])

        observation_loss = torch.log(cumsums + 1e-10) - preds_sorted_btarget_minus_max

        observation_loss[mask] = 0.0

        return torch.mean(torch.sum(observation_loss, dim=1))


class LearnToRankWithRegressionLoss(nn.Module):
    def __init__(
        self,
        regression_weight=1.0,
        ranking_weight=1.0,
        regression_loss_name="MSELoss",
        ranking_loss_name="LambdaLoss",
        regression_loss_kwargs=None,
        ranking_loss_kwargs=None,
        device="cuda",
        **loss_kwargs,
    ):
        super().__init__()
        if regression_loss_kwargs is None:
            regression_loss_kwargs = {}
        if ranking_loss_kwargs is None:
            ranking_loss_kwargs = {}
        self.device = device

        self.regression_loss = create_loss(regression_loss_name, regression_loss_kwargs)
        self.ranking_loss = create_loss(ranking_loss_name, ranking_loss_kwargs)
        self.regression_weight = regression_weight
        self.ranking_weight = ranking_weight

    def forward(
        self,
        input,
        target,
        relevancy_scores_input=None,
        relevancy_scores_target=None,
        only_regression=False,
        subplan=False,
    ):
        """
        if only_regression is True, only the regression loss is computed on the subplans
            if only_regression and regression_weight is 0, the regression loss should be computed without weighting
        otherwise weighted regression and ranking loss are computed
        """
        if input is None or target is None:
            print("Input or target is None, returning 0.0 loss")
            return torch.tensor(0.0, device=self.device)

        regression_loss, ranking_loss = (
            torch.tensor(0.0, device=self.device),
            torch.tensor(0.0, device=self.device),
        )
        if (self.regression_weight != 0) or (only_regression):
            regression_loss = self.regression_loss(
                input.reshape(-1, 1), target.reshape(-1, 1)
            )
            if not (only_regression and self.regression_weight == 0):
                regression_loss = regression_loss * self.regression_weight
        if self.ranking_weight != 0 and not only_regression:
            if (
                relevancy_scores_input is not None
                and relevancy_scores_target is not None
            ):
                ranking_loss = (
                    self.ranking_loss(relevancy_scores_input, relevancy_scores_target)
                    * self.ranking_weight
                )
            else:
                ranking_loss = self.ranking_loss(input, target) * self.ranking_weight
        total_loss = regression_loss + ranking_loss
        return total_loss


class PairwiseLoss(nn.Module):
    def __init__(
        self, loss_name: str, inverse_train=True, diff_train=True, **loss_kwargs
    ):
        super().__init__()
        self.loss_name = loss_name
        self.inverse_train = inverse_train
        self.diff_train = diff_train
        self.loss = create_loss(loss_name, loss_kwargs)

    def forward(self, input, target):
        loss = self.loss(input, target)
        return loss
