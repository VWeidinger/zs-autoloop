from itertools import product

import torch
from fast_soft_sort.pytorch_ops import soft_rank
from torch import nn

# ranking loss functions are adapted from https://github.com/allegro/allRank/tree/master


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
        print(
            f"Using LambdaLoss with weighing scheme {weighing_scheme}, k={k}, sigma={sigma}, mu={mu}, reduction={reduction}, reduction_log={reduction_log}"
        )

    def forward(self, input, target, **kwargs):
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

        # removed ndcg_at_k_mask from here to include all pairs in the loss calculation
        if self.reduction == "sum":
            loss = -torch.sum(losses[padded_pairs_mask])
        elif self.reduction == "mean":
            loss = -torch.mean(losses[padded_pairs_mask])
        else:
            raise ValueError("Reduction method can be either sum or mean")

        return loss


def ndcgLoss1(G, D, *args):
    return (G / D)[:, :, None]


def ndcgLoss2(G, D, *args):
    pos_idxs = torch.arange(1, G.shape[1] + 1, device=G.device)
    delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])
    deltas = torch.abs(
        torch.pow(torch.abs(D[0, delta_idxs - 1]), -1.0)
        - torch.pow(torch.abs(D[0, delta_idxs]), -1.0)
    )
    deltas.diagonal().zero_()

    return deltas[None, :, :] * torch.abs(G[:, :, None] - G[:, None, :])


def lambdaRank(G, D, *args):
    return torch.abs(
        torch.pow(D[:, :, None], -1.0) - torch.pow(D[:, None, :], -1.0)
    ) * torch.abs(G[:, :, None] - G[:, None, :])


def ndcgLoss2PP(G, D, *args):
    return args[0] * ndcgLoss2(G, D) + lambdaRank(G, D)


def rankNet(G, D, *args):
    return 1.0


def rankNetWeightedByGTDiff(G, D, *args):
    return torch.abs(args[1][:, :, None] - args[1][:, None, :])


def rankNetWeightedByGTDiffPowed(G, D, *args):
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

    def forward(self, input, target, **kwargs):
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


class BCELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, input, target, **kwargs):
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

    def forward(self, input, target, **kwargs):
        input = torch.reshape(input, (-1,))
        corr = spearman(
            target, input
        )  # invert so that the optimizer minimizes the loss
        corr = -1 * (corr + 1)  # offset with 1 to make it positive, invert to minimize
        return corr
