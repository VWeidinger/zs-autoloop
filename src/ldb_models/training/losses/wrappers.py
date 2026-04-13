import torch
import torch.nn as nn


class LearnFromTimeoutLossWrapper(nn.Module):
    def __init__(self, timeout_threshold, loss, **kwargs):
        super().__init__()
        self.timeout_threshold = timeout_threshold
        self.loss = loss

    def forward(self, input, target, **kwargs):
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

        # import here to avoid circular import with factory.create_loss
        from .factory import create_loss

        self.regression_loss = create_loss(regression_loss_name, regression_loss_kwargs)
        self.ranking_loss = create_loss(ranking_loss_name, ranking_loss_kwargs)
        self.regression_weight = regression_weight
        self.ranking_weight = ranking_weight
        self.last_breakdown = {}

    def forward(
        self,
        input,
        target,
        relevancy_scores_input=None,
        relevancy_scores_target=None,
        only_regression_on_subplans=False,
        inputs_are_already_log=False,
        subplan=False,
        group_function=None,
        debug=False,
    ):
        """
        if only_regression_on_subplans is True, only the regression loss is computed on the subplans
        if only_regression and regression_weight is 0, the regression loss should be computed without weighting
        otherwise weighted regression and ranking loss are computed
        """
        self.last_breakdown = {}
        if input is None or target is None:
            print("Input or target is None, returning 0.0 loss")
            return torch.tensor(0.0, device=self.device)

        regression_loss, ranking_loss = (
            torch.tensor(0.0, device=self.device),
            torch.tensor(0.0, device=self.device),
        )
        compute_only_regression = only_regression_on_subplans and subplan
        if (self.regression_weight != 0) or (compute_only_regression):
            regression_loss = self.regression_loss(
                input,
                target,
                inputs_are_already_log=inputs_are_already_log,
            )
            if debug:
                print(
                    f"Regression loss: {regression_loss.item()}, With weight: {regression_loss.item() * self.regression_weight}"
                )

            self.last_breakdown["regression_loss"] = regression_loss.item()
            if self.regression_weight != 0:
                weighted_regression_loss = regression_loss * self.regression_weight
                self.last_breakdown["regression_loss_weighted"] = (
                    weighted_regression_loss.item()
                )
                regression_loss = weighted_regression_loss

        if self.ranking_weight != 0 and not compute_only_regression:
            if (
                relevancy_scores_input is not None
                and relevancy_scores_target is not None
            ):
                ranking_loss = self.ranking_loss(
                    group_function(relevancy_scores_input),
                    group_function(relevancy_scores_target),
                )
            else:
                ranking_loss = self.ranking_loss(
                    group_function(input),
                    group_function(target),
                )
            if debug:
                print(
                    f"Ranking loss: {ranking_loss.item()}, With weight: {ranking_loss.item() * self.ranking_weight}"
                )

            self.last_breakdown["ranking_loss"] = ranking_loss.item()
            ranking_loss = ranking_loss * self.ranking_weight
            self.last_breakdown["ranking_loss_weighted"] = ranking_loss.item()

        total_loss = regression_loss + ranking_loss
        return total_loss
