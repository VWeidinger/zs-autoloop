import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, input, target, return_aggregated: bool = True, **kwargs):
        if return_aggregated:
            loss = F.mse_loss(input.view(-1), target.view(-1), reduction="mean")
        else:
            loss = F.mse_loss(input.view(-1), target.view(-1), reduction="none")
        return loss


class MSELogLoss(nn.Module):
    """
    Computes the Mean Squared Error in log space.
    It uses log1p (log(1+x)) for numerical stability with zero-cost values.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        return_aggregated: bool = True,
        **kwargs,
    ):
        # Ensure predictions are non-negative before log transform
        # Using relu is a simple way to enforce this.
        input_log = torch.log1p(F.relu(input))
        target_log = torch.log1p(target)

        if return_aggregated:
            loss = F.mse_loss(input_log, target_log, reduction="mean")
        else:
            loss = F.mse_loss(input_log, target_log, reduction="none")
        return loss


class BiasedLog1pLoss(nn.Module):
    """
    A parameter-free regression loss for DB Query Optimization.

    Features:
    1. Log1p: Handles inputs < 1 (linear regime) and inputs > 1M (log regime).
    2. Parameter-Free Asymmetry: Automatically penalizes underestimates more
    3. Zero-Safe: Loss is 0.0 when label == input.
    """

    def __init__(self, device="cuda", **kwargs):
        super().__init__()
        self.device = device

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        inputs_are_already_log: bool = False,
        return_aggregated: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        # 1. Robust Log Transform (log1p)
        # We clamp input at 0.0 to prevent NaNs from negative predictions
        if not inputs_are_already_log:
            pred_log = torch.log1p(F.relu(input))
            target_log = torch.log1p(target)
        else:
            pred_log = input
            target_log = target

        # 2. Calculate Residual
        # diff > 0 : Overestimation
        # diff < 0 : Underestimation
        diff = pred_log - target_log

        # 3. Standard Squared Error (Symmetric)
        mse_part = diff**2

        # 4. Parameter-Free Asymmetry
        # We extract the magnitude of underestimation only.
        # If diff is positive (overestimate), -diff is negative, relu makes it 0.
        # If diff is negative (underestimate), -diff is positive.
        # We square the underestimation to penalize it more.
        underestimation_part = F.relu(-diff) ** 2

        # 5. Total Loss
        loss = mse_part + underestimation_part

        if return_aggregated:
            return torch.mean(loss)
        else:
            return loss


class ExpUELog1pLoss(nn.Module):
    """
    A parameter-free regression loss for DB Query Optimization.

    Features:
    1. Log1p: Handles inputs < 1 (linear regime) and inputs > 1M (log regime).
    2. Parameter-Free Asymmetry: Automatically penalizes underestimates more
    3. Zero-Safe: Loss is 0.0 when label == input.
    """

    def __init__(self, device="cuda", **kwargs):
        super().__init__()
        self.device = device

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        inputs_are_already_log: bool = False,
        return_aggregated: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        # 1. Robust Log Transform (log1p)
        # We clamp input at 0.0 to prevent NaNs from negative predictions
        if not inputs_are_already_log:
            pred_log = torch.log1p(input)
            target_log = torch.log1p(target)
        else:
            pred_log = input
            target_log = target

        # 2. Calculate Residual
        # diff > 0 : Overestimation
        # diff < 0 : Underestimation
        diff = pred_log - target_log

        # 3. Standard Squared Error (Symmetric)
        # for overestimates mse, for underestimates exponential
        loss = torch.where(diff >= 0, diff**2, torch.expm1(-diff))

        if return_aggregated:
            return torch.mean(loss)
        else:
            return loss


class QLoss(nn.Module):
    """
    Regression loss that minimizes the q-error for each prediction
    """

    def __init__(self, min_val=1e-3, penalty_negative=1e5, **kwargs):
        self.min_val = min_val
        self.penalty_negative = penalty_negative
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        return_aggregated: bool = True,
        **kwargs,
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

    def forward(self, input, target, return_aggregated: bool = True, **kwargs):
        return (
            torch.log(
                self.qloss(input, target, return_aggregated=return_aggregated, **kwargs)
            )
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

    def forward(self, input, target, return_aggregated: bool = True, **kwargs):
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

    def forward(self, input, target, return_aggregated: bool = True, **kwargs):
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

    def forward(self, input, target, return_aggregated: bool = True, **kwargs):
        # Calculate the log-based q-error
        loss = torch.abs(torch.log1p(input) - torch.log1p(target))

        if return_aggregated:
            return torch.mean(loss)
        else:
            return loss


class Log1pQLossScaled(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, input, target, return_aggregated: bool = True, **kwargs):
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

    def forward(self, input, target, return_aggregated: bool = True, **kwargs):
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

    def forward(self, input, target, return_aggregated: bool = True, **kwargs):
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

    def forward(self, input, target, return_aggregated: bool = True, **kwargs):
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
