import torch
import torch.nn as nn


class DaceLoss(nn.Module):
    def __init__(self, **loss_kwargs):
        super().__init__()
        self.loss_masks = None
        self.preds = None
        self.real_run_times = None

    def forward(self, **kwargs):
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
