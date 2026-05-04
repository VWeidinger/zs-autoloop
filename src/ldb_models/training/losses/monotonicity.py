import torch
import torch.nn as nn
import torch.nn.functional as F

from ldb_models.training.losses.regression import QLoss


class LessThanLoss(nn.Module):
    # This loss is used to check if the model predicts a value smaller than the target
    # input1 should be smaller than input2 - in this case 0 will be returned
    # otherwise the loss will be the difference between input1 and input2
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, input1, input2, **kwargs):
        return F.relu(input1 - input2).mean()


class RelativeLessThanLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.qerror = QLoss(**kwargs)

    def forward(self, input1, input2, **kwargs):
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
