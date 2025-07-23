from torch import nn
from torch.nn import CELU, SELU, LeakyReLU, ReLU, Tanh
from torch.nn import functional as F

LeakyReLU
CELU
SELU
ReLU
Tanh


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)
