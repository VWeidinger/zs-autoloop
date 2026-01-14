from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class ModelOutput:
    """A structured container for the model's predictions."""

    plan_predictions: torch.Tensor
    # for subplan models
    subplan_predictions: Optional[torch.Tensor] = None
    subplan_graphs: Optional[Dict] = None
    subplan_lengths: Optional[torch.Tensor] = None
    # for cardinality estimation
    subplan_card_predictions: Optional[torch.Tensor] = None


@dataclass
class LossOutput:
    """A structured container for the loss outputs."""

    loss: torch.Tensor
    plan_loss: Optional[torch.Tensor] = None
    subplan_loss: Optional[torch.Tensor] = None
    card_loss: Optional[torch.Tensor] = None
    monotonicity_loss: Optional[torch.Tensor] = None
    # for ranking models
    gouped_output: Optional[torch.Tensor] = None
    grouped_labels: Optional[torch.Tensor] = None
