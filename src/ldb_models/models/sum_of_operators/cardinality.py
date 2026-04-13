import torch
from torch import Tensor


def to_linear_cardinality(
    card_values: Tensor,
    log_transform_card_labels: bool,
) -> Tensor:
    """
    Convert cardinality values from model space to linear space.

    Args:
        card_values: Tensor containing cardinalities in either log1p space or
            already-linear space.
        log_transform_card_labels: Whether the model stores cardinalities in
            log1p space.

    Returns:
        Tensor with values clamped to a small positive minimum so later divisions
        remain numerically stable.
    """
    if log_transform_card_labels:
        # Clamp before expm1 to avoid overflow when labels are large.
        linear_cards = torch.expm1(torch.clamp(card_values, min=0.0, max=30.0))
        return torch.clamp(linear_cards, min=1e-6)
    return torch.clamp(card_values, min=1e-6)


def from_linear_cardinality(
    linear_card_values: Tensor,
    log_transform_card_labels: bool,
) -> Tensor:
    """
    Convert linear cardinalities back into the model's training space.

    Args:
        linear_card_values: Cardinality tensor in linear space.
        log_transform_card_labels: Whether to map the values back with log1p.

    Returns:
        Tensor in model space, clamped to keep the logarithm well defined.
    """
    linear_card_values = torch.clamp(linear_card_values, min=1e-6)
    if log_transform_card_labels:
        return torch.log1p(linear_card_values)
    return linear_card_values
