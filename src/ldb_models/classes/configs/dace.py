from typing import Callable
import attrs as attrs
from .base import ModelConfig
from .constants import InputFormats, ModelName, ModelType, Name

from training.batch_to_funcs import dace_batch_to
from training.featurizations import (
    Featurization,
    DACEActCardFeaturization,
    DACEFeaturization,
    DACEFeaturizationNoCosts,
)


@attrs.define(frozen=True, slots=False)
class DACEModelConfig(ModelConfig):
    name: Name = ModelName.DACE
    type: ModelType = ModelType.WL_AGNOSTIC
    node_length: int = (
        10 + 3
    )  # This needs to be equals to the number of operator types plus the remaining node feats.
    hidden_dim: int = 128
    output_dim: int = 1
    mlp_activation: str = "ReLU"
    transformer_activation: str = "gelu"
    mlp_dropout: float = 0.3
    transformer_dropout: float = 0.2
    max_runtime: int = 30000
    pad_length: int = 30
    loss_weight: float = 0.5
    batch_to_func: Callable = dace_batch_to
    loss_class_name: str = "DaceLoss"
    loss_class_kwargs: dict = dict()
    featurization: Featurization = DACEFeaturization
    input_format: InputFormats = InputFormats.parsed
    optimizer_class_name: str = "Adam"
    optimizer_kwargs: dict = dict(lr=1e-3)
    device: str = "cuda:[socket]"


@attrs.define(frozen=True, slots=False)
class DACEStratifiedModelConfig(DACEModelConfig):
    name: Name = ModelName.DACE_STRATIFIED
    stratified_workloads: bool = True


@attrs.define(frozen=True, slots=False)
class DACEModelNoCostsConfig(DACEModelConfig):
    name: Name = ModelName.DACE_NO_COSTS
    node_length: int = 21  # This needs to be equals to the number of operator types plus the remaining node feats.
    featurization: Featurization = DACEFeaturizationNoCosts


@attrs.define(frozen=True, slots=False)
class DACEModelActCardConfig(DACEModelConfig):
    name: Name = ModelName.DACE_ACT_CARDS
    node_length: int = 22
    featurization: Featurization = DACEActCardFeaturization


@attrs.define(frozen=True, slots=False)
class DACEModelWHintsConfig(DACEModelConfig):
    name: Name = ModelName.DACE_WITH_HINTS
    include_hint_workloads: bool = True


@attrs.define(frozen=True, slots=False)
class DACEModelOnlyHints(DACEModelConfig):
    name: Name = ModelName.DACE_ONLY_HINTS
    only_hint_workloads: bool = True
