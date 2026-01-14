from typing import Callable
import attrs as attrs
from .base import ModelConfig
from .constants import InputFormats, ModelName, ModelType, Name
from training.batch_to_funcs import simple_batch_to
from training.featurizations import (
    Featurization,
    QPPNetActCardsFeaturization,
    QPPNetFeaturization,
    QPPNetNoCostsFeaturization,
)


@attrs.define(frozen=True, slots=False)
class QPPNetModelConfig(ModelConfig):
    name: Name = ModelName.QPP_NET
    type: ModelType = ModelType.WL_DRIVEN
    optimizer_kwargs: dict = dict(lr=1e-3)
    featurization: Featurization = QPPNetFeaturization
    loss_class_name: str = "QPPLoss"
    loss_class_kwargs: dict = dict()
    hidden_dim_plan: int = 128
    batch_to_func: Callable = simple_batch_to
    device: str = "cpu"
    input_format: str = InputFormats.json
    column_statistics: bool = True


@attrs.define(frozen=True, slots=False)
class QPPModelNoCostsConfig(QPPNetModelConfig):
    name: Name = ModelName.QPP_NET_NO_COSTS
    featurization: Featurization = QPPNetNoCostsFeaturization


@attrs.define(frozen=True, slots=False)
class QPPModelActCardsConfig(QPPNetModelConfig):
    name: Name = ModelName.QPP_NET_ACT_CARDS
    featurization: Featurization = QPPNetActCardsFeaturization
