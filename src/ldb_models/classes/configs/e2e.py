import attrs
from typing import Callable

from .base import ModelConfig
from .constants import InputFormats, ModelName, ModelType, Name

from training.batch_to_funcs import wl_driven_batch_to
from models.workload_driven.dataset.plan_tree_batching import baseline_plan_collator


@attrs.define(frozen=True, slots=False)
class E2EModelConfig(ModelConfig):
    name: Name = ModelName.E2E
    type: ModelType = ModelType.WL_DRIVEN
    hidden_dim_plan: int = 256
    hidden_dim_pred: int = 256
    batch_to_func: Callable = wl_driven_batch_to
    collator_func: Callable = baseline_plan_collator
    input_format: str = InputFormats.baseline_parsed
    column_statistics: bool = True
    word_embeddings: bool = True
    device: str = "cuda:[socket]"


@attrs.define(frozen=True, slots=False)
class TPoolModelConfig(E2EModelConfig):
    model_name: Name = ModelName.E2E


@attrs.define(frozen=True, slots=False)
class TlstmModelConfig(E2EModelConfig):
    model_name: Name = ModelName.E2E


@attrs.define(frozen=True, slots=False)
class TestSumModelConfig(E2EModelConfig):
    model_name: Name = Name("testsum", "TestSum")
