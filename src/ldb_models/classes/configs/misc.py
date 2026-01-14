from typing import Callable
import attrs as attrs
from .base import ModelConfig
from .constants import InputFormats, ModelName, ModelType, Name
from .tabular import TabularModelConfig
from models.t3.query_plan import CardType
from models.workload_driven.dataset.mscn_batching import mscn_plan_collator
from training.batch_to_funcs import (
    neo_batch_to,
    query_former_batch_to,
    zs_batch_to,
)


@attrs.define(frozen=True, slots=False)
class MSCNModelConfig(ModelConfig):
    name: Name = ModelName.MSCN
    type: ModelType = ModelType.WL_DRIVEN
    mscn_enc_layers: int = 1
    mscn_est_layers: int = 1
    batch_to_func: Callable = zs_batch_to
    collator_func: Callable = mscn_plan_collator
    input_format: InputFormats = InputFormats.baseline_parsed
    column_statistics: bool = True
    word_embeddings: bool = True
    hidden_dim_plan: int = 256
    device: str = "cuda:[socket]"


@attrs.define(frozen=True, slots=False)
class QueryFormerModelConfig(ModelConfig):
    name: Name = ModelName.QUERY_FORMER
    type: ModelType = ModelType.WL_DRIVEN
    hidden_dim_plan: int = 256
    input_format: InputFormats = InputFormats.parsed
    column_statistics: bool = True
    word_embeddings: bool = True
    embedding_size: int = 64
    ffn_dim: int = 32
    head_size: int = 8
    dropout: float = 0.1
    attention_dropout_rate: float = 0.1
    n_layers: int = 8
    use_sample: bool = True
    use_histogram: bool = True
    histogram_bin_number: int = 10
    hidden_dim_prediction: int = 128
    max_num_filters: int = 6
    batch_to_func: Callable = query_former_batch_to
    loss_class_name: str = "QLoss"  # Originally MSELoss
    optimizer_class_name: str = "AdamW"  # Originally Adam
    device: str = "cuda:[socket]"


@attrs.define(frozen=True, slots=False)
class T3ModelConfig(TabularModelConfig):
    name: Name = ModelName.T3
    type: ModelType = ModelType.WL_AGNOSTIC
    device: str = "cuda:[socket]"
    input_format: InputFormats = InputFormats.parsed
    card_type: CardType = CardType.pg


@attrs.define(frozen=True, slots=False)
class T3TreeModelConfig(T3ModelConfig):
    name: Name = ModelName.T3Tree


@attrs.define(frozen=True, slots=False)
class T3FlatModelConfig(T3ModelConfig):
    name: Name = ModelName.T3Flat


@attrs.define(frozen=True, slots=False)
class NeoModelConfig(ModelConfig):
    name: Name = ModelName.NEO
    device: str = "cuda:[socket]"
    type: ModelType = ModelType.WL_DRIVEN
    mscn_enc_layers: int = 1
    mscn_est_layers: int = 1
    batch_to_func: Callable = zs_batch_to
    collator_func: Callable = mscn_plan_collator
    input_format: InputFormats = InputFormats.parsed
    column_statistics: bool = False
    word_embeddings: bool = False
    batch_to_func: callable = neo_batch_to
