import attrs as attrs
from attr import field

from .base import ModelConfig
from .constants import InputFormats, ModelName, ModelType, Name
from training.featurizations import (
    Featurization,
    FlatModelActCardFeaturization,
    FlatModelFeaturization,
)


@attrs.define(frozen=True, slots=False)
class TabularModelConfig(ModelConfig):
    gen_dataset_func: staticmethod = field(default=None)


@attrs.define(frozen=True, slots=False)
class FlatModelConfig(TabularModelConfig):
    name: ModelName = ModelName.FLAT
    type: ModelType = ModelType.WL_AGNOSTIC
    device: str = "cpu"
    input_format: str = InputFormats.parsed
    featurization: Featurization = FlatModelFeaturization


@attrs.define(frozen=True, slots=False)
class FlatStratifiedModelConfig(FlatModelConfig):
    name: ModelName = ModelName.FLAT_STRATIFIED
    stratified_workloads: bool = True


@attrs.define(frozen=True, slots=False)
class FlatModelActCardModelConfig(FlatModelConfig):
    name: Name = ModelName.FLAT_ACT_CARDS
    input_format: str = InputFormats.parsed
    featurization: Featurization = FlatModelActCardFeaturization


@attrs.define(frozen=True, slots=False)
class SPNModelConfig(TabularModelConfig):
    name: Name = ModelName.SPN
    type: ModelType = ModelType.WL_AGNOSTIC
    device: str = "cpu"
    input_format: str = InputFormats.parsed
    featurization: Featurization = FlatModelFeaturization


@attrs.define(frozen=True, slots=False)
class ScaledPostgresModelConfig(TabularModelConfig):
    name: Name = ModelName.POSTGRES
    type: ModelType = ModelType.WL_AGNOSTIC
    device: str = "cpu"
    input_format: str = InputFormats.parsed


@attrs.define(frozen=True, slots=False)
class ScaledPostgresStratifiedModelConfig(ScaledPostgresModelConfig):
    name: Name = ModelName.POSTGRES_STRATIFIED
    stratified_workloads: bool = True


@attrs.define(frozen=True, slots=False)
class AnalyticalEstCardModelConfig(TabularModelConfig):
    model_name: Name = Name("analytical_est_card", "Analytical (est. card)")


@attrs.define(frozen=True, slots=False)
class AnalyticalActCardModelConfig(TabularModelConfig):
    model_name: Name = Name("analytical_est_card", "Analytical (act. card)")
