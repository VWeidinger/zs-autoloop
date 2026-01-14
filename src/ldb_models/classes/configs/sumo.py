import attrs as attrs
from training.featurizations import Featurization, PostgresEstSystemNoCardDetail

from .constants import ModelName, Name
from .mamba import MambaRankingModelConfig
from .zs_ranking import ZeroShotRankingModelBaseConfig


@attrs.define(frozen=True, slots=False)
class SumOfOperatorsModelConfig(ZeroShotRankingModelBaseConfig):
    """Configuration for the SumOfOperators model."""

    name: Name = ModelName.SUM_OF_OPERATORS

    include_complex_many_joins_workloads: bool = True
    include_hint_idx_workloads: bool = True
    include_complex_ranking_workloads: bool = True

    # This model is trained by predicting subplan operation times, so subplan_estimation is essential.
    subplan_estimation: bool = True
    subplans_reg_loss: bool = True
    subplan_weight: float = 1

    # This custom flag will signal the data collator to calculate op_time for labels.
    subplan_label_op_time: bool = True

    # Options for Cardinality Estimation
    cardinality_estimation: bool = False
    cardinality_weight: float = 0.1
    log_transform_card_labels: bool = True
    use_card_feature: bool = attrs.field(default=False)

    regression_loss_name: str = "Log1pQLossScaled"
    ranking_loss_name: str = "LambdaLoss"
    regression_weight: float = 0.0
    ranking_weight: float = 1.0
    loss_class_name: str = "LearnToRankWithRegressionLoss"
    loss_class_kwargs: dict = attrs.field(init=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        object.__setattr__(
            self,
            "loss_class_kwargs",
            dict(
                ranking_loss_name=self.ranking_loss_name,
                regression_loss_name=self.regression_loss_name,
                regression_weight=self.regression_weight,
                ranking_weight=self.ranking_weight,
                subplan_weight=self.subplan_weight,
                cardinality_weight=self.cardinality_weight,
                ranking_loss_kwargs=dict(),
                regression_loss_kwargs=dict(),
            ),
        )


@attrs.define(frozen=True, slots=False)
class SumOfOperatorsModelWithRelevancyConfig(SumOfOperatorsModelConfig):
    """Configuration for the SumOfOperators model with relevancy."""

    name: Name = ModelName.SUM_OF_OPERATORS_RELEVANCY

    adaptive_generator: bool = False
    use_relevancy_generator: bool = True
    predict_relevancy_scores: bool = False
    rl_use_log_transform: bool = True
    relevancy_generator_args: dict = attrs.field(init=False)

    rl_scalar: float = 50.0
    rl_data_min: float = 1e-8
    rl_data_max: float = 180.0
    rl_log_degree: float = 5.0

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        object.__setattr__(
            self,
            "relevancy_generator_args",
            dict(
                rl_data_min=self.rl_data_min,
                rl_data_max=self.rl_data_max,
                rl_scalar=self.rl_scalar,
                rl_log_degree=self.rl_log_degree,
                adaptive=self.adaptive_generator,
                rl_use_log_transform=self.rl_use_log_transform,
            ),
        )


@attrs.define(frozen=True, slots=False)
class SumOfOperatorsModelWithNoCardConfig(SumOfOperatorsModelWithRelevancyConfig):
    """Configuration for the SumOfOperators model without cardinality estimation."""

    name: Name = ModelName.SUM_OF_OPERATORS_NO_CARD

    cardinality_estimation: bool = True
    cardinality_weight: float = 0.1
    use_card_feature: bool = True
    log_transform_card_labels: bool = True

    featurization: Featurization = PostgresEstSystemNoCardDetail


@attrs.define(frozen=True, slots=False)
class SUMOModelConfig(ZeroShotRankingModelBaseConfig):
    """Configuration for the SumOfOperators model."""

    name: Name = ModelName.SUMO

    include_complex_many_joins_workloads: bool = True
    include_hint_idx_workloads: bool = True
    include_complex_ranking_workloads: bool = True

    # This model is trained by predicting subplan operation times, so subplan_estimation is essential.
    subplan_estimation: bool = True
    subplans_reg_loss: bool = True
    subplan_weight: float = 1.0

    # This custom flag will signal the data collator to calculate op_time for labels.
    subplan_label_op_time: bool = True

    # Options for Cardinality Estimation
    cardinality_estimation: bool = True
    cardinality_weight: float = 0.1
    log_transform_card_labels: bool = True
    use_card_feature: bool = attrs.field(default=False)
    use_runtime_feature: bool = attrs.field(default=False)

    # Loss Parameters
    regression_loss_name: str = "Log1pQLossScaled"
    ranking_loss_name: str = "LambdaLoss"
    regression_weight: float = 0.0
    ranking_weight: float = 1.0
    loss_class_name: str = "LearnToRankWithRegressionLoss"
    loss_class_kwargs: dict = attrs.field(init=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        object.__setattr__(
            self,
            "loss_class_kwargs",
            dict(
                ranking_loss_name=self.ranking_loss_name,
                regression_loss_name=self.regression_loss_name,
                regression_weight=self.regression_weight,
                ranking_weight=self.ranking_weight,
                subplan_weight=self.subplan_weight,
                cardinality_weight=self.cardinality_weight,
                ranking_loss_kwargs=dict(),
                regression_loss_kwargs=dict(),
            ),
        )


@attrs.define(frozen=True, slots=False)
class SUMOModelWithRelevancyConfig(SUMOModelConfig):
    """Configuration for the SumOfOperators model with relevancy."""

    name: Name = ModelName.SUMO_RELEVANCY
    # Relevancy Parameters
    adaptive_generator: bool = False
    use_relevancy_generator: bool = True
    predict_relevancy_scores: bool = False
    rl_use_log_transform: bool = True
    relevancy_generator_args: dict = attrs.field(init=False)

    # Relevancy generator parameters
    rl_scalar: float = 50.0
    rl_data_min: float = 1e-8
    rl_data_max: float = 180.0
    rl_log_degree: float = 5.0

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        object.__setattr__(
            self,
            "relevancy_generator_args",
            dict(
                rl_data_min=self.rl_data_min,
                rl_data_max=self.rl_data_max,
                rl_scalar=self.rl_scalar,
                rl_log_degree=self.rl_log_degree,
                adaptive=self.adaptive_generator,
                rl_use_log_transform=self.rl_use_log_transform,
            ),
        )


@attrs.define(frozen=True, slots=False)
class MambaSumOfOperatorsModelConfig(MambaRankingModelConfig):
    """Configuration for the Mamba SumOfOperators model that supports ranking."""

    name: Name = ModelName.MAMBA_SUM_OF_OPERATORS

    # This model is trained by predicting subplan operation times, so subplan_estimation is essential.
    subplan_estimation: bool = True
    subplans_reg_loss: bool = True
    subplan_weight: float = 1.0

    # This custom flag will signal the data collator to calculate op_time for labels.
    subplan_label_op_time: bool = True

    # Options for Cardinality Estimation
    cardinality_estimation: bool = False
    cardinality_weight: float = 0.1
    log_transform_card_labels: bool = True
    use_card_feature: bool = attrs.field(default=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        object.__setattr__(
            self,
            "loss_class_kwargs",
            dict(
                ranking_loss_name=self.ranking_loss_name,
                regression_loss_name=self.regression_loss_name,
                regression_weight=self.regression_weight,
                ranking_weight=self.ranking_weight,
                subplan_weight=self.subplan_weight,
                cardinality_weight=self.cardinality_weight,
                ranking_loss_kwargs=dict(),
                regression_loss_kwargs=dict(),
            ),
        )
