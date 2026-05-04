import attrs as attrs
from ldb_models.training.featurizations import (
    Featurization,
    PostgresEstSystemNoCardDetail,
)

from ldb_models.classes.configs.constants import ModelName, Name
from ldb_models.classes.configs.mamba import MambaRankingModelConfig
from ldb_models.classes.configs.relevancy import RelevancyConfigMixin
from ldb_models.classes.configs.zs_ranking import ZeroShotRankingModelBaseConfig


@attrs.define(frozen=True, slots=False)
class SumOfOperatorsModelConfig(ZeroShotRankingModelBaseConfig):
    """Configuration for the SumOfOperators model."""

    name: Name = ModelName.SUM_OF_OPERATORS

    # This model is trained by predicting subplan operation times, so subplan_estimation is essential.
    subplan_estimation: bool = True
    subplans_reg_loss: bool = True
    subplan_weight: float = 1.0

    # This custom flag will signal the data collator to calculate only the operator time for each subplan.
    subplan_label_op_time: bool = True

    # Options for Cardinality Estimation
    cardinality_estimation: bool = False
    cardinality_weight: float = 0.1
    # Whether to log transform cardinality labels.
    # If the loss function also log transforms predictions, there is an option to avoid logging labels to prevent double log transformation.
    log_transform_card_labels: bool = True
    dont_log_if_log_transformed: bool = True
    use_card_feature: bool = attrs.field(default=False)

    # SUMO-LOOP
    sumo_loop: bool = False
    sumo_loop_multiply_est_error: bool = False
    sumo_loop_loss_on_scaled: bool = False

    # Loss Parameters
    regression_loss_name: str = "BiasedLog1pLoss"

    # How to aggregate child messages for join nodes.
    use_per_child_join_aggregation: bool = False


@attrs.define(frozen=True, slots=False)
class SumOfOperatorsNoRankModelConfig(SumOfOperatorsModelConfig):
    """Configuration for the SumOfOperators model."""

    name: Name = ModelName.SUM_OF_OPERATORS_NO_RANKING

    stable: bool = True

    # Workload inclusion flags
    include_complex_ranking_workloads: bool = False

    # Options for Cardinality Estimation
    cardinality_estimation: bool = False
    cardinality_weight: float = 0.0

    # Loss Parameters
    regression_weight: float = 1.0
    ranking_weight: float = 0.0
    subplan_weight: float = 2.0

    # Grouping and batching
    group_size: int = 256
    batch_group_size: int = 1

    group_similar: bool = False
    group_by: str = None  # both/sql/none
    random_percentage: float = None


@attrs.define(frozen=True, slots=False)
class SUMOBaselineModelConfig(SumOfOperatorsNoRankModelConfig):
    """Configuration for the SUMO Baseline model that predicts total query runtime."""

    name: Name = ModelName.SUMO_BASELINE

    include_hint_idx_workloads: bool = False
    include_pg_opt_workloads: bool = False

    subplan_weight: float = 5.0


@attrs.define(frozen=True, slots=False)
class SumOfOperatorsLoopModelConfig(SumOfOperatorsNoRankModelConfig):
    # SUMO-LOOP: Loop-aware operator runtime prediction
    name: Name = ModelName.SUM_OF_OPERATORS_LOOP

    include_complex_many_joins_workloads: bool = False
    # Computes loss for subplans on the labels divided by the actual cardinality of the sibling subplan.
    # This forces the model to predict per loop iteration runtime for the right child of a join.
    # During inference, this per loop runtime is scaled by the predicted estimated cardinality of the sibling subplan.

    # Subplan Loss = Loss(right_pred_single_loop_runtime, op_time/act_card)
    # right_pred_runtime = left_pred_card * right_pred_single_loop_runtime
    sumo_loop: bool = True
    # instead of dividing the label by the actual cardinality,
    #   we can multiply the prediction by the ratio of predicted to estimated cardinality for the sibling subplan.
    # This stablizes training because per_loop runtime given by the optimizer is extremely low.

    # pg_off_multiplier = left_pred_card / left_est_card
    # Subplan Loss = Loss(right_pred_est_loop_runtime * pg_off_multiplier, op_time)
    # right_pred_runtime = pg_off_multiplier * right_pred_est_loop_runtime

    # This is also applied for right card pred
    # Card Loss = Loss(card_pred * pg_off_multiplier, op_time)
    # right_pred_card = pg_off_multiplier * right_pred_est_loop_card
    sumo_loop_multiply_est_error: bool = False
    # instead of dividing the label by the actual cardinality,
    #   we can multiply the prediction by the predicted cardinality of the sibling subplan.
    #   and compute loss on this scaled prediction against the original label.

    # Subplan Loss = Loss(pred * sibling_pred_card, op_time)
    # right_pred_runtime = sibling_pred_card * right_pred_single_loop_runtime

    # This is also applied for right card pred
    # Card Loss = Loss(right_pred_est_loop_card * sibling_pred_card, op_time)
    # right_pred_card = sibling_pred_card * right_pred_est_loop_card
    sumo_loop_loss_on_scaled: bool = True

    # Options for Cardinality Estimation (required for loop multiplier calculation)
    cardinality_estimation: bool = True
    cardinality_weight: float = 0.0005

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.sumo_loop and not self.cardinality_estimation:
            raise ValueError(
                "sumo_loop=True requires cardinality_estimation=True. "
                "Loop multipliers depend on sibling cardinality predictions."
            )
        if self.sumo_loop_multiply_est_error and not self.sumo_loop:
            raise ValueError(
                "sumo_loop_multiply_est_error=True requires sumo_loop=True."
            )
        if self.sumo_loop_loss_on_scaled and not self.sumo_loop:
            raise ValueError("sumo_loop_loss_on_scaled=True requires sumo_loop=True.")
        if self.sumo_loop_multiply_est_error and self.sumo_loop_loss_on_scaled:
            raise ValueError(
                "sumo_loop_multiply_est_error and sumo_loop_loss_on_scaled are mutually exclusive."
            )


@attrs.define(frozen=True, slots=False)
class SumOfOperatorsModelWithRelevancyConfig(
    RelevancyConfigMixin, SumOfOperatorsModelConfig
):
    """Configuration for the SumOfOperators model with relevancy."""

    name: Name = ModelName.SUM_OF_OPERATORS_RELEVANCY

    pass


@attrs.define(frozen=True, slots=False)
class SumOfOperatorsModelWithNoCardConfig(SumOfOperatorsModelWithRelevancyConfig):
    """Configuration for the SumOfOperators model without cardinality estimation."""

    name: Name = ModelName.SUM_OF_OPERATORS_NO_CARD

    cardinality_estimation: bool = True
    use_card_feature: bool = True

    featurization: Featurization = PostgresEstSystemNoCardDetail


@attrs.define(frozen=True, slots=False)
class SUMOModelConfig(ZeroShotRankingModelBaseConfig):
    """Configuration for the SumOfOperators model."""

    name: Name = ModelName.SUMO

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

    # SUMO-LOOP: Loop-aware operator runtime prediction
    sumo_loop: bool = False
    # SUMO-LOOP mode: use (pred_card / est_card) * est_loop_runtime for right child.
    sumo_loop_multiply_est_error: bool = False
    # SUMO-LOOP mode: use sibling predicted cardinality as multiplier and compute subplan loss on scaled preds.
    sumo_loop_loss_on_scaled: bool = False

    # Loss Parameters
    regression_loss_name: str = "Log1pQLossScaled"

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        # Validate SUMO-LOOP configuration
        if self.sumo_loop and not self.cardinality_estimation:
            raise ValueError(
                "sumo_loop=True requires cardinality_estimation=True. "
                "Loop multipliers depend on sibling cardinality predictions."
            )
        if self.sumo_loop_multiply_est_error and not self.sumo_loop:
            raise ValueError(
                "sumo_loop_multiply_est_error=True requires sumo_loop=True."
            )
        if self.sumo_loop_loss_on_scaled and not self.sumo_loop:
            raise ValueError("sumo_loop_loss_on_scaled=True requires sumo_loop=True.")
        if self.sumo_loop_multiply_est_error and self.sumo_loop_loss_on_scaled:
            raise ValueError(
                "sumo_loop_multiply_est_error and sumo_loop_loss_on_scaled are mutually exclusive."
            )
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
class SUMOModelWithRelevancyConfig(RelevancyConfigMixin, SUMOModelConfig):
    """Configuration for the SumOfOperators model with relevancy."""

    name: Name = ModelName.SUMO_RELEVANCY

    # Relevancy generator parameters
    rl_scalar: float = 50.0

    # Preserve previous behavior: SUMO relevancy args did not include this key.
    _include_mask_single_plan_queries = False


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
