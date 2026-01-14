import attrs as attrs
from training.featurizations import (
    Featurization,
    PostgresEstWithNumLoops,
    PostgresTrueCardDetail,
)

from .constants import ModelName, Name
from .zeroshot import ZeroShotModelConfig


@attrs.define(frozen=True, slots=False)
class ZeroShotRankingModelBaseConfig(ZeroShotModelConfig):
    name: Name = ModelName.ZEROSHOT_RANKING_BASE

    featurization: Featurization = PostgresEstWithNumLoops

    # Dataset Parameters
    group_size: int = 64  # number of plans in a group for ranking
    batch_group_size: int = 32  # number of groups in a batch
    batch_size: int = attrs.field(init=False)
    lr: float = 5e-4
    early_stopping_patience: int = 75
    epochs: int = 75
    include_hint_idx_workloads: bool = attrs.field(default=True)
    include_complex_ranking_workloads: bool = attrs.field(default=True)
    include_complex_many_joins_workloads: bool = attrs.field(default=False)
    shuffle_train_data: bool = attrs.field(default=True)
    # Loss Parameters
    regression_loss_name: str = "Log1pQLossScaled"
    ranking_loss_name: str = "LambdaLoss"
    regression_weight: float = 0.0
    ranking_weight: float = 1.0
    loss_class_name: str = "LearnToRankWithRegressionLoss"
    loss_class_kwargs: dict = attrs.field(init=False)

    def __attrs_post_init__(self):
        object.__setattr__(self, "batch_size", self.batch_group_size * self.group_size)
        object.__setattr__(
            self,
            "loss_class_kwargs",
            dict(
                ranking_loss_name=self.ranking_loss_name,
                regression_loss_name=self.regression_loss_name,
                regression_weight=self.regression_weight,
                ranking_weight=self.ranking_weight,
                ranking_loss_kwargs=dict(),
                regression_loss_kwargs=dict(),
            ),
        )
        object.__setattr__(self, "optimizer_kwargs", dict(lr=self.lr))


@attrs.define(frozen=True, slots=False)
class ZeroShotRankingModelConfig(ZeroShotRankingModelBaseConfig):
    name: Name = ModelName.ZEROSHOT_RANKING

    scheduler_class_name: str = "NoScheduler"


@attrs.define(frozen=True, slots=False)
class ZeroShotRankingModelConfigWithActCard(ZeroShotRankingModelBaseConfig):
    name: Name = ModelName.ZEROSHOT_RANKING_ACT_CARD
    featurization: Featurization = PostgresTrueCardDetail


@attrs.define(frozen=True, slots=False)
class ZeroShotRankingModelConfigWithSubplan(ZeroShotRankingModelBaseConfig):
    name: Name = ModelName.ZEROSHOT_RANKING_SUBPLAN
    subplans_reg_loss: bool = True
    subplan_estimation: bool = True
    subplan_weight: float = 0.1

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
                ranking_loss_kwargs=dict(),
                regression_loss_kwargs=dict(),
                subplan_weight=self.subplan_weight,
            ),
        )


@attrs.define(frozen=True, slots=False)
class ZeroShotRankingModelConfigWithRelevancy(ZeroShotRankingModelBaseConfig):
    name: Name = ModelName.ZEROSHOT_RANKING_RELEVANCY

    adaptive_generator: bool = False
    use_relevancy_generator: bool = True
    predict_relevancy_scores: bool = True
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
class LatentZeroShotRankingModelConfig(ZeroShotRankingModelBaseConfig):
    name: Name = ModelName.ZEROSHOT_RANKING_LATENT
    latent: bool = True
    # whether to apply LayerNorm to latent representations before attention
    use_layernorm: bool = True
    # attention parameters
    attention_heads: int = 4
    attention_layers: int = 2
    # whether to sample batches such that all plans in a batch belong to the same query in test
    all_same_batch_sampler: bool = True

    group_similar: bool = True
    random_percentage: float = 0.25
    group_by: str = "sql"  # sql/plan_runtime_ms

    latent_strategy: str = "grouped"  # grouped/unified/tournament

    subplan_estimation: bool = False
    subplan_weight: float = 0.0
    subplans_reg_loss: bool = False

    def __attrs_post_init__(self):
        super().__attrs_post_init__()


@attrs.define(frozen=True, slots=False)
class LatentZeroShotRankingWithRelevancyConfig(LatentZeroShotRankingModelConfig):
    name: Name = ModelName.ZEROSHOT_RANKING_LATENT_RELEVANCY

    use_relevancy_generator: bool = True
    predict_relevancy_scores: bool = False
    adaptive_generator: bool = False
    rl_use_log_transform: bool = True
    relevancy_generator_args: dict = attrs.field(init=False)
    rl_scalar: float = 50.0
    rl_data_min: float = 1e-8
    rl_data_max: float = 180.0
    rl_log_degree: float = 5.0

    group_similar: bool = True
    group_by: str = "sql"  # sql/plan_runtime_ms
    random_percentage: float = 0.25
    batch_group_size: int = 32
    group_size: int = 64
    batch_size: int = attrs.field(init=False)

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
class TournamentZeroShotRankingModelConfig(LatentZeroShotRankingWithRelevancyConfig):
    name: Name = ModelName.ZEROSHOT_RANKING_LATENT_TOURNAMENT

    latent_strategy: str = "tournament"  # grouped/unified/tournament
    all_same_batch_sampler: bool = True


@attrs.define(frozen=True, slots=False)
class UnifiedZeroShotRankingModelConfig(LatentZeroShotRankingWithRelevancyConfig):
    name: Name = ModelName.ZEROSHOT_RANKING_LATENT_UNIFIED

    latent_strategy: str = "unified"  # grouped/unified/tournament
    all_same_batch_sampler: bool = True


@attrs.define(frozen=True, slots=False)
class CascadeRankingModelConfig(LatentZeroShotRankingWithRelevancyConfig):
    """
    Configuration for a cascade ranking model.

    This model uses a two-stage approach for evaluation:
    1. A fast, pretrained 'filter' model makes initial predictions on all candidate plans.
    2. The top-k candidates are then re-ranked by a more complex 'rank' model (the main model
       defined by this config).
    """

    name: Name = ModelName.ZEROSHOT_RANKING_LATENT_CASCADE

    use_pretrained_embeddings: bool = False
    all_same_batch_sampler: bool = True

    latent_strategy: str = "grouped"  # grouped/unified/tournament

    # Parameters for the Filter Model
    cascade_filter_model_dir: str = attrs.field(
        default="/mnt/labstore/SIGs/ML/learned_db/models/pm-zs_sop_rl-s50-ld3-rl_use_log_transform/imdb"
    )
    cascade_filter_model_name: str = attrs.field(
        default="pm-zs_sop_rl-s50-ld3-rl_use_log_transform_4_rankmodel"
    )
    # /mnt/labstore/SIGs/ML/learned_db/models/pm-zs_sop_rl-s50-ld3-rl_use_log_transform/imdb/pm-zs_sop_rl-s50-ld3-rl_use_log_transform_4_rankmodel.pt
    cascade_filter_model_type: str = attrs.field(default="zs_sop_rl")
