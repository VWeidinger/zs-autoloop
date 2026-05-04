import attrs as attrs
from ldb_models.training.featurizations import (
    Featurization,
    PostgresEstWithNumLoops,
    PostgresTrueCardDetail,
)

from ldb_models.classes.configs.constants import ModelName, Name
from ldb_models.classes.configs.relevancy import RelevancyConfigMixin
from ldb_models.classes.configs.zeroshot import ZeroShotModelConfig


@attrs.define(frozen=True, slots=False)
class ZeroShotRankingModelBaseConfig(ZeroShotModelConfig):
    name: Name = ModelName.ZEROSHOT_RANKING_BASE

    featurization: Featurization = PostgresEstWithNumLoops

    # Dataset Parameters
    group_size: int = 64  # number of plans in a group for ranking
    batch_group_size: int = 32  # number of groups in a batch
    batch_size: int = attrs.field(init=False)

    # Optimizer Parameters
    lr: float = 5e-4
    weight_decay: float = 1e-2
    scheduler_class_name: str = "CosineAnnealingWarmRestarts"
    warmup_epochs: int = 5

    # Training Parameters
    early_stopping_patience: int = 75
    epochs: int = 75

    # Data Inclusion Parameters
    include_hint_idx_workloads: bool = attrs.field(default=True)
    include_complex_ranking_workloads: bool = attrs.field(default=True)
    include_complex_many_joins_workloads: bool = attrs.field(default=False)

    # Data Sampling Parameters
    group_similar: bool = attrs.field(default=True)
    group_by: str = attrs.field(default="sql")  # both/sql/plan_runtime_ms
    random_percentage: float = attrs.field(default=0.0)
    shuffle_train_data: bool = attrs.field(default=True)

    # Loss Parameters
    regression_loss_name: str = "Log1pQLoss"
    ranking_loss_name: str = "LambdaLoss"
    regression_weight: float = 0.0
    ranking_weight: float = 1.0
    loss_class_name: str = "LearnToRankWithRegressionLoss"
    loss_class_kwargs: dict = attrs.field(init=False)

    # Ranking Loss Parameters
    weighing_scheme: str = "ndcgLoss2PP"
    k: int = None

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        object.__setattr__(self, "batch_size", self.batch_group_size * self.group_size)
        ranking_loss_kwargs = dict()
        if self.ranking_weight != 0.0:
            if self.k is not None:
                ranking_loss_kwargs["k"] = self.k
            if self.weighing_scheme is not None:
                ranking_loss_kwargs["weighing_scheme"] = self.weighing_scheme
        if (
            not self.group_similar
            and self.group_by is not None
            and self.random_percentage is not None
        ):
            object.__setattr__(self, "group_by", None)
            object.__setattr__(self, "random_percentage", None)

        object.__setattr__(
            self,
            "loss_class_kwargs",
            dict(
                ranking_loss_name=self.ranking_loss_name,
                regression_loss_name=self.regression_loss_name,
                regression_weight=self.regression_weight,
                ranking_weight=self.ranking_weight,
                ranking_loss_kwargs=dict(**ranking_loss_kwargs),
                regression_loss_kwargs=dict(),
            ),
        )
        object.__setattr__(
            self, "optimizer_kwargs", dict(lr=self.lr, weight_decay=self.weight_decay)
        )


@attrs.define(frozen=True, slots=False)
class ZeroShotRankingModelConfig(ZeroShotRankingModelBaseConfig):
    name: Name = ModelName.ZEROSHOT_RANKING


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
class ZeroShotRankingModelConfigWithRelevancy(
    RelevancyConfigMixin, ZeroShotRankingModelBaseConfig
):
    name: Name = ModelName.ZEROSHOT_RANKING_RELEVANCY

    pass


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

    latent_strategy: str = "grouped"  # grouped/unified/tournament

    def __attrs_post_init__(self):
        super().__attrs_post_init__()


@attrs.define(frozen=True, slots=False)
class LatentZeroShotRankingWithRelevancyConfig(
    RelevancyConfigMixin, LatentZeroShotRankingModelConfig
):
    name: Name = ModelName.ZEROSHOT_RANKING_LATENT_RELEVANCY

    pass


@attrs.define(frozen=True, slots=False)
class TournamentZeroShotRankingModelConfig(LatentZeroShotRankingWithRelevancyConfig):
    name: Name = ModelName.ZEROSHOT_RANKING_LATENT_TOURNAMENT

    latent_strategy: str = "tournament"  # grouped/unified/tournament


@attrs.define(frozen=True, slots=False)
class UnifiedZeroShotRankingModelConfig(LatentZeroShotRankingWithRelevancyConfig):
    name: Name = ModelName.ZEROSHOT_RANKING_LATENT_UNIFIED

    latent_strategy: str = "unified"  # grouped/unified/tournament


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

    # Parameters for the Filter Model
    cascade_filter_model_dir: str = attrs.field(
        default="/mnt/labstore/SIGs/ML/learned_db/models/pm-zs_sop_rl-s50-ld3-rl_use_log_transform/imdb"
    )
    cascade_filter_model_name: str = attrs.field(
        default="pm-zs_sop_rl-s50-ld3-rl_use_log_transform_4_rankmodel"
    )
    # /mnt/labstore/SIGs/ML/learned_db/models/pm-zs_sop_rl-s50-ld3-rl_use_log_transform/imdb/pm-zs_sop_rl-s50-ld3-rl_use_log_transform_4_rankmodel.pt
    cascade_filter_model_type: str = attrs.field(default="zs_sop_rl")
