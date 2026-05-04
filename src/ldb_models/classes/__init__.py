from ldb_models.classes import configs

# Re-export all model config classes into the `classes` package namespace
# so that code which inspects `classes` (via `import classes as cls`) can
# discover ModelConfig subclasses by iterating over `cls.__dict__`.
from ldb_models.classes.configs.base import *  # noqa: F403
from ldb_models.classes.configs.constants import *  # noqa: F403
from ldb_models.classes.configs.dace import *  # noqa: F403
from ldb_models.classes.configs.e2e import *  # noqa: F403
from ldb_models.classes.configs.mamba import *  # noqa: F403
from ldb_models.classes.configs.misc import *  # noqa: F403
from ldb_models.classes.configs.qppnet import *  # noqa: F403
from ldb_models.classes.configs.sumo import *  # noqa: F403
from ldb_models.classes.configs.tabular import *  # noqa: F403
from ldb_models.classes.configs.zeroshot import *  # noqa: F403
from ldb_models.classes.configs.zs_ranking import *  # noqa: F403

MODEL_CONFIGS = [
    configs.tabular.ScaledPostgresModelConfig(),
    configs.tabular.FlatModelConfig(),
    configs.misc.MSCNModelConfig(),
    configs.e2e.E2EModelConfig(),
    configs.qppnet.QPPNetModelConfig(),
    configs.misc.QueryFormerModelConfig(),
    configs.zeroshot.ZeroShotModelConfig(),
    configs.zeroshot.ZSRankLossConfig(),
    configs.dace.DACEModelConfig(),
]

NO_COSTS_MODEL_CONFIGS = MODEL_CONFIGS + [
    configs.qppnet.QPPModelNoCostsConfig(),
    configs.dace.DACEModelNoCostsConfig(),
]

ACT_CARD_MODEL_CONFIGS = [
    configs.tabular.ScaledPostgresModelConfig(),
    configs.tabular.FlatModelConfig(),
    configs.tabular.FlatModelActCardModelConfig(),
    configs.qppnet.QPPNetModelConfig(),
    configs.qppnet.QPPModelActCardsConfig(),
    configs.zeroshot.ZeroShotModelConfig(),
    configs.zeroshot.ZeroShotModelActCardConfig(),
    configs.dace.DACEModelConfig(),
    configs.dace.DACEModelActCardConfig(),
]

ACT_CARD_ALL_MODEL_CONFIGS = [
    configs.tabular.ScaledPostgresModelConfig(),
    configs.tabular.FlatModelConfig(),
    configs.misc.MSCNModelConfig(),
    configs.e2e.E2EModelConfig(),
    configs.qppnet.QPPNetModelConfig(),
    configs.misc.QueryFormerModelConfig(),
    configs.zeroshot.ZeroShotModelConfig(),
    configs.dace.DACEModelConfig(),
    configs.tabular.FlatModelActCardModelConfig(),
    configs.qppnet.QPPModelActCardsConfig(),
    configs.zeroshot.ZeroShotModelActCardConfig(),
    configs.dace.DACEModelActCardConfig(),
]
