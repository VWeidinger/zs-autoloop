from . import configs

# Re-export all model config classes into the `classes` package namespace
# so that code which inspects `classes` (via `import classes as cls`) can
# discover ModelConfig subclasses by iterating over `cls.__dict__`.
from .configs.base import *
from .configs.constants import *
from .configs.dace import *
from .configs.e2e import *
from .configs.mamba import *
from .configs.misc import *
from .configs.qppnet import *
from .configs.sumo import *
from .configs.tabular import *
from .configs.zeroshot import *
from .configs.zs_ranking import *

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
