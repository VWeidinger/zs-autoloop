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
    ScaledPostgresModelConfig(),
    FlatModelConfig(),
    MSCNModelConfig(),
    E2EModelConfig(),
    QPPNetModelConfig(),
    QueryFormerModelConfig(),
    ZeroShotModelConfig(),
    ZSRankLossConfig(),
    DACEModelConfig(),
]

NO_COSTS_MODEL_CONFIGS = MODEL_CONFIGS + [
    QPPModelNoCostsConfig(),
    DACEModelNoCostsConfig(),
]

ACT_CARD_MODEL_CONFIGS = [
    ScaledPostgresModelConfig(),
    FlatModelConfig(),
    FlatModelActCardModelConfig(),
    QPPNetModelConfig(),
    QPPModelActCardsConfig(),
    ZeroShotModelConfig(),
    ZeroShotModelActCardConfig(),
    DACEModelConfig(),
    DACEModelActCardConfig(),
]

ACT_CARD_ALL_MODEL_CONFIGS = [
    ScaledPostgresModelConfig(),
    FlatModelConfig(),
    MSCNModelConfig(),
    E2EModelConfig(),
    QPPNetModelConfig(),
    QueryFormerModelConfig(),
    ZeroShotModelConfig(),
    DACEModelConfig(),
    FlatModelActCardModelConfig(),
    QPPModelActCardsConfig(),
    ZeroShotModelActCardConfig(),
    DACEModelActCardConfig(),
]
