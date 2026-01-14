from typing import Callable
import attrs as attrs
from .base import ModelConfig
from .constants import InputFormats, ModelName, ModelType, Name
from models.white_mamba.mamba_dataset import mamba_plan_collator
from training.batch_to_funcs import mamba_batch_to
from training.featurizations import Featurization, MambaFull


@attrs.define(frozen=True, slots=False)
class MambaModelConfig(ModelConfig):
    name: Name = ModelName.MAMBA
    type: ModelType = ModelType.WL_AGNOSTIC
    # featurization: Featurization = MambaFeaturization
    featurization: Featurization = MambaFull
    loss_class_name: str = "QLoss"
    loss_class_kwargs: dict = dict()

    batch_to_func: Callable = mamba_batch_to
    collator_func: Callable = mamba_plan_collator
    input_format: InputFormats = InputFormats.parsed
    device: str = "cuda:[socket]"  # Device selection is handled by octopus

    hidden_dim: int = 128
    output_dim: int = 1

    fc_out_kwargs: dict = dict(
        activation_class_name="LeakyReLU",
        activation_class_kwargs={},
        norm_class_name="Identity",
        norm_class_kwargs={},
        residual=False,
        dropout=False,
        p_dropout=0.0,
        activation=True,
        inplace=True,
    )
    final_mlp_kwargs: dict = dict(
        width_factor=1.5,
        n_layers=4,
    )
    msg_layer_kwargs: dict = dict(width_factor=0.6, n_layers=2)
    """node_type_kwargs: dict = dict(
        width_factor=1.5,
        n_layers=4,
        one_hot_embeddings=True,
        max_emb_dim=32,
        drop_whole_embeddings=False,
        apply_optimized_nte=True,
    )"""
    node_type_kwargs: dict = dict(
        width_factor=1.5,
        n_layers=4,
        one_hot_embeddings=True,
        max_emb_dim=32,
        drop_whole_embeddings=False,
        activation_class_name="ReLU",
        activation_class_kwargs={},
        norm_class_name="Identity",
        norm_class_kwargs={},
        residual=False,
        dropout=False,
        p_dropout=0.0,
        activation=True,
        inplace=True,
    )
    final_mlp_kwargs.update(**fc_out_kwargs)
    msg_layer_kwargs.update(**fc_out_kwargs)
    node_type_kwargs.update(**fc_out_kwargs)

    # layer_norm: bool = False  # use layer norm in the message passing layers
    layer_norm: bool = True
    add_child_type_info_to_messages: bool = False  # add child type information to the messages in the message passing layers
    use_dcn: bool = False  # use deep cross network in the feature extraction layers
    apply_layer_norm_before_dcn: bool = (
        False  # apply layer norm before the dcn in the feature extraction layers
    )
    residual_connection: bool = False  # use residual connections skipping the message aggregation layer (before layer norm)
    scaler_type: str = "robust"  # type of scaler to use for the input features - either 'robust' or 'minmax'
    # filter_condense_operators: bool = False  # whether to condense filter operators (reduce number of different ops)
    filter_condense_operators: bool = True
    subplan_monotonicity_loss: bool = False  # add a loss term that enforces the subplan runtime to be monotonically increasing
    subplan_monotonicity_loss_fn: str = "lt_loss"  # loss function for the subplan monotonicity loss - either lt_loss or relative_lt_loss
    scan_nte_relu: bool = False  # use ReLU activation in the scan node-type-encoder (join nte will not be affected by this setting)


class MambaModelConfigOHEM(MambaModelConfig):
    # OHEM - online hard example mining
    online_hard_example_mining: bool = False
    ohem_initialize_difficulty: bool = False  # initialize difficulty for OHEM with model before first epoch - otherwise all inf
    ohem_resort_after_num_epochs: int = (
        2  # resort OHEM dataset after this number of epochs
    )
    ohem_slice_by: str = "file"  # file, dataset, none
    ohem_ratio: float = 1  # ratio of training samples participating in OHEM (1 means all are sampled with OHEM, 50% means half is randomly sampled (static) and the other half is sampled with OHEM)


@attrs.define(frozen=True, slots=False)
class MambaModelConfigTanH(MambaModelConfig):
    name: Name = ModelName.MAMBA_TANH

    node_type_kwargs: dict = dict(
        width_factor=1.5,
        n_layers=4,
        one_hot_embeddings=True,
        max_emb_dim=32,
        drop_whole_embeddings=False,
        activation_class_name="Tanh",
        activation_class_kwargs={},
        norm_class_name="Identity",
        norm_class_kwargs={},
        residual=False,
        dropout=False,
        p_dropout=0.0,
        activation=True,
        inplace=True,
    )


@attrs.define(frozen=True, slots=False)
class MambaModelConfigRelu(MambaModelConfig):
    name: Name = ModelName.MAMBA_RELU

    node_type_kwargs: dict = dict(
        width_factor=1.5,
        n_layers=4,
        one_hot_embeddings=True,
        max_emb_dim=32,
        drop_whole_embeddings=False,
        activation_class_name="ReLU",
        activation_class_kwargs={},
        norm_class_name="Identity",
        norm_class_kwargs={},
        residual=False,
        dropout=False,
        p_dropout=0.0,
        activation=True,
        inplace=True,
    )


@attrs.define(frozen=True, slots=False)
class MambaRankingModelConfig(MambaModelConfig):
    name: Name = ModelName.MAMBA_RANKING

    # Dataset Parameters
    batch_size: int = attrs.field(init=False)
    group_size: int = 16
    batch_group_size: int = 16
    include_hint_idx_workloads: bool = attrs.field(default=True)
    include_complex_ranking_workloads: bool = attrs.field(default=True)
    include_complex_many_joins_workloads: bool = attrs.field(default=False)

    # Training Parameters
    lr: float = 1e-3
    early_stopping_patience: int = 75
    epochs: int = 75

    # Loss Parameters
    regression_loss_name: str = "QLoss"
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
