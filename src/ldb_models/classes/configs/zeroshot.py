from typing import Callable

import attrs as attrs
from ldb_models.models.zeroshot.postgres_plan_batching import postgres_plan_collator
from ldb_models.training.batch_to_funcs import zs_batch_to
from ldb_models.training.featurizations import (
    Featurization,
    PostgresEstSystemCardDetail,
    PostgresEstWithPGCost,
    PostgresTrueCardDetail,
    with_jit_plan_features,
)

from ldb_models.classes.configs.base import ModelConfig
from ldb_models.classes.configs.constants import InputFormats, ModelName, ModelType, Name


@attrs.define(frozen=True, slots=False)
class ZeroShotModelConfig(ModelConfig):
    name: Name = ModelName.ZEROSHOT
    type: ModelType = ModelType.WL_AGNOSTIC
    featurization: Featurization = PostgresEstSystemCardDetail
    loss_class_name: str = "QLoss"
    loss_class_kwargs: dict = dict()

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
    tree_layer_kwargs: dict = dict(width_factor=0.6, n_layers=2)
    node_type_kwargs: dict = dict(
        width_factor=1.5,
        n_layers=4,
        one_hot_embeddings=True,
        max_emb_dim=32,
        drop_whole_embeddings=False,
    )
    final_mlp_kwargs.update(**fc_out_kwargs)
    tree_layer_kwargs.update(**fc_out_kwargs)
    node_type_kwargs.update(**fc_out_kwargs)

    hidden_dim: int = 128
    output_dim: int = 1
    batch_size: int = 256
    tree_layer_name: str = "MscnConv"  # GATConv MscnConv
    # how to aggregate child messages for join-like nodes. Options:
    # - 'sum' (default): sum up all child messages (existing behavior)
    # - 'proj_then_concat': project each child message to half hidden dim, concat, then reduce
    # - 'concat_reduce': concatenate child messages as-is and reduce with a linear layer
    # - 'dot_gate': use a dot product gating mechanism to aggregate child messages
    # - 'per_dim_gate': use a per-dimension gating mechanism to aggregate child messages
    # - 'pad_single': pad a single child message with zeros to match expected dim
    join_repr_mode: str = "pad_single"
    # when enabled, intra_plan join aggregation preserves individual child messages
    # per destination node instead of pre-summing by source node type.
    use_per_child_join_aggregation: bool = False
    # when enabled, plan nodes on the deepest level of a batch (and plan depths that get no
    # to_plan edge in the batch) receive the same zero-message update as every other leaf.
    # Without it, a plan's prediction depends on whether its batch holds a deeper plan
    # (e.g. batch size 1 vs 256). Shortened to "RMP" in model names.
    reproducible_message_passing: bool = False
    # when enabled, the JIT information of the plan parser (JIT used, inlining, optimization, number of compiled
    # functions, node paying the lazy JIT compilation) is added to the plan node features (featurization {..}WithJIT).
    # Requires parsed_plans_version 50. Shortened to "JIT" in model names.
    use_jit_features: bool = False
    batch_to_func: Callable = zs_batch_to
    collator_func: Callable = postgres_plan_collator
    input_format: InputFormats = InputFormats.parsed
    device: str = "cuda:[socket]"  # Device selection is handled by octopus
    skip_top_aggregation: bool = (
        False  # removes any aggregation node from the top of the tree
    )
    filter_inline_col_feats: bool = (
        True  # by default zs is trained with inline column features
    )
    # we have > and < in test data but not in training data, so by default we rewrite them to >= and <= to match the training data distribution.
    # whether to rewrite operators to canonical forms (e.g. < to <=)
    do_operator_rewriting: bool = True
    annotate_base_table_card_to_scan_op: bool = False  # annotate the base table card as children_card at scan ops (instead of setting it to 1 by default)
    subplan_monotonicity_loss: bool = False  # add a loss term that enforces the subplan runtime to be monotonically increasing
    subplan_monotonicity_loss_fn: str = "lt_loss"  # loss function for the subplan monotonicity loss - either lt_loss or relative_lt_loss
    scaler_type: str = "robust"  # type of scaler to use for the input features - either 'robust' or 'minmax'
    # whether to apply log1p transform to input cardinality features (est/act)
    # This is distinct from the sum-of-operators specific option `log_transform_card_labels`.
    use_log_input_card: bool = False
    subplan_estimation: bool = False  # perform subplan loss - i.e. compute the subplan runtimes and add a loss term for them
    subplans_reg_loss: bool = False  # Only compute a regression loss on the subplans

    online_hard_example_mining: bool = False

    stable: bool = False
    stable_layernorm: bool = False
    stable_residuals: bool = True
    stable_dropout: bool = True

    def __attrs_post_init__(self):
        if self.stable:
            # Apply stable defaults without overriding explicitly provided kwargs
            # (e.g., values passed via manual config parsing / CLI overrides).
            attr_defaults = {
                attr.name: attr.default for attr in type(self).__attrs_attrs__
            }
            norm_class_name = "LayerNorm" if self.stable_layernorm else "Identity"
            dropout = self.stable_dropout
            residual = self.stable_residuals
            stable_fc_out_defaults: dict = dict(
                activation_class_name="LeakyReLU",
                activation_class_kwargs={},
                norm_class_name=norm_class_name,
                norm_class_kwargs={},
                residual=residual,
                dropout=dropout,
                p_dropout=0.1,
                activation=True,
                inplace=True,
            )

            default_fc_out_kwargs = attr_defaults.get("fc_out_kwargs")
            default_node_type_kwargs = attr_defaults.get("node_type_kwargs")
            default_tree_layer_kwargs = attr_defaults.get("tree_layer_kwargs")
            default_final_mlp_kwargs = attr_defaults.get("final_mlp_kwargs")

            fc_out_kwargs = dict(self.fc_out_kwargs)
            fc_out_kwargs_is_default = self.fc_out_kwargs is default_fc_out_kwargs
            for key, value in stable_fc_out_defaults.items():
                if key not in fc_out_kwargs or fc_out_kwargs_is_default:
                    fc_out_kwargs[key] = value

            node_type_kwargs = dict(self.node_type_kwargs)
            tree_layer_kwargs = dict(self.tree_layer_kwargs)
            final_mlp_kwargs = dict(self.final_mlp_kwargs)
            node_type_kwargs_is_default = (
                self.node_type_kwargs is default_node_type_kwargs
            )
            tree_layer_kwargs_is_default = (
                self.tree_layer_kwargs is default_tree_layer_kwargs
            )
            final_mlp_kwargs_is_default = (
                self.final_mlp_kwargs is default_final_mlp_kwargs
            )

            for key, value in fc_out_kwargs.items():
                if key not in node_type_kwargs or node_type_kwargs_is_default:
                    node_type_kwargs[key] = value
                if key not in tree_layer_kwargs or tree_layer_kwargs_is_default:
                    tree_layer_kwargs[key] = value
                if key not in final_mlp_kwargs or final_mlp_kwargs_is_default:
                    final_mlp_kwargs[key] = value

            object.__setattr__(self, "fc_out_kwargs", fc_out_kwargs)
            object.__setattr__(self, "final_mlp_kwargs", final_mlp_kwargs)
            object.__setattr__(self, "tree_layer_kwargs", tree_layer_kwargs)
            object.__setattr__(self, "node_type_kwargs", node_type_kwargs)
        # parsed_plans50 = parsed_plans3 + JIT information
        if self.parsed_plans_version in (3, 50):
            object.__setattr__(self, "do_operator_rewriting", True)
        else:
            object.__setattr__(self, "do_operator_rewriting", False)
        if self.use_jit_features:
            if self.parsed_plans_version != 50:
                raise ValueError(
                    f"use_jit_features requires parsed_plans_version 50 (parsed plans with JIT information), "
                    f"got {self.parsed_plans_version}"
                )
            object.__setattr__(
                self, "featurization", with_jit_plan_features(self.featurization)
            )


@attrs.define(frozen=True, slots=False)
class ZeroShotModelConfigOHEM(ZeroShotModelConfig):
    # OHEM - online hard example mining
    online_hard_example_mining: bool = False
    ohem_initialize_difficulty: bool = False  # initialize difficulty for OHEM with model before first epoch - otherwise all inf
    ohem_resort_after_num_epochs: int = (
        2  # resort OHEM dataset after this number of epochs
    )
    ohem_slice_by: str = "file"  # file, dataset, none
    ohem_ratio: float = 1  # ratio of training samples participating in OHEM (1 means all are sampled with OHEM, 50% means half is randomly sampled (static) and the other half is sampled with OHEM)


@attrs.define(frozen=True, slots=False)
class ZeroShotLowMin(ZeroShotModelConfig):
    name: Name = ModelName.ZS_LOW_MIN
    loss_class_kwargs: dict = {"min_val": 1e-6}


@attrs.define(frozen=True, slots=False)
class ZeroShotXLConfig(ZeroShotModelConfig):
    name: Name = ModelName.ZS_XL
    final_mlp_kwargs: dict = dict(
        width_factor=2, n_layers=6, **ZeroShotModelConfig().fc_out_kwargs
    )
    tree_layer_kwargs: dict = dict(
        width_factor=1.2, n_layers=4, **ZeroShotModelConfig().fc_out_kwargs
    )
    hidden_dim: int = 256
    include_scan_only_hint_workloads: bool = True
    include_hint_idx_workloads: bool = True
    include_complex_workloads: bool = True
    loss_class_name: str = "QLogSqLoss"


@attrs.define(frozen=True, slots=False)
class ZeroShotXLLayerNormConfig(ZeroShotXLConfig):
    name: Name = ModelName.ZS_XL_LAYER_NORM
    tree_layer_kwargs: dict = dict(
        width_factor=1.2,
        n_layers=4,
        layer_norm=True,
        **ZeroShotModelConfig().fc_out_kwargs,
    )


@attrs.define(frozen=True, slots=False)
class ZeroShotLayerNormConfig(ZeroShotModelConfig):
    name: Name = ModelName.ZS_LAYER_NORM
    tree_layer_kwargs: dict = dict(
        width_factor=0.6,
        n_layers=2,
        layer_norm=True,
        **ZeroShotModelConfig().fc_out_kwargs,
    )

    include_scan_only_hint_workloads: bool = True
    include_hint_idx_workloads: bool = True
    include_complex_workloads: bool = True


@attrs.define(frozen=True, slots=False)
class ZSStratifiedModelConfig(ZeroShotModelConfig):
    name: Name = ModelName.ZS_STRATIFIED
    stratified_workloads: bool = True


@attrs.define(frozen=True, slots=False)
class ZSRankLossConfig(ZeroShotModelConfig):
    name: Name = ModelName.ZS_RANK_LOSS
    loss_class_name: str = "RankLoss"


@attrs.define(frozen=True, slots=False)
class ZSLearnFromTimeoutConfig(ZeroShotModelConfig):
    name: Name = ModelName.ZS_FROM_TIMEOUT
    include_timeout_queries: bool = True
    loss_class_kwargs: dict = {"learn_from_timeout": True, "timeout_threshold": 30}


@attrs.define(frozen=True, slots=False)
class ZeroShotModel2Dim(ZeroShotModelConfig):
    name: Name = ModelName.ZS_2DIM
    hidden_dim: int = 2


@attrs.define(frozen=True, slots=False)
class ZSStratifiedPGCostWithHintPlan(ZeroShotModelConfig):
    name: Name = ModelName.ZS_STRATIFIED_PG_COST_WITH_HINT_PLAN
    stratified_workloads: bool = True
    include_hint_workloads: bool = True
    featurization: Featurization = PostgresEstWithPGCost


@attrs.define(frozen=True, slots=False)
class ZSPGCostWithHintPlan(ZeroShotModelConfig):
    name: Name = ModelName.ZS_PG_COST_WITH_HINT_PLAN
    include_hint_workloads: bool = True
    featurization: Featurization = PostgresEstWithPGCost


@attrs.define(frozen=True, slots=False)
class ZeroShotModelActCardConfig(ZeroShotModelConfig):
    name: Name = ModelName.ZEROSHOT_ACT_CARD
    featurization: Featurization = PostgresTrueCardDetail


@attrs.define(frozen=True, slots=False)
class ZeroShotWithPostgresConfig(ZeroShotModelConfig):
    name: Name = ModelName.ZS_PGCOST
    featurization: Featurization = PostgresEstWithPGCost


@attrs.define(frozen=True, slots=False)
class ZeroShotModelWHintsConfig(ZeroShotModelConfig):
    name: Name = ModelName.ZS_WITH_HINTS
    include_hint_workloads: bool = True


@attrs.define(frozen=True, slots=False)
class ZeroShotModelOnlyHints(ZeroShotModelConfig):
    name: Name = ModelName.ZS_ONLY_HINTS
    only_hint_workloads: bool = True


@attrs.define(frozen=True, slots=False)
class ZeroShotVAEModelConfig(ZeroShotModelConfig):
    name: Name = ModelName.ZS_VAE
    vae_dim: int = 128
    beta: float = 0.1
    num_samples: int = 100
