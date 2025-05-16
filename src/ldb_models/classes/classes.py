from pathlib import Path
from typing import List, Callable, Optional

import attrs as attrs
from attr import field
from ldb_schema.datasets import Dataset
from ldb_schema.paths import Paths, ClusterPaths

from models.query_former.dataloader import query_former_batch_to
from models.t3.query_plan import CardType
from models.workload_driven.dataset.mscn_batching import mscn_plan_collator
from models.workload_driven.dataset.plan_tree_batching import plan_batch_to, baseline_plan_collator
from models.zeroshot.postgres_plan_batching import postgres_plan_collator
from training.batch_to_funcs import dace_batch_to, simple_batch_to, batch_to, neo_batch_to
from training.featurizations import Featurization, DACEFeaturization, QPPNetFeaturization, \
    PostgresEstSystemCardDetail, DACEFeaturizationNoCosts, QPPNetNoCostsFeaturization, FlatModelFeaturization, \
    FlatModelActCardFeaturization, QPPNetActCardsFeaturization, PostgresTrueCardDetail, DACEActCardFeaturization, \
    PostgresEstWithPGCost


class InputFormats:
    parsed = "parsed_plans"
    baseline_parsed = 'augmented_plans'
    json = 'json'


@attrs.define(frozen=True, slots=False)
class Name:
    NAME: str = field(default=None)
    DISPLAY_NAME: str = field(default=None)


class ModelName:
    POSTGRES: Name = Name("postgres", "Sc. Postgres")
    POSTGRES_STRATIFIED = Name("postgres_stratified", "Sc. Postgres (stratified)")

    FLAT: Name = Name("flat", "Flat Vector")
    FLAT_ACT_CARDS = Name("flat_act_cards", "Flat Vector (act. card.)")
    FLAT_STRATIFIED = Name("flat_stratified", "Flat Vector (stratified)")

    MSCN = Name("mscn", "MSCN")
    E2E = Name("e2e", "E2E")
    ZEROSHOT = Name("zeroshot", "Zero-Shot")
    ZS_LOW_MIN = Name("zeroshot_low_min", "Zero-Shot (low min)")
    ZEROSHOT_ACT_CARD = Name("zeroshot_act_card", "Zero-Shot (act. card)")
    ZS_FROM_TIMEOUT = Name('zeroshot_learn_from_timeout', 'Zero-Shot (learn from timeout)')
    ZS_RANK_LOSS = Name('zeroshot_rank_loss', 'Zero-Shot (Rank Loss)')
    ZS_STRATIFIED = Name("zeroshot_stratified", "Zero-Shot (stratified)")
    ZS_XL = Name("zeroshot_xl", "Zero-Shot (XL)")
    ZS_XL_LAYER_NORM = Name("zeroshot_xl_layer_norm", "Zero-Shot (XL, Layer Norm)")
    ZS_LAYER_NORM = Name("zeroshot_layer_norm", "Zero-Shot (Layer Norm)")
    ZS_PGCOST = Name("zeroshot_pgcost", "Zero-Shot (pgcost)")
    ZS_WITH_HINTS = Name("zeroshot_with_hints", "Zero-Shot (with hints)")
    ZS_ONLY_HINTS = Name("zeroshot_only_hints", "Zero-Shot (only hints)")
    ZS_2DIM = Name("zeroshot_2dim", "Zero-Shot (2dim)")
    ZS_STRATIFIED_PG_COST_WITH_HINT_PLAN = Name("zeroshot_stratified_pg_cost_with_hint_plan",
                                                "Zero-Shot (stratified, PG cost, with hint plan)")
    ZS_PG_COST_WITH_HINT_PLAN = Name("zeroshot_pg_cost_with_hint_plan",
                                     "Zero-Shot (PG cost, with hint plan)")

    QPP_NET = Name("qppnet", "QPP-Net")
    QPP_NET_NO_COSTS = Name("qppnet_no_costs", "QPP-Net\n(w/o PG costs)")
    QPP_NET_ACT_CARDS = Name("qppnet_act_cards", "QPP-Net (act. card.)")
    DACE = Name("dace", "DACE")
    DACE_ACT_CARDS = Name("dace_act_cards", "DACE (act. card.)")
    DACE_NO_COSTS = Name("dace_no_costs", "DACE\n(w/o PG costs)")
    DACE_STRATIFIED = Name("dace_stratified", "DACE (stratified)")
    DACE_WITH_HINTS = Name("dace_with_hints", "DACE (with hints)")
    DACE_ONLY_HINTS = Name("dace_only_hints", "DACE (only hints)")

    QUERY_FORMER = Name("query_former", "QueryFormer")
    T3 = Name("t3", "T3")
    T3Flat = Name("t3_flat", "T3 (Flat)")
    T3Tree = Name("t3_tree", "T3 (Tree)")

    NEO = Name("neo", "Neo")


class ModelType:
    WL_DRIVEN = "wl_driven"
    WL_AGNOSTIC = "wl_agnostic"


def shorten_model_name(name: str) -> str:
    # shorten wandb name
    name = name.replace('include_complex_workloads', 'complex')
    name = name.replace('include_pg_opt_workloads', 'pgopt')
    name = name.replace('include_hint_workloads', 'hint')
    name = name.replace('include_hint_idx_workloads', 'hintidx')
    name = name.replace('include_scan_only_hint_workloads', 'scanhint')
    name = name.replace('train_on_subplans', 'subplans')
    name = name.replace('PostgresEstWithIsIndexCond', 'FeatPGEstIsIdxC')
    name = name.replace('limit_plans_per_file', 'limitpf')
    name = name.replace('PostgresEstWithIsJoinCond', 'isjoincond')
    name = name.replace('filter_inline_col_featsFalse', 'nofiltercolinline')
    name = name.replace('AdamWScheduleFree', 'AdamWSF')
    name = name.replace('annotate_base_table_card_to_scan_op', 'tablecardatscan')
    name = name.replace('online_hard_example_mining', 'ohem')
    name = name.replace('ohem_initialize_difficulty', 'ohem_init')
    name = name.replace('tablecardatscan', 'tcardscan')
    name = name.replace('PostgresEstWithAdditionalFeats', 'addfeats')
    name = name.replace('include_complex_many_joins_workloads', 'manyjoins')
    return name


@attrs.define(frozen=True, slots=False)
class ModelConfig:
    name: Name = field(default=None)
    type: ModelType = field(default=None)
    device: str = field(default=None)
    optimizer_class_name: str = "AdamW"
    optimizer_kwargs: dict = dict(lr=1e-3)
    batch_size: int = 64
    seed: int = field(default=0)
    epochs: int = field(default=200)
    early_stopping_patience: int = 20
    num_workers: int = field(default=20)
    max_epoch_tuples: int = field(default=None)
    cap_training_samples: int = field(default=None)
    loss_class_name: str = "QLoss"
    loss_class_kwargs: dict = field(default=dict())
    limit_plans_per_file: int = field(default=10000)
    limit_plans_overall: int = field(default=None)
    include_timeout_queries: bool = False
    input_format: InputFormats = field(default=None)
    hyperparameter: Path = None
    column_statistics: bool = False
    word_embeddings: bool = False
    stratified_workloads: bool = False
    include_pg_opt_workloads: bool = False
    include_hint_workloads: bool = False
    include_hint_idx_workloads: bool = False
    include_gt30_hints: bool = False
    include_scan_only_hint_workloads: bool = False
    include_complex_workloads: bool = False
    include_complex_ranking_workloads: bool = False
    include_complex_many_joins_workloads: bool = False
    wandb_group: str = field(default=None)
    train_on_subplans: bool = False
    merge_nlj_and_inlj_to_same_op: bool = False  # merge NLJ and INLJ to the same operator / do not differentiate between them
    remove_join_cond_from_plan: bool = False  # remove join conditions from the plan / i.e. train without join conditions

    # model should be pretrained
    pretrained_model_dir: str = None
    pretrained_model_name: str = None

    # OHEM - online hard example mining
    online_hard_example_mining: bool = False
    ohem_initialize_difficulty: bool = False  # initialize difficulty for OHEM with model before first epoch - otherwise all inf
    ohem_resort_after_num_epochs: int = 2  # resort OHEM dataset after this number of epochs
    ohem_slice_by: str = 'file'  # file, dataset, none

    enforce_aggregation_node_at_root: bool = False  # enforce that the root node is an aggregation node - will be done at the beginning of collator
    prohibit_aggregation_node_at_root: bool = False  # prohibit that the root node is an aggregation node - will be done at the beginning of collator

    apply_outcol_pruning: bool = False  # perform projection pushdown on the plans - sometimes postgres is probably doing this but not annotating the query plans. i.e. all table accesses which are a left child of a join are annotated to output all columns of that table. This is unrealistic to assume, and confuses the model (is annotated, but has not impact on performance). This also increases the overall number of columns in the graph unreasonably.

    def get_model_dir(self, source_path: Paths, database: Dataset) -> Path:
        # shorten model name - otherwise we are hitting file-name limits
        model_name = shorten_model_name(self.name.NAME)
        return source_path.Data.models / model_name / database.db_name

    def get_eval_dir(self, source_path: Paths, database: Dataset) -> Path:
        # shorten model name - otherwise we are hitting file-name limits
        model_name = shorten_model_name(self.name.NAME)
        return source_path.Data.evaluation / model_name / database.db_name

    def get_statistics(self, source_path: Paths, dataset: Dataset, format: InputFormats) -> Path:
        if format == InputFormats.baseline_parsed:
            format = InputFormats.parsed
        if self.type == ModelType.WL_DRIVEN:
            return source_path.Data.root / 'feature_statistics' / f'{format}_feature_statistics_{dataset.db_name}.json'
        elif self.type == ModelType.WL_AGNOSTIC:
            return source_path.Data.root / 'feature_statistics' / f'{format}_feature_statistics.json'

    def get_column_stats(self, dataset: Dataset) -> Optional[Path]:
        if self.column_statistics:
            return ClusterPaths().Data.root / "dataset_statistics" / dataset.db_name / f"column_statistics_scaled{dataset.scale}.json"
        else:
            return None

    def get_word_embeddings(self, source_path: Paths, database: Dataset) -> Optional[Path]:
        if self.word_embeddings:
            return source_path.Data.sentences / f"{database.db_name}_scaled{database.scale}"/'word2vec.m'
        else:
            return None

    def get_training_workloads(self, training_datasets: List[Dataset], target_dataset: Dataset,
                               stratified_workload: bool = False,
                               romans_runs: bool = False) -> List[str]:

        # select the workload to be used
        workload_names = []

        if self.include_pg_opt_workloads:
            workload_names.append('workload_200k_s1')

        if self.include_hint_workloads:
            workload_names.append('workload_200k_s1_full_hints_lt30')

        if self.include_hint_idx_workloads:
            workload_names.append('workload_200k_s1_full_hints_idx')

        if self.include_gt30_hints:
            workload_names.append('workload_200k_s1_full_hints_gt30')

        if self.include_complex_workloads:
            workload_names.append('complex_workload_200k_s1')

        if self.include_scan_only_hint_workloads:
            workload_names.append('scan_only_s1_access_path_hints')

        if self.include_complex_ranking_workloads:
            workload_names.append('full_qo_complex_workload_200k_s1')

        if self.include_complex_many_joins_workloads:
            workload_names.append('complex_workload_many_joins_hints')

        if self.type == ModelType.WL_AGNOSTIC:
            training_workloads = []
            for db in training_datasets:
                if db.db_name != target_dataset.db_name:
                    for wl in workload_names:

                        if wl == 'complex_workload_many_joins_hints' and db.db_name not in ['baseball', 'credit',
                                                                                            'financial', 'geneea',
                                                                                            'imdb', 'tournament']:
                            continue

                        path = self.get_data_base_dir(
                            stratified_workloads=stratified_workload,
                            romans_runs=romans_runs) / f"{db.db_name}_scaled{db.scale}" / f"{wl}.json.gz"
                        training_workloads.append(path.as_posix())

            # assert len(training_workloads) == 19, f"Found {len(training_workloads)}, but it should be 19"
            assert len(training_workloads) > 0, f'No training workloads found for {self.name.NAME}'
            return training_workloads

        elif self.type == ModelType.WL_DRIVEN:
            training_workloads = []
            for wl in workload_names:
                path = self.get_data_base_dir(
                    stratified_workloads=stratified_workload,
                    romans_runs=romans_runs) / f"{target_dataset.db_name}_scaled{target_dataset.scale}" / f"{wl}.json.gz"
                training_workloads.append(path.as_posix())
            assert len(training_workloads) > 0, f'No training workloads found for {self.name.NAME}'
            return training_workloads

    def assemble_wl_path(self, target_db: Dataset, wl_name, stratified_workloads: bool = True,
                         romans_runs: bool = False) -> str:
        return f'{self.get_data_base_dir(stratified_workloads=stratified_workloads, romans_runs=romans_runs).as_posix()}/{target_db.db_name}_scaled{target_db.scale}/{wl_name}.json.gz'

    def get_test_workloads(self, target_db: Dataset, romans_runs: bool) -> str:
        test_paths = []

        # always use this workload - this is forward backward compatibility / comparing with old runs
        test_paths.append(
            self.assemble_wl_path(target_db, "workload_200k_s1", stratified_workloads=True, romans_runs=romans_runs))

        # always test on scan-only: check understanding of idx/seq scan costs
        test_paths.append(self.assemble_wl_path(target_db, "scan_only_s1_access_path_hints", stratified_workloads=True,
                                                romans_runs=romans_runs))

        # also test on complex ranking
        # test_paths.append(self.assemble_wl_path(target_db, "full_qo_complex_workload_200k_s1", stratified_workloads=True, romans_runs=romans_runs))

        return ' '.join(test_paths)

    def get_data_base_dir(self, stratified_workloads: bool, romans_runs: bool = False) -> Path:
        if stratified_workloads and not romans_runs:  # we do not have roman's runs stratified
            if self.input_format == InputFormats.baseline_parsed:
                return ClusterPaths().Data.augmented_plans
            elif self.input_format == InputFormats.parsed:
                return ClusterPaths().Data.parsed_plans_stratified2
            elif self.input_format == InputFormats.json:
                return ClusterPaths().Data.json_plans_stratified
        else:
            if self.input_format == InputFormats.baseline_parsed:
                return ClusterPaths().Data.augmented_plans
            elif self.input_format == InputFormats.parsed:
                if not romans_runs:
                    return ClusterPaths().Data.parsed_plans2
                else:
                    return ClusterPaths().Data.root / "runs_roman" / "parsed_plans"
            elif self.input_format == InputFormats.json:
                return ClusterPaths().Data.json_plans


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


@attrs.define(frozen=True, slots=False)
class QPPNetModelConfig(ModelConfig):
    name: Name = ModelName.QPP_NET
    type: ModelType = ModelType.WL_DRIVEN
    optimizer_kwargs: dict = dict(lr=1e-3)
    featurization: Featurization = QPPNetFeaturization
    loss_class_name: str = "QPPLoss"
    loss_class_kwargs: dict = dict()
    hidden_dim_plan: int = 128
    batch_to_func: Callable = simple_batch_to
    device: str = "cpu"
    input_format: str = InputFormats.json
    column_statistics: bool = True


@attrs.define(frozen=True, slots=False)
class QPPModelNoCostsConfig(QPPNetModelConfig):
    name: Name = ModelName.QPP_NET_NO_COSTS
    featurization: Featurization = QPPNetNoCostsFeaturization


@attrs.define(frozen=True, slots=False)
class QPPModelActCardsConfig(QPPNetModelConfig):
    name: Name = ModelName.QPP_NET_ACT_CARDS
    featurization: Featurization = QPPNetActCardsFeaturization


@attrs.define(frozen=True, slots=False)
class E2EModelConfig(ModelConfig):
    name: Name = ModelName.E2E
    type: ModelType = ModelType.WL_DRIVEN
    hidden_dim_plan: int = 256
    hidden_dim_pred: int = 256
    batch_to_func: Callable = plan_batch_to
    collator_func: Callable = baseline_plan_collator
    input_format: str = InputFormats.parsed
    column_statistics: bool = True
    word_embeddings: bool = True
    device: str = "cuda:[socket]"


@attrs.define(frozen=True, slots=False)
class TPoolModelConfig(E2EModelConfig):
    model_name: Name = ModelName.E2E


@attrs.define(frozen=True, slots=False)
class TlstmModelConfig(E2EModelConfig):
    model_name: Name = ModelName.E2E


@attrs.define(frozen=True, slots=False)
class TestSumModelConfig(E2EModelConfig):
    model_name: Name = Name("testsum", "TestSum")


@attrs.define(frozen=True, slots=False)
class ZeroShotModelConfig(ModelConfig):
    name: Name = ModelName.ZEROSHOT
    type: ModelType = ModelType.WL_AGNOSTIC
    featurization: Featurization = PostgresEstSystemCardDetail
    loss_class_name: str = "QLoss"
    loss_class_kwargs: dict = dict()

    fc_out_kwargs: dict = dict(activation_class_name='LeakyReLU',
                               activation_class_kwargs={},
                               norm_class_name='Identity',
                               norm_class_kwargs={},
                               residual=False,
                               dropout=False,
                               p_dropout=0.0,
                               activation=True,
                               inplace=True)
    final_mlp_kwargs: dict = dict(width_factor=1.5,
                                  n_layers=4, )
    tree_layer_kwargs: dict = dict(width_factor=0.6,
                                   n_layers=2)
    node_type_kwargs: dict = dict(width_factor=1.5,
                                  n_layers=4,
                                  one_hot_embeddings=True,
                                  max_emb_dim=32,
                                  drop_whole_embeddings=False)
    final_mlp_kwargs.update(**fc_out_kwargs)
    tree_layer_kwargs.update(**fc_out_kwargs)
    node_type_kwargs.update(**fc_out_kwargs)

    hidden_dim: int = 128
    output_dim: int = 1
    batch_size: int = 256
    tree_layer_name: str = 'MscnConv'  # GATConv MscnConv
    batch_to_func: Callable = batch_to
    collator_func: Callable = postgres_plan_collator
    input_format: InputFormats = InputFormats.parsed
    device: str = "cuda:[socket]"  # Device selection is handled by octopus
    skip_top_aggregation: bool = False  # removes any aggregation node from the top of the tree
    filter_inline_col_feats: bool = True  # by default zs is trained with inline column features
    annotate_base_table_card_to_scan_op: bool = False  # annotate the base table card as children_card at scan ops (instead of setting it to 1 by default)
    subplan_monotonicity_loss: bool = False  # add a loss term that enforces the subplan runtime to be monotonically increasing
    subplan_monotonicity_loss_fn: str = 'lt_loss'  # loss function for the subplan monotonicity loss - either lt_loss or relative_lt_loss


@attrs.define(frozen=True, slots=False)
class ZeroShotLowMin(ZeroShotModelConfig):
    name: Name = ModelName.ZS_LOW_MIN
    loss_class_kwargs: dict = {
        'min_val': 1e-6
    }

@attrs.define(frozen=True, slots=False)
class ZeroShotXLConfig(ZeroShotModelConfig):
    name: Name = ModelName.ZS_XL
    final_mlp_kwargs: dict = dict(width_factor=2,
                                  n_layers=6, **ZeroShotModelConfig().fc_out_kwargs)
    tree_layer_kwargs: dict = dict(width_factor=1.2,
                                   n_layers=4, **ZeroShotModelConfig().fc_out_kwargs)
    hidden_dim: int = 256
    include_scan_only_hint_workloads: bool = True
    include_hint_idx_workloads: bool = True
    include_complex_workloads: bool = True
    loss_class_name: str = 'QLogSqLoss'


@attrs.define(frozen=True, slots=False)
class ZeroShotXLLayerNormConfig(ZeroShotXLConfig):
    name: Name = ModelName.ZS_XL_LAYER_NORM
    tree_layer_kwargs: dict = dict(width_factor=1.2,
                                   n_layers=4, layer_norm=True, **ZeroShotModelConfig().fc_out_kwargs)


@attrs.define(frozen=True, slots=False)
class ZeroShotLayerNormConfig(ZeroShotModelConfig):
    name: Name = ModelName.ZS_LAYER_NORM
    tree_layer_kwargs: dict = dict(width_factor=0.6,
                                   n_layers=2, layer_norm=True, **ZeroShotModelConfig().fc_out_kwargs)

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
    loss_class_name: str = 'RankLoss'


@attrs.define(frozen=True, slots=False)
class ZSLearnFromTimeoutConfig(ZeroShotModelConfig):
    name: Name = ModelName.ZS_FROM_TIMEOUT
    include_timeout_queries: bool = True
    loss_class_kwargs: dict = {'learn_from_timeout': True, 'timeout_threshold': 30}


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
class MSCNModelConfig(ModelConfig):
    name: Name = ModelName.MSCN
    type: ModelType = ModelType.WL_DRIVEN
    mscn_enc_layers: int = 1
    mscn_est_layers: int = 1
    batch_to_func: Callable = batch_to
    collator_func: Callable = mscn_plan_collator
    input_format: InputFormats = InputFormats.baseline_parsed
    column_statistics: bool = True
    word_embeddings: bool = True
    hidden_dim_plan: int = 256
    device: str = "cuda:[socket]"

@attrs.define(frozen=True, slots=False)
class DACEModelConfig(ModelConfig):
    name: Name = ModelName.DACE
    type: ModelType = ModelType.WL_AGNOSTIC
    node_length: int = 10 + 3  # This needs to be equals to the number of operator types plus the remaining node feats.
    hidden_dim: int = 128
    output_dim: int = 1
    mlp_activation: str = "ReLU"
    transformer_activation: str = "gelu"
    mlp_dropout: float = 0.3
    transformer_dropout: float = 0.2
    max_runtime: int = 30000
    pad_length: int = 30
    loss_weight: float = 0.5
    batch_to_func: Callable = dace_batch_to
    loss_class_name: str = "DaceLoss"
    loss_class_kwargs: dict = dict()
    featurization: Featurization = DACEFeaturization
    input_format: InputFormats = InputFormats.parsed
    optimizer_class_name: str = "Adam"
    optimizer_kwargs: dict = dict(lr=1e-3)
    device: str = "cuda:[socket]"


@attrs.define(frozen=True, slots=False)
class DACEStratifiedModelConfig(DACEModelConfig):
    name: Name = ModelName.DACE_STRATIFIED
    stratified_workloads: bool = True


@attrs.define(frozen=True, slots=False)
class DACEModelNoCostsConfig(DACEModelConfig):
    name: Name = ModelName.DACE_NO_COSTS
    node_length: int = 21  # This needs to be equals to the number of operator types plus the remaining node feats.
    featurization: Featurization = DACEFeaturizationNoCosts


@attrs.define(frozen=True, slots=False)
class DACEModelActCardConfig(DACEModelConfig):
    name: Name = ModelName.DACE_ACT_CARDS
    node_length: int = 22
    featurization: Featurization = DACEActCardFeaturization


@attrs.define(frozen=True, slots=False)
class DACEModelWHintsConfig(DACEModelConfig):
    name: Name = ModelName.DACE_WITH_HINTS
    include_hint_workloads: bool = True


@attrs.define(frozen=True, slots=False)
class DACEModelOnlyHints(DACEModelConfig):
    name: Name = ModelName.DACE_ONLY_HINTS
    only_hint_workloads: bool = True


@attrs.define(frozen=True, slots=False)
class QueryFormerModelConfig(ModelConfig):
    name: Name = ModelName.QUERY_FORMER
    type: ModelType = ModelType.WL_DRIVEN
    hidden_dim_plan: int = 256
    input_format: InputFormats = InputFormats.parsed
    column_statistics: bool = True
    word_embeddings: bool = True
    embedding_size: int = 64
    ffn_dim: int = 32
    head_size: int = 8
    dropout: float = 0.1
    attention_dropout_rate: float = 0.1
    n_layers: int = 8
    use_sample: bool = True
    use_histogram: bool = True
    histogram_bin_number: int = 10
    hidden_dim_prediction: int = 128
    max_num_filters: int = 6
    batch_to_func: Callable = query_former_batch_to
    loss_class_name: str = "QLoss"  # Originally MSELoss
    optimizer_class_name: str = "AdamW"  # Originally Adam
    device: str = "cuda:[socket]"


@attrs.define(frozen=True, slots=False)
class T3ModelConfig(TabularModelConfig):
    name: Name = ModelName.T3
    type: ModelType = ModelType.WL_AGNOSTIC
    device: str = "cuda:[socket]"
    input_format: InputFormats = InputFormats.parsed
    card_type: CardType = CardType.pg


@attrs.define(frozen=True, slots=False)
class T3TreeModelConfig(T3ModelConfig):
    name: Name = ModelName.T3Tree


@attrs.define(frozen=True, slots=False)
class T3FlatModelConfig(T3ModelConfig):
    name: Name = ModelName.T3Flat

@attrs.define(frozen=True, slots=False)
class NeoModelConfig(ModelConfig):
    name: Name = ModelName.NEO
    batch_to_func: Callable = dace_batch_to
    device: str = "cuda:[socket]"
    type: ModelType = ModelType.WL_DRIVEN
    mscn_enc_layers: int = 1
    mscn_est_layers: int = 1
    batch_to_func: Callable = batch_to
    collator_func: Callable = mscn_plan_collator
    input_format: InputFormats = InputFormats.parsed
    column_statistics: bool = False
    word_embeddings: bool = False
    batch_to_func: callable = neo_batch_to


@attrs.define(frozen=True, slots=False)
class DataLoaderOptions:
    val_ratio: float = 0.20
    shuffle: bool = False
    pin_memory: bool = False
    dim_bitmaps: int = 1000  # ToDo move to another class?
    stratify_dataset: bool = False
    stratify_per_wl_file: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = True


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

NO_COSTS_MODEL_CONFIGS = MODEL_CONFIGS + [QPPModelNoCostsConfig(), DACEModelNoCostsConfig()]

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
    DACEModelActCardConfig()
]
