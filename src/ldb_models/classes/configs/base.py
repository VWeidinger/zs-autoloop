from pathlib import Path
from typing import List, Optional

import attrs as attrs
from attr import field
from ldb_schema.datasets import Dataset
from ldb_schema.paths import ClusterPaths, Paths

from ldb_models.classes.configs.constants import InputFormats, ModelType, Name

# parsed_plans3 + JIT information (ldb_wlexecutor branch jit_extraction) - not (yet) part of ldb_schema's DataPaths
PARSED_PLANS50_DIRNAME = "parsed_plans50"


def shorten_model_name(name: str) -> str:
    # shorten wandb name
    name = name.replace("include_complex_workloads", "complex")
    name = name.replace("include_pg_opt_workloads", "pg_opt")
    name = name.replace("include_hint_workloads", "hint")
    name = name.replace("include_hint_idx_workloads", "hi")
    name = name.replace("include_complex_ranking_workloads", "cr")
    name = name.replace("include_scan_only_hint_workloads", "scanhint")
    name = name.replace("include_complex_many_joins_workloads", "mj")
    name = name.replace("include_noisy_workloads", "no")

    name = name.replace("train_on_subplans", "subplans")
    name = name.replace("PostgresEstWithIsIndexCond", "FeatPGEstIsIdxC")
    name = name.replace("PostgresEstWithNumLoops", "EstLoops")
    name = name.replace("PostgresEstSystemCardDetail", "NoLoops")

    name = name.replace("limit_plans_per_file", "limitpf")
    name = name.replace("PostgresEstWithIsJoinCond", "isjoincond")
    name = name.replace("filter_inline_col_featsFalse", "nofiltercolinline")
    name = name.replace("AdamWScheduleFree", "AdamWSF")
    name = name.replace("annotate_base_table_card_to_scan_op", "tablecardatscan")
    name = name.replace("online_hard_example_mining", "ohem")
    name = name.replace("ohem_initialize_difficulty", "ohem_init")
    name = name.replace("tablecardatscan", "tcardscan")
    name = name.replace("PostgresEstWithAdditionalFeats", "addfeats")
    name = name.replace("cardinality_estimation", "CAE")
    name = name.replace("log_transform_card_labels", "LCL")
    name = name.replace("dont_log_if_log_transformed", "DLT")
    name = name.replace("use_runtime_feature", "RF")

    name = name.replace("group_similar", "gs")
    name = name.replace("random_percentage", "p")

    name = name.replace("enable_top_down", "ETD")
    name = name.replace("enable_bottom_up", "EBU")
    name = name.replace("update_est_loops", "UEL")

    name = name.replace("sumo_loop_multiply_est_error", "MEST")
    name = name.replace("sumo_loop_loss_on_scaled", "LOS")

    name = name.replace("join_repr_mode", "JOIN")
    name = name.replace("use_per_child_join_aggregation", "PCJOIN")
    name = name.replace("reproducible_message_passing", "RMP")
    name = name.replace("stable_literal_hash", "SLH")
    name = name.replace("use_jit_features", "JIT")
    name = name.replace("parsed_plans_version", "PPV")

    name = name.replace("use_pretrained_embeddings", "PTE")
    name = name.replace("latent_strategy", "LS")

    name = name.replace("use_relevancy_generator", "REL")
    name = name.replace("mask_single_plan_queries", "MSPQ")
    name = name.replace("rl_scalar", "s")
    name = name.replace("rl_log_degree", "ld")

    name = name.replace("subplan_estimation", "SP")
    name = name.replace("subplans_reg_loss", "SPR")

    name = name.replace("batch_group_size", "bgs")
    name = name.replace("group_size", "gs")

    name = name.replace("stable_layernorm", "sLN")
    name = name.replace("stable_dropout", "sDO")
    name = name.replace("stable_residual", "sRE")

    # loss weights
    name = name.replace("regression_weight", "regrW")
    name = name.replace("ranking_weight", "rankW")
    name = name.replace("subplan_weight", "subpW")
    name = name.replace("cardinality_weight", "cardW")

    # lambdaloss options
    name = name.replace("weighing_scheme", "WS")

    # scheduler names
    name = name.replace("CosineAnnealingLR", "CosA")
    name = name.replace("CosineAnnealingWarmRestarts", "CosAWR")
    name = name.replace("NoScheduler", "NoSch")

    # loss names
    name = name.replace("loss_class_name", "")
    name = name.replace("regression_loss_name", "")
    name = name.replace("ranking_loss_name", "")

    name = name.replace("layer_norm", "layern")
    name = name.replace("add_child_type_info_to_messages", "childtypeinfo")
    name = name.replace("filter_condense_operators", "filtercondensed")
    name = name.replace("scanhint_stratified", "scanstrat")

    name = name.replace("apply_cardinality_estimation_correction_logic", "cardcor")
    name = name.replace("cardinality_estimation_correction_logic_params", "ccmode")
    return name


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
    drop_last: bool = False


@attrs.define(frozen=True, slots=False)
class ModelConfig:
    name: Name = field(default=None)
    type: ModelType = field(default=None)
    device: str = field(default=None)
    optimizer_class_name: str = "AdamW"
    optimizer_kwargs: dict = dict(lr=1e-3)
    batch_size: int = 64
    seed: int = field(default=0)
    version: int = field(default=1)
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
    hyperparameter: Path | None = None
    column_statistics: bool = False
    word_embeddings: bool = False
    stratified_workloads: bool = False
    include_pg_opt_workloads: bool = False
    include_noisy_workloads: bool = False
    include_hint_workloads: bool = False
    include_hint_idx_workloads: bool = False
    include_gt30_hints: bool = False
    include_scan_only_hint_workloads: bool = False
    include_scan_only_hint_workloads_stratified: bool = False
    include_complex_workloads: bool = False
    include_complex_ranking_workloads: bool = False
    include_complex_many_joins_workloads: bool = False
    shuffle_train_data: bool = field(default=False)
    shuffle_before_limiting: bool = field(default=False)
    wandb_group: str = field(default=None)
    merge_nlj_and_inlj_to_same_op: bool = False  # merge NLJ and INLJ to the same operator / do not differentiate between them
    remove_join_cond_from_plan: bool = False  # remove join conditions from the plan / i.e. train without join conditions

    cardinality_estimation: bool = False
    # whether to apply log1p transform to input cardinality features (est/act)
    # This is distinct from any sum-of-operators specific option.
    use_log_input_card: bool = False
    train_on_subplans: bool = False
    cardinality_weight: float = 0.0
    subplan_estimation: bool = False

    # scheduling
    scheduler_class_name: str = "NoScheduler"
    warmup_epochs: int = 0

    # whether to batch plans by query
    all_same_batch_sampler: bool = False  # all plans in a batch are from the same query (only for testing/evaluation, not training)
    group_similar: bool = (
        False  # group plans with similar runtimes in the same batch (only for training)
    )
    random_percentage: float = 0.25  # percentage of plans in a batch which are randomly sampled (only for training, only used if group_similar is True)
    group_by: str = "sql"  # sql/plan_runtime_ms

    # model should be pretrained
    pretrained_model_dir: str | None = None
    pretrained_model_name: str | None = None

    enforce_aggregation_node_at_root: bool = False  # enforce that the root node is an aggregation node - will be done at the beginning of collator
    prohibit_aggregation_node_at_root: bool = False  # prohibit that the root node is an aggregation node - will be done at the beginning of collator

    apply_outcol_pruning: bool = False  # perform projection pushdown on the plans - sometimes postgres is probably doing this but not annotating the query plans. i.e. all table accesses which are a left child of a join are annotated to output all columns of that table. This is unrealistic to assume, and confuses the model (is annotated, but has not impact on performance). This also increases the overall number of columns in the graph unreasonably.

    # apply cardinality estimation correction logic
    apply_cardinality_estimation_correction_logic: bool = False
    cardinality_estimation_correction_logic_params: str | None = None

    # parsed plans and feature statistics version mapping
    # Version 2: parsed_plans2, feature_stats _v20260107
    # Version 3: parsed_plans3, feature_stats _v20260309
    # Version 50: parsed_plans50 (= parsed_plans3 + JIT information, no stratified copy), feature_stats _v20261104
    parsed_plans_version: int = field(default=3)

    def get_feature_stats_suffix(self) -> str:
        """Map parsed_plans_version to the corresponding feature stats suffix."""
        version_suffix_map = {
            2: "_v20260107",
            3: "_v20260309",
            50: "_v20261104",
        }
        if self.parsed_plans_version not in version_suffix_map:
            raise ValueError(
                f"Unknown parsed_plans_version: {self.parsed_plans_version}. "
                f"Supported versions: {list(version_suffix_map.keys())}"
            )
        return version_suffix_map[self.parsed_plans_version]

    def get_model_dir(self, source_path: Paths, database: Dataset) -> Path:
        # shorten model name - otherwise we are hitting file-name limits
        model_name = shorten_model_name(self.name.NAME)
        return source_path.Data.models / model_name / database.db_name

    def get_eval_dir(self, source_path: Paths, database: Dataset) -> Path:
        # shorten model name - otherwise we are hitting file-name limits
        model_name = shorten_model_name(self.name.NAME)
        return source_path.Data.evaluation / model_name / database.db_name

    def get_statistics(
        self,
        source_path: Paths,
        dataset: Dataset,
        format: InputFormats,
        suffix: str = "",
    ) -> Path:
        # map format always to parsed for getting statistics, as the statistics are the same for both formats
        if format == InputFormats.baseline_parsed:
            format = InputFormats.parsed

        if format == InputFormats.parsed:
            # Use the appropriate parsed_plans path based on version
            if self.parsed_plans_version == 2:
                parsed_plans_path = source_path.Data.parsed_plans2
            elif self.parsed_plans_version == 3:
                parsed_plans_path = source_path.Data.parsed_plans3
            elif self.parsed_plans_version == 50:
                parsed_plans_path = source_path.Data.runs / PARSED_PLANS50_DIRNAME
            else:
                raise ValueError(
                    f"Unknown parsed_plans_version: {self.parsed_plans_version}. "
                    f"Supported versions: 2, 3, 50"
                )
            format_str = parsed_plans_path.name
        elif format == InputFormats.json:
            format_str = source_path.Data.json_plans.name
        else:
            raise Exception(f"Unknown input format: {format}")

        if self.type == ModelType.WL_DRIVEN:
            return (
                source_path.Data.root
                / "feature_statistics"
                / f"{format_str}_feature_statistics{suffix}_{dataset.db_name}.json"
            )
        elif self.type == ModelType.WL_AGNOSTIC:
            return (
                source_path.Data.root
                / "feature_statistics"
                / f"{format_str}_feature_statistics{suffix}.json"
            )
        else:
            raise ValueError(f"Unknown model type: {self.type}")

    def get_column_stats(self, dataset: Dataset) -> Optional[Path]:
        if self.column_statistics:
            return (
                ClusterPaths().Data.root
                / "dataset_statistics"
                / dataset.db_name
                / f"column_statistics_scaled{dataset.scale}.json"
            )
        else:
            return None

    def get_word_embeddings(
        self, source_path: Paths, database: Dataset
    ) -> Optional[Path]:
        if self.word_embeddings:
            return (
                source_path.Data.sentences
                / f"{database.db_name}_scaled{database.scale}"
                / "word2vec.m"
            )
        else:
            return None

    def get_training_workloads(
        self,
        training_datasets: List[Dataset],
        target_dataset: Dataset,
        stratified_workload: bool = False,
        romans_runs: bool = False,
    ) -> list[str]:
        # select the workload to be used
        workload_names = []

        if self.include_pg_opt_workloads:
            workload_names.append("workload_200k_s1")

        if self.include_noisy_workloads:
            workload_names.append("workload_200k_s1_noisy")

        if self.include_hint_workloads:
            workload_names.append("workload_200k_s1_full_hints_lt30")

        if self.include_hint_idx_workloads:
            workload_names.append("workload_200k_s1_full_hints_idx")

        if self.include_gt30_hints:
            workload_names.append("workload_200k_s1_full_hints_gt30")

        if self.include_complex_workloads:
            workload_names.append("complex_workload_200k_s1")

        if self.include_scan_only_hint_workloads:
            workload_names.append("scan_only_s1_access_path_hints")

        if self.include_scan_only_hint_workloads_stratified:
            workload_names.append("scan_only_s1_access_path_hints_stratified")

        if self.include_complex_ranking_workloads:
            workload_names.append("full_qo_complex_workload_200k_s1")

        if self.include_complex_many_joins_workloads:
            workload_names.append("complex_workload_many_joins_hints")

        print(f"Using training workloads: {workload_names}")

        if self.type == ModelType.WL_AGNOSTIC:
            training_workloads: list[str] = []
            for db in training_datasets:
                if db.db_name != target_dataset.db_name:
                    for wl in workload_names:
                        if (
                            wl == "complex_workload_many_joins_hints"
                            and db.db_name
                            not in [
                                "baseball",
                                "credit",
                                "financial",
                                "geneea",
                                "imdb",
                                "tournament",
                            ]
                        ):
                            continue

                        path = (
                            self.get_data_base_dir(
                                stratified_workloads=stratified_workload,
                                romans_runs=romans_runs,
                            )
                            / f"{db.db_name}_scaled{db.scale}"
                            / f"{wl}.json"
                        )
                        training_workloads.append(path.as_posix())

            # assert len(training_workloads) == 19, f"Found {len(training_workloads)}, but it should be 19"
            assert len(training_workloads) > 0, (
                f"No training workloads found for {self.name.NAME}"
            )
            return training_workloads

        elif self.type == ModelType.WL_DRIVEN:
            training_workloads = []
            for wl in workload_names:
                path = (
                    self.get_data_base_dir(
                        stratified_workloads=stratified_workload,
                        romans_runs=romans_runs,
                    )
                    / f"{target_dataset.db_name}_scaled{target_dataset.scale}"
                    / f"{wl}.json"
                )
                training_workloads.append(path.as_posix())
            assert len(training_workloads) > 0, (
                f"No training workloads found for {self.name.NAME}"
            )
            return training_workloads
        else:
            raise ValueError(f"Unknown model type: {self.type}")

    def assemble_wl_path(
        self,
        target_db: Dataset,
        wl_name,
        stratified_workloads: bool = True,
        romans_runs: bool = False,
    ) -> str:
        path = f"{self.get_data_base_dir(stratified_workloads=stratified_workloads, romans_runs=romans_runs).as_posix()}/{target_db.db_name}_scaled{target_db.scale}/{wl_name}.json"

        return path

    def get_test_workloads(self, target_db: Dataset, romans_runs: bool) -> str:
        test_paths = []

        # always use this workload - this is forward backward compatibility / comparing with old runs
        test_paths.append(
            self.assemble_wl_path(
                target_db,
                "workload_200k_s1",
                stratified_workloads=True,
                romans_runs=romans_runs,
            )
        )

        # # always test on scan-only: check understanding of idx/seq scan costs
        # test_paths.append(self.assemble_wl_path(target_db, "scan_only_s1_access_path_hints", stratified_workloads=True,
        #                                         romans_runs=romans_runs))

        # also test on complex ranking
        # test_paths.append(self.assemble_wl_path(target_db, "full_qo_complex_workload_200k_s1", stratified_workloads=True, romans_runs=romans_runs))

        return " ".join(test_paths)

    def get_data_base_dir(
        self, stratified_workloads: bool, romans_runs: bool = False
    ) -> Path:
        if (
            stratified_workloads and not romans_runs
        ):  # we do not have roman's runs stratified
            if self.input_format == InputFormats.baseline_parsed:
                return ClusterPaths().Data.augmented_plans
            elif self.input_format == InputFormats.parsed:
                if self.parsed_plans_version == 2:
                    return ClusterPaths().Data.parsed_plans_stratified2
                elif self.parsed_plans_version == 3:
                    return ClusterPaths().Data.parsed_plans_stratified3
                elif self.parsed_plans_version == 50:
                    # there is no stratified copy of parsed_plans50 - use the full workloads
                    return ClusterPaths().Data.runs / PARSED_PLANS50_DIRNAME
                else:
                    raise ValueError(
                        f"Unknown parsed_plans_version: {self.parsed_plans_version}. Supported versions: 2, 3, 50"
                    )
            elif self.input_format == InputFormats.json:
                return ClusterPaths().Data.json_plans_stratified
            else:
                raise ValueError(f"Unknown input format: {self.input_format}")
        else:
            if self.input_format == InputFormats.baseline_parsed:
                return ClusterPaths().Data.augmented_plans
            elif self.input_format == InputFormats.parsed:
                if not romans_runs:
                    if self.parsed_plans_version == 2:
                        return ClusterPaths().Data.parsed_plans2
                    elif self.parsed_plans_version == 3:
                        return ClusterPaths().Data.parsed_plans3
                    elif self.parsed_plans_version == 50:
                        return ClusterPaths().Data.runs / PARSED_PLANS50_DIRNAME
                    else:
                        raise ValueError(
                            f"Unknown parsed_plans_version: {self.parsed_plans_version}. Supported versions: 2, 3, 50"
                        )
                else:
                    return ClusterPaths().Data.root / "runs_roman" / "parsed_plans"
            elif self.input_format == InputFormats.json:
                return ClusterPaths().Data.json_plans
            else:
                raise ValueError(f"Unknown input format: {self.input_format}")
