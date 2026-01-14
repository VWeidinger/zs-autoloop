import concurrent.futures
import functools
import math
import time
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from classes import (
    DataLoaderOptions,
    InputFormats,
    MambaModelConfig,
    MambaRankingModelConfig,
    ModelConfig,
    SumOfOperatorsModelConfig,
    ZeroShotModelConfig,
    ZeroShotRankingModelConfig,
)
from classes.workload_runs import WorkloadRuns
from ldb_schema.load_schema import get_json_schema, load_json
from models.white_mamba.mamba_dataset import (
    augment_feature_stats_for_mamba,
)
from models.zeroshot.postgres_plan_batching import add_numerical_scalers
from models.zeroshot.query_batch_sampler import (
    AllPlansForQuerySampler,
    BatchByQuerySampler,
    SimilarSampler,
)
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.dataset.card_adjustment import apply_cardinality_estimation_logic
from training.dataset.plan_dataset import (
    OHEMPlanDataset,
    PlanDataset,
    PreBatchedDataset,
)
from training.dataset.pushdown_projections import apply_projection_pushdown
from training.dataset.statification import stratify_plans


# worker fn to read a file
def read_fn(
    path: str,
):
    if path is None or str(path) == "None":
        return None

    try:
        run = load_json(str(path))
    except JSONDecodeError:
        print(f"Error reading {path} - skip")
        return None

    return run


def recursively_rewrite_inlj_to_nlj(plan: Dict):
    if plan["plan_parameters"]["op_name"] == "Index Nested Loop":
        plan["plan_parameters"]["op_name"] = "Nested Loop"

    if "children" in plan:
        for child in plan["children"]:
            recursively_rewrite_inlj_to_nlj(child)


def recursively_remove_join_cond_from_plan(plan: Dict):
    if "join" in plan["plan_parameters"]:
        plan["plan_parameters"].pop("join")

    if "children" in plan:
        for child in plan["children"]:
            recursively_remove_join_cond_from_plan(child)


def read_workload_runs(
    workload_run_paths: List[Path],
    limit_plans_per_file: int,
    limit_plans_overall: int,
    input_format: InputFormats = InputFormats.parsed,
    include_timeout_queries: bool = False,
    include_hint_not_obeyed_queries: bool = False,
    include_zero_card_queries: bool = False,
    stratify_per_wl_file: bool = False,
    stratify_dataset: bool = False,
    augment_with_subplans: bool = False,
    return_per_file: bool = False,
    merge_nlj_and_inlj_to_same_op: bool = False,
    remove_join_cond_from_plan: bool = False,
    apply_outcol_pruning: bool = False,
    apply_card_correction_logic: bool = False,
    card_correction_logic_params: Optional[str] = None,
) -> Tuple[List[dict], dict]:
    # reads several workload runs
    plans = []
    database_statistics = dict()

    timeout_thresholds_set = set()

    stratify_file_blacklist = [
        "join_order",
        "physical_operator",
        "access_path_selection",
    ]
    limit_plans_blacklist = ["join_order", "physical_operator", "access_path_selection"]

    # load the workload runs in parallel
    t0 = time.perf_counter()
    parallelize = True

    if parallelize:
        num_workers = 10
        """
        print(
            f"Start loading workload runs in parallel ({num_workers} workers, {len(workload_run_paths)} files)",
            flush=True,
        )
        """
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            run_list = list(executor.map(read_fn, workload_run_paths))
    else:
        run_list = [read_fn(path) for path in tqdm(workload_run_paths)]

    # print(f"Loaded {len(run_list)} files in {time.perf_counter() - t0:.2f}s", flush=True)

    assert len(run_list) == len(workload_run_paths)
    for path_index, (path, run) in enumerate(zip(workload_run_paths, run_list)):
        database_statistics[path_index] = run["database_stats"]

        if input_format in [InputFormats.parsed, InputFormats.baseline_parsed]:
            assert "parsed_plans" in run, (
                f"parsed_plans not in {path}, did you maybe used JSON query plans?"
            )
            file_plans = run["parsed_plans"]
            if len(file_plans) == 0:
                print(f"No plans in {path}, skipping", flush=True)
                db_name = "unknown"
            else:
                db_name = file_plans[0]["dataset"]
        elif input_format == InputFormats.json:
            if "query_list" not in run:
                raise ValueError(
                    f"This model requires json format, are you sure {path} is in json format?"
                )
            file_plans = run["query_list"]

            if len(file_plans) == 0:
                print(f"No plans in {path}, skipping", flush=True)
                db_name = "unknown"
            else:
                parent_dir = Path(path).parent.name
                assert "_scaled" in parent_dir, (
                    f"parent dir {parent_dir} does not contain _scaled"
                )
                db_name = parent_dir.split("_scaled")[0]
        else:
            raise ValueError(f"Input format {input_format} not supported")

        # extract db_name from run
        database_statistics[path_index]["database_name"] = db_name

        # check if the file is in the blacklist
        if any([blacklist in str(path) for blacklist in limit_plans_blacklist]):
            file_in_limit_blacklist = True
        else:
            file_in_limit_blacklist = False

        if "join_order_job" in str(path):
            # overwrite flag, to include zero card queries
            include_zero_card_queries = True

        hint_not_obeyed_ctr = 0
        timeout_ctr = 0
        include_zero_card_queries_ctr = 0
        per_file_plans = []

        for p_id, plan in enumerate(file_plans):
            if input_format in [InputFormats.parsed, InputFormats.baseline_parsed]:
                if (
                    not include_timeout_queries and plan["timeout"]
                ):  # Skip timeout queries
                    timeout_ctr += 1
                    continue
                if (
                    not include_hint_not_obeyed_queries and plan["invalid_hint"]
                ):  # Skip queries with invalid hints
                    hint_not_obeyed_ctr += 1
                    continue
                if (
                    not include_zero_card_queries and plan["zero_card"]
                ):  # skip zero card queries
                    include_zero_card_queries_ctr += 1
                    continue
            elif input_format == InputFormats.json:
                if (
                    not include_timeout_queries and plan["timeout"]
                ):  # Skip timeout queries
                    timeout_ctr += 1
                    continue
                if (
                    not include_hint_not_obeyed_queries and plan["invalid_hint"]
                ):  # Skip queries with invalid hints
                    hint_not_obeyed_ctr += 1
                    continue
                #  TODO this annotation is not 100% correct in the json plans (plan parsed is fixing this)!
                # if not include_zero_card_queries and plan[
                #    "zero_card"]:  #
                #    include_zero_card_queries_ctr += 1
                #    continue
            else:
                raise ValueError(f"Input format {input_format} not supported")

            plan["run_file_id"] = path_index

            if merge_nlj_and_inlj_to_same_op:
                # replace Index Nested Loop with Nested Loop Join
                recursively_rewrite_inlj_to_nlj(plan)

            if remove_join_cond_from_plan:
                recursively_remove_join_cond_from_plan(plan)

            if apply_outcol_pruning:
                try:
                    plan = apply_projection_pushdown(
                        plan, database_statistics=database_statistics
                    )[0]
                except Exception as e:
                    print(f"full plan: {plan}", flush=True)
                    raise e

            if apply_card_correction_logic:
                # apply card correction logic for plans

                enable_bu = (
                    "bu" in card_correction_logic_params
                    if card_correction_logic_params
                    else True
                )
                enable_td = (
                    "td" in card_correction_logic_params
                    if card_correction_logic_params
                    else True
                )
                update_est_loops = (
                    "ul" in card_correction_logic_params
                    if card_correction_logic_params
                    else True
                )
                plan = apply_cardinality_estimation_logic(
                    plan,
                    enable_bu=enable_bu,
                    enable_td=enable_td,
                    update_est_loops=update_est_loops,
                )

            per_file_plans.append(plan)

            if (
                limit_plans_per_file is not None
                and len(per_file_plans) >= limit_plans_per_file
                and not file_in_limit_blacklist
            ):
                print(
                    f"Limiting plans per file to {limit_plans_per_file} for {path}",
                    flush=True,
                )
                break

        # extract the timeout thresholds used during query execution
        # per_file_timeout_thresholds = set([p['timeout_threshold'] for p in per_file_plans])  # in ms
        # if len(per_file_timeout_thresholds) != 1:
        #    print(f"Multiple timeout thresholds in {path}: {per_file_timeout_thresholds}")

        # timeout_thresholds_set.update(per_file_timeout_thresholds)

        if augment_with_subplans:
            # augment the plans with subplans
            print("Augmenting plans with subplans", flush=True)
            subplans = augment_subplans(per_file_plans)
            print(
                f"Extracted {len(subplans)} subplans for the {len(per_file_plans)} plans in {path}",
                flush=True,
            )

            # shuffle the subplans
            np.random.shuffle(subplans)

            # apply limit
            if (
                limit_plans_per_file is not None
                and len(per_file_plans) >= limit_plans_per_file
                and not file_in_limit_blacklist
            ):
                per_file_plans = per_file_plans[: int(limit_plans_per_file / 2)]
                subplans = subplans[: int(limit_plans_per_file / 2)]

                print(
                    f"Reducing number of plans to {len(per_file_plans)} full-plans and {len(subplans)} sub-plans for {path}",
                    flush=True,
                )
            else:
                subplans = subplans[: len(per_file_plans)]

            # add the subplans to the plans - make sure it is 50/50 split (full-plans vs. sub-plans)
            per_file_plans = per_file_plans + subplans

            # shuffle the plans
            np.random.shuffle(per_file_plans)

        if stratify_per_wl_file:
            print(f"Stratify per wl file: {stratify_per_wl_file}")

        if any([blacklist in str(path) for blacklist in stratify_file_blacklist]):
            # Do not stratify if the file is in the blacklist
            pass
        # elif stratify_per_wl_file:
        #    per_file_plans = stratify_plans(
        #        plans=per_file_plans, max_runtime_ms=per_file_timeout_thresholds[0], output_identifier_hint=str(path),
        #        print_distr_stats=True, shuffle_before=True
        #    )

        """
        print(
            f"Loaded {len(per_file_plans)}/{len(file_plans)} plans from {path} (include timeout: {include_timeout_queries} (excluded: {timeout_ctr}), "
            f" include invalid hint: {include_hint_not_obeyed_queries} (excluded: {hint_not_obeyed_ctr}), include zero card: {include_zero_card_queries} (excluded: {include_zero_card_queries_ctr}), file in limit blacklist: {file_in_limit_blacklist}, limit file: {limit_plans_per_file})",
            flush=True,
        )
        """

        if return_per_file:
            plans.append(per_file_plans)
        else:
            plans.extend(per_file_plans)

    if len(workload_run_paths) == 1 and any(
        [
            blacklist in str(workload_run_paths[0])
            for blacklist in stratify_file_blacklist
        ]
    ):
        # Do not stratify if only a single file is present and it is in the blacklist
        pass
    elif stratify_dataset:
        # ideally there is just a single timeout threshold
        if len(timeout_thresholds_set) > 1:
            print(
                f"Multiple timeout thresholds in dataset: {timeout_thresholds_set}. Continuing stratification with max threshold."
            )
        max_runtime = max(timeout_thresholds_set)

        assert not return_per_file, "Not yet implemented for return_per_file=True"
        plans = stratify_plans(
            plans=plans,
            max_runtime_ms=max_runtime,
            output_identifier_hint="overall",
            print_distr_stats=True,
            shuffle_before=True,
        )

    if limit_plans_overall is not None:
        assert not return_per_file, "Not yet implemented for return_per_file=True"
        # shuffle plans
        np.random.shuffle(plans)
        plans = plans[:limit_plans_overall]

    return plans, database_statistics


def check_plan_runtime_validity(parsed_plan: Dict, parent_runtime: float) -> bool:
    # make sure that act_time of this  node is always smaller than runtime of parent
    act_time = parsed_plan["plan_parameters"]["act_time"]
    operator = parsed_plan["plan_parameters"]["op_name"]
    children = parsed_plan["children"] if "children" in parsed_plan else []

    if operator == "Materialize":
        # skip
        assert len(children) == 1
        return check_plan_runtime_validity(children[0], parent_runtime)

    if act_time > parent_runtime:
        return False

    child_runtimes = []
    for child in children:
        if not check_plan_runtime_validity(child, act_time):
            return False

        if child["plan_parameters"]["op_name"] == "Materialize":
            child_runtimes.append(child["children"][0]["plan_parameters"]["act_time"])
        else:
            child_runtimes.append(child["plan_parameters"]["act_time"])

    if sum(child_runtimes) > act_time:
        return False

    return True


def augment_subplans(
    plans: List[dict], return_only_subplans: bool = True
) -> List[dict]:
    augmented_plans = []
    for plan in tqdm(plans, desc="Augmenting subplans"):
        if not return_only_subplans:
            augmented_plans.append(plan)

        # check if subplan runtimes are consistent - if not: do not augment
        if not check_plan_runtime_validity(
            plan, parent_runtime=plan["plan_runtime_ms"]
        ):
            continue

        root_info = {
            "timeout": plan["timeout"],
            "invalid_hint": plan["invalid_hint"],
            "zero_card": plan["zero_card"],
            "run_file_id": plan["run_file_id"],
        }

        # extract subplans
        augmented_plans.extend(recursively_augment_plan(plan, root_info))

    return augmented_plans


def recursively_augment_plan(
    plan: Dict, root_information: Dict, depth: int = 0
) -> List[Dict]:
    if "children" in plan and len(plan["children"]) > 0:
        children = plan["children"]
    else:
        return []

    # filter out children with runtime 0
    children = [child for child in children if child["plan_parameters"]["act_time"] > 0]

    # copy over all children ([:] is important to create a copy!)
    subplans = children[:]

    # iterate over all children and augment them
    for child in children:
        child["plan_runtime_ms"] = plan["plan_parameters"]["act_time"]

        child.update(root_information)

        subplans.extend(
            recursively_augment_plan(child, root_information, depth=depth + 1)
        )

    return subplans


def _inv_log1p(x):
    return np.exp(x) - 1


def split_plans_in_train_val(
    plans: List[dict],
    val_ratio: float,
    shuffle_before_split: bool,
    cap_training_samples: int,
) -> (List[dict], List[dict]):
    # default behavior
    no_plans = len(plans)

    plan_indexes = list(range(no_plans))
    if shuffle_before_split:
        np.random.shuffle(plan_indexes)

    train_ratio = 1 - val_ratio
    split_train = int(no_plans * train_ratio)
    train_indexes = plan_indexes[:split_train]
    # Limit number of training samples. To have comparable batch sizes, replicate remaining indexes.
    if cap_training_samples is not None:
        print(f"Limiting dataset to {cap_training_samples}")
        prev_train_length = len(train_indexes)
        train_indexes = train_indexes[:cap_training_samples]
        replicate_factor = max(prev_train_length // len(train_indexes), 1)
        train_indexes = train_indexes * replicate_factor

    if val_ratio > 0:
        val_indexes = plan_indexes[split_train:]
    else:
        val_indexes = None

    return train_indexes, val_indexes


@functools.lru_cache(maxsize=None)
def get_db_schema_info(db_name: str) -> Dict:
    return get_json_schema(db_name)


def create_datasets(
    workload_run_paths,
    model_config: ModelConfig,
    val_ratio=0.15,
    shuffle_before_split=True,
    is_test_dataset: bool = False,
    do_not_limit_file: bool = False,
    stratify_dataset: bool = False,
    stratify_per_wl_file: bool = False,
    inject_plans: List[Dict] = None,
    inject_database_statistics: Dict = None,
    include_zero_card_queries: bool = False,
    assemble_graphs: bool = False,
    plan_collate_fn=None,
) -> Tuple[Pipeline, PlanDataset, PlanDataset, PlanDataset, Dict]:
    # check if we need to apply OHEM
    apply_ohem = (
        hasattr(model_config, "online_hard_example_mining")
        and model_config.online_hard_example_mining
        and not is_test_dataset
    )  # do not apply OHEM on test datasets - we want to have a static dataset

    if inject_plans is None or inject_database_statistics is None:
        plans, database_statistics = read_workload_runs(
            workload_run_paths=workload_run_paths,
            limit_plans_per_file=model_config.limit_plans_per_file
            if not do_not_limit_file and not apply_ohem
            else None,
            limit_plans_overall=model_config.limit_plans_overall
            if not do_not_limit_file and not apply_ohem
            else None,
            input_format=model_config.input_format,
            include_timeout_queries=model_config.include_timeout_queries
            and not is_test_dataset,
            include_zero_card_queries=include_zero_card_queries,
            stratify_dataset=stratify_dataset,
            stratify_per_wl_file=stratify_per_wl_file,
            augment_with_subplans=model_config.train_on_subplans
            and not is_test_dataset,
            return_per_file=apply_ohem,
            merge_nlj_and_inlj_to_same_op=model_config.merge_nlj_and_inlj_to_same_op,
            remove_join_cond_from_plan=model_config.remove_join_cond_from_plan,
            apply_outcol_pruning=model_config.apply_outcol_pruning,
            apply_card_correction_logic=model_config.apply_cardinality_estimation_correction_logic,
            card_correction_logic_params=model_config.cardinality_estimation_correction_logic_params,
        )
    else:
        plans = inject_plans
        database_statistics = inject_database_statistics

    # extract database schema information
    db_names = dict(
        (run_file_id, value["database_name"])
        for run_file_id, value in database_statistics.items()
    )
    db_schema_info = dict(
        (run_file_id, get_json_schema(db_name))
        for run_file_id, db_name in db_names.items()
        if db_name != "unknown"
    )

    # derive label normalization
    if apply_ohem:
        # flatten list[list[plan]] to list[plan]
        runtime_plans = []
        for p_list in plans:
            runtime_plans.extend(p_list)
    else:
        runtime_plans = plans

    for p in runtime_plans:
        assert "timeout" in p, (
            f"plan {p} does not have timeout information / {workload_run_paths}"
        )

    runtimes = np.array(
        [p["plan_runtime_ms"] / 1000 for p in runtime_plans if not p["timeout"]]
    )
    label_norm = derive_label_normalizer(model_config.loss_class_name, runtimes)

    if apply_ohem:
        # special case handling for OHEM

        # do the train/val split for each file separately
        train_indexes_lists = []
        val_indexes_lists = []

        for plan_list in plans:
            train_indexes, val_indexes = split_plans_in_train_val(
                plans=plan_list,
                val_ratio=val_ratio,
                shuffle_before_split=shuffle_before_split,
                cap_training_samples=model_config.cap_training_samples,
            )
            train_indexes_lists.append(train_indexes)
            val_indexes_lists.append(val_indexes)

        # overwrite global args
        train_indexes = train_indexes_lists  # this is now a list of lists!
        val_indexes = val_indexes_lists  # this is now a list of lists!

    else:
        train_indexes, val_indexes = split_plans_in_train_val(
            plans=plans,
            val_ratio=val_ratio,
            shuffle_before_split=shuffle_before_split,
            cap_training_samples=model_config.cap_training_samples,
        )

    def create_dataset(
        plans,
        indexes,
        is_validation_set: bool = False,
        assemble_graphs: bool = False,
        assemble_graph_collate_fn=None,
    ):
        if apply_ohem:
            assert not assemble_graphs, (
                "with ohem we cannot make the graphs up front, because in each epoch we reselect the plans used for training"
            )
            assert model_config.cap_training_samples is None, (
                "ensuring that always this number of samples is included is not yet supported - this would require replication in case the limit is too high! - not yet implemented"
            )

            # gather plans using indexes
            gathered_plans = []
            for p_list, idx_list in zip(plans, indexes):
                gathered_plans.append([p_list[i] for i in idx_list])

            if is_validation_set:
                # since we limit the datasets after we do the split into train/val, we need to adjust the val_ratio
                # otherwise we would have a different ratio in the validation set
                ratio = val_ratio / (1 - val_ratio)
            else:
                ratio = 1

            # compute limit values
            l_plans_per_file, l_plans_overall = None, None
            if not do_not_limit_file and model_config.limit_plans_per_file is not None:
                l_plans_per_file = int(model_config.limit_plans_per_file * ratio)
            if not do_not_limit_file and model_config.limit_plans_overall is not None:
                l_plans_overall = int(model_config.limit_plans_overall * ratio)

            # use a different dataset
            dataset = OHEMPlanDataset(
                plan_lists=gathered_plans,
                plan_idxs_lists=indexes,
                workload_file_names=workload_run_paths,
                limit_plans_per_file=l_plans_per_file,
                limit_plans_overall=l_plans_overall,
                slice_by=model_config.ohem_slice_by,
                ohem_ratio=model_config.ohem_ratio,
            )
        else:
            if assemble_graphs:
                # apply plan selection
                indexed_plans = [
                    plans[i] for i in indexes
                ]  # not sure if this is really necessary - I think can also be ommitted

                # manually create batches of plans
                batch_size = model_config.batch_size
                batches = []
                num_batches = math.ceil(len(indexed_plans) / batch_size)

                parallelize = True
                if parallelize:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = [
                            executor.submit(
                                assemble_graph_collate_fn,
                                list(zip(indexes, indexed_plans[i : i + batch_size])),
                            )
                            for i in range(0, len(indexed_plans), batch_size)
                        ]
                        for future in tqdm(
                            concurrent.futures.as_completed(futures),
                            desc="Creating batches",
                            total=num_batches,
                        ):
                            batches.append(future.result())
                else:
                    for i in tqdm(
                        range(0, len(indexed_plans), batch_size),
                        desc="Creating batches",
                        total=num_batches,
                    ):
                        # create batch of plans
                        batch = indexed_plans[i : i + batch_size]
                        batch = assemble_graph_collate_fn(list(zip(indexes, batch)))

                        # add to batch list
                        batches.append(batch)

                # compute some checks - make sure all plans are included in a batch
                assert len(batches) == num_batches, (
                    f"our batch number computation is incorrect, {len(batches)} == {num_batches}"
                )
                assert len(batches) * batch_size >= len(indexed_plans), (
                    f"not all plans are included in a batch, {len(batches) * batch_size} >= {len(indexed_plans)}"
                )

                # create dataset
                dataset = PreBatchedDataset(batches=batches, batch_size=batch_size)
            else:
                # apply gather operation on the plans - necessary for replication / shuffling (see above)
                dataset = PlanDataset([plans[i] for i in indexes], indexes)

        return dataset

    if assemble_graphs:
        collate_fn = functools.partial(
            plan_collate_fn,
            db_statistics=database_statistics,
            db_schema_info=db_schema_info,
        )
    else:
        collate_fn = None

    train_dataset = create_dataset(
        plans,
        train_indexes,
        assemble_graphs=assemble_graphs,
        assemble_graph_collate_fn=collate_fn,
    )

    val_dataset = None
    if val_ratio > 0:
        val_dataset = create_dataset(
            plans,
            val_indexes,
            is_validation_set=True,
            assemble_graphs=assemble_graphs,
            assemble_graph_collate_fn=collate_fn,
        )

    return label_norm, train_dataset, val_dataset, database_statistics, db_schema_info


def derive_label_normalizer(loss_class_name, y) -> Pipeline:
    if loss_class_name == "MSELoss":
        log_transformer = preprocessing.FunctionTransformer(
            np.log1p, _inv_log1p, validate=True
        )
        scale_transformer = preprocessing.MinMaxScaler()
        pipeline = Pipeline([("log", log_transformer), ("scale", scale_transformer)])
        pipeline.fit(y.reshape(-1, 1))

    elif loss_class_name == "QLoss":
        scale_transformer = preprocessing.MinMaxScaler(feature_range=(1e-2, 1))
        pipeline = Pipeline([("scale", scale_transformer)])
        pipeline.fit(y.reshape(-1, 1))

    else:
        pipeline = None

    return pipeline


def create_zeroshot_dataloader(
    workload_runs: WorkloadRuns,
    statistics_file: Path,
    model_config: ZeroShotModelConfig,
    data_loader_options: DataLoaderOptions,
    quick_run: bool = False,
    assemble_graphs_up_front: bool = False,
) -> (
    Optional[Pipeline],
    dict,
    Optional[DataLoader],
    Optional[DataLoader],
    List[Optional[DataLoader]],
):
    # Postgres_plan_collator does the heavy lifting of creating the graphs and extracting the features and thus
    # requires both database statistics but also feature statistics
    label_norm = Optional[Pipeline]

    group_similar = (
        hasattr(model_config, "group_similar") and model_config.group_similar
    )
    batch_by_query = (
        hasattr(model_config, "batch_by_query") and model_config.batch_by_query
    )

    if batch_by_query:
        # This custom batching is not compatible with pre-batching or OHEM.
        assert not assemble_graphs_up_front, (
            "batch_by_query is not compatible with assemble_graphs_up_front=True"
        )
        assert (
            not hasattr(model_config, "online_hard_example_mining")
            or not model_config.online_hard_example_mining
        ), "batch_by_query is not compatible with online_hard_example_mining"

    feature_statistics = load_json(str(statistics_file), namespace=False)
    assert feature_statistics != {}, "Feature statistics file is empty!"

    plan_collator = model_config.collator_func

    add_pseudo_entries_to_feature_statistics = True
    if add_pseudo_entries_to_feature_statistics:
        # base model has never seen these operator types - add them to the feature statistics to avoid one-hot encoder errors
        if "MIN" not in feature_statistics["aggregation"]["value_dict"]:
            feature_statistics["aggregation"]["value_dict"]["MIN"] = 2
        if "MAX" not in feature_statistics["aggregation"]["value_dict"]:
            feature_statistics["aggregation"]["value_dict"]["MAX"] = 2
        if "IN" not in feature_statistics["operator"]["value_dict"]:
            feature_statistics["operator"]["value_dict"]["IN"] = 3
        if "LIKE" not in feature_statistics["operator"]["value_dict"]:
            feature_statistics["operator"]["value_dict"]["LIKE"] = 3
        if "NOT LIKE" not in feature_statistics["operator"]["value_dict"]:
            feature_statistics["operator"]["value_dict"]["NOT LIKE"] = 3
        if "IS NOT NULL" not in feature_statistics["operator"]["value_dict"]:
            feature_statistics["operator"]["value_dict"]["IS NOT NULL"] = 3
        if "IS NULL" not in feature_statistics["operator"]["value_dict"]:
            feature_statistics["operator"]["value_dict"]["IS NULL"] = 3
        # Ensure logical operators are present to avoid runtime assertion when parsing predicates
        if "OR" not in feature_statistics["operator"]["value_dict"]:
            feature_statistics["operator"]["value_dict"]["OR"] = 3
        if "AND" not in feature_statistics["operator"]["value_dict"]:
            feature_statistics["operator"]["value_dict"]["AND"] = 3

    dataloader_args = dict(
        batch_size=model_config.batch_size,
        shuffle=data_loader_options.shuffle,
        num_workers=model_config.num_workers,
        pin_memory=data_loader_options.pin_memory,
    )

    collator_args = {"scaler_type": model_config.scaler_type}
    if isinstance(
        model_config,
        (ZeroShotModelConfig, ZeroShotRankingModelConfig, SumOfOperatorsModelConfig),
    ):
        collator_args["skip_top_aggregation"] = model_config.skip_top_aggregation
        collator_args["filter_inline_col_feats"] = model_config.filter_inline_col_feats
        collator_args["annotate_base_table_card_to_scan_op"] = (
            model_config.annotate_base_table_card_to_scan_op
        )
        collator_args["enforce_aggregation_node_at_root"] = (
            model_config.enforce_aggregation_node_at_root
        )
        collator_args["prohibit_aggregation_node_at_root"] = (
            model_config.prohibit_aggregation_node_at_root
        )
        # whether to apply log1p to input cardinality features (est/act) before scaling
        collator_args["use_log_input_card"] = (
            hasattr(model_config, "use_log_input_card")
            and model_config.use_log_input_card
        )

    elif isinstance(model_config, (MambaModelConfig, MambaRankingModelConfig)):
        if model_config.filter_condense_operators:
            # overwrite part of the featurization
            scan_features = model_config.featurization.SCAN_FEATURES

            # remove flat_vector
            assert "filter_flat_vector" in scan_features, (
                f'cannot apply condense, "filter_flat_vector" not in {scan_features}'
            )

            scan_features.remove("filter_flat_vector")
            scan_features.append("filter_flat_vector_condensed")

            # overwrite the scan features
            model_config.featurization.SCAN_FEATURES = scan_features
        # augment feature statistics for Mamba
        # doing this outside collate to avoid parallelization issues
        # (e.g. if we have multiple workers, we would augment the feature statistics multiple times)
        augment_feature_stats_for_mamba(feature_statistics)
    add_numerical_scalers(feature_statistics, scaler_type=model_config.scaler_type)

    if workload_runs.train_workload_runs and len(workload_runs.train_workload_runs) > 0:
        # Split plans into train/test/validation set
        print("Creating dataloader for training and validation data")
        collator_args["subplan_label_op_time"] = getattr(
            model_config, "subplan_label_op_time", False
        )
        collator_args["log_transform_card_labels"] = (
            hasattr(model_config, "log_transform_card_labels")
            and model_config.log_transform_card_labels
        )

        base_collate_fn = functools.partial(
            plan_collator,
            feature_statistics=feature_statistics,
            plan_featurization=model_config.featurization,
            **collator_args,
            output_tensors=assemble_graphs_up_front,
        )

        label_norm, train_dataset, val_dataset, database_statistics, db_schema_info = (
            create_datasets(
                workload_run_paths=workload_runs.train_workload_runs,
                model_config=model_config,
                val_ratio=data_loader_options.val_ratio,
                assemble_graphs=assemble_graphs_up_front,
                plan_collate_fn=base_collate_fn,
            )
        )

        # Create the collate functions
        train_collate_fn = functools.partial(
            base_collate_fn,
            db_statistics=database_statistics,
            db_schema_info=db_schema_info,
        )
        valid_collate_fn = functools.partial(
            base_collate_fn,
            db_statistics=database_statistics,
            db_schema_info=db_schema_info,
        )

        # Configure and create the Training DataLoader
        train_loader_args = dataloader_args.copy()

        if assemble_graphs_up_front:
            # Pre-batched graphs. Each item in the dataset is a batch.
            train_loader_args["batch_size"] = None
            train_loader_args.pop(
                "collate_fn", None
            )  # collate_fn is not needed for pre-batched data
        elif group_similar:
            train_loader_args.pop("batch_size", None)
            train_loader_args.pop("shuffle", None)
            train_loader_args.pop("drop_last", None)
            key = (
                getattr(model_config, "group_by", "plan_runtime_ms")
                if hasattr(model_config, "group_by")
                else "plan_runtime_ms"
            )
            train_batch_sampler = SimilarSampler(
                plans=train_dataset.plans,
                batch_size=model_config.batch_size,
                key=key,
                random_percentage=getattr(model_config, "random_percentage", 0.1),
                group_size=getattr(model_config, "group_size", 16),
            )
            train_loader_args["batch_sampler"] = train_batch_sampler
            train_loader_args["collate_fn"] = train_collate_fn
        elif batch_by_query:
            # Custom batching with QueryBatchSampler.
            train_loader_args.pop("batch_size", None)
            train_loader_args.pop("shuffle", None)
            train_loader_args.pop("drop_last", None)

            train_batch_sampler = BatchByQuerySampler(
                plans=train_dataset.plans, max_batch_size=model_config.batch_size
            )
            train_loader_args["batch_sampler"] = train_batch_sampler
            train_loader_args["collate_fn"] = train_collate_fn
        else:
            # Standard batching.
            train_loader_args["collate_fn"] = train_collate_fn

        train_loader: DataLoader = DataLoader(train_dataset, **train_loader_args)

        # Configure and create the Validation DataLoader
        val_loader = None
        if val_dataset:
            valid_loader_args = dataloader_args.copy()
            valid_loader_args["shuffle"] = False
            valid_loader_args["drop_last"] = False

            if assemble_graphs_up_front:
                valid_loader_args["batch_size"] = None
                valid_loader_args.pop("collate_fn", None)
            else:
                valid_loader_args["collate_fn"] = valid_collate_fn

            val_loader = DataLoader(val_dataset, **valid_loader_args)

        # 3. Configure and create the Validation DataLoader
        val_loader = None
        if val_dataset:
            valid_loader_args = dataloader_args.copy()
            valid_loader_args["shuffle"] = False
            valid_loader_args["drop_last"] = False

            if assemble_graphs_up_front:
                valid_loader_args["batch_size"] = None
                valid_loader_args.pop("collate_fn", None)
            else:
                valid_loader_args["collate_fn"] = valid_collate_fn

            val_loader = DataLoader(val_dataset, **valid_loader_args)

        # Update OHEM dataset if necessary with the correct loader args
        if (
            hasattr(model_config, "online_hard_example_mining")
            and model_config.online_hard_example_mining
        ):
            train_dataset.dataloader_args = dataloader_args.copy()  # create copy, because following test dataset will reference and overwrite dataloader_args
            val_dataset.dataloader_args = dataloader_args.copy()

        try:
            print(
                f"Created datasets of size: "
                f"train {len(train_dataset)}, "
                f"validation: {len(val_dataset)}"
            )
        except Exception as e:
            print(e)
            print("Problem retrieving size of train or val dataset")
    else:
        train_loader, val_loader = None, None

    if workload_runs.test_workload_runs and len(workload_runs.test_workload_runs) > 0:
        # For each test workload run create a distinct test loader
        print("Creating dataloader for test data")
        test_loaders = []

        all_same_batch_sampler = (
            hasattr(model_config, "all_same_batch_sampler")
            and model_config.all_same_batch_sampler
        )
        test_loader_args = dataloader_args.copy()
        test_loader_args["shuffle"] = False
        for test_path in workload_runs.test_workload_runs:
            if "join_order" in str(test_path):
                include_zero_card = True
            else:
                include_zero_card = False

            test_collate_fn = functools.partial(
                plan_collator,
                feature_statistics=feature_statistics,
                plan_featurization=model_config.featurization,
                **collator_args,
                output_tensors=assemble_graphs_up_front,
                # in case we assemble the graphs up front, we directly want to convert to tensors
            )

            _, test_dataset, _, test_database_statistics, test_db_schema_info = (
                create_datasets(
                    [test_path],
                    model_config=model_config,
                    shuffle_before_split=False,
                    val_ratio=0.0,
                    is_test_dataset=True,
                    do_not_limit_file=True and not quick_run,
                    # in case of quick_run we still want to limit this file
                    # do not limit test files - e.g. for QO we need scores for all plans
                    include_zero_card_queries=include_zero_card,
                    assemble_graphs=assemble_graphs_up_front,
                    plan_collate_fn=test_collate_fn,
                )
            )

            test_collate_fn = functools.partial(
                test_collate_fn,
                db_statistics=test_database_statistics,
                db_schema_info=test_db_schema_info,
            )
            test_loader_args["collate_fn"] = test_collate_fn
            if all_same_batch_sampler:
                test_loader_args.pop("batch_size", None)
                test_loader_args.pop("shuffle", None)
                test_loader_args.pop("drop_last", None)

                test_batch_sampler = AllPlansForQuerySampler(
                    plans=test_dataset.plans, batch_size=model_config.batch_size
                )
                test_loader_args["batch_sampler"] = test_batch_sampler
            elif assemble_graphs_up_front:
                # overwrite batch size to 1 - since we have already pre-batched everything
                test_loader_args["batch_size"] = None

            # turn off shuffle for test set
            test_loader_args["shuffle"] = False
            test_loader_args["drop_last"] = False

            test_loader = DataLoader(test_dataset, **test_loader_args)
            test_loaders.append(test_loader)

            # update ohem dataset if necessary - this is done post creation of the dataset! (this is a bit hacky, information was at init time not available)
            if (
                hasattr(model_config, "online_hard_example_mining")
                and model_config.online_hard_example_mining
            ):
                test_dataset.dataloader_args = test_loader_args.copy()

            print(f"Created dataset of size: {len(test_dataset)} for {test_path}")
    else:
        test_loaders = None

    return label_norm, feature_statistics, train_loader, val_loader, test_loaders
