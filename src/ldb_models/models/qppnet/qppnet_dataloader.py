import functools
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
from classes.classes import DataLoaderOptions, ModelConfig, QPPNetModelConfig
from classes.workload_runs import WorkloadRuns
from ldb_schema.load_schema import get_json_schema, load_json
from models.white_mamba.extract_card_error_utils import extract_card_error_from_plan
from models.zeroshot.annotate_additional_plan_info import assign_additional_plan_info
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader
from training.dataset.dataset_creation import (
    derive_label_normalizer,
    read_workload_runs,
)
from training.dataset.plan_dataset import PlanDataset
from training.training.operator_tree import OperatorTree, operator_tree_from_json


def create_qppnet_dataloader(
    workload_runs: WorkloadRuns,
    statistics_file: Path,
    model_config: QPPNetModelConfig,
    column_statistics: Path,
    data_loader_options: DataLoaderOptions,
) -> (
    Optional[Pipeline],
    dict,
    Optional[DataLoader],
    Optional[DataLoader],
    List[Optional[DataLoader]],
):
    feature_statistics = load_json(str(statistics_file), namespace=False)
    assert feature_statistics != {}, "Feature statistics file is empty!"

    train_loader, val_loader, test_loaders = (
        Optional[DataLoader],
        Optional[DataLoader],
        List[Optional[DataLoader]],
    )

    plan_collator = qppnet_collator

    dataloader_args = dict(
        batch_size=model_config.batch_size,
        shuffle=data_loader_options.shuffle,
        num_workers=model_config.num_workers,
        pin_memory=data_loader_options.pin_memory,
    )

    column_statistics = load_json(str(column_statistics), namespace=False)

    if workload_runs.train_workload_runs:
        print("Create dataloader for training, validation and test data")

        label_norm, train_dataset, val_dataset, database_statistics, db_schema_info = (
            create_datasets(
                workload_run_paths=workload_runs.train_workload_runs,
                model_config=model_config,
                val_ratio=data_loader_options.val_ratio,
                stratify_dataset=data_loader_options.stratify_dataset,
                stratify_per_wl_file=data_loader_options.stratify_per_wl_file,
            )
        )

        test_dataset, val_dataset = val_dataset.split(0.5)
        print(
            f"Created datasets of size: "
            f"train {len(train_dataset)}, "
            f"validation: {len(val_dataset)}, "
            f"test: {len(test_dataset)}"
        )

        # remove timeout entries from test_dataset
        test_dataset = test_dataset.remove_timeouts()

        train_collate_fn = functools.partial(
            plan_collator,
            db_statistics=database_statistics,
            db_schema_info=db_schema_info,
            feature_statistics=feature_statistics,
            column_statistics=column_statistics,
            plan_featurization=model_config.featurization,
        )

        dataloader_args.update(collate_fn=train_collate_fn)
        train_loader = DataLoader(train_dataset, **dataloader_args)
        val_loader = DataLoader(val_dataset, **dataloader_args)
        test_loaders = [DataLoader(test_dataset, **dataloader_args)]

    if workload_runs.test_workload_runs:
        test_loaders = []
        for test_workload in workload_runs.test_workload_runs:
            print(f"Create dataloader for test data: {test_workload}")
            _, _, test_dataset, database_statistics, db_schema_info = create_datasets(
                workload_run_paths=[test_workload],
                model_config=model_config,
                val_ratio=1.0,
                is_test_dataset=True,
                stratify_dataset=data_loader_options.stratify_dataset,
                stratify_per_wl_file=data_loader_options.stratify_per_wl_file,
            )

            train_collate_fn = functools.partial(
                plan_collator,
                db_statistics=database_statistics,
                feature_statistics=feature_statistics,
                column_statistics=column_statistics,
                plan_featurization=model_config.featurization,
                db_schema_info=db_schema_info,
            )

            dataloader_args.update(collate_fn=train_collate_fn)
            test_loaders.append(DataLoader(test_dataset, **dataloader_args))

    return None, feature_statistics, train_loader, val_loader, test_loaders


def create_datasets(
    workload_run_paths,
    model_config: ModelConfig,
    val_ratio=0.15,
    shuffle_before_split=True,
    is_test_dataset: bool = False,
    stratify_dataset: bool = False,
    stratify_per_wl_file: bool = False,
) -> (Pipeline, PlanDataset, PlanDataset, PlanDataset, dict):
    plans, database_statistics = read_workload_runs(
        workload_run_paths=workload_run_paths,
        limit_plans_per_file=model_config.limit_plans_per_file,
        limit_plans_overall=model_config.limit_plans_overall,
        input_format=model_config.input_format,
        include_timeout_queries=model_config.include_timeout_queries
        and not is_test_dataset,
        stratify_dataset=stratify_dataset,
        stratify_per_wl_file=stratify_per_wl_file,
        merge_nlj_and_inlj_to_same_op=model_config.merge_nlj_and_inlj_to_same_op,
        remove_join_cond_from_plan=model_config.remove_join_cond_from_plan,
    )

    no_plans = len(plans)
    plan_indexes = list(range(no_plans))
    if shuffle_before_split:
        np.random.shuffle(plan_indexes)

    train_ratio = 1 - val_ratio
    split_train = int(no_plans * train_ratio)
    train_indexes = plan_indexes[:split_train]
    # Limit number of training samples. To have comparable batch sizes, replicate remaining indexes.
    if model_config.cap_training_samples is not None:
        print(f"Limiting dataset to {model_config.cap_training_samples}")
        prev_train_length = len(train_indexes)
        train_indexes = train_indexes[: model_config.cap_training_samples]
        replicate_factor = max(prev_train_length // len(train_indexes), 1)
        train_indexes = train_indexes * replicate_factor

    train_dataset = PlanDataset([plans[i] for i in train_indexes], train_indexes)

    val_dataset = None
    if val_ratio > 0:
        val_indexes = plan_indexes[split_train:]
        val_dataset = PlanDataset([plans[i] for i in val_indexes], val_indexes)

    # derive label normalization
    runtimes = np.array(
        [p["analyze_plans"][0]["Execution Time"] / 1000 for p in plans]
    )  # ToDo: Average over different plans, where else we do this?
    label_norm = derive_label_normalizer(model_config.loss_class_name, runtimes)

    # extract database schema information
    db_names = dict(
        (db_id, value["database_name"]) for db_id, value in database_statistics.items()
    )
    db_schema_info = dict(
        (db_id, get_json_schema(db_name))
        for db_id, db_name in db_names.items()
        if db_name != "unknown"
    )

    return label_norm, train_dataset, val_dataset, database_statistics, db_schema_info


def qppnet_collator(
    plans,
    feature_statistics: dict = None,
    db_statistics: dict = None,
    db_schema_info: dict = None,
    column_statistics: dict = None,
    plan_featurization=None,
):
    labels = []
    query_plans = []
    errors = []

    # iterate over plans and create lists of edges and features per node
    query_stats_dict = defaultdict(list)

    for sample_idx, p in plans:
        query_stats_dict["sample_idxs"].append(sample_idx)
        query_stats_dict["sql"].append(p["sql"])
        query_stats_dict["hint"].append(p["hint"])
        assign_additional_plan_info(
            plan=p,
            db_statistics=db_statistics,
            db_schema_info=db_schema_info,
            is_parsed_plan=False,
        )

        # Replacing with dummy values to make program work
        query_stats_dict["selectivity"].append(p["selectivity"])
        query_stats_dict["scan_op"].append(p["scan_op"])
        query_stats_dict["join_op"].append(p["join_op"])
        query_stats_dict["num_tables"].append(p["num_tables"])
        query_stats_dict["plan_depth"].append(None)

        # Extract cardinality errors
        mse, qerror = extract_card_error_from_plan(p, mode="root", is_parsed_plan=False)
        mse_avg, qerror_avg = extract_card_error_from_plan(
            p, mode="avg", is_parsed_plan=False
        )

        query_stats_dict["card_error_root_mse"].append(mse)
        query_stats_dict["card_error_root_qerror"].append(qerror)
        query_stats_dict["card_error_avg_mse"].append(mse_avg)
        query_stats_dict["card_error_avg_qerror"].append(qerror_avg)

        analyze_plan = p["analyze_plans"][0]

        # Process and parse the plan
        query_plan: OperatorTree = operator_tree_from_json(analyze_plan)
        try:
            query_plan.encode_recursively(
                column_statistics, feature_statistics, plan_featurization
            )
            labels.append(query_plan.runtime)
            query_plans.append(query_plan)
        except ValueError as e:
            errors.append(e)

    if errors:
        print(errors)

    return (query_plans, labels, query_stats_dict)
