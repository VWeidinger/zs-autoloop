import functools
from json import JSONDecodeError
from pathlib import Path
from typing import List, Optional

import numpy as np
from ldb_schema.load_schema import load_json
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader

from classes.classes import ZeroShotModelConfig, DataLoaderOptions, ModelConfig, InputFormats
from classes.workload_runs import WorkloadRuns
from models.zeroshot.postgres_plan_batching import postgres_plan_collator
from training.dataset.plan_dataset import PlanDataset
from training.dataset.statification import stratify_plans


def read_workload_runs(workload_run_paths: List[Path],
                       limit_queries: int = None,
                       limit_queries_affected_wl: int = None,
                       input_format: InputFormats = InputFormats.parsed,
                       include_timeout_queries: bool = False,
                       include_hint_not_obeyed_queries: bool = False,
                       stratify_per_wl_file: bool = False,
                       stratify_dataset: bool = False):

    # reads several workload runs
    plans = []
    database_statistics = dict()

    timeout_thresholds = []

    stratify_file_blacklist = ['join_order', 'physical_operator', 'access_path_selection']

    for path_index, path in enumerate(workload_run_paths):
        per_file_plans = []

        try:
            run = load_json(str(path))
        except JSONDecodeError:
            raise ValueError(f"Error reading {path}")
        database_statistics[path_index] = run["database_stats"]

        limit_per_ds = None
        if limit_queries is not None:
            if path_index >= len(workload_run_paths) - limit_queries_affected_wl:
                limit_per_ds = limit_queries // limit_queries_affected_wl
                print(f"Capping workload {path} after {limit_per_ds} queries")

        if input_format == InputFormats.parsed:
            for p_id, plan in enumerate(run["parsed_plans"]):

                if not include_timeout_queries and plan["timeout"]:
                    # Skip timeout queries
                    continue

                if not include_hint_not_obeyed_queries and plan["invalid_hint"]:
                    # Skip queries with invalid hints
                    continue

                plan["database_id"] = path_index
                per_file_plans.append(plan)
                if limit_per_ds is not None and p_id > limit_per_ds:
                    print("Stopping now")
                    break
        elif input_format == InputFormats.json:
            if "query_list" not in run:
                raise ValueError(f"This model requires json format, are you sure {path} is in json format?")

            for plan in run["query_list"]:
                # Only learn from valid plans
                plan["database_id"] = path_index

                if not include_timeout_queries and plan["timeout"]:
                    # Skip timeout queries
                    continue

                if not include_hint_not_obeyed_queries and plan["invalid_hint"]:
                    # Skip queries with invalid hints
                    continue

                if plan["zero_card"] != True and plan["invalid_hint"] != True:
                    assert "Execution Time" in plan["analyze_plans"][0].keys(), f"Execution Time not in plan {plan}"
                    per_file_plans.append(plan)
        else:
            raise ValueError(f"Input format {input_format} not supported")

        # extract the timeout thresholds used during query execution
        per_file_timeout_thresholds = [p['timeout_threshold'] for p in per_file_plans]  # in ms
        assert len(
            set(per_file_timeout_thresholds)) == 1, f"Multiple timeout thresholds in {path}: {set(per_file_timeout_thresholds)}"
        timeout_thresholds.append(per_file_timeout_thresholds[0])

        print(f'Stratify per wl file: {stratify_per_wl_file}')

        if any([blacklist in str(path) for blacklist in stratify_file_blacklist]):
            # Do not stratify if the file is in the blacklist
            pass
        elif stratify_per_wl_file:
            per_file_plans = stratify_plans(
                plans=per_file_plans, max_runtime_ms=per_file_timeout_thresholds[0], output_identifier_hint=str(path),
                print_distr_stats=True, shuffle_before=True
            )

        plans.extend(per_file_plans)

    if len(workload_run_paths) == 1 and any(
            [blacklist in str(workload_run_paths[0]) for blacklist in stratify_file_blacklist]):
        # Do not stratify if only a single file is present and it is in the blacklist
        pass
    elif stratify_dataset:
        # ideally there is just a single timeout threshold
        if len(set(timeout_thresholds)) > 1:
            print(
                f"Multiple timeout thresholds in dataset: {set(timeout_thresholds)}. Continuing stratification with max threshold.")
        max_runtime = max(timeout_thresholds)

        plans = stratify_plans(
            plans=plans, max_runtime_ms=max_runtime, output_identifier_hint="overall", print_distr_stats=True,
            shuffle_before=True
        )

    return plans, database_statistics


def _inv_log1p(x):
    return np.exp(x) - 1


def create_datasets(workload_run_paths,
                    model_config: ModelConfig,
                    val_ratio=0.15,
                    shuffle_before_split=True, is_test_dataset: bool = False,
                    stratify_dataset: bool = False, stratify_per_wl_file: bool = False) -> (
        Pipeline, PlanDataset, PlanDataset, PlanDataset, dict):
    plans, database_statistics = read_workload_runs(workload_run_paths=workload_run_paths,
                                                    limit_queries=model_config.limit_queries,
                                                    limit_queries_affected_wl=model_config.limit_queries_affected_wl,
                                                    input_format=model_config.input_format,
                                                    include_timeout_queries=model_config.include_timeout_queries and not is_test_dataset,
                                                    stratify_dataset=stratify_dataset,
                                                    stratify_per_wl_file=stratify_per_wl_file)

    no_plans = len(plans)
    plan_indexes = list(range(no_plans))
    if shuffle_before_split:
        np.random.shuffle(plan_indexes)

    train_ratio = 1 - val_ratio
    split_train = int(no_plans * train_ratio)
    train_indexes = plan_indexes[:split_train]
    # Limit number of training samples. To have comparable batch sizes, replicate remaining indexes.
    if model_config.cap_training_samples is not None:
        print(f'Limiting dataset to {model_config.cap_training_samples}')
        prev_train_length = len(train_indexes)
        train_indexes = train_indexes[:model_config.cap_training_samples]
        replicate_factor = max(prev_train_length // len(train_indexes), 1)
        train_indexes = train_indexes * replicate_factor

    train_dataset = PlanDataset([plans[i] for i in train_indexes], train_indexes)

    val_dataset = None
    if val_ratio > 0:
        val_indexes = plan_indexes[split_train:]
        val_dataset = PlanDataset([plans[i] for i in val_indexes], val_indexes)

    # derive label normalization
    runtimes = np.array([p["plan_runtime_ms"] / 1000 for p in plans if not p['timeout']])
    label_norm = derive_label_normalizer(model_config.loss_class_name, runtimes)

    return label_norm, train_dataset, val_dataset, database_statistics


def derive_label_normalizer(loss_class_name, y) -> Pipeline:
    if loss_class_name == "MSELoss":
        log_transformer = preprocessing.FunctionTransformer(np.log1p, _inv_log1p, validate=True)
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


def create_zeroshot_dataloader(workload_runs: WorkloadRuns,
                               statistics_file: Path,
                               model_config: ZeroShotModelConfig,
                               data_loader_options: DataLoaderOptions) \
        -> (Optional[Pipeline], dict, Optional[DataLoader], Optional[DataLoader], List[Optional[DataLoader]]):
    # Postgres_plan_collator does the heavy lifting of creating the graphs and extracting the features and thus
    # requires both database statistics but also feature statistics
    label_norm = Optional[Pipeline]
    train_loader, val_loader, test_loaders = Optional[DataLoader], Optional[DataLoader], List[Optional[DataLoader]]

    feature_statistics = load_json(str(statistics_file), namespace=False)
    assert feature_statistics != {}, "Feature statistics file is empty!"
    plan_collator = postgres_plan_collator

    dataloader_args = dict(batch_size=model_config.batch_size,
                           shuffle=data_loader_options.shuffle,
                           num_workers=model_config.num_workers,
                           pin_memory=data_loader_options.pin_memory)

    if workload_runs.train_workload_runs:
        # Split plans into train/test/validation set
        print("Creating dataloader for training and validation data")
        label_norm, train_dataset, val_dataset, database_statistics \
            = create_datasets(workload_run_paths=workload_runs.train_workload_runs,
                              model_config=model_config,
                              val_ratio=data_loader_options.val_ratio,
                              stratify_dataset=data_loader_options.stratify_dataset,
                              stratify_per_wl_file=data_loader_options.stratify_per_wl_file)

        train_collate_fn = functools.partial(plan_collator,
                                             db_statistics=database_statistics,
                                             feature_statistics=feature_statistics,
                                             plan_featurization=model_config.featurization,
                                             test_mode=False)

        dataloader_args.update(collate_fn=train_collate_fn)
        train_loader: DataLoader = DataLoader(train_dataset, **dataloader_args)
        val_loader: DataLoader = DataLoader(val_dataset, **dataloader_args)

    if workload_runs.test_workload_runs:
        # For each test workload run create a distinct test loader
        print("Creating dataloader for test data")
        test_loaders = []
        for test_path in workload_runs.test_workload_runs:
            _, test_dataset, _, test_database_statistics = create_datasets([test_path],
                                                                           model_config=model_config,
                                                                           shuffle_before_split=False,
                                                                           val_ratio=0.0,
                                                                           is_test_dataset=True,
                                                                           stratify_dataset=data_loader_options.stratify_dataset,
                                                                           stratify_per_wl_file=data_loader_options.stratify_per_wl_file)
            # test dataset
            test_collate_fn = functools.partial(plan_collator,
                                                db_statistics=test_database_statistics,
                                                feature_statistics=feature_statistics,
                                                plan_featurization=model_config.featurization,
                                                test_mode=True)

            # previously shuffle=False but this resulted in bugs
            dataloader_args.update(collate_fn=test_collate_fn)
            test_loader = DataLoader(test_dataset, **dataloader_args)
            test_loaders.append(test_loader)

    print(f"Created datasets of size: "
          f"train {len(train_dataset)}, "
          f"validation: {len(val_dataset)}, t"
          f"est: {len(test_dataset)}")

    return label_norm, feature_statistics, train_loader, val_loader, test_loaders
