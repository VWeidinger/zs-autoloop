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
                       limit_plans_per_file: int,
                       limit_plans_overall: int,
                       input_format: InputFormats = InputFormats.parsed,
                       include_timeout_queries: bool = False,
                       include_hint_not_obeyed_queries: bool = False,
                       include_zero_card_queries: bool = False,
                       stratify_per_wl_file: bool = False,
                       stratify_dataset: bool = False, ) -> (List[dict], dict):
    # reads several workload runs
    plans = []
    database_statistics = dict()

    timeout_thresholds = []

    stratify_file_blacklist = ['join_order', 'physical_operator', 'access_path_selection']
    limit_plans_blacklist = ['join_order', 'physical_operator', 'access_path_selection']

    for path_index, path in enumerate(workload_run_paths):
        if path is None or str(path) =='None':
            continue

        per_file_plans = []

        try:
            run = load_json(str(path))
        except JSONDecodeError:
            print(f'Error reading {path} - skip')
            continue
        database_statistics[path_index] = run["database_stats"]

        if input_format == InputFormats.parsed:
            assert "parsed_plans" in run, f"parsed_plans not in {path}, did you maybe used JSON query plans?"
            file_plans = run['parsed_plans']
        elif input_format == InputFormats.json:
            if "query_list" not in run:
                raise ValueError(f"This model requires json format, are you sure {path} is in json format?")
            file_plans = run['query_list']
        else:
            raise ValueError(f"Input format {input_format} not supported")

        # check if the file is in the blacklist
        if any([blacklist in str(path) for blacklist in limit_plans_blacklist]):
            file_in_limit_blacklist = True
        else:
            file_in_limit_blacklist = False

        if 'join_order_job' in str(path):
            # overwrite flag, to include zero card queries
            include_zero_card_queries = True

        hint_not_obeyed_ctr = 0
        timeout_ctr = 0
        include_zero_card_queries_ctr = 0

        for p_id, plan in enumerate(file_plans):
            if input_format == InputFormats.parsed:
                if not include_timeout_queries and plan["timeout"]:  # Skip timeout queries
                    timeout_ctr += 1
                    continue
                if not include_hint_not_obeyed_queries and plan["invalid_hint"]:  # Skip queries with invalid hints
                    hint_not_obeyed_ctr += 1
                    continue
                if not include_zero_card_queries and plan["zero_card"]:  # skip zero card queries
                    include_zero_card_queries_ctr += 1
                    continue
            elif input_format == InputFormats.json:
                if not include_timeout_queries and plan["timeout"]:  # Skip timeout queries
                    timeout_ctr += 1
                    continue
                if not include_hint_not_obeyed_queries and plan["invalid_hint"]:  # Skip queries with invalid hints
                    hint_not_obeyed_ctr += 1
                    continue
                if not include_zero_card_queries and plan[
                    "zero_card"]:  # TODO this annotation is not 100% correct in the json plans (plan parsed is fixing this)!
                    include_zero_card_queries_ctr += 1
                    continue
            else:
                raise ValueError(f"Input format {input_format} not supported")

            assert plan['plan_runtime_ms'] is not None, f"Missing plan_runtime_ms in {path} for {plan}"

            plan["database_id"] = path_index
            per_file_plans.append(plan)

            if limit_plans_per_file is not None and len(
                    per_file_plans) >= limit_plans_per_file and not file_in_limit_blacklist:
                print(f'Limiting plans per file to {limit_plans_per_file} for {path}', flush=True)
                break

        # extract the timeout thresholds used during query execution
        per_file_timeout_thresholds = [p['timeout_threshold'] for p in per_file_plans]  # in ms
        if len(per_file_timeout_thresholds) != 1:
            print(f"Multiple timeout thresholds in {path}: {set(per_file_timeout_thresholds)}")
        # assert len(
        #     set(per_file_timeout_thresholds)) == 1, f"Multiple timeout thresholds in {path}: {set(per_file_timeout_thresholds)}"
        timeout_thresholds.append(per_file_timeout_thresholds[0])

        if stratify_per_wl_file:
            print(f'Stratify per wl file: {stratify_per_wl_file}')

        if any([blacklist in str(path) for blacklist in stratify_file_blacklist]):
            # Do not stratify if the file is in the blacklist
            pass
        elif stratify_per_wl_file:
            per_file_plans = stratify_plans(
                plans=per_file_plans, max_runtime_ms=per_file_timeout_thresholds[0], output_identifier_hint=str(path),
                print_distr_stats=True, shuffle_before=True
            )

        print(
            f'Loaded {len(per_file_plans)}/{len(file_plans)} plans from {path} (include timeout: {include_timeout_queries} (excluded: {timeout_ctr}), '
            f' include invalid hint: {include_hint_not_obeyed_queries} (excluded: {hint_not_obeyed_ctr}), include zero card: {include_zero_card_queries} (excluded: {include_zero_card_queries_ctr}), file in limit blacklist: {file_in_limit_blacklist}, limit file: {limit_plans_per_file})',
            flush=True)

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

    if limit_plans_overall is not None:
        # shuffle plans
        np.random.shuffle(plans)
        plans = plans[:limit_plans_overall]

    return plans, database_statistics


def _inv_log1p(x):
    return np.exp(x) - 1


def create_datasets(workload_run_paths,
                    model_config: ModelConfig,
                    val_ratio=0.15,
                    shuffle_before_split=True, is_test_dataset: bool = False,
                    do_not_limit_file: bool = False,
                    stratify_dataset: bool = False, stratify_per_wl_file: bool = False) -> (
        Pipeline, PlanDataset, PlanDataset, PlanDataset, dict):
    plans, database_statistics = read_workload_runs(workload_run_paths=workload_run_paths,
                                                    limit_plans_per_file=model_config.limit_plans_per_file if not do_not_limit_file else None,
                                                    limit_plans_overall=model_config.limit_plans_overall if not do_not_limit_file else None,
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

    add_pseudo_entries_to_feature_statistics = True
    if add_pseudo_entries_to_feature_statistics:
        # base model has never seen these operator types - add them to the feature statistics to avoid one-hot encoder errors
        if 'MIN' not in feature_statistics['aggregation']['value_dict']:
            feature_statistics['aggregation']['value_dict']['MIN'] = 2
        if 'MAX' not in feature_statistics['aggregation']['value_dict']:
            feature_statistics['aggregation']['value_dict']['MAX'] = 2
        if 'IN' not in feature_statistics['operator']['value_dict']:
            feature_statistics['operator']['value_dict']['IN'] = 3
        if 'LIKE' not in feature_statistics['operator']['value_dict']:
            feature_statistics['operator']['value_dict']['LIKE'] = 3
        if 'NOT LIKE' not in feature_statistics['operator']['value_dict']:
            feature_statistics['operator']['value_dict']['NOT LIKE'] = 3
        if 'IS NOT NULL' not in feature_statistics['operator']['value_dict']:
            feature_statistics['operator']['value_dict']['IS NOT NULL'] = 3
        if 'IS NULL' not in feature_statistics['operator']['value_dict']:
            feature_statistics['operator']['value_dict']['IS NULL'] = 3


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
                              val_ratio=data_loader_options.val_ratio)

        train_collate_fn = functools.partial(plan_collator,
                                             db_statistics=database_statistics,
                                             feature_statistics=feature_statistics,
                                             plan_featurization=model_config.featurization,
                                             test_mode=False)

        dataloader_args.update(collate_fn=train_collate_fn)
        train_loader: DataLoader = DataLoader(train_dataset, **dataloader_args)
        val_loader: DataLoader = DataLoader(val_dataset, **dataloader_args)

        print(f"Created datasets of size: "
              f"train {len(train_dataset)}, "
              f"validation: {len(val_dataset)}")
    else:
        train_loader, val_loader = None, None

    if workload_runs.test_workload_runs:
        # For each test workload run create a distinct test loader
        print("Creating dataloader for test data")
        test_loaders = []
        for test_path in workload_runs.test_workload_runs:
            _, test_dataset, _, test_database_statistics = create_datasets([test_path],
                                                                           model_config=model_config,
                                                                           shuffle_before_split=False,
                                                                           val_ratio=0.0,
                                                                           is_test_dataset=True)
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
                  f"test: {len(test_dataset)}")

    return label_norm, feature_statistics, train_loader, val_loader, test_loaders