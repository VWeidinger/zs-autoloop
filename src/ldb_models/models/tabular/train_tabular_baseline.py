import functools
import math
import os
import time
from pathlib import Path
from typing import Optional, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import torch
from ldb_schema.load_schema import load_json
from sklearn.linear_model import LinearRegression
from wandb.apis.public import Run

from classes.classes import FlatModelConfig, ModelConfig, ScaledPostgresModelConfig, \
    AnalyticalEstCardModelConfig, AnalyticalActCardModelConfig, DataLoaderOptions, T3ModelConfig, T3FlatModelConfig, \
    T3TreeModelConfig
from classes.workload_runs import WorkloadRuns
from models.t3.model import T3Model, TreeModel, FlatTreeModel, PerTupleTreeModel
from models.t3.t3_dataloader import gen_t3_dataset
from models.tabular.slice_train_val import slice_train_val
from models.zeroshot.postgres_plan_batching import assign_additional_plan_info
from training.bound_predictions_with_history import PredictionBounder
from training.dataset.dataset_creation import read_workload_runs
from training.evaluate_query_optimization import evaluate_join_order, evaluate_physical_plan_selection, \
    evaluate_access_path_selection
from training.featurizations import Featurization
from training.training.checkpoint import save_csv
from training.training.metrics import init_metrics
from training.training.utils import prefix_dict
from training.training.validate import to_rows


def train_tabular_model(workload_runs: WorkloadRuns,
                        config: ModelConfig,
                        statistics_file: Path,
                        model_dir: Path,
                        cap_training_samples: int = None,
                        early_stopping_rounds: int = 20,
                        num_boost_round: int = 1000,
                        bound_predictions: bool = False,
                        prediction_bounder: Optional[PredictionBounder] = None,
                        dataloader_options: DataLoaderOptions = None,
                        verbose: bool = True
                        ):
    if bound_predictions or prediction_bounder:
        raise NotImplementedError("prediction bounding needs to be implemented for flat models")

    # seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    os.makedirs(model_dir, exist_ok=True)

    param_dict = {}
    # create tabular representation using the corresponding method
    if isinstance(config, FlatModelConfig):
        gen_dataset_func = gen_flattened_dataset
    elif isinstance(config, ScaledPostgresModelConfig):
        gen_dataset_func = gen_postgres_cost_dataset
    elif isinstance(config, AnalyticalEstCardModelConfig):
        gen_dataset_func = functools.partial(gen_analytical_flattened_dataset, act_card=False)
    elif isinstance(config, AnalyticalActCardModelConfig):
        gen_dataset_func = functools.partial(gen_analytical_flattened_dataset, act_card=True)
    elif isinstance(config, T3ModelConfig):
        gen_dataset_func = functools.partial(gen_t3_dataset, model_config=config, verbose=verbose)
    else:
        raise RuntimeError("Not supported")

    if config.type == 'wl_driven':
        x_train, x_test, y_train, y_test, _, _ = gen_dataset_func(workload_runs=workload_runs.train_workload_runs,
                                                                  statistics_file=statistics_file,
                                                                  cap_training_samples=cap_training_samples,
                                                                  val_ratio=0.20,
                                                                  shuffle=True,
                                                                  stratify_dataset=dataloader_options.stratify_dataset,
                                                                  stratify_per_wl_file=dataloader_options.stratify_per_wl_file,
                                                                  limit_plans_per_file=config.limit_plans_per_file,
                                                                  limit_plans_overall=config.limit_plans_overall,
                                                                  augment_with_subplans=config.train_on_subplans)
        raise NotImplementedError("This needs to be solved for workload driven model due to the correct split")
    elif config.type == 'wl_agnostic':

        if hasattr(config, 'featurization'):
            featurization = config.featurization
        else:
            featurization = None
        x_train, x_test, y_train, y_test, sqls, hints, selectivities, scan_op, join_op = gen_dataset_func(
            workload_runs=workload_runs.train_workload_runs,
            statistics_file=statistics_file,
            cap_training_samples=cap_training_samples,
            featurization=featurization,
            val_ratio=0.20,
            shuffle=True,
            limit_plans_per_file=config.limit_plans_per_file,
            limit_plans_overall=config.limit_plans_overall,
            augment_with_subplans=config.train_on_subplans,
        )
    else:
        raise RuntimeError("Not supported")

    if isinstance(x_train, list):
        print(f"Training model {config.name.NAME} "
              f"with seed {config.seed} "
              f"and {len(x_train)} training points "
              f"and {len(x_test)} validation points")

    start_t = time.perf_counter()
    model = None
    if isinstance(config, FlatModelConfig):
        train_data = lgb.Dataset(x_train, label=y_train)
        val_data = lgb.Dataset(x_test, label=y_test, reference=train_data)
        param = dict(metric='mse',
                     objective='regression',
                     random_state=config.seed,
                     seed=config.seed,
                     bagging_seed=config.seed,
                     feature_fraction_seed=config.seed)
        bst = lgb.train(param, train_data, num_boost_round=num_boost_round, valid_sets=[val_data])
        param_dict.update(num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds)
        bst.save_model(f'{model_dir}_{config.seed}.txt')
        print(f'Model saved to {model_dir}_{config.seed}.txt')

    elif isinstance(config, (ScaledPostgresModelConfig, AnalyticalEstCardModelConfig, AnalyticalActCardModelConfig)):
        m = LinearRegression()
        m.fit(x_train, y_train)
        joblib.dump(m, f'{model_dir}_{config.seed}.pkl')
        print(f'Model saved to {model_dir}_{config.seed}.pkl')
        os.chmod(f'{model_dir}_{config.seed}.pkl', 0o777)
    elif isinstance(config, T3ModelConfig):
        param = {"objective": "mape", "verbose": 2 if verbose else -1}
        bst = lgb.Booster(param, x_train)
        bst.add_valid(x_test, "val_data")
        if verbose:
            print(bst.eval_train())
        for i in range(200):
            bst.update()
            if verbose:
                print(i + 1, bst.eval_train(), bst.eval_valid())

        # assemble model path
        model_path = f'{model_dir}_{config.seed}_{config.name}.txt'
        bst.save_model(model_path)
        print(f'Model saved to {model_dir}_{config.seed}.txt')

        # wrap the model in the T3Model class
        if isinstance(config, T3FlatModelConfig):
            model = FlatTreeModel(bst)
        elif isinstance(config, T3TreeModelConfig):
            model = TreeModel(bst)
        else:
            model = PerTupleTreeModel(bst)
    else:
        raise NotImplementedError('Not supported tabular model')

    param_dict.update(training_time=time.perf_counter() - start_t,
                      no_train_points=len(x_train) if isinstance(x_train, list) else -1,
                      no_val_points=len(x_test) if isinstance(x_test, list) else -1, )

    save_csv([param_dict], str(model_dir) + f'_{config.seed}_{config.name}_params.csv')
    return model


def predict_tabular_model(config: ModelConfig,
                          workload_runs: WorkloadRuns,
                          statistics_file: Path,
                          model_dir: Path,
                          cap_training_samples: bool = None,
                          wandb_run: Optional[Run] = None,
                          bound_predictions: bool = False,
                          prediction_bounder: Optional[PredictionBounder] = None,
                          log_all_queries: bool = False,
                          t3_model: T3Model = None):
    if bound_predictions or prediction_bounder:
        raise NotImplementedError("prediction bounding needs to be implemented for flat models")

    # seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if isinstance(config, FlatModelConfig):
        print(f"Loading model from {model_dir}_{config.seed}.txt")
        bst = lgb.Booster(model_file=f'{model_dir}_{config.seed}.txt')

        # Define a function to apply lower bound
        def predict_with_lower_bound(data, lower_bound, bst=bst):
            predictions = bst.predict(data, num_iteration=bst.best_iteration)
            predictions = [max(prediction, lower_bound) for prediction in predictions]
            return predictions

        # Partially apply the predict_with_lower_bound function
        pred_func = functools.partial(predict_with_lower_bound, lower_bound=0.01)

        gen_dataset_func = gen_flattened_dataset

    elif isinstance(config, ScaledPostgresModelConfig):
        print(f"Loading model from {model_dir}_{config.seed}.pkl")
        m = joblib.load(filename=f'{model_dir}_{config.seed}.pkl')
        pred_func = m.predict
        gen_dataset_func = gen_postgres_cost_dataset
    elif isinstance(config, T3ModelConfig):
        pred_func = t3_model.estimate_runtime_wrapper  # use the list-predict wrapper
        gen_dataset_func = functools.partial(gen_t3_dataset, model_config=config, verbose=False, return_dataset=False)
    else:
        raise NotImplementedError('Not supported tabular model')

    if hasattr(config, 'featurization'):
        featurization = config.featurization
    else:
        featurization = None

    test_datasets = []
    for test_wl in workload_runs.test_workload_runs:
        x_test, _, y_test, _, test_sqls, test_hints, test_selectivities, test_scan_op_list, test_join_op_list \
            = gen_dataset_func(workload_runs=[test_wl],
                               statistics_file=statistics_file,
                               cap_training_samples=cap_training_samples,
                               featurization=featurization,
                               val_ratio=0,
                               shuffle=False,
                               include_timeout_queries=False,
                               limit_plans_overall=config.limit_plans_overall,
                               limit_plans_per_file=config.limit_plans_per_file,
                               augment_with_subplans=False,  # nether augment test dataset
                               )
        test_datasets.append(
            (x_test, y_test, test_sqls, test_hints, test_selectivities, test_scan_op_list, test_join_op_list))

    # retrieve metrics similar to the ones used for QLoss
    metrics = init_metrics('QLoss')

    for (x_test, y_test, sqls, hints, selectivities, scan_ops, join_ops), test_path in zip(test_datasets,
                                                                                           workload_runs.target_test_csv_paths):
        print(f"Starting validation for {test_path}")
        test_stats = dict()
        start_t = time.perf_counter()
        pred_runtimes = pred_func(x_test)
        test_stats.update(
            test_time=time.perf_counter() - start_t,
            no_test_points=len(y_test),
            # test_labels=y_test,
            # test_preds=pred_runtimes
        )

        for metric in metrics:
            metric.evaluate(metrics_dict=test_stats,
                            model=None,
                            labels=y_test,
                            preds=pred_runtimes,
                            probs=None)
        save_csv([test_stats], target_csv_path=str(test_path.with_suffix('')) + f"_test_stats.csv")

        # Logging all queries
        rows = to_rows(
            indexes=[int(i) for i in range(0, len(y_test))],
            labels=[float(f) for f in y_test],
            predictions=[float(f) for f in pred_runtimes],
            sqls=sqls,
            hints=hints,
            selectivitys=selectivities,
            scan_ops=scan_ops,
            join_ops=join_ops)
        if log_all_queries:
            save_csv(rows, target_csv_path=str(test_path) + "_test_pred.csv")

        # Log to wandb
        test_file_name = test_path.stem
        # remove _seed from name (this is different from the _s0 suffix)
        test_file_name = '_'.join(test_file_name.split('_')[:-1])

        if wandb_run:
            wandb_run.summary.update(prefix_dict(test_stats, f'{test_file_name}.json/test_'))

        if "join_order" in str(test_path):
            if test_file_name == 'join_order':
                wl_name = 'job_light'  # job_light
            elif test_file_name in ['join_order_job', 'join_order_job_w_idxs_combined']:
                wl_name = 'job'
            else:
                raise ValueError(f"Unknown workload name: {test_file_name}")
            evaluate_join_order(predictions=rows, wandb_run=wandb_run, wl_name=wl_name)
        if "physical_operator" in str(test_path):
            evaluate_physical_plan_selection(predictions=rows, wandb_run=wandb_run,
                                             with_idxs='_w_idx' in str(test_path))
        if "access_path_selection" in str(test_path):
            evaluate_access_path_selection(predictions=rows, wandb_run=wandb_run)
    return


def gen_postgres_cost_dataset(workload_runs: List[Path],
                              cap_training_samples: Optional[bool],
                              statistics_file: Path,
                              featurization: Optional,
                              shuffle: bool,
                              limit_plans_overall: int,
                              limit_plans_per_file: int,
                              val_ratio: float = 0.15,
                              include_timeout_queries: bool = False,
                              stratify_dataset: bool = False,
                              stratify_per_wl_file: bool = False,
                              augment_with_subplans: bool = False, ) -> Tuple[np.ndarray]:
    plans, database_statistics = read_workload_runs(workload_runs, include_timeout_queries=include_timeout_queries,
                                                    stratify_dataset=stratify_dataset,
                                                    stratify_per_wl_file=stratify_per_wl_file,
                                                    limit_plans_overall=limit_plans_overall,
                                                    limit_plans_per_file=limit_plans_per_file,
                                                    augment_with_subplans=augment_with_subplans)
    feature_vecs = []
    labels = []
    sqls, hints = [], []  # Required to later evaluate against join order
    selectivities, join_ops, scan_ops = [], [], []
    for plan in plans:
        # append to all feature vectors and runtime labels
        feature_vecs.append([plan["plan_parameters"]["pg_cost"]])
        assert plan['plan_runtime_ms'] is not None, f"Plan runtime is None for {plan}"
        labels.append(plan["plan_runtime_ms"] / 1000)
        sqls.append(plan["sql"])
        hints.append(plan["hint"])
        assign_additional_plan_info(plan=plan, db_statistics=database_statistics)
        selectivities += plan["selectivity"]
        join_ops += plan["join_op"]
        scan_ops += plan["scan_op"]

    X_train, X_val, Y_train, Y_val = slice_train_val(cap_training_samples,
                                                     feature_vecs,
                                                     labels,
                                                     plans,
                                                     val_ratio,
                                                     shuffle)

    return X_train, X_val, Y_train, Y_val, sqls, hints, selectivities, scan_ops, join_ops


def gen_flattened_dataset(workload_runs: List[Path],
                          statistics_file: Path,
                          cap_training_samples: Optional[bool],
                          featurization: Featurization,
                          limit_plans_per_file: int,
                          limit_plans_overall: int,
                          val_ratio: float = 0.15,
                          shuffle: bool = True,
                          include_timeout_queries: bool = False,
                          augment_with_subplans: bool = False,
                          ):
    feature_statistics = load_json(str(statistics_file), namespace=False)
    op_idx_dict = feature_statistics['op_name']['value_dict']
    no_ops = len(op_idx_dict)
    plans, database_statistics = read_workload_runs(workload_runs, include_timeout_queries=include_timeout_queries,
                                                    limit_plans_per_file=limit_plans_per_file,
                                                    limit_plans_overall=limit_plans_overall,
                                                    augment_with_subplans=augment_with_subplans)
    feature_vecs = []
    labels = []
    selectivities = []
    join_ops = []
    scan_ops = []

    sqls, hints = [], []  # Required to later evaluate against query optimization
    for plan in plans:
        sqls.append(plan["sql"])
        hints.append(plan["hint"])
        assign_additional_plan_info(plan=plan, db_statistics=database_statistics)
        selectivities += plan["selectivity"]
        join_ops += plan["join_op"]
        scan_ops += plan["scan_op"]

        # number an operator occurs
        feature_num_vec = np.zeros(no_ops)
        # number of rows per op (estimated by postgres)
        feature_row_vec = np.zeros(no_ops)

        def extract_features(p, featurization):
            op_name = p["plan_parameters"]["op_name"]
            op_idx = op_idx_dict[op_name]
            feature_num_vec[op_idx] += 1
            if featurization.PLAN_FEATURE == 'est_card':
                if 'est_card' in p["plan_parameters"].keys():
                    feature_row_vec[op_idx] += p["plan_parameters"]["est_card"]
                else:
                    feature_row_vec[op_idx] += p["plan_parameters"]["est_rows"]
            elif featurization.PLAN_FEATURE == 'act_card':
                if 'act_card' in p["plan_parameters"].keys():
                    feature_row_vec[op_idx] += p["plan_parameters"]["act_card"]
                elif 'act_rows' in p["plan_parameters"].keys():
                    feature_row_vec[op_idx] += p["plan_parameters"]["act_rows"]
                else:
                    feature_row_vec[op_idx] += p["plan_parameters"]["est_card"]

            for c in p["children"]:
                extract_features(c, featurization)

        extract_features(plan, featurization)

        # append to all feature vectors and runtime labels
        feature_vecs.append(np.concatenate((feature_num_vec, feature_row_vec)))
        labels.append(plan["plan_runtime_ms"] / 1000)
    X_train, X_val, Y_train, Y_val = slice_train_val(cap_training_samples, feature_vecs, labels, plans, val_ratio,
                                                     shuffle=shuffle)
    return X_train, X_val, Y_train, Y_val, sqls, hints, selectivities, scan_ops, join_ops


def extract_card(p, act_card):
    if not act_card:
        if 'est_card' in p["plan_parameters"].keys():
            card = p["plan_parameters"]["est_card"]
        else:
            card = p["plan_parameters"]["est_rows"]
    else:
        if 'est_card' in p["plan_parameters"].keys():
            if 'act_card' in p["plan_parameters"].keys():
                card = p["plan_parameters"]["act_card"]
            # fallback: for some plan operators, pg does not annotate act cards
            else:
                card = p["plan_parameters"]["est_card"]
        else:
            card = p["plan_parameters"]["act_rows"]
    return card


def gen_analytical_flattened_dataset(workload_runs, statistics_file, cap_training_samples, val_ratio=0.15,
                                     act_card=False, include_timeout_queries: bool = False,
                                     stratify_dataset: bool = False, stratify_per_wl_file: bool = False,
                                     limit_plans_per_file: int = None, limit_plans_overall: int = None,
                                     augment_with_subplans: bool = False):
    feature_statistics = load_json(str(statistics_file), namespace=False)
    op_idx_dict = feature_statistics['op_name']['value_dict']
    no_ops = len(op_idx_dict)
    plans, database_statistics = read_workload_runs(workload_run_paths=workload_runs,
                                                    include_timeout_queries=include_timeout_queries,
                                                    stratify_dataset=stratify_dataset,
                                                    stratify_per_wl_file=stratify_per_wl_file,
                                                    limit_plans_per_file=limit_plans_per_file,
                                                    limit_plans_overall=limit_plans_overall,
                                                    augment_with_subplans=augment_with_subplans)
    feature_vecs = []
    labels = []
    for plan in plans:
        # number an operator occurs
        feature_num_vec = np.zeros(no_ops)
        # number of rows per op (estimated by postgres)
        feature_row_vec = np.zeros(no_ops)
        feature_row_vec_left_input = np.zeros(no_ops)
        feature_row_vec_right_input = np.zeros(no_ops)
        feature_row_vec_prod_input = np.zeros(no_ops)

        def extract_features(p):
            op_name = p["plan_parameters"]["op_name"]
            op_idx = op_idx_dict[op_name]
            feature_num_vec[op_idx] += 1

            # output
            feature_row_vec[op_idx] += extract_card(p, act_card)

            child_card = [extract_card(c, act_card) for c in p["children"]]
            if len(child_card) >= 1:
                feature_row_vec_left_input[op_idx] += child_card[0]
                feature_row_vec_prod_input[op_idx] += math.prod(child_card)
            if len(child_card) >= 2:
                feature_row_vec_right_input[op_idx] += child_card[1]

            for c in p["children"]:
                extract_features(c)

        extract_features(plan)

        # append to all feature vectors and runtime labels
        feature_vecs.append(np.concatenate((feature_num_vec, feature_row_vec, feature_row_vec_left_input,
                                            feature_row_vec_right_input, feature_row_vec_prod_input)))
        labels.append(plan["plan_runtime_ms"] / 1000)
    X_train, X_val, Y_train, Y_val = slice_train_val(cap_training_samples, feature_vecs, labels, plans, val_ratio)
    return X_train, X_val, Y_train, Y_val
