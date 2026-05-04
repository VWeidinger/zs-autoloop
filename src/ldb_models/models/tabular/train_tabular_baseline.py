import functools
import math
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import torch
from ldb_schema.load_schema import get_json_schema, load_json
from sklearn.linear_model import LinearRegression
from spn.algorithms.LearningWrappers import learn_parametric
from spn.algorithms.MPE import mpe
from spn.structure.Base import Context
from spn.structure.leaves.parametric.Parametric import Gaussian
from wandb.apis.public import Run

from ldb_models.classes import (
    AnalyticalActCardModelConfig,
    AnalyticalEstCardModelConfig,
    DataLoaderOptions,
    FlatModelConfig,
    ModelConfig,
    ScaledPostgresModelConfig,
    SPNModelConfig,
    T3FlatModelConfig,
    T3ModelConfig,
    T3TreeModelConfig,
)
from ldb_models.classes.workload_runs import WorkloadRuns
from ldb_models.models.t3.model import (
    FlatTreeModel,
    PerTupleTreeModel,
    T3Model,
    TreeModel,
)
from ldb_models.models.t3.t3_dataloader import gen_t3_dataset
from ldb_models.models.tabular.slice_train_val import slice_train_val
from ldb_models.models.white_mamba.extract_query_stats import (
    extract_query_stats_for_plan,
)
from ldb_models.models.zeroshot.postgres_plan_batching import (
    assign_additional_plan_info,
)
from ldb_models.training.bound_predictions_with_history import PredictionBounder
from ldb_models.training.dataset.dataset_creation import read_workload_runs
from ldb_models.training.evaluate_query_optimization import (
    evaluate_access_path_selection,
    evaluate_join_order,
    evaluate_physical_plan_selection,
)
from ldb_models.training.featurizations import Featurization
from ldb_models.training.training.checkpoint import save_csv
from ldb_models.training.training.metrics import init_metrics
from ldb_models.training.training.outputs import ModelOutput
from ldb_models.training.training.utils import prefix_dict
from ldb_models.training.training.validate import to_rows


def train_tabular_model(
    workload_runs: WorkloadRuns,
    config: ModelConfig,
    statistics_file: Path,
    model_dir: Path,
    cap_training_samples: int = None,
    early_stopping_rounds: int = 20,
    num_boost_round: int = 1000,
    bound_predictions: bool = False,
    prediction_bounder: Optional[PredictionBounder] = None,
    dataloader_options: DataLoaderOptions = None,
    verbose: bool = True,
):
    if bound_predictions or prediction_bounder:
        raise NotImplementedError(
            "prediction bounding needs to be implemented for flat models"
        )

    # seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    os.makedirs(model_dir, exist_ok=True)

    param_dict = {}
    # create tabular representation using the corresponding method
    if isinstance(config, FlatModelConfig) or isinstance(config, SPNModelConfig):
        gen_dataset_func = gen_flattened_dataset
    elif isinstance(config, ScaledPostgresModelConfig):
        gen_dataset_func = gen_postgres_cost_dataset
    elif isinstance(config, AnalyticalEstCardModelConfig):
        gen_dataset_func = functools.partial(
            gen_analytical_flattened_dataset, act_card=False
        )
    elif isinstance(config, AnalyticalActCardModelConfig):
        gen_dataset_func = functools.partial(
            gen_analytical_flattened_dataset, act_card=True
        )
    elif isinstance(config, T3ModelConfig):
        gen_dataset_func = functools.partial(gen_t3_dataset, verbose=verbose)
    else:
        raise RuntimeError("Not supported")

    if config.type == "wl_driven":
        x_train, x_test, y_train, y_test, _, _ = gen_dataset_func(
            workload_runs=workload_runs.train_workload_runs,
            statistics_file=statistics_file,
            cap_training_samples=cap_training_samples,
            val_ratio=0.20,
            shuffle=True,
            stratify_dataset=dataloader_options.stratify_dataset,
            stratify_per_wl_file=dataloader_options.stratify_per_wl_file,
            model_config=config,
        )
        raise NotImplementedError(
            "This needs to be solved for workload driven model due to the correct split"
        )
    elif config.type == "wl_agnostic":
        if hasattr(config, "featurization"):
            featurization = config.featurization
        else:
            featurization = None
        (
            x_train,
            x_test,
            y_train,
            y_test,
            query_stats_dict,
        ) = gen_dataset_func(
            workload_runs=workload_runs.train_workload_runs,
            statistics_file=statistics_file,
            cap_training_samples=cap_training_samples,
            featurization=featurization,
            val_ratio=0.20,
            shuffle=True,
            model_config=config,
        )
    else:
        raise RuntimeError("Not supported")

    if isinstance(x_train, list):
        print(
            f"Training model {config.name.NAME} "
            f"with seed {config.seed} "
            f"and {len(x_train)} training points "
            f"and {len(x_test)} validation points"
        )

    start_t = time.perf_counter()
    model = None
    if isinstance(config, FlatModelConfig):
        train_data = lgb.Dataset(x_train, label=y_train)
        val_data = lgb.Dataset(x_test, label=y_test, reference=train_data)
        param = dict(
            metric="mse",
            objective="regression",
            random_state=config.seed,
            seed=config.seed,
            bagging_seed=config.seed,
            feature_fraction_seed=config.seed,
        )
        bst = lgb.train(
            param, train_data, num_boost_round=num_boost_round, valid_sets=[val_data]
        )
        param_dict.update(
            num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds
        )
        bst.save_model(f"{model_dir}_{config.seed}v{config.version}.txt")
        print(f"Model saved to {model_dir}_{config.seed}v{config.version}.txt")

    elif isinstance(
        config,
        (
            ScaledPostgresModelConfig,
            AnalyticalEstCardModelConfig,
            AnalyticalActCardModelConfig,
        ),
    ):
        m = LinearRegression()
        m.fit(x_train, y_train)
        model_path = f"{model_dir}_{config.seed}v{config.version}.pkl"
        joblib.dump(m, model_path)
        print(f"Model saved to {model_path}")
        os.chmod(model_path, 0o777)

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
        model_path = (
            f"{model_dir}_{config.seed}_{config.name.NAME}v{config.version}.txt"
        )
        bst.save_model(model_path)
        print(f"Model saved to {model_path}")

        # wrap the model in the T3Model class
        if isinstance(config, T3FlatModelConfig):
            model = FlatTreeModel(bst)
        elif isinstance(config, T3TreeModelConfig):
            model = TreeModel(bst)
        else:
            model = PerTupleTreeModel(bst)

    elif isinstance(config, SPNModelConfig):
        # Merge x_train and y_train into a single dataset (see SPN documentation)
        if len(x_train.shape) == 1:
            x_train = x_train.reshape(-1, 1)
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
        train = np.hstack((x_train, y_train))
        ds_context = Context(parametric_types=[Gaussian] * train.shape[1]).add_domains(
            train
        )

        print("Learning SPN with Gaussian leaves...")
        spn = learn_parametric(train, ds_context, min_instances_slice=20)

        print("Saving SPN model...")
        model_path = (
            f"{model_dir}_{config.seed}_{config.name.NAME}v{config.version}.pkl"
        )

        with open(f"{model_path}", "wb") as f:
            pickle.dump(spn, f)
        print(f"Model saved to {model_path}")

    else:
        raise NotImplementedError("Not supported tabular model")

    param_dict.update(
        training_time=time.perf_counter() - start_t,
        no_train_points=len(x_train) if isinstance(x_train, list) else -1,
        no_val_points=len(x_test) if isinstance(x_test, list) else -1,
    )

    save_csv(
        [param_dict],
        str(model_dir)
        + f"_{config.seed}v{config.version}_{config.name.NAME}_params.csv",
    )
    return model


def predict_tabular_model(
    config: ModelConfig,
    workload_runs: WorkloadRuns,
    statistics_file: Path,
    model_dir: Path,
    cap_training_samples: bool = None,
    wandb_run: Optional[Run] = None,
    bound_predictions: bool = False,
    prediction_bounder: Optional[PredictionBounder] = None,
    log_all_queries: bool = True,
    t3_model: T3Model = None,
):
    if bound_predictions or prediction_bounder:
        raise NotImplementedError(
            "prediction bounding needs to be implemented for flat models"
        )

    # seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if isinstance(config, FlatModelConfig):
        model_path = f"{model_dir}_{config.seed}v{config.version}.txt"
        print(f"Loading model from {model_path}")
        bst = lgb.Booster(model_file=model_path)

        # Define a function to apply lower bound
        def predict_with_lower_bound(data, lower_bound, bst=bst):
            predictions = bst.predict(data, num_iteration=bst.best_iteration)
            predictions = [max(prediction, lower_bound) for prediction in predictions]
            return ModelOutput(plan_predictions=predictions)

        # Partially apply the predict_with_lower_bound function
        pred_func = functools.partial(predict_with_lower_bound, lower_bound=0.01)
        gen_dataset_func = gen_flattened_dataset

    elif isinstance(config, ScaledPostgresModelConfig):
        model_path = f"{model_dir}_{config.seed}v{config.version}.pkl"
        print(f"Loading model from {model_path}")
        m = joblib.load(filename=model_path)
        pred_func = m.predict
        gen_dataset_func = gen_postgres_cost_dataset

    elif isinstance(config, T3ModelConfig):
        model_path = (
            f"{model_dir}_{config.seed}_{config.name.NAME}v{config.version}.txt"
        )
        print(f"Loading model from {model_path}")
        bst = lgb.Booster(model_file=model_path)
        print(f"Model saved to {model_path}")

        # wrap the model in the T3Model class
        if isinstance(config, T3FlatModelConfig):
            t3_model = FlatTreeModel(bst)
        elif isinstance(config, T3TreeModelConfig):
            t3_model = TreeModel(bst)
        else:
            t3_model = PerTupleTreeModel(bst)
        assert t3_model is not None
        pred_func = t3_model.estimate_runtime_wrapper  # use the list-predict wrapper
        gen_dataset_func = functools.partial(
            gen_t3_dataset, model_config=config, verbose=False, return_dataset=False
        )

    elif isinstance(config, SPNModelConfig):
        model_path = (
            f"{model_dir}_{config.seed}_{config.name.NAME}v{config.version}.pkl"
        )
        print(f"Loading model from {model_path}")
        with open(model_path, "rb") as f:
            spn_loaded = pickle.load(f)
        assert spn_loaded is not None, f"SPN model {model_path} could not be loaded"
        pred_func = lambda x: mpe(spn_loaded, x)  # noqa: E731
        gen_dataset_func = gen_flattened_dataset

    else:
        raise NotImplementedError("Not supported tabular model")

    if hasattr(config, "featurization"):
        featurization = config.featurization
    else:
        featurization = None

    test_datasets = []
    for test_wl in workload_runs.test_workload_runs:
        (
            x_test,
            _,
            y_test,
            _,
            query_stats_dict,
        ) = gen_dataset_func(
            workload_runs=[test_wl],
            statistics_file=statistics_file,
            cap_training_samples=cap_training_samples,
            featurization=featurization,
            val_ratio=0,
            shuffle=False,
            model_config=config,
            is_test=True,
        )
        test_datasets.append((x_test, y_test, query_stats_dict))

    # retrieve metrics similar to the ones used for QLoss
    metrics = init_metrics("QLoss")

    # initialize variables for aggregated reporting across all experiments
    job_total_selected_runtime = -1
    job_avg_slowdown = None
    job_light_total_selected_runtime = -1
    job_light_avg_slowdown = None
    job_complex_total_selected_runtime = -1
    _job_complex_avg_slowdown = None  # For compatibility with older benchmark they are not included in the total slowdown calculation
    physop_selected_runtime = -1
    physop_avg_slowdown = None
    physop_idx_selected_runtime = -1
    physop_idx_avg_slowdown = None
    accesspath_selected_runtime = -1
    accesspath_avg_slowdown = None
    accesspath_w_agg_selected_runtime = -1
    _accesspath_w_agg_avg_slowdown = None  # For compatibility with older benchmark they are not included in the total slowdown calculation

    for (x_test, y_test, query_stats_dict), test_path in zip(
        test_datasets, workload_runs.target_test_csv_paths
    ):
        print(f"Starting validation for {test_path}")
        test_stats = dict()
        start_t = time.perf_counter()
        if isinstance(config, SPNModelConfig):
            if len(x_test.shape) == 1:
                x_test = x_test.reshape(-1, 1)
            test = np.hstack((x_test, np.full((x_test.shape[0], 1), np.nan)))
            pred_runtimes = mpe(spn_loaded, test)
            pred_runtimes = pred_runtimes[:, -1]

        else:
            pred_runtimes = pred_func(x_test)
        if isinstance(pred_runtimes, ModelOutput):
            pred_runtimes = pred_runtimes.plan_predictions
        test_stats.update(
            test_time=time.perf_counter() - start_t,
            no_test_points=len(y_test),
            # test_labels=y_test,
            # test_preds=pred_runtimes
        )

        for metric in metrics:
            metric.evaluate(
                metrics_dict=test_stats,
                model=None,
                labels=y_test,
                preds=pred_runtimes,
                probs=None,
            )
        save_csv(
            [test_stats],
            target_csv_path=str(test_path.with_suffix("")) + "_test_stats.csv",
        )

        # Logging all queries
        rows = to_rows(
            dict(
                label=[float(f) for f in y_test],
                prediction=[float(f) for f in pred_runtimes],
                **query_stats_dict,
            )
        )
        if log_all_queries:
            save_csv(rows, target_csv_path=str(test_path) + "_test_pred.csv")

        # Log to wandb
        test_file_name = test_path.stem
        # remove _seed from name (this is different from the _s0 suffix)
        test_file_name = "_".join(test_file_name.split("_")[:-1])

        if wandb_run:
            wandb_run.summary.update(
                prefix_dict(test_stats, f"{test_file_name}.json/test_")
            )

        if (
            "join_order" in str(test_path)
            or "job_" in str(test_path)
            or "JOBExtended" in test_file_name
        ):
            if test_file_name == "join_order":
                wl_name = "job_light"  # job_light
            elif test_file_name in [
                "join_order_job",
                "join_order_job_w_idxs_combined",
                "job_w_optimal_w_idxs",
            ]:
                wl_name = "job"
            elif "JOBExtended" in test_file_name or "job_complex" in test_file_name:
                wl_name = "job_complex"
            else:
                raise ValueError(f"Unknown workload name: {test_file_name}")
            rt, slowdown = evaluate_join_order(
                predictions=rows,
                wandb_run=wandb_run,
                wl_name=wl_name,
                is_per_epoch=False,
            )
            if wl_name == "job":
                job_total_selected_runtime = rt
                job_avg_slowdown = slowdown
            elif wl_name == "job_light":
                job_light_total_selected_runtime = rt
                job_light_avg_slowdown = slowdown
            elif wl_name == "job_complex":
                job_complex_total_selected_runtime = rt
                _job_complex_avg_slowdown = slowdown
            else:
                raise ValueError(f"Unknown workload name: {wl_name}")

        if "physical_operator" in str(test_path):
            with_idxs = "_w_idx" in str(test_path)
            rt, slowdown = evaluate_physical_plan_selection(
                predictions=rows,
                wandb_run=wandb_run,
                with_idxs=with_idxs,
                is_per_epoch=False,
            )
            if with_idxs:
                physop_idx_selected_runtime = rt
                physop_idx_avg_slowdown = slowdown
            else:
                physop_selected_runtime = rt
                physop_avg_slowdown = slowdown
        if "access_path_selection" in str(test_path):
            if "with_aggregate" in str(test_path):
                ap_w_aggregate = True
                wl_name = "w_aggregate"
            else:
                ap_w_aggregate = False
                wl_name = ""
            res = evaluate_access_path_selection(
                predictions=rows, wandb_run=wandb_run, is_per_epoch=False
            )
            if not ap_w_aggregate:
                (
                    accesspath_selected_runtime,
                    accesspath_avg_slowdown,
                ) = res
            else:
                (
                    accesspath_w_agg_selected_runtime,
                    _accesspath_w_agg_avg_slowdown,
                ) = res

    # aggregate metrics
    if (
        job_total_selected_runtime != -1
        and job_light_total_selected_runtime != -1
        and job_complex_total_selected_runtime != -1
        and physop_selected_runtime != -1
        and physop_idx_selected_runtime != -1
        and accesspath_selected_runtime != -1
        and accesspath_w_agg_selected_runtime != -1
    ):
        # if not all 5 workloads are present, do not aggregate
        assert job_total_selected_runtime is not None
        assert job_light_total_selected_runtime is not None
        assert job_complex_total_selected_runtime is not None
        assert physop_selected_runtime is not None
        assert physop_idx_selected_runtime is not None
        assert accesspath_selected_runtime is not None
        assert accesspath_w_agg_selected_runtime is not None

        join_order_total_selected_runtime = (
            job_total_selected_runtime
            + job_light_total_selected_runtime
            + job_complex_total_selected_runtime
        )
        access_path_total_selected_runtime = (
            accesspath_w_agg_selected_runtime + accesspath_selected_runtime
        )
        physop_total_selected_runtime = (
            physop_selected_runtime + physop_idx_selected_runtime
        )

        jo_ap_po_total_selected_runtime = (
            join_order_total_selected_runtime
            + access_path_total_selected_runtime
            + physop_total_selected_runtime
        )

        ac_po_total_selected_runtime = (
            access_path_total_selected_runtime + physop_total_selected_runtime
        )

        total_selected_runtime = (
            job_total_selected_runtime
            + job_light_total_selected_runtime
            + physop_selected_runtime
            + physop_idx_selected_runtime
            + accesspath_selected_runtime
        )

        avg_slowdown = (
            job_avg_slowdown
            + job_light_avg_slowdown
            + physop_avg_slowdown
            + physop_idx_avg_slowdown
            + accesspath_avg_slowdown
        ) / 5
        if wandb_run is not None:
            # We intentionally backfill the same final Qo totals across all epochs so they
            # appear as horizontal reference lines in W&B epoch-based plots.
            for epoch in range(config.epochs):
                wandb_run.log(
                    {
                        "qo_total_per_epoch/selected_runtime": total_selected_runtime,
                        "qo_total_per_epoch/avg_slowdown": avg_slowdown,
                        "qo_total_with_complex_per_epoch/selected_runtime": total_selected_runtime
                        + job_complex_total_selected_runtime,
                        "qo_total_per_epoch/jo_ap_po_selected_runtime": jo_ap_po_total_selected_runtime,
                        "qo_total_per_epoch/ac_po_selected_runtime": ac_po_total_selected_runtime,
                        "qo_total_per_epoch/jo_selected_runtime": join_order_total_selected_runtime,
                        "qo_total_per_epoch/ap_selected_runtime": access_path_total_selected_runtime,
                        "qo_total_per_epoch/po_selected_runtime": physop_total_selected_runtime,
                        "join_order_job_eval-per_epoch/total_selected_runtime": job_total_selected_runtime,
                        "join_order_job_light_eval-per_epoch/total_selected_runtime": job_light_total_selected_runtime,
                        "join_order_job_complex_eval-per_epoch/total_selected_runtime": job_complex_total_selected_runtime,
                        "physop_eval-per_epoch/total_selected_runtime": physop_selected_runtime,
                        "physop_eval_with_idxs-per_epoch/total_selected_runtime": physop_idx_selected_runtime,
                        "accesspath_eval-per_epoch/total_selected_runtime": accesspath_selected_runtime,
                        "accesspath_eval_w_aggregate-per_epoch/total_selected_runtime": accesspath_w_agg_selected_runtime,
                    },
                    step=epoch,
                )
                time.sleep(1)  # to avoid wandb logging issues
                if epoch % 10 == 0:
                    print(f"Logged epoch {epoch} Qo results to wandb")
                    time.sleep(10)  # to avoid wandb logging issues
                wandb_run.summary.update(
                    {
                        "qo_total/selected_runtime": total_selected_runtime,
                        "qo_total/avg_slowdown": avg_slowdown,
                        "qo_total_with_complex/selected_runtime": total_selected_runtime
                        + job_complex_total_selected_runtime,
                        "qo_total/jo_ap_po_selected_runtime": jo_ap_po_total_selected_runtime,
                        "qo_total/ac_po_selected_runtime": ac_po_total_selected_runtime,
                        "qo_total/jo_selected_runtime": join_order_total_selected_runtime,
                        "qo_total/ap_selected_runtime": access_path_total_selected_runtime,
                        "qo_total/po_selected_runtime": physop_total_selected_runtime,
                    }
                )
    return


def gen_postgres_cost_dataset(
    workload_runs: List[Path],
    cap_training_samples: Optional[bool],
    statistics_file: Path,
    featurization: Optional[Featurization],
    shuffle: bool,
    model_config: ModelConfig,
    val_ratio: float = 0.15,
    stratify_dataset: bool = False,
    stratify_per_wl_file: bool = False,
    is_test: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    plans, database_statistics = read_workload_runs(
        workload_runs,
        include_timeout_queries=model_config.include_timeout_queries and not is_test,
        stratify_dataset=stratify_dataset,
        stratify_per_wl_file=stratify_per_wl_file,
        limit_plans_overall=model_config.limit_plans_overall,
        limit_plans_per_file=None if is_test else model_config.limit_plans_per_file,
        augment_with_subplans=model_config.train_on_subplans and not is_test,
        remove_join_cond_from_plan=model_config.remove_join_cond_from_plan,
    )

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

    feature_vecs = []
    labels = []

    query_stats_dict = defaultdict(list)

    for sample_idx, plan in enumerate(plans):
        # append to all feature vectors and runtime labels
        feature_vecs.append([plan["plan_parameters"]["pg_cost"]])
        assert plan["plan_runtime_ms"] is not None, f"Plan runtime is None for {plan}"
        labels.append(plan["plan_runtime_ms"] / 1000)

        assign_additional_plan_info(
            plan=plan, db_statistics=database_statistics, db_schema_info=db_schema_info
        )

        # extract query stats
        extract_query_stats_for_plan(
            query_stats_dict,
            plan,
            sample_idx,
        )

    X_train, X_val, Y_train, Y_val = slice_train_val(
        cap_training_samples, feature_vecs, labels, plans, val_ratio, shuffle
    )

    return (X_train, X_val, Y_train, Y_val, query_stats_dict)


def gen_flattened_dataset(
    workload_runs: List[Path],
    statistics_file: Path,
    cap_training_samples: Optional[bool],
    featurization: Featurization,
    model_config: ModelConfig,
    val_ratio: float = 0.15,
    shuffle: bool = True,
    is_test: bool = False,
):
    feature_statistics = load_json(str(statistics_file), namespace=False)
    op_idx_dict = feature_statistics["op_name"]["value_dict"]
    no_ops = len(op_idx_dict)
    plans, database_statistics = read_workload_runs(
        workload_runs,
        include_timeout_queries=model_config.include_timeout_queries and not is_test,
        limit_plans_per_file=model_config.limit_plans_per_file,
        limit_plans_overall=model_config.limit_plans_overall,
        augment_with_subplans=model_config.train_on_subplans and not is_test,
        merge_nlj_and_inlj_to_same_op=model_config.merge_nlj_and_inlj_to_same_op,
        remove_join_cond_from_plan=model_config.remove_join_cond_from_plan,
        apply_card_correction_logic=model_config.apply_cardinality_estimation_correction_logic,
        card_correction_logic_params=model_config.cardinality_estimation_correction_logic_params,
    )

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

    feature_vecs = []
    labels = []

    query_stats_dict = defaultdict(list)

    for i, plan in enumerate(plans):
        assign_additional_plan_info(
            plan=plan, db_statistics=database_statistics, db_schema_info=db_schema_info
        )

        # number an operator occurs
        feature_num_vec = np.zeros(no_ops)
        # number of rows per op (estimated by postgres)
        feature_row_vec = np.zeros(no_ops)

        def extract_features(p, featurization):
            op_name = p["plan_parameters"]["op_name"]
            op_idx = op_idx_dict[op_name]
            feature_num_vec[op_idx] += 1
            if featurization.PLAN_FEATURE == "est_card":
                if "est_card" in p["plan_parameters"].keys():
                    feature_row_vec[op_idx] += p["plan_parameters"]["est_card"]
                else:
                    feature_row_vec[op_idx] += p["plan_parameters"]["est_rows"]
            elif featurization.PLAN_FEATURE == "act_card":
                if "act_card" in p["plan_parameters"].keys():
                    feature_row_vec[op_idx] += p["plan_parameters"]["act_card"]
                elif "act_rows" in p["plan_parameters"].keys():
                    feature_row_vec[op_idx] += p["plan_parameters"]["act_rows"]
                else:
                    feature_row_vec[op_idx] += p["plan_parameters"]["est_card"]

            for c in p["children"]:
                extract_features(c, featurization)

        extract_features(plan, featurization)

        # append to all feature vectors and runtime labels
        feature_vecs.append(np.concatenate((feature_num_vec, feature_row_vec)))
        labels.append(plan["plan_runtime_ms"] / 1000)

        # extract query stats
        extract_query_stats_for_plan(
            query_stats_dict,
            plan,
            i,
        )

    X_train, X_val, Y_train, Y_val = slice_train_val(
        cap_training_samples, feature_vecs, labels, plans, val_ratio, shuffle=shuffle
    )

    return (X_train, X_val, Y_train, Y_val, query_stats_dict)


def extract_card(p, act_card):
    if not act_card:
        if "est_card" in p["plan_parameters"].keys():
            card = p["plan_parameters"]["est_card"]
        else:
            card = p["plan_parameters"]["est_rows"]
    else:
        if "est_card" in p["plan_parameters"].keys():
            if "act_card" in p["plan_parameters"].keys():
                card = p["plan_parameters"]["act_card"]
            # fallback: for some plan operators, pg does not annotate act cards
            else:
                card = p["plan_parameters"]["est_card"]
        else:
            card = p["plan_parameters"]["act_rows"]
    return card


def gen_analytical_flattened_dataset(
    workload_runs,
    statistics_file,
    cap_training_samples,
    model_config: ModelConfig,
    val_ratio=0.15,
    act_card=False,
    stratify_dataset: bool = False,
    stratify_per_wl_file: bool = False,
    is_test: bool = False,
):
    feature_statistics = load_json(str(statistics_file), namespace=False)
    op_idx_dict = feature_statistics["op_name"]["value_dict"]
    no_ops = len(op_idx_dict)
    plans, database_statistics = read_workload_runs(
        workload_run_paths=workload_runs,
        include_timeout_queries=model_config.include_timeout_queries and not is_test,
        stratify_dataset=stratify_dataset,
        stratify_per_wl_file=stratify_per_wl_file,
        limit_plans_per_file=model_config.limit_plans_per_file,
        limit_plans_overall=model_config.limit_plans_overall,
        augment_with_subplans=model_config.train_on_subplans,
        merge_nlj_and_inlj_to_same_op=model_config.merge_nlj_and_inlj_to_same_op
        and not is_test,
        remove_join_cond_from_plan=model_config.remove_join_cond_from_plan,
    )
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
        feature_vecs.append(
            np.concatenate(
                (
                    feature_num_vec,
                    feature_row_vec,
                    feature_row_vec_left_input,
                    feature_row_vec_right_input,
                    feature_row_vec_prod_input,
                )
            )
        )
        labels.append(plan["plan_runtime_ms"] / 1000)
    X_train, X_val, Y_train, Y_val = slice_train_val(
        cap_training_samples, feature_vecs, labels, plans, val_ratio
    )
    return X_train, X_val, Y_train, Y_val
