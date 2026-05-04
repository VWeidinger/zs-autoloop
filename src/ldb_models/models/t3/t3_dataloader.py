from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
from ldb_models.classes import T3FlatModelConfig, T3ModelConfig, T3TreeModelConfig
from ldb_schema.load_schema import get_json_schema
from ldb_models.models.t3.benchmarked_query import BenchmarkedQuery
from ldb_models.models.t3.features import FeatureMapper
from ldb_models.models.t3.query_plan import QueryPlan
from ldb_models.models.tabular.slice_train_val import slice_train_val
from ldb_models.models.white_mamba.extract_query_stats import extract_query_stats_for_plan
from ldb_models.models.zeroshot.postgres_plan_batching import assign_additional_plan_info
from ldb_models.training.dataset.dataset_creation import (
    check_plan_runtime_validity,
    read_workload_runs,
)
from ldb_models.training.featurizations import Featurization


def annotate_op_id(parsed_plan: Dict, id: int = -1) -> int:
    id = id + 1
    parsed_plan["plan_parameters"]["op_id"] = id + 1  # 1-indexed
    parsed_plan["plan_parameters"]["analyze_plan_id"] = (
        id  # 0-indexed # not sure why they also have this info in the plans
    )

    # iterate over children
    if "input" in parsed_plan:
        id = annotate_op_id(parsed_plan["input"], id)

    if "left" in parsed_plan:
        id = annotate_op_id(parsed_plan["left"], id)

        assert "right" in parsed_plan
        # in case of an index NLJ the right side is pruned / can be none
        if parsed_plan["right"] is not None:
            id = annotate_op_id(parsed_plan["right"], id)

    return id


def add_table_stats_dict(db_statistics: Dict):
    table_stats = db_statistics["table_stats"]

    table_stats_dict = {}
    for entry in table_stats:
        table_stats_dict[entry["relname"]] = entry
    db_statistics["table_stats_dict"] = table_stats_dict


def construct_pseudo_pipeline(op_ids: List[int], runtime: int) -> Dict:
    assert runtime >= 0, f"{runtime}"

    return {
        "duration": runtime,
        "parallelism": None,
        "operators": op_ids,
        # add the orderingl ater
    }


def add_order_to_pipelines(pipelines: List[Dict]):
    # first pipeline is executed first, then all the others

    start_ts = 0

    # start iterating from the end
    for i in range(len(pipelines) - 1, -1, -1):
        pipeline = pipelines[i]

        pipeline["start"] = start_ts

        # add duration to start_ts
        start_ts += pipeline["duration"]
        pipeline["stop"] = start_ts


def extract_pipeline_infos(
    parsed_plan: Dict, pipelines: List[Dict], root: bool = True
) -> Tuple[List[int], float]:
    operator = parsed_plan["plan_parameters"]["op_name"]
    analyze_plan_id = parsed_plan["plan_parameters"][
        "analyze_plan_id"
    ]  # unique identifier of this node in the plan
    act_time = parsed_plan["plan_parameters"]["act_time"]  # total time of the subplan

    if root:
        pipeline_op_ids, child_pipelines_runtime = extract_pipeline_infos(
            parsed_plan, pipelines, root=False
        )

        total_plan_runtime = parsed_plan["plan_runtime_ms"]

        # add to pipelines - insert at begin: first pipeline is executed last
        pipelines.insert(
            0,
            construct_pseudo_pipeline(
                op_ids=pipeline_op_ids,
                runtime=total_plan_runtime - child_pipelines_runtime,
            ),
        )

        # make sure all pipeline runtimes sum up to total plan runtime
        pipeline_runtimes = [p["duration"] for p in pipelines]
        assert np.isclose(np.sum(pipeline_runtimes), total_plan_runtime), (
            f"{np.sum(pipeline_runtimes)} / {total_plan_runtime}"
        )

        add_order_to_pipelines(pipelines)

        return None, total_plan_runtime

    # call fn on child nodes
    if "input" in parsed_plan:
        child_stream, child_pipelines_runtime = extract_pipeline_infos(
            parsed_plan["input"], pipelines=pipelines, root=False
        )
        child_streams = [child_stream]
        child_pipelines_runtimes = [child_pipelines_runtime]
    elif "left" in parsed_plan:
        left, left_child_pipelines_runtime = extract_pipeline_infos(
            parsed_plan["left"], pipelines=pipelines, root=False
        )
        child_streams = [left]
        child_pipelines_runtimes = [left_child_pipelines_runtime]

        # in case of an index NLJ the right side is not represented as a separate pipeline
        if operator not in ["Index Nested Loop"]:
            right, right_child_pipelines_runtimes = extract_pipeline_infos(
                parsed_plan["right"], pipelines=pipelines, root=False
            )
            child_streams.append(right)
            child_pipelines_runtimes.append(right_child_pipelines_runtimes)
    else:
        child_streams = []
        child_pipelines_runtimes = [0]

    # make sure each element in child_pipelines_runtimes is > 0
    assert all([x >= 0 for x in child_pipelines_runtimes]), (
        f"{child_pipelines_runtimes} / {parsed_plan}"
    )

    # check for pipeline breaking operations
    if operator in ["Sort", "Aggregate", "Simple Aggregate", "Finalize Aggregate"]:
        assert len(child_streams) == 1, f"{len(child_streams)} / {parsed_plan}"
        op_ids = child_streams[0]
        op_ids = [analyze_plan_id] + op_ids

        # add to pipelines - insert at begin: first pipeline is executed last
        assert len(child_pipelines_runtimes) == 1, f"{child_pipelines_runtimes}"
        pipelines.insert(
            0,
            construct_pseudo_pipeline(
                op_ids=op_ids, runtime=act_time - child_pipelines_runtimes[0]
            ),
        )

        return [analyze_plan_id], act_time
    elif len(child_streams) == 2:
        # handle joins
        if operator in ["Hash Join"]:
            # hash pipeline finished
            hash_pipeline_runtime = parsed_plan["plan_parameters"]["left_runtime"]
            hash_pipeline_runtime = (
                hash_pipeline_runtime - child_pipelines_runtimes[0]
            )  # subtract child runtime

            hash_pipeline = [analyze_plan_id] + child_streams[0]
            pipelines.insert(
                0,
                construct_pseudo_pipeline(hash_pipeline, runtime=hash_pipeline_runtime),
            )

            # continue other pipeline
            pipeline_op_ids = [analyze_plan_id] + child_streams[1]
            return pipeline_op_ids, hash_pipeline_runtime + child_pipelines_runtimes[
                0
            ] + child_pipelines_runtimes[
                1
            ]  # runtime of hash pipeline + runtimes of all sub pipelines
        elif (
            operator in ["Index Nested Loop"]
        ):  # the right child is pruned beforehand - i.e. only one child / no need to break the pipeline or finalize a pipeline here
            assert False, "Index Nested Loop should not have two incoming pipelines!"
            # # index pipeline finished
            # index_pipeline = [analyze_plan_id] + child_streams[1]
            # pipelines.insert(0, construct_pseudo_pipeline(index_pipeline, runtime=runtime))
            #
            # # continue other pipeline
            # pipeline_op_ids = [analyze_plan_id] + child_streams[0]
            # return pipeline_op_ids
        elif operator in ["Merge Join"]:
            # merge join is a blocking operator - we need to finalize both pipelines
            left_pipeline = [analyze_plan_id] + child_streams[0]
            left_runtime = parsed_plan["plan_parameters"]["left_runtime"]
            left_runtime = (
                left_runtime - child_pipelines_runtimes[0]
            )  # subtract child pipeline runtime
            assert left_runtime >= 0, (
                f"{left_runtime} / {child_pipelines_runtimes}, {parsed_plan}"
            )
            pipelines.insert(
                0, construct_pseudo_pipeline(left_pipeline, runtime=left_runtime)
            )

            right_pipeline = [analyze_plan_id] + child_streams[1]
            right_runtime = parsed_plan["plan_parameters"]["right_runtime"]
            right_runtime = (
                right_runtime - child_pipelines_runtimes[1]
            )  # subtract child pipeline runtime
            if right_runtime == 0:
                # in case of a sort, we need to add a small runtime to the right pipeline (since we are creating a pipeline between sort and the merge)
                right_runtime += 0.00001
            assert right_runtime >= 0, (
                f"{right_runtime} / {child_pipelines_runtimes}, {parsed_plan}"
            )
            pipelines.insert(
                0, construct_pseudo_pipeline(right_pipeline, runtime=right_runtime)
            )

            # continue new pipeline
            pipeline_op_ids = [analyze_plan_id]
            return (
                pipeline_op_ids,
                left_runtime
                + right_runtime
                + child_pipelines_runtimes[0]
                + child_pipelines_runtimes[1],
            )  # runtime of both pipelines + runtimes of all sub pipelines
        elif operator in ["Nested Loop"]:
            # left child is the inner pipeline
            left_pipeline = [analyze_plan_id] + child_streams[0]
            left_runtime = parsed_plan["plan_parameters"]["left_runtime"]
            left_runtime = (
                left_runtime - child_pipelines_runtimes[0]
            )  # subtract child pipeline runtime

            if left_runtime < 0 and left_runtime > -0.05:
                # sometimes we have small inconsistencies. We make it fit.
                left_runtime = 0

            assert left_runtime >= 0, (
                f"{left_runtime} / {child_pipelines_runtimes}, {parsed_plan}"
            )
            pipelines.insert(
                0, construct_pseudo_pipeline(left_pipeline, runtime=left_runtime)
            )

            # right input is the outer pipeline (i.e. we model this as probe side)
            right_pipeline = child_streams[1]
            pipeline_op_ids = [analyze_plan_id] + right_pipeline
            return pipeline_op_ids, left_runtime + child_pipelines_runtimes[
                0
            ] + child_pipelines_runtimes[
                1
            ]  # runtime of both pipelines + runtimes of all sub pipelines
        else:
            # other joins not supported
            raise Exception
    elif len(child_streams) == 0:
        # leaf node
        return [analyze_plan_id], 0
    else:
        assert len(child_streams) == 1, f"{parsed_plan}"
        pipeline_op_ids = [analyze_plan_id] + child_streams[0]

        return pipeline_op_ids, child_pipelines_runtimes[0]


class UnsupportedOperatorException(Exception):
    def __init__(self, message):
        super().__init__(message)


def prune_ops(node: Dict):
    # prune specific operators
    if node["plan_parameters"]["op_name"] in ["Materialize"]:
        # remove the materialize operator
        assert len(node["children"]) == 1, (
            f"Materialize operator should have only one child: {node}"
        )
        node = node["children"][0]

        node = prune_ops(node)

    return node


def rewrite_children(parsed_plan: Dict):
    # make sure in hashjoin children the first side is always the hash

    if "children" in parsed_plan and len(parsed_plan["children"]) > 0:
        children = parsed_plan.pop("children")
        if len(children) == 2:
            # join
            if parsed_plan["plan_parameters"]["op_name"] in ["Hash Join"]:
                # make sure first child is of op_name hash
                child_op_names = [c["plan_parameters"]["op_name"] for c in children]

                assert "Hash" in child_op_names, (
                    f"No hash in children found: {child_op_names} / {parsed_plan}"
                )
                assert child_op_names[1] == "Hash", (
                    f"Postgres usually gives the hash at the second position: {child_op_names} / {parsed_plan}"
                )

                # prune the hash side
                hash_op = children[1]
                assert len(hash_op["children"]) == 1, (
                    f"Hash operator should have only one child: {hash_op}"
                )
                children[1] = hash_op["children"][0]

                # annotate hash pipeline runtime to this node
                parsed_plan["plan_parameters"]["left_runtime"] = hash_op[
                    "plan_parameters"
                ]["act_time"]

                # set children
                left = children[1]  # build side / hash side
                right = children[0]  # probe side

            elif parsed_plan["plan_parameters"]["op_name"] in ["Index Nested Loop"]:
                # make sure first child is of op_name index
                child_op_names = [c["plan_parameters"]["op_name"] for c in children]

                assert "Index Scan" in child_op_names or "Index Only Scan", (
                    f"No index in children found: {child_op_names} / {parsed_plan}"
                )
                assert child_op_names[1] in ["Index Scan", "Index Only Scan"], (
                    f"Postgres usually gives the index at the second position: {child_op_names} / {parsed_plan}"
                )

                # copy information from index scan to this node - we will prune the index scan
                parsed_plan["plan_parameters"]["idx_scan"] = children[1][
                    "plan_parameters"
                ]

                # set children
                left = children[0]  # probe side
                right = children[
                    1
                ]  # index side - only to convey information - no pipeline will be constructed
            elif parsed_plan["plan_parameters"]["op_name"] in ["Merge Join"]:
                # make sure botch children are sorts
                child_op_names = [c["plan_parameters"]["op_name"] for c in children]

                # it not always has to be a sort - e.g. if the data is already sorted
                # assert 'Sort' in child_op_names, f'No sort in children found: {child_op_names} / {parsed_plan}'
                # assert child_op_names[0] == 'Sort', f'Both children have to be a sort: {child_op_names} / {parsed_plan}'
                # assert child_op_names[1] == 'Sort', f'Both children have to be a sort: {child_op_names} / {parsed_plan}'

                left = children[0]  # left sort
                right = children[1]  # right sort
            elif parsed_plan["plan_parameters"]["op_name"] in ["Nested Loop"]:
                # swap left/right: in pg we have the outer on the left and the inner on the right
                # but we want to represent the inner as the "build" / blocking side and the outer as the "probe" side
                # therefor (to be consistent with HashJoin/...) we swap them - build side is always left
                left = children[1]  # left side
                right = children[0]  # right side
            else:
                raise UnsupportedOperatorException(
                    f"Operator not supported: {parsed_plan['plan_parameters']['op_name']}"
                )

            # copy left runtime / right runtime
            if "left_runtime" not in parsed_plan["plan_parameters"]:
                # hash join is already setting this parameter
                left_time = left["plan_parameters"]["act_time"]
                if left["plan_parameters"]["op_name"] in ["Materialize"]:
                    # materialize operator runtime annotation does not include child runtimes - we need to take the child runtime
                    left_time = left[
                        "children"
                    ][
                        0
                    ][
                        "plan_parameters"
                    ][
                        "act_time"
                    ]  # + left_time - do not take materialzie time into account. Mostly they are wrong / off / inconsistent
                    assert left_time < parsed_plan["plan_parameters"]["act_time"], (
                        f"{parsed_plan} / {left}"
                    )
                assert left_time >= 0, left
                parsed_plan["plan_parameters"]["left_runtime"] = left_time
            assert "right_runtime" not in parsed_plan["plan_parameters"], (
                f"Right runtime already set: {parsed_plan}"
            )
            parsed_plan["plan_parameters"]["right_runtime"] = right["plan_parameters"][
                "act_time"
            ]

            # prune specific operators
            left = prune_ops(left)
            right = prune_ops(right)

            # annotate left/right
            parsed_plan["left"] = left
            parsed_plan["right"] = right

            # recursively call on left/right
            rewrite_children(left)
            rewrite_children(right)

        else:
            assert len(children) == 1, (
                f"Operator not supported: {parsed_plan['plan_parameters']['op_name']}"
            )

            # extract op_name
            op_name = parsed_plan["plan_parameters"]["op_name"]

            if op_name in ["Finalize Aggregate"]:
                # remove all children until we find a non-gather / non-partial aggregate op
                def find_non_gather(node):
                    if node["plan_parameters"]["op_name"] in [
                        "Gather",
                        "Partial Aggregate",
                    ]:
                        children = node["children"]
                        assert len(children) == 1, (
                            f"Gather / Partial Aggregate should have only one child: {node}"
                        )
                        return find_non_gather(children[0])
                    else:
                        return node

                # skip both gather and partial aggregate
                input = find_non_gather(children[0])

                # rename the operator
                parsed_plan["plan_parameters"]["op_name"] = "Aggregate"
            elif op_name in ["Simple Aggregate"]:
                child_op = children[0]["plan_parameters"]["op_name"]
                assert child_op not in ["Gather", "Partial Aggregate"], (
                    f"Unexpected operator: {parsed_plan}"
                )
                input = children[0]  # there is no left/right

                # rename the operator
                parsed_plan["plan_parameters"]["op_name"] = "Aggregate"
            else:
                input = children[0]  # there is no left/right

            # copy intput runtime
            parsed_plan["plan_parameters"]["input_runtime"] = input["plan_parameters"][
                "act_time"
            ]

            # prune ops
            input = prune_ops(input)

            parsed_plan["input"] = input

            # recursively call on left
            rewrite_children(input)


def assemble_x_y(queries: List[BenchmarkedQuery], model_config: T3ModelConfig):
    feature_mapper = FeatureMapper()
    x_vectors = []
    y_values = []

    if isinstance(model_config, T3TreeModelConfig):
        for query in queries:
            for x, y in query.get_pipeline_runtime_data(feature_mapper):
                x_vectors.append(x)
                y_values.append(y)
        x = np.vstack(x_vectors)
        y = np.array(y_values)
    elif isinstance(model_config, T3FlatModelConfig):
        for query in queries:
            current_x = []
            current_y = []
            for x, y in query.get_pipeline_runtime_data(feature_mapper):
                current_x.append(x)
                current_y.append(y)
            x_vectors.append(np.sum(current_x, axis=0))
            y_values.append(float(np.sum(current_y)))
        x = np.vstack(x_vectors)
        y = np.array(y_values)
    else:
        # default t3 model (t3 per tuple)
        for query in queries:
            for x, y in query.get_per_tuple_pipeline_runtime_data(feature_mapper):
                if np.any(x != 0):
                    x_vectors.append(x)
                    y_values.append(y)
        x = np.vstack(x_vectors)
        y = np.array(y_values)
        # log scale improves training
        y = np.maximum(y, 1e-15)
        y = -np.log(y)

    return x, y, feature_mapper


def gen_t3_dataset(
    workload_runs: List[Path],
    statistics_file: Path,
    cap_training_samples: Optional[bool],
    featurization: Featurization,
    val_ratio: float = 0.15,
    shuffle: bool = True,
    model_config: T3ModelConfig = None,
    verbose: bool = False,
    return_dataset: bool = True,
    is_test: bool = False,
):
    # read files
    plans, db_statistics = read_workload_runs(
        workload_run_paths=workload_runs,
        include_timeout_queries=model_config.include_timeout_queries and not is_test,
        limit_plans_per_file=model_config.limit_plans_per_file,
        limit_plans_overall=model_config.limit_plans_overall,
        augment_with_subplans=model_config.train_on_subplans and not is_test,
        merge_nlj_and_inlj_to_same_op=model_config.merge_nlj_and_inlj_to_same_op,
        remove_join_cond_from_plan=model_config.remove_join_cond_from_plan,
        apply_card_correction_logic=model_config.apply_cardinality_estimation_correction_logic,
        card_correction_logic_params=model_config.cardinality_estimation_correction_logic_params,
    )

    # augment database statistics
    for run_file_id, stats in db_statistics.items():
        assert isinstance(stats, Dict), f"Expected dict, got {type(stats)}"
        add_table_stats_dict(stats)

    # extract database schema information
    db_names = dict(
        (run_file_id, value["database_name"])
        for run_file_id, value in db_statistics.items()
    )
    db_schema_info = dict(
        (run_file_id, get_json_schema(db_name))
        for run_file_id, db_name in db_names.items()
        if db_name != "unknown"
    )

    # create dataset info
    benchmark_queries = []
    labels = []
    query_stats_dict = defaultdict(list)

    filtered_plans = []

    skip_plan_inconsistent_runtimes_ctr = 0
    for i, plan in enumerate(plans):
        if not check_plan_runtime_validity(
            plan, parent_runtime=plan["plan_runtime_ms"]
        ):
            skip_plan_inconsistent_runtimes_ctr += 1
            continue

        assign_additional_plan_info(
            plan=plan, db_statistics=db_statistics, db_schema_info=db_schema_info
        )

        # extract query stats
        extract_query_stats_for_plan(
            query_stats_dict,
            plan,
            i,
        )

        # reorder hashjoin children (hash first child)
        try:
            rewrite_children(plan)
        except UnsupportedOperatorException as e:
            print(f"Unsupported Operator: {e}")
            continue

        # assign unique IDs to each operator in the plan
        annotate_op_id(plan)

        # extract pipeline infos
        pipelines = []
        extract_pipeline_infos(plan, pipelines=pipelines)

        # create BenchmarkQuery
        qp = QueryPlan(
            plan,
            card_type=model_config.card_type,
            db_statistics=db_statistics[plan["run_file_id"]],
        )
        qp.build_pipelines(pipelines=pipelines)

        # extract runtimes
        runtime = plan["plan_runtime_ms"]
        sql = plan["sql"]

        # create BenchmarkQuery
        benchmark_queries.append(
            BenchmarkedQuery(
                query_plan=qp,
                total_runtimes=[runtime],
                name=None,
                query_text=sql,
                query_category=None,
            )
        )

        labels.append(plan["plan_runtime_ms"] / 1000)
        filtered_plans.append(plans)

    print(
        f"Skipped {skip_plan_inconsistent_runtimes_ctr}/{len(plans)} plans due to inconsistent runtimes"
    )

    labels = np.asarray(labels)

    assert len(labels) == len(filtered_plans)
    assert len(labels) == len(benchmark_queries)

    if return_dataset:
        # assemble x, y
        print(f"Num queries: {len(benchmark_queries)}")
        X, Y, feature_mapper = assemble_x_y(benchmark_queries, model_config)

        # slice into train/eval
        X_train, X_val, Y_train, Y_val = slice_train_val(
            cap_training_samples, X, Y, filtered_plans, val_ratio, shuffle=shuffle
        )

        # create datasets
        param = {"objective": "mape", "verbose": 2 if verbose else -1}
        if isinstance(model_config, T3TreeModelConfig) or isinstance(
            model_config, T3FlatModelConfig
        ):
            train_data = lgb.Dataset(X_train, label=Y_train, params=param)
            val_data = lgb.Dataset(
                X_val, label=Y_val, reference=train_data, params=param
            )
        else:
            # default t3 per tuple tree model
            train_data = lgb.Dataset(
                X_train,
                label=Y_train,
                feature_name=FeatureMapper.get_names(),
                params=param,
            )
            val_data = lgb.Dataset(
                X_val, label=Y_val, reference=train_data, params=param
            )

        return (train_data, val_data, Y_train, Y_val, query_stats_dict)
    else:
        return (benchmark_queries, None, labels, None, query_stats_dict)
