from pathlib import Path
from typing import List, Dict, Optional

import lightgbm as lgb
import numpy as np

from classes.classes import T3ModelConfig, T3TreeModelConfig, T3FlatModelConfig
from models.t3.benchmarked_query import BenchmarkedQuery
from models.t3.features import FeatureMapper
from models.t3.query_plan import QueryPlan
from models.tabular.slice_train_val import slice_train_val
from models.zeroshot.postgres_plan_batching import assign_additional_plan_info
from training.dataset.dataset_creation import read_workload_runs
from training.featurizations import Featurization


def annotate_op_id(parsed_plan: Dict, id: int = -1) -> int:
    id = id + 1
    parsed_plan['plan_parameters']['op_id'] = id + 1  # 1-indexed
    parsed_plan['plan_parameters'][
        'analyze_plan_id'] = id  # 0-indexed # not sure why they also have this info in the plans

    # iterate over children
    if 'input' in parsed_plan:
        id = annotate_op_id(parsed_plan['input'], id)

    if 'left' in parsed_plan:
        id = annotate_op_id(parsed_plan['left'], id)
        id = annotate_op_id(parsed_plan['right'], id)

    return id


def add_table_stats_dict(db_statistics: Dict):
    table_stats = db_statistics['table_stats']

    table_stats_dict = {}
    for entry in table_stats:
        table_stats_dict[entry['relname']] = entry
    db_statistics['table_stats_dict'] = table_stats_dict


def construct_pseudo_pipeline(op_ids: List[int], runtime: int) -> Dict:
    return {
        'start': max(op_ids) * 10,  # we do not know the real ordering in pg - we start with the left most pipeline
        'stop': max(op_ids) * 10 + 5,
        'duration': runtime,
        'parallelism': None,
        'operators': op_ids
    }


def extract_pipeline_infos(parsed_plan: Dict, pipelines: List[Dict], root: bool = True) -> List[int]:
    operator = parsed_plan['plan_parameters']['op_name']
    analyze_plan_id = parsed_plan['plan_parameters']['analyze_plan_id']  # unique identifier of this node in the plan
    runtime = parsed_plan['plan_parameters']['act_time']  # total time of the subplan

    if root:
        pipeline_op_ids = extract_pipeline_infos(parsed_plan, pipelines, root=False)
        # add to pipelines - insert at begin: first pipeline is executed last
        pipelines.insert(0, construct_pseudo_pipeline(op_ids=pipeline_op_ids, runtime=runtime))

        return None

    # call fn on child nodes
    if 'input' in parsed_plan:
        child_streams = [extract_pipeline_infos(parsed_plan['input'], pipelines=pipelines, root=False)]
    elif 'left' in parsed_plan:
        left = extract_pipeline_infos(parsed_plan['left'], pipelines=pipelines, root=False)
        right = extract_pipeline_infos(parsed_plan['right'], pipelines=pipelines, root=False)
        child_streams = [left, right]
    else:
        child_streams = []

    # check for pipeline breaking operations
    if operator in ['Sort', 'Aggregate', 'Simple Aggregate', 'Finalize Aggregate']:
        assert len(child_streams) == 1
        op_ids = child_streams[0]
        op_ids = [analyze_plan_id] + op_ids

        # add to pipelines - insert at begin: first pipeline is executed last
        pipelines.insert(0, construct_pseudo_pipeline(op_ids=op_ids, runtime=runtime))

        return [analyze_plan_id]
    elif len(child_streams) == 2:
        # handle joins
        if operator in ['Hash Join']:
            # hash pipeline finished
            assert 'hash_pipeline_runtime' in parsed_plan['plan_parameters']
            hash_pipeline_runtime = parsed_plan['plan_parameters']['hash_pipeline_runtime']
            hash_pipeline = [analyze_plan_id] + child_streams[0]
            pipelines.insert(0, construct_pseudo_pipeline(hash_pipeline, runtime=hash_pipeline_runtime))

            # continue other pipeline
            pipeline_op_ids = [analyze_plan_id] + child_streams[1]
            return pipeline_op_ids
        elif operator in ['Index Nested Loop']:  # TODO - not sure if this makes sense!
            # index pipeline finished
            index_pipeline = [analyze_plan_id] + child_streams[0]
            pipelines.insert(0, construct_pseudo_pipeline(index_pipeline, runtime=runtime))

            # continue other pipeline
            pipeline_op_ids = [analyze_plan_id] + child_streams[1]
            return pipeline_op_ids
        else:
            # other joins not supported
            raise Exception
    elif len(child_streams) == 0:
        # leaf node
        return [analyze_plan_id]
    else:
        assert len(child_streams) == 1, f'{parsed_plan}'
        pipeline_op_ids = [analyze_plan_id] + child_streams[0]

        return pipeline_op_ids


def prune_nodes_to_ignore(parsed_plan: Dict):
    op_name = parsed_plan['plan_parameters']['op_name']

    # recursively clean plan
    if 'left' in parsed_plan:
        parsed_plan['left'] = prune_nodes_to_ignore(parsed_plan['left'])
        parsed_plan['right'] = prune_nodes_to_ignore(parsed_plan['right'])
    if 'input' in parsed_plan:
        parsed_plan['input'] = prune_nodes_to_ignore(parsed_plan['input'])

    # potentially prune this node
    if op_name in ['Aggregate', 'Simple Aggregate', 'Hash']:

        assert 'input' in parsed_plan and not 'left' in parsed_plan and not 'right' in parsed_plan, f'Unexpected operator: {parsed_plan}'
        child = parsed_plan['input']

        # args to copy over if existing
        arg_names = ['plan_runtime_ms', 'sql', 'database_id']
        for arg in arg_names:
            if arg in parsed_plan:
                child[arg] = parsed_plan[arg]

        return child

    return parsed_plan


class UnsupportedOperatorException(Exception):
    def __init__(self, message):
        super().__init__(message)


def rewrite_children(parsed_plan: Dict):
    # make sure in hashjoin children the first side is always the hash

    if 'children' in parsed_plan and len(parsed_plan['children']) > 0:
        children = parsed_plan.pop('children')
        if len(children) == 2:
            # join
            if parsed_plan['plan_parameters']['op_name'] in ['Hash Join']:
                # make sure first child is of op_name hash
                child_op_names = [c['plan_parameters']['op_name'] for c in children]

                assert 'Hash' in child_op_names, f'No hash in children found: {child_op_names} / {parsed_plan}'
                assert child_op_names[
                           1] == 'Hash', f'Postgres usually gives the hash at the second position: {child_op_names} / {parsed_plan}'

                # set children
                left = children[1]  # build side
                right = children[0]  # probe side

                # annotate hash pipeline runtime to this node
                parsed_plan['plan_parameters']['hash_pipeline_runtime'] = left['plan_parameters']['act_time']
            elif parsed_plan['plan_parameters']['op_name'] in ['Index Nested Loop']:
                # make sure first child is of op_name index
                child_op_names = [c['plan_parameters']['op_name'] for c in children]

                assert 'Index Scan' in child_op_names or 'Index Only Scan', f'No index in children found: {child_op_names} / {parsed_plan}'
                assert child_op_names[1] in ['Index Scan',
                                             'Index Only Scan'], f'Postgres usually gives the index at the second position: {child_op_names} / {parsed_plan}'

                # set children
                left = children[0]  # probe side
                right = children[1]  # index side

                raise UnsupportedOperatorException(
                    f'Operator not supported: {parsed_plan["plan_parameters"]["op_name"]}')
            elif parsed_plan['plan_parameters']['op_name'] in ['Merge Join']:
                raise UnsupportedOperatorException(
                    f'Operator not supported: {parsed_plan["plan_parameters"]["op_name"]}')
            else:
                raise UnsupportedOperatorException(f'Operator not supported: {parsed_plan["plan_parameters"]["op_name"]}')

            # annotate left/right
            parsed_plan['left'] = left
            parsed_plan['right'] = right

            # recursively call on left/right
            rewrite_children(left)
            rewrite_children(right)

        else:
            assert len(children) == 1, f'Operator not supported: {parsed_plan["plan_parameters"]["op_name"]}'
            input = children[0]  # there is no left/right
            parsed_plan['input'] = input

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


def gen_t3_dataset(workload_runs: List[Path],
                   statistics_file: Path,
                   cap_training_samples: Optional[bool],
                   featurization: Featurization,
                   limit_plans_per_file: int,
                   limit_plans_overall: int,
                   val_ratio: float = 0.15,
                   shuffle: bool = True,
                   include_timeout_queries: bool = False,
                   model_config: T3ModelConfig = None,
                   verbose: bool = False,
                   return_dataset: bool = True
                   ):
    # read files
    plans, db_statistics = read_workload_runs(workload_run_paths=workload_runs,
                                              include_timeout_queries=include_timeout_queries,
                                              limit_plans_per_file=limit_plans_per_file,
                                              limit_plans_overall=limit_plans_overall)

    # augment database statistics
    for db_id, stats in db_statistics.items():
        assert isinstance(stats, Dict), f'Expected dict, got {type(stats)}'
        add_table_stats_dict(stats)

    # create dataset info
    benchmark_queries = []
    labels = []
    selectivities = []
    join_ops = []
    scan_ops = []

    sqls, hints = [], []  # Required to later evaluate against query optimization

    filtered_plans = []
    for plan in plans:
        assign_additional_plan_info(plan=plan, db_statistics=db_statistics)

        # reorder hashjoin children (hash first child)
        try:
            rewrite_children(plan)
        except UnsupportedOperatorException as e:
            # print(f'Unsupported Operator: {e}')
            continue

        sqls.append(plan["sql"])
        hints.append(plan["hint"])
        selectivities += plan["selectivity"]
        join_ops += plan["join_op"]
        scan_ops += plan["scan_op"]

        # prune nodes to ignore - prune only afterward HJ reordering - we still need the hash op
        plan = prune_nodes_to_ignore(plan)

        # assign unique IDs to each operator in the plan
        annotate_op_id(plan)

        # extract pipeline infos
        pipelines = []
        extract_pipeline_infos(plan, pipelines=pipelines)

        # create BenchmarkQuery
        qp = QueryPlan(plan, card_type=model_config.card_type, db_statistics=db_statistics[plan['database_id']])
        qp.build_pipelines(pipelines=pipelines)

        # extract runtimes
        runtime = plan['plan_runtime_ms']
        sql = plan['sql']

        # create BenchmarkQuery
        benchmark_queries.append(
            BenchmarkedQuery(query_plan=qp, total_runtimes=[runtime], name=None, query_text=sql, query_category=None))

        labels.append(plan['plan_runtime_ms'] / 1000)
        filtered_plans.append(plans)

    labels = np.asarray(labels)

    assert len(labels) == len(filtered_plans)
    assert len(labels) == len(benchmark_queries)
    assert len(labels) == len(sqls)
    assert len(labels) == len(hints)
    assert len(labels) == len(selectivities)
    assert len(labels) == len(join_ops)
    assert len(labels) == len(scan_ops)

    if return_dataset:
        # assemble x, y
        print(f'Num queries: {len(benchmark_queries)}')
        X, Y, feature_mapper = assemble_x_y(benchmark_queries, model_config)

        # slice into train/eval
        X_train, X_val, Y_train, Y_val = slice_train_val(cap_training_samples, X, Y, filtered_plans, val_ratio,
                                                         shuffle=shuffle)

        # create datasets
        param = {"objective": "mape", "verbose": 2 if verbose else -1}
        if isinstance(model_config, T3TreeModelConfig) or isinstance(model_config, T3FlatModelConfig):
            train_data = lgb.Dataset(X_train, label=Y_train, params=param)
            val_data = lgb.Dataset(X_val, label=Y_val, reference=train_data, params=param)
        else:
            # default t3 per tuple tree model
            train_data = lgb.Dataset(X_train, label=Y_train, feature_name=FeatureMapper.get_names(), params=param)
            val_data = lgb.Dataset(X_val, label=Y_val, reference=train_data, params=param)

        return train_data, val_data, Y_train, Y_val, sqls, hints, selectivities, scan_ops, join_ops
    else:
        return benchmark_queries, None, labels, None, sqls, hints, selectivities, scan_ops, join_ops
