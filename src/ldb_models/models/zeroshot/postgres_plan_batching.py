import collections
from typing import List, Tuple

import dgl
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler

from training.dataset.operator import Operator
from training.featurizations import Featurization
from training.preprocessing.feature_statistics import FeatureType


def encode(column, plan_params, feature_statistics):
    #  fallback in case actual cardinality is not in plan parameters
    if column == 'act_card' and column not in plan_params:
        value = 0
    else:
        value = plan_params[column]
    if feature_statistics[column].get('type') == str(FeatureType.numeric):
        enc_value = feature_statistics[column]['scaler'].transform(np.array([[value]])).item()
    elif feature_statistics[column].get('type') == str(FeatureType.categorical):
        value_dict = feature_statistics[column]['value_dict']
        assert len(list(value_dict.keys())) > 1, f'Value dict has only one key: {value_dict} for feature {column}'
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        assert str(value) in value_dict, f'{value} not in {value_dict.keys()}'
        enc_value = value_dict[str(value)]
    else:
        raise NotImplementedError
    return enc_value


def assign_additional_plan_info(plan: dict, db_statistics: dict):
    selectivity_list = []
    scan_op_list = []
    join_op_list = []

    # extract additional info
    if plan['num_tables'] == 1:
        # compute selectivity
        def get_leaf_node(p):
            if len(p['children']) == 0:
                return p
            else:
                return get_leaf_node(p['children'][0])

        leaf_node = get_leaf_node(plan)
        scan_op = leaf_node['plan_parameters']['op_name']
        out_card = leaf_node['plan_parameters']['act_card']

        # compute base table size
        base_table_name = leaf_node['plan_parameters']['table_name']
        base_table_id = leaf_node['plan_parameters']['table_id']

        assert 'database_id' in plan, f'No database_id in plan: {plan}'
        assert plan['database_id'] in db_statistics
        assert 'table_stats' in db_statistics[plan['database_id']]

        assert 'reltuples' in db_statistics[plan['database_id']]['table_stats'][
            base_table_id], f'table_size not in {db_statistics[plan["database_id"]]["table_stats"][base_table_id].keys()}'

        base_table_size = db_statistics[plan['database_id']]['table_stats'][base_table_id]['reltuples']

        # compute selectivity
        selectivity_list.append(out_card / base_table_size)
        scan_op_list.append(scan_op)
        join_op_list.append(None)

    elif plan['num_tables'] == 2:
        # compute join info
        selectivity_list.append(None)
        scan_op_list.append(None)

        def get_top_join_node(p):
            if len(p['children']) == 2:
                return p
            elif len(p['children']) == 1:
                return get_top_join_node(p['children'][0])
            else:
                raise ValueError('Invalid plan structure')

        join_node = get_top_join_node(plan)
        join_op_list.append(join_node['plan_parameters']['op_name'])

    else:
        selectivity_list.append(None)
        scan_op_list.append(None)
        join_op_list.append(None)

    plan["selectivity"] = selectivity_list
    plan["scan_op"] = scan_op_list
    plan["join_op"] = join_op_list
    return


def plan_to_graph(node, database_id, plan_depths, plan_features, plan_to_plan_edges, db_statistics, feature_statistics,
                  filter_to_plan_edges, predicate_col_features, output_column_to_plan_edges, output_column_features,
                  column_to_output_column_edges, column_features, table_features, table_to_plan_edges,
                  output_column_idx, column_idx, table_idx, plan_featurization, predicate_depths, intra_predicate_edges,
                  logical_preds, parent_node_id=None, depth=0, skip_top_aggregation:bool=False):

    plan_params = node["plan_parameters"]

    if 'Aggregate' in plan_params['op_name'] and skip_top_aggregation:
        assert len(node["children"]) == 1, f'Aggregation has more than one child: {node}'
        plan_to_graph(node["children"][0], database_id, plan_depths, plan_features, plan_to_plan_edges, db_statistics,
                      feature_statistics, filter_to_plan_edges, predicate_col_features, output_column_to_plan_edges,
                      output_column_features, column_to_output_column_edges, column_features, table_features,
                      table_to_plan_edges, output_column_idx, column_idx, table_idx, plan_featurization,
                      predicate_depths, intra_predicate_edges, logical_preds, parent_node_id=parent_node_id, depth=depth, skip_top_aggregation=False)
        return

    # add plan features
    curr_plan_features = [encode(column, plan_params, feature_statistics) for column in
                          plan_featurization.PLAN_FEATURES]
    plan_features.append(curr_plan_features)

    # compute plan node id
    plan_node_id = len(plan_depths)
    plan_depths.append(depth)

    # encode output columns which can in turn have several columns as a product in the aggregation
    output_columns = plan_params.get('output_columns')
    if output_columns is not None:
        for output_column in output_columns:
            output_column_id_list = [col['col_id'] for col in output_column['columns']]
            output_column_node_id = output_column_idx.get(
                (output_column["aggregation"], str(output_column_id_list), database_id))

            # if not, create
            if output_column_node_id is None:
                curr_output_column_features = [encode(column, output_column, feature_statistics)
                                               for column in plan_featurization.OUTPUT_COLUMN_FEATURES]

                output_column_node_id = len(output_column_features)
                output_column_features.append(curr_output_column_features)
                output_column_idx[(output_column["aggregation"], str(output_column_id_list), database_id)] \
                    = output_column_node_id

                # featurize product of columns if there are any
                db_column_features = db_statistics[database_id]["column_stats"]
                assert db_column_features is not None
                for column in output_column["columns"]:
                    column_node_id = column_idx.get((str(column['col_id']), database_id))
                    if column_node_id is None:
                        assert isinstance(column, dict), f'Column is not a dict: {column} / {node}'
                        try:
                            curr_column_features = [
                                encode(feature_name, db_column_features[column["col_id"]], feature_statistics)
                                for feature_name in plan_featurization.COLUMN_FEATURES]
                        except Exception as e:
                            print(
                                f'Error in column: {column} / {db_column_features[column["col_id"]]} / {str(feature_statistics)[:100]}',
                                flush=True)
                            raise e
                        column_node_id = len(column_features)
                        column_features.append(curr_column_features)
                        column_idx[(str(column['col_id']), database_id)] = column_node_id
                    column_to_output_column_edges.append((column_node_id, output_column_node_id))

            # in any case add the corresponding edge
            output_column_to_plan_edges.append((output_column_node_id, plan_node_id))

    # filter_columns (we do not reference the filter columns to columns since we anyway have to create a node per filter
    #  node)
    filter = plan_params.get('filter')
    if filter is not None:
        assert 'is_index_cond' in filter, f'No is_index_cond in filter: {filter} / {node}'
        db_column_features = db_statistics[database_id]["column_stats"]
        try:
            parse_predicates(db_column_features, feature_statistics, filter, filter_to_plan_edges,
                         plan_featurization, predicate_col_features, predicate_depths, intra_predicate_edges,
                         logical_preds, plan_node_id=plan_node_id)
        except Exception as e:
            print(f'Error in filter: {filter}', flush=True)
            raise e

    # featurize join conditions
    join = plan_params.get('join')
    if join is not None:
        assert 'is_index_cond' in join, f'No is_index_cond in join: {join} / {node}'
        db_column_features = db_statistics[database_id]["column_stats"]
        try:
            parse_predicates(db_column_features, feature_statistics, join, filter_to_plan_edges,
                         plan_featurization, predicate_col_features, predicate_depths, intra_predicate_edges,
                         logical_preds, plan_node_id=plan_node_id, is_join=True)
        except Exception as e:
            print(f'Error in join: {join}', flush=True)
            raise e

    # tables
    table_id = plan_params.get('table_id')
    if 'Scan' in plan_params['op_name'] and plan_params['op_name'] != 'Bitmap Index Scan':
        assert table_id is not None, f'No table in scan: {plan_params}'
    if table_id is not None:
        assert isinstance(table_id, int), f'Table id is not an int: {table_id} / {plan_params}'
        table_node_id = table_idx.get((table_id, database_id))
        db_table_statistics = db_statistics[database_id]['table_stats']

        if table_node_id is None:
            curr_table_features = [encode(feature_name, db_table_statistics[table_id], feature_statistics)
                                   for feature_name in plan_featurization.TABLE_FEATURES]
            table_node_id = len(table_features)
            table_features.append(curr_table_features)
            table_idx[(table_id, database_id)] = table_node_id

        table_to_plan_edges.append((table_node_id, plan_node_id))

    # add edge to parent
    if parent_node_id is not None:
        plan_to_plan_edges.append((plan_node_id, parent_node_id))

    # continue recursively
    for c in node["children"]:
        plan_to_graph(c, database_id, plan_depths, plan_features, plan_to_plan_edges, db_statistics, feature_statistics,
                      filter_to_plan_edges, predicate_col_features, output_column_to_plan_edges, output_column_features,
                      column_to_output_column_edges, column_features, table_features, table_to_plan_edges,
                      output_column_idx, column_idx, table_idx, plan_featurization, predicate_depths,
                      intra_predicate_edges, logical_preds, parent_node_id=plan_node_id, depth=depth + 1, skip_top_aggregation=skip_top_aggregation)


def parse_predicates(db_column_features, feature_statistics, filter, filter_to_plan_edges, plan_featurization,
                     predicate_col_features, predicate_depths, intra_predicate_edges, logical_preds, plan_node_id=None,
                     parent_filter_node_id=None, depth=0, is_join: bool = False):
    """
    Recursive parsing of predicate columns

    :param db_column_features:
    :param feature_statistics:
    :param filter:
    :param filter_to_plan_edges:
    :param plan_featurization:
    :param plan_node_id:
    :param predicate_col_features:
    :return:
    """
    filter_node_id = len(predicate_depths)
    predicate_depths.append(depth)

    assert 'operator' in filter, f'No operator in filter: {filter}'

    # gather features
    if filter['operator'] in {str(op) for op in list(Operator)}:
        curr_filter_features = [encode(feature_name, filter, feature_statistics)
                                for feature_name in plan_featurization.FILTER_FEATURES]

        if is_join:
            filter['col_id'] = filter['col_id1']

        if 'col_name' in filter or 'column_name' in filter:
            assert 'col_id' in filter, f'No col_id in filter: {filter}'

        if 'col_id' in filter and filter['col_id'] is not None:
            curr_filter_col_feats = [
                encode(column, db_column_features[filter['col_id']], feature_statistics)
                for column in plan_featurization.COLUMN_FEATURES]
        # hack for cases in which we have no base filter column (e.g., in a having clause where the column is some
        # result column of a subquery/groupby). In the future, this should be replaced by some graph model that also
        # encodes the structure of this output column
        else:
            curr_filter_col_feats = [0 for _ in plan_featurization.COLUMN_FEATURES]
        curr_filter_features += curr_filter_col_feats
        logical_preds.append(False)

    else:
        curr_filter_features = [encode(feature_name, filter, feature_statistics)
                                for feature_name in plan_featurization.FILTER_FEATURES]
        logical_preds.append(True)

    predicate_col_features.append(curr_filter_features)

    # add edge either to plan or inside predicates
    if depth == 0:
        assert plan_node_id is not None
        # in any case add the corresponding edge
        filter_to_plan_edges.append((filter_node_id, plan_node_id))

    else:
        assert parent_filter_node_id is not None
        intra_predicate_edges.append((filter_node_id, parent_filter_node_id))

    # recurse
    if 'children' in filter:
        for c in filter['children']:
            parse_predicates(db_column_features, feature_statistics, c, filter_to_plan_edges,
                         plan_featurization, predicate_col_features, predicate_depths, intra_predicate_edges,
                             logical_preds, parent_filter_node_id=filter_node_id, depth=depth + 1, is_join=is_join)


def postgres_plan_collator(plans: List[Tuple], feature_statistics: dict[dict] = None, db_statistics: dict = None,
                           plan_featurization: Featurization = None, test_mode: bool = False, skip_top_aggregation:bool=False):
    """
    Combines physical plans into a large graph that can be fed into ML models.
    :return:
    """

    # output:
    #   - list of labels (i.e., plan runtimes)
    #   - feature dictionaries
    #       - column_features: matrix
    #       - output_column_features: matrix
    #       - filter_column_features: matrix
    #       - plan_node_features: matrix
    #       - table_features: matrix
    #       - logical_pred_features: matrix
    #   - edges
    #       - table_to_output
    #       - column_to_output
    #       - filter_to_plan
    #       - output_to_plan
    #       - plan_to_plan
    #       - intra predicate (e.g., column to AND)
    plan_depths = []
    plan_features = []
    plan_to_plan_edges = []
    filter_to_plan_edges = []
    filter_features = []
    output_column_to_plan_edges = []
    output_column_features = []
    column_to_output_column_edges = []
    column_features = []
    table_features = []
    table_to_plan_edges = []
    labels = []
    predicate_depths = []
    intra_predicate_edges = []
    logical_preds = []

    output_column_idx = dict()
    column_idx = dict()
    table_idx = dict()

    # In case of test mode, we need sql_statements and hints to match corresponding queries that belong
    # to the same plan
    sql_statements = []
    hints = []

    # other test information
    selectivity_list = []
    scan_op_list = []
    join_op_list = []

    # prepare robust encoder for the numerical fields
    add_numerical_scalers(feature_statistics)

    # iterate over plans and create lists of edges and features per node
    sample_idxs = []
    for sample_idx, p in plans:
        sample_idxs.append(sample_idx)
        labels.append(p["plan_runtime_ms"])

        if test_mode:
            sql_statements.append(p["sql"])
            hints.append(p["hint"])
            assign_additional_plan_info(p, db_statistics)
            selectivity_list += p["selectivity"]
            scan_op_list +=p["scan_op"]
            join_op_list+= p["join_op"]

        # Parse the plan as graph
        plan_to_graph(p, p["database_id"], plan_depths, plan_features, plan_to_plan_edges, db_statistics,
                      feature_statistics, filter_to_plan_edges, filter_features, output_column_to_plan_edges,
                      output_column_features, column_to_output_column_edges, column_features, table_features,
                      table_to_plan_edges, output_column_idx, column_idx, table_idx,
                      plan_featurization, predicate_depths, intra_predicate_edges, logical_preds,
                      skip_top_aggregation=skip_top_aggregation)


    assert len(labels) == len(plans)
    assert len(plan_depths) == len(plan_features)

    data_dict, nodes_per_depth, plan_dict = create_node_types_per_depth(plan_depths, plan_to_plan_edges)

    # similarly create node types:
    #   pred_node_{depth}, filter column
    pred_dict = dict()
    nodes_per_pred_depth = collections.defaultdict(int)
    no_filter_columns = 0
    for pred_node, d in enumerate(predicate_depths):
        # predicate node
        if logical_preds[pred_node]:
            pred_dict[pred_node] = (nodes_per_pred_depth[d], d)
            nodes_per_pred_depth[d] += 1
        # filter column
        else:
            pred_dict[pred_node] = no_filter_columns
            no_filter_columns += 1

    adapt_predicate_edges(data_dict, filter_to_plan_edges, intra_predicate_edges, logical_preds, plan_dict, pred_dict,
                          pred_node_type_id)

    # we additionally have filters, tables, columns, output_columns and plan nodes as node types
    # Attention: The following code assumes that column_to_output_column_edges is not none. This
    # However might be the case for very simple queries.
    # This can be mitigated by larger batch sizes
    # However, another fix would be to add a check here and
    # adapt the Message Passing scheme for the case of missing column-nodes
    data_dict[('column', 'col_output_col', 'output_column')] = column_to_output_column_edges
    for u, v in output_column_to_plan_edges:
        v_node_id, d_v = plan_dict[v]
        data_dict[('output_column', 'to_plan', f'plan{d_v}')].append((u, v_node_id))
    for u, v in table_to_plan_edges:
        v_node_id, d_v = plan_dict[v]
        data_dict[('table', 'to_plan', f'plan{d_v}')].append((u, v_node_id))

    # also pass number of nodes per type
    max_depth, max_pred_depth = get_depths(plan_depths, predicate_depths)
    num_nodes_dict = {
        'column': len(column_features),
        'table': len(table_features),
        'output_column': len(output_column_features),
        'filter_column': len(logical_preds) - sum(logical_preds),
    }
    num_nodes_dict = update_node_counts(max_depth, max_pred_depth, nodes_per_depth, nodes_per_pred_depth,
                                        num_nodes_dict)

    # create graph
    try:
        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    except Exception as e:
        print(f'Error in creating graph: {num_nodes_dict}')
        raise e
    graph.max_depth = max_depth
    graph.max_pred_depth = max_pred_depth

    features = collections.defaultdict(list)
    features.update(dict(column=column_features, table=table_features, output_column=output_column_features,
                         filter_column=[f for f, log_pred in zip(filter_features, logical_preds) if not log_pred]))
    # sort the plan features based on the depth
    for u, plan_feat in enumerate(plan_features):
        u_node_id, d_u = plan_dict[u]
        features[f'plan{d_u}'].append(plan_feat)

    # sort the predicate features based on the depth
    for pred_node_id, pred_feat in enumerate(filter_features):
        if not logical_preds[pred_node_id]:
            continue
        node_type, _ = pred_node_type_id(logical_preds, pred_dict, pred_node_id)
        features[node_type].append(pred_feat)

    features = postprocess_feats(features, num_nodes_dict)

    # rather deal with runtimes in secs
    labels = postprocess_labels(labels)

    if test_mode:
        return graph, features, labels, sample_idxs, sql_statements, hints, selectivity_list, scan_op_list, join_op_list
    else:
        return graph, features, labels, sample_idxs, [], [], [], [], []


def postprocess_labels(labels):
    labels = np.array(labels, dtype=np.float32)
    labels /= 1000
    # we do this later
    # labels = torch.from_numpy(labels)
    return labels


def postprocess_feats(features, num_nodes_dict):
    # convert to tensors, replace nan with 0®
    for k in features.keys():
        v = features[k]
        v = np.array(v, dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0)
        v = torch.from_numpy(v)
        features[k] = v
    # filter out any node type with zero nodes
    features = {k: v for k, v in features.items() if k in num_nodes_dict}
    return features


def update_node_counts(max_depth, max_pred_depth, nodes_per_depth, nodes_per_pred_depth, num_nodes_dict):
    num_nodes_dict.update({f'plan{d}': nodes_per_depth[d] for d in range(max_depth + 1)})
    num_nodes_dict.update({f'logical_pred_{d}': nodes_per_pred_depth[d] for d in range(max_pred_depth + 1)})
    # filter out any node type with zero nodes
    num_nodes_dict = {k: v for k, v in num_nodes_dict.items() if v > 0}
    return num_nodes_dict


def get_depths(plan_depths, predicate_depths):
    max_depth = max(plan_depths)
    max_pred_depth = 0
    if len(predicate_depths) > 0:
        max_pred_depth = max(predicate_depths)
    return max_depth, max_pred_depth


def adapt_predicate_edges(data_dict, filter_to_plan_edges, intra_predicate_edges, logical_preds, plan_dict, pred_dict,
                          pred_node_type_id_func):
    # convert to plan edges
    for u, v in filter_to_plan_edges:
        # transform plan node to right id and depth
        v_node_id, d_v = plan_dict[v]
        # transform predicate node to right node type and id
        node_type, u_node_id = pred_node_type_id_func(logical_preds, pred_dict, u)

        data_dict[(node_type, 'to_plan', f'plan{d_v}')].append((u_node_id, v_node_id))
    # convert intra predicate edges (e.g. column to AND)
    for u, v in intra_predicate_edges:
        u_node_type, u_node_id = pred_node_type_id_func(logical_preds, pred_dict, u)
        v_node_type, v_node_id = pred_node_type_id_func(logical_preds, pred_dict, v)
        data_dict[(u_node_type, 'intra_predicate', v_node_type)].append((u_node_id, v_node_id))


def create_node_types_per_depth(plan_depths, plan_to_plan_edges):
    # now create heterograph with node types: table, column, filter_column, logical_pred, output_column, plan{depth}
    # for this, first create mapping of old plan node id -> depth and node id for depth
    plan_dict = dict()
    nodes_per_depth = collections.defaultdict(int)
    for plan_node, d in enumerate(plan_depths):
        plan_dict[plan_node] = (nodes_per_depth[d], d)
        nodes_per_depth[d] += 1
    # create edge and node types depending on depth in the plan
    data_dict = collections.defaultdict(list)
    for u, v in plan_to_plan_edges:
        u_node_id, d_u = plan_dict[u]
        v_node_id, d_v = plan_dict[v]
        assert d_v < d_u
        data_dict[(f'plan{d_u}', f'intra_plan', f'plan{d_v}')].append((u_node_id, v_node_id))
    return data_dict, nodes_per_depth, plan_dict


def add_numerical_scalers(feature_statistics):
    for k, v in feature_statistics.items():
        if v.get('type') == str(FeatureType.numeric):
            scaler = RobustScaler()
            scaler.center_ = v['center']
            scaler.scale_ = v['scale']
            feature_statistics[k]['scaler'] = scaler


def pred_node_type_id(logical_preds, pred_dict, u):
    if logical_preds[u]:
        u_node_id, depth = pred_dict[u]
        node_type = f'logical_pred_{depth}'
    else:
        u_node_id = pred_dict[u]
        node_type = f'filter_column'
    return node_type, u_node_id
