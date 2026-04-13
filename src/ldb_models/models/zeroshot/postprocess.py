import collections

import numpy as np
import torch


def postprocess_labels(labels):
    labels = np.array(labels, dtype=np.float32)
    labels /= 1000
    # we do this later
    # labels = torch.from_numpy(labels)
    return labels


def postprocess_feats(features, num_nodes_dict):
    # convert to tensors, replace nan with 0®
    for k, v in features.items():
        v = np.array(v, dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0)
        v = torch.from_numpy(v)
        features[k] = v
    # filter out any node type with zero nodes
    features = {k: v for k, v in features.items() if k in num_nodes_dict}
    return features


def update_node_counts(
    max_depth, max_pred_depth, nodes_per_depth, nodes_per_pred_depth, num_nodes_dict
):
    num_nodes_dict.update(
        {f"plan{d}": nodes_per_depth[d] for d in range(max_depth + 1)}
    )
    num_nodes_dict.update(
        {
            f"logical_pred_{d}": nodes_per_pred_depth[d]
            for d in range(max_pred_depth + 1)
        }
    )
    return num_nodes_dict


def adapt_predicate_edges(
    data_dict,
    filter_to_plan_edges,
    intra_predicate_edges,
    logical_preds,
    plan_dict,
    pred_dict,
    pred_node_type_id_func,
    column_to_filter_edges,
):
    # convert to plan edges
    for u, v in filter_to_plan_edges:
        # transform plan node to right id and depth
        v_node_id, d_v = plan_dict[v]
        # transform predicate node to right node type and id
        node_type, u_node_id = pred_node_type_id_func(logical_preds, pred_dict, u)

        data_dict[(node_type, "to_plan", f"plan{d_v}")].append((u_node_id, v_node_id))
    # convert intra predicate edges (e.g. column to AND)
    for u, v in intra_predicate_edges:
        u_node_type, u_node_id = pred_node_type_id_func(logical_preds, pred_dict, u)
        v_node_type, v_node_id = pred_node_type_id_func(logical_preds, pred_dict, v)
        data_dict[(u_node_type, "intra_predicate", v_node_type)].append(
            (u_node_id, v_node_id)
        )

    for u, v in column_to_filter_edges:
        v_node_type, v_node_id = pred_node_type_id_func(logical_preds, pred_dict, v)
        data_dict[("column", "col_filter", v_node_type)].append((u, v_node_id))


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
        data_dict[(f"plan{d_u}", "intra_plan", f"plan{d_v}")].append(
            (u_node_id, v_node_id)
        )
    return data_dict, nodes_per_depth, plan_dict


def get_depths(plan_depths, predicate_depths):
    max_depth = max(plan_depths)
    max_pred_depth = 0
    if len(predicate_depths) > 0:
        max_pred_depth = max(predicate_depths)
    return max_depth, max_pred_depth
