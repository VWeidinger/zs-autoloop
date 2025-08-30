import collections

import dgl
import numpy as np
import torch
from dgl import DGLHeteroGraph
from training.training.operator_tree import OperatorTree


def recursive_to(obj, device):
    if isinstance(obj, (dgl.DGLGraph, DGLHeteroGraph)):
        return obj.to(device, non_blocking=True)
    elif isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    elif isinstance(obj, (collections.abc.Mapping, dict)):
        return type(obj)({k: recursive_to(v, device) for k, v in obj.items()})
    elif isinstance(obj, OperatorTree):
        obj.encoded_features = obj.encoded_features.to(device, non_blocking=True)
        for c in obj.children:
            c = recursive_to(c, device)
        return obj
    elif isinstance(obj, list):
        return [recursive_to(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(v, device) for v in obj)
    else:
        raise Exception(f"Unsupported type: {type(obj)}")


def dace_batch_to(batch, device, label_norm):
    (
        seq_encodings,
        attention_masks,
        loss_masks,
        run_times,
        labels,
        query_stats,
    ) = batch
    seq_encodings = recursive_to(seq_encodings, device)
    attention_masks = recursive_to(attention_masks, device)
    run_times = recursive_to(run_times, device)
    loss_masks = recursive_to(loss_masks, device)
    labels = recursive_to(labels, device)

    return (
        (seq_encodings, attention_masks, loss_masks, run_times),
        labels,
        query_stats,
    )


def simple_batch_to(batch, device, label_norm):
    (query_plans, labels, query_stats) = batch
    if label_norm is not None:
        labels = label_norm.transform(np.asarray(labels.reshape(-1, 1)))
        labels = labels.reshape(-1)
    labels = torch.as_tensor(labels, device=device, dtype=torch.float)
    query_plans = recursive_to(query_plans, device)
    labels = recursive_to(labels, device)
    return (query_plans, labels, query_stats)


def zs_batch_to(batch, device, label_norm):
    (
        graph,
        features,
        label,
        query_stats,
    ) = batch

    # normalize the labels for training
    if label_norm is not None:
        label = label_norm.transform(np.asarray(label).reshape(-1, 1))
        label = label.reshape(-1)

    label_tensor = torch.as_tensor(label, device=device, dtype=torch.float)

    features = recursive_to(features, device)
    label_tensor = recursive_to(label_tensor, device)
    # recursive_to(graph, device)
    graph = graph.to(device, non_blocking=True)
    return (
        (graph, features),
        label_tensor,
        query_stats,
    )


def neo_batch_to(batch, device, label_norm):
    # Unpack the batch
    (
        query_encodings,
        trees,
        indexes,
        labels,
        query_stats,
    ) = batch

    # Normalize the labels for training
    if label_norm is not None:
        labels = label_norm.transform(np.asarray(labels).reshape(-1, 1))
        labels = labels.reshape(-1)

    # Convert to tensors and move to device
    labels = torch.as_tensor(labels, device=device, dtype=torch.float)
    trees = recursive_to(trees, device)
    query_encodings = recursive_to(query_encodings, device)
    indexes = recursive_to(indexes, device)
    labels = recursive_to(labels, device)
    return ((query_encodings, trees, indexes), labels, query_stats)


def mamba_batch_to(batch, device, label_norm):
    (
        edge_ids,
        edge_left_right_information,
        plan_ids,
        node_depths,
        node_type_lookup,
        node_type_id_lookup,
        join_features,
        scan_features,
        labels,
        query_stats,
    ) = batch

    # normalize the labels for training
    if label_norm is not None:
        labels = label_norm.transform(np.asarray(labels).reshape(-1, 1))
        labels = labels.reshape(-1)

    labels = torch.as_tensor(labels, device=device, dtype=torch.float)

    # Batch device transfer for all tensors/lists and overwrite variables
    data = (
        edge_ids,
        edge_left_right_information,
        plan_ids,
        node_depths,
        node_type_lookup,
        node_type_id_lookup,
        join_features,
        scan_features,
    )
    data = recursive_to(data, device)

    return (data, labels, query_stats)


def query_former_batch_to(batch, device, label_norm):
    (
        (features, join_ids, attention_bias, rel_pos, node_heights),
        label,
        query_stats,
    ) = batch
    if label_norm is not None:
        label = label_norm.transform(np.asarray(label).reshape(-1, 1))
        label = label.reshape(-1)

    label = torch.as_tensor(label, device=device, dtype=torch.float)
    features = recursive_to(features, device)
    join_ids = recursive_to(join_ids, device)
    attention_bias = recursive_to(attention_bias, device)
    rel_pos = recursive_to(rel_pos, device)
    node_heights = recursive_to(node_heights, device)
    return (
        (features, join_ids, attention_bias, rel_pos, node_heights),
        label,
        query_stats,
    )


def wl_driven_batch_to(batch, device, label_norm):
    # label, plan_graph, features, pred_graph, plan_pred_mapping, sample_plan_idxs = batch
    (labels, plan_graph, features, pred_graph, plan_pred_mapping, query_stats) = batch
    # graph, features, label = batch
    features = recursive_to(features, device)
    labels = recursive_to(labels, device)
    plan_pred_mapping = recursive_to(plan_pred_mapping, device)

    # recursive_to(graph, device)
    plan_graph = plan_graph.to(device)
    pred_graph = pred_graph.to(device)

    return ((plan_graph, features, pred_graph, plan_pred_mapping), labels, query_stats)
