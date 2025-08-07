import numpy as np
import torch
from training.training.utils import recursive_to


def dace_batch_to(batch, device, label_norm):
    (
        seq_encodings,
        attention_masks,
        loss_masks,
        run_times,
        labels,
        query_stats,
    ) = batch
    recursive_to(seq_encodings, device)
    recursive_to(attention_masks, device)
    recursive_to(run_times, device)
    recursive_to(loss_masks, device)
    recursive_to(labels, device)
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
    recursive_to(query_plans, device)
    recursive_to(labels, device)
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
    recursive_to(features, device)
    recursive_to(label_tensor, device)
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
    recursive_to(trees, device)
    recursive_to(query_encodings, device)
    recursive_to(indexes, device)
    recursive_to(labels, device)
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
