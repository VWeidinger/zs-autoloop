from typing import List, Tuple

import torch
from ldb_models.models.zeroshot.subplan_runtime_helper import SubplanClass


def mamba_compute_subplan_runtimes(
    all_node_states: torch.Tensor, model_input: Tuple, pred_fn: callable
) -> Tuple[List[SubplanClass], torch.Tensor]:
    # predict runtime for each subplan i.e. for each plan node in the graph
    (
        edge_ids,
        edge_left_right_information,
        plan_ids,
        node_depths,
        node_type_lookup_tensor,
        node_type_id_lookup,
        join_features,
        scan_features,
    ) = model_input

    # compute runtime predictions for all nodes
    all_predictions = pred_fn(all_node_states, node_type_lookup_tensor)
    assert all_predictions.shape[0] == node_type_lookup_tensor.shape[0], (
        f"Expected {node_type_lookup_tensor.shape[0]} predictions, got {all_predictions.shape[0]}"
    )

    # reshape predictions to 1D tensor
    all_predictions = all_predictions.view(-1)

    # create parent-child info
    subplan_obj_idx = {}

    # move tensors to CPU and convert to list
    node_type_lookup = node_type_lookup_tensor.cpu().tolist()

    # create all subplan objects
    for i in range(len(node_type_lookup)):
        node_type = node_type_lookup[i]
        subplan_obj = SubplanClass(node_type, i)
        subplan_obj_idx[(node_type, i)] = subplan_obj

    # add child-parent connections to subplan objects
    # iterate over all edges
    left_id_tensor, right_id_tensor = edge_ids
    left_id = left_id_tensor.cpu().tolist()
    right_id = right_id_tensor.cpu().tolist()

    for src, dst in zip(left_id, right_id):
        # extract node types
        src_node_type = node_type_lookup[src]
        dst_node_type = node_type_lookup[dst]

        assert (dst_node_type, dst) != (src_node_type, src)

        # add child to parent
        subplan_obj_idx[(dst_node_type, dst)].children.append(
            subplan_obj_idx[(src_node_type, src)]
        )

        # check that we are not forming a cycle
        assert (
            subplan_obj_idx[(dst_node_type, dst)]
            not in subplan_obj_idx[(src_node_type, src)].children
        ), f"Cycle detected between {src_node_type} {src} and {dst_node_type} {dst}"

    # extract all subplan objects at root
    subplan_graphs = []
    for i in range(len(all_predictions)):
        if node_depths[i] == 0:  # root nodes are at depth 0
            subplan_graphs.append(subplan_obj_idx[(node_type_lookup[i], i)])

    assert len(subplan_graphs) > 0, "No root nodes found in the subplan graphs"

    return subplan_graphs, all_predictions
