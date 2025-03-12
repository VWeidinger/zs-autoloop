from typing import List, Dict, Tuple

import torch


class SubplanClass:

    def __init__(self, node_type: str, pred_idx: int):
        self.node_type = node_type
        self.pred_idx = pred_idx

        # initialize children with empty list
        self.children: List["SubplanClass"] = []


def compute_subplan_runtimes(graph, all_node_states: Dict, pred_fn):
    # predict runtime for each subplan i.e. for each plan node in the graph - this works because we did topological message passing before
    predictions_dict = {}
    for node_type, node_states in all_node_states.items():
        if node_type.startswith('plan'):
            # run predict fn
            predictions_dict[node_type] = pred_fn(node_states)

    # create subplan objects
    subplan_obj_idx = {}
    predictions_ctr = 0

    predictions_list = []
    for node_type, predictions in predictions_dict.items():
        for i in range(predictions.shape[0]):
            subplan_obj = SubplanClass(node_type, i + predictions_ctr)
            subplan_obj_idx[(node_type, i)] = subplan_obj

        # increment counter
        predictions_ctr += predictions.shape[0]

        assert len(predictions.shape) == 2, f"Expected 2D tensor, got {predictions.shape}"
        assert predictions.shape[1] == 1, f"Expected dim 1 to be 1, got {predictions.shape[1]}"

        # reshape to 1D
        predictions = predictions.view(-1)

        # add predictions to list
        predictions_list.append(predictions)

    # concatenate predictions
    predictions_list = torch.cat(predictions_list)

    # add child-parent connections

    # iterate over all edges
    for edge_type in graph.canonical_etypes:
        if edge_type[1] != 'intra_plan':
            # skip
            continue

        # iterate over all edges
        src_ids, dst_ids = graph.edges(etype=edge_type)

        # convert to list
        src_ids = src_ids.tolist()
        dst_ids = dst_ids.tolist()

        for src, dst in zip(src_ids, dst_ids):
            # extract node types
            src_node_type = edge_type[0]
            dst_node_type = edge_type[2]

            assert (dst_node_type, dst) != (src_node_type, src)

            # add child to parent
            subplan_obj_idx[(dst_node_type, dst)].children.append(subplan_obj_idx[(src_node_type, src)])

            # check that we are not forming a cycle
            assert subplan_obj_idx[(dst_node_type, dst)] not in subplan_obj_idx[(src_node_type, src)].children

    # extract all root nodes (plan0) of the dags
    root_nodes = [subplan_obj_idx[('plan0', i)] for i in range(len(predictions_dict['plan0']))]
    assert len(root_nodes) > 0

    return root_nodes, predictions_list


def recursively_get_parent_child_pairs(subplan: SubplanClass) -> List[Tuple[int, int]]:
    # iterate over all children
    parent_child_pairs = []
    for child in subplan.children:
        # add direct parent-child pair
        parent_child_pairs.append((subplan.pred_idx, child.pred_idx))

        # recursively call
        parent_child_pairs.extend(recursively_get_parent_child_pairs(child))

    return parent_child_pairs


def extract_smaller_larger_preds_tensor_for_subplan_graphs(subplan_graphs: List[SubplanClass],
                                                           predictions: torch.Tensor) -> Tuple[
    torch.Tensor, torch.Tensor]:
    # create parent-child pairs
    parent_child_pairs_list = []
    for subplan in subplan_graphs:
        parent_child_pairs_list.extend(recursively_get_parent_child_pairs(subplan))

    # extract all child idxs
    child_idxs = [child for _, child in parent_child_pairs_list]
    child_idxs = torch.as_tensor(child_idxs, device=predictions.device)

    # extract all parent idxs
    parent_idxs = [parent for parent, _ in parent_child_pairs_list]
    parent_idxs = torch.as_tensor(parent_idxs, device=predictions.device)

    # perform gather
    smaller_tensor = torch.gather(predictions, 0, parent_idxs)
    larger_tensor = torch.gather(predictions, 0, child_idxs)

    return smaller_tensor, larger_tensor
