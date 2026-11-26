import collections
from typing import DefaultDict, Dict, List, Tuple

import dgl
import numpy as np
import torch
from ldb_models.models.white_mamba.extract_query_stats import extract_query_stats_for_plan
from ldb_models.models.zeroshot.annotate_additional_plan_info import assign_additional_plan_info
from ldb_models.training.featurizations import JIT_GLOBAL_PLAN_FEATURES, Featurization

from ldb_models.models.zeroshot.encoding import apply_feature_scalers, pred_node_type_id
from ldb_models.models.zeroshot.graph_construction import plan_to_graph
from ldb_models.models.zeroshot.postprocess import (
    adapt_predicate_edges,
    create_node_types_per_depth,
    get_depths,
    postprocess_feats,
    postprocess_labels,
    update_node_counts,
)


def postgres_plan_collator(
    plans: List[Tuple],
    feature_statistics: Dict,
    db_statistics: Dict,
    db_schema_info: Dict,
    plan_featurization: Featurization,
    skip_top_aggregation: bool = False,
    filter_inline_col_feats: bool = True,
    annotate_base_table_card_to_scan_op: bool = False,
    enforce_aggregation_node_at_root: bool = False,
    prohibit_aggregation_node_at_root: bool = False,
    output_tensors: bool = False,
    subplan_label_op_time: bool = False,
    log_transform_card_labels: bool = False,
    use_log_input_card: bool = False,
    cardinality_estimation: bool = False,
    subplan_estimation: bool = False,
    sumo_loop: bool = False,
    sumo_loop_multiply_est_error: bool = False,
    sumo_loop_loss_on_scaled: bool = False,
    scaler_type: str = "robust",
):
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
    column_to_filter_edges = []
    column_features = []
    table_features = []
    table_to_plan_edges = []
    labels = []
    subplan_labels = []
    subplan_card_labels = []
    subplan_est_card_labels = []
    loop_multiplier_map = (
        [],
        [],
    )  # SUMO-LOOP: Batch-level loop multiplier mapping (node_ids, sibling_ids)

    predicate_depths = []
    intra_predicate_edges = []
    logical_preds = []

    output_column_idx = dict()
    column_idx = dict()
    table_idx = dict()

    # query stats
    query_stats_dict = collections.defaultdict(list)

    # iterate over plans and create lists of edges and features per node
    sample_idxs = []
    for idx, (sample_idx, p) in enumerate(plans):
        sample_idxs.append(sample_idx)
        # print(f'Before extracting plan runtime: {p}', flush=True)
        labels.append(p["plan_runtime_ms"])

        # annotate additional info
        assign_additional_plan_info(p, db_statistics, db_schema_info)

        if enforce_aggregation_node_at_root:
            # check if plan has aggregation node at root
            original_root_plan_parameters = p["plan_parameters"]

            if original_root_plan_parameters["op_name"] not in [
                "Simple Aggregate",
                "Aggregate",
            ]:
                # create an artifical aggregate node that sits on top
                other_args = {
                    k: p.pop(k)
                    for k in list(p.keys())
                    if k not in ["plan_parameters", "children"]
                }

                aggregate_node = {
                    "plan_parameters": {
                        "act_card": 1,
                        "act_time": original_root_plan_parameters["act_time"],
                        "act_loops": 1,
                        "est_card": 1,
                        "est_width": 8,
                        "pg_cost": original_root_plan_parameters["pg_cost"],
                        "est_loops": 1,
                        "op_name": "Simple Aggregate",
                        "output_columns": [
                            {
                                "aggregation": "COUNT",
                                "columns": [],
                                "udf_name": None,
                                "udf_output": "False",
                                "child_ref": False,
                            }
                        ],
                        "act_children_card": original_root_plan_parameters["act_card"],
                        "est_children_card": original_root_plan_parameters["est_card"],
                        "dd_est_card": 1,
                        "dd_est_children_card": original_root_plan_parameters[
                            "dd_est_card"
                        ]
                        if "dd_est_card" in original_root_plan_parameters
                        else -1,
                        "above_udf_filter": False,
                        "is_udf_filter": False,
                        # global JIT features (parsed_plans50) of the query, compiled in the original plan
                        **{
                            k: original_root_plan_parameters[k]
                            for k in JIT_GLOBAL_PLAN_FEATURES
                            if k in original_root_plan_parameters
                        },
                        "jit_compiled_here": False,
                    },
                    "children": [p],
                    **other_args,
                }

                # overwrite the plan with the new one
                p = aggregate_node
        elif prohibit_aggregation_node_at_root:
            if p["plan_parameters"]["op_name"] in ["Simple Aggregate", "Aggregate"]:
                # remove the aggregation node
                root_args = {
                    k: p.pop(k)
                    for k in list(p.keys())
                    if k not in ["plan_parameters", "children"]
                }
                children = p["children"]
                assert len(children) == 1, (
                    f"Aggregation node has more than one child: {p}"
                )
                p = children[0]

                # copy over root args
                for k, v in root_args.items():
                    p[k] = v

        plan_depth_start_idx = len(plan_depths)

        # SUMO-LOOP: Initialize per-plan loop_multiplier_map for this batch iteration
        plan_loop_multiplier_map = ([], [])

        # Parse the plan as graph
        plan_to_graph(
            p,
            p["run_file_id"],
            plan_depths,
            plan_features,
            subplan_labels,
            subplan_card_labels,
            subplan_est_card_labels,
            plan_to_plan_edges,
            db_statistics,
            feature_statistics,
            filter_to_plan_edges,
            filter_features,
            output_column_to_plan_edges,
            output_column_features,
            column_to_output_column_edges,
            column_to_filter_edges,
            column_features,
            table_features,
            table_to_plan_edges,
            output_column_idx,
            column_idx,
            table_idx,
            plan_featurization,
            predicate_depths,
            intra_predicate_edges,
            logical_preds,
            skip_top_aggregation=skip_top_aggregation,
            filter_inline_col_feats=filter_inline_col_feats,
            annotate_base_table_card_to_scan_op=annotate_base_table_card_to_scan_op,
            subplan_label_op_time=subplan_label_op_time,
            sumo_loop=sumo_loop,
            sumo_loop_multiply_est_error=sumo_loop_multiply_est_error,
            sumo_loop_loss_on_scaled=sumo_loop_loss_on_scaled,
            loop_multiplier_map=plan_loop_multiplier_map,
        )

        # SUMO-LOOP: Merge plan-level loop_multiplier_map into batch map.
        # Node IDs from plan_to_graph are already batch-global (recorded using the shared
        # plan_depths list which accumulated entries from prior plans), so no offset needed.
        if sumo_loop and plan_loop_multiplier_map[0]:
            loop_multiplier_map[0].extend(plan_loop_multiplier_map[0])
            loop_multiplier_map[1].extend(plan_loop_multiplier_map[1])

        # extract query stats
        extract_query_stats_for_plan(
            query_stats_dict,
            p,
            sample_idx,
            node_depth=max(plan_depths[plan_depth_start_idx:], default=0),
        )

    assert len(labels) == len(plans)
    assert len(subplan_labels) == len(plan_features)
    assert len(plan_depths) == len(plan_features)

    # Optionally apply log1p to input cardinality features (est/act) before scaling.
    if use_log_input_card:
        # find indices of plan features that contain 'card' in their name
        card_feature_indices = [
            i for i, f in enumerate(plan_featurization.PLAN_FEATURES) if "card" in f
        ]
        if len(card_feature_indices) > 0 and len(plan_features) > 0:
            # apply np.log1p in-place to the specific columns
            plan_features_np = np.array(plan_features, dtype=np.float32)
            plan_features_np[:, card_feature_indices] = np.log1p(
                plan_features_np[:, card_feature_indices]
            )
            plan_features = plan_features_np.tolist()

    # run the feature scaling
    plan_features = apply_feature_scalers(
        plan_features,
        feature_statistics=feature_statistics,
        feature_names=plan_featurization.PLAN_FEATURES,
    )
    output_column_features = apply_feature_scalers(
        output_column_features,
        feature_statistics=feature_statistics,
        feature_names=plan_featurization.OUTPUT_COLUMN_FEATURES,
    )
    table_features = apply_feature_scalers(
        table_features,
        feature_statistics=feature_statistics,
        feature_names=plan_featurization.TABLE_FEATURES,
    )

    # assemble the filter features
    if filter_inline_col_feats:
        filter_feats_list = (
            plan_featurization.FILTER_FEATURES + plan_featurization.COLUMN_FEATURES
        )
    else:
        filter_feats_list = plan_featurization.FILTER_FEATURES

    filter_features = apply_feature_scalers(
        filter_features,
        feature_statistics=feature_statistics,
        feature_names=filter_feats_list,
    )

    column_features = apply_feature_scalers(
        column_features,
        feature_statistics=feature_statistics,
        feature_names=plan_featurization.COLUMN_FEATURES,
    )

    data_dict, nodes_per_depth, plan_dict = create_node_types_per_depth(
        plan_depths, plan_to_plan_edges
    )

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

    adapt_predicate_edges(
        data_dict,
        filter_to_plan_edges,
        intra_predicate_edges,
        logical_preds,
        plan_dict,
        pred_dict,
        pred_node_type_id,
        column_to_filter_edges,
    )

    # we additionally have filters, tables, columns, output_columns and plan nodes as node types
    # Attention: The following code assumes that column_to_output_column_edges is not none. This
    # However might be the case for very simple queries.
    # This can be mitigated by larger batch sizes
    # However, another fix would be to add a check here and
    # adapt the Message Passing scheme for the case of missing column-nodes
    data_dict[("column", "col_output_col", "output_column")] = (
        column_to_output_column_edges
    )

    for u, v in output_column_to_plan_edges:
        v_node_id, d_v = plan_dict[v]
        data_dict[("output_column", "to_plan", f"plan{d_v}")].append((u, v_node_id))
    for u, v in table_to_plan_edges:
        v_node_id, d_v = plan_dict[v]
        data_dict[("table", "to_plan", f"plan{d_v}")].append((u, v_node_id))

    # also pass number of nodes per type
    max_depth, max_pred_depth = get_depths(plan_depths, predicate_depths)
    num_nodes_dict = {
        "column": len(column_features),
        "table": len(table_features),
        "output_column": len(output_column_features),
        "filter_column": len(logical_preds) - sum(logical_preds),
    }
    num_nodes_dict = update_node_counts(
        max_depth, max_pred_depth, nodes_per_depth, nodes_per_pred_depth, num_nodes_dict
    )

    # create graph
    try:
        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)  # type: ignore
    except Exception as e:
        print(
            f"Error in creating graph: {num_nodes_dict} \n {data_dict.keys()}",
            flush=True,
        )
        raise e
    graph.max_depth = max_depth
    graph.max_pred_depth = max_pred_depth

    features: DefaultDict = collections.defaultdict(list)
    features.update(
        {
            "column": column_features,
            "table": table_features,
            "output_column": output_column_features,
            "filter_column": [
                f for f, log_pred in zip(filter_features, logical_preds) if not log_pred
            ],
        }
    )

    # Invert plan_dict to map the model's internal node index back to our original ID.
    # The original ID corresponds to the label index.
    inv_plan_dict = {}
    for orig_id, (type_idx, depth) in plan_dict.items():
        ntype = f"plan{depth}"
        if ntype not in inv_plan_dict:
            inv_plan_dict[ntype] = {}
        inv_plan_dict[ntype][type_idx] = orig_id

    # Attach the original_ids as data to each plan node in the graph.
    for ntype in graph.ntypes:
        if ntype.startswith("plan"):
            # Check if we have nodes of this type in the current batch
            if ntype in inv_plan_dict and graph.num_nodes(ntype) > 0:
                type_map = inv_plan_dict[ntype]
                # Create a list of original_ids, sorted by the DGL node index (0, 1, 2, ...).
                # This assumes DGL node IDs for a given type are contiguous from 0.
                if graph.num_nodes(ntype) != len(type_map):
                    # This might happen if a plan has no nodes of a certain depth but the graph was constructed
                    # to potentially have them. Only process if they match.
                    continue

                sorted_orig_ids = [type_map[i] for i in range(graph.num_nodes(ntype))]
                graph.nodes[ntype].data["original_id"] = torch.tensor(
                    sorted_orig_ids, dtype=torch.long
                )

    # sort the plan features based on the depth
    for u, plan_feat in enumerate(plan_features):
        u_node_id, d_u = plan_dict[u]
        features[f"plan{d_u}"].append(plan_feat)

    # sort the predicate features based on the depth
    for pred_node_id, pred_feat in enumerate(filter_features):
        if not logical_preds[pred_node_id]:
            continue

        # we have padded the logical features - remove the padding
        pred_feat = pred_feat[
            : len(plan_featurization.FILTER_FEATURES)
        ]  # we can do this since the embedding / one-hot encoding / ... will be done afterwards in the node type encoder

        node_type, _ = pred_node_type_id(logical_preds, pred_dict, pred_node_id)
        features[node_type].append(pred_feat)

    postprocessed_features = postprocess_feats(features, num_nodes_dict)

    # rather deal with runtimes in secs
    labels = postprocess_labels(labels)
    subplan_labels = postprocess_labels(subplan_labels)
    subplan_card_labels = np.array(subplan_card_labels, dtype=np.float32)
    subplan_est_card_labels = np.array(subplan_est_card_labels, dtype=np.float32)
    if log_transform_card_labels:
        subplan_card_labels = np.log1p(subplan_card_labels)
        subplan_est_card_labels = np.log1p(subplan_est_card_labels)

    if output_tensors:
        # create tensors
        labels = torch.from_numpy(labels)
        subplan_labels = torch.from_numpy(subplan_labels)
        sample_idxs = torch.from_numpy(np.array(sample_idxs, dtype=np.int64))
        subplan_card_labels = torch.from_numpy(subplan_card_labels, dtype=np.float32)
        subplan_est_card_labels = torch.from_numpy(
            subplan_est_card_labels, dtype=np.float32
        )

    if cardinality_estimation:
        query_stats_dict["subplan_card_labels"] = subplan_card_labels
        query_stats_dict["subplan_est_card_labels"] = subplan_est_card_labels
    if subplan_estimation:
        query_stats_dict["subplan_labels"] = subplan_labels
    if sumo_loop:
        query_stats_dict["loop_multiplier_map"] = loop_multiplier_map

    return (
        graph,
        postprocessed_features,
        labels,
        {
            "sample_idxs": sample_idxs,
            **query_stats_dict,
            "plan_dict": plan_dict,
        },
    )
