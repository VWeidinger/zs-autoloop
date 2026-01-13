import collections
from typing import Dict, Tuple

import dgl
import numpy as np
import torch
from models.utils import get_general_table_col_to_ids_dict
from models.white_mamba.extract_query_stats import extract_query_stats_for_plan
from models.zeroshot.postgres_plan_batching import assign_additional_plan_info
from training.dataset.cmp_operator import LogicalOperator


class WorkloadDrivenGraph:
    def __init__(self):
        self.labels = []
        # plan representation
        self.plan_depths = []
        self.plan_features = []
        self.plan_edges = []
        self.plan_to_filter_mapping = []

        # predicate graph representation
        self.predicate_depths = []
        self.predicate_edges = []
        self.predicate_operators = []
        self.predicate_features = []


def plan_to_graphs(
    column_statistics,
    word_embeddings,
    feature_statistics,
    node,
    wl_graph,
    table_to_id_dict: Dict[str, int],
    col_id_dict: Dict[Tuple[str, str], int],
    depth=0,
    parent_node_id=None,
    dim_word_hash=None,
    dim_word_emdb=None,
    dim_bitmaps=None,
):
    plan_node_id = len(wl_graph.plan_depths)
    wl_graph.plan_depths.append(depth)

    plan_params = node["plan_parameters"]

    # add Operation, Metadata (=used tables + columns) and Sample Bitmap
    # operation
    operation_vec = encode_one_hot(
        "op_name", feature_statistics, plan_params["op_name"]
    )

    # metadata (used tables)
    dim_cols = len(col_id_dict)
    dim_tables = len(table_to_id_dict)

    col_vec = np.zeros(dim_cols)
    if "output_columns" in plan_params.keys():
        output_columns = plan_params["output_columns"]
        for output_column in output_columns:
            for c_idx in output_column["columns"]:
                table_name = c_idx["table_name"]
                col_name = c_idx["col_name"]
                col_vec[col_id_dict[(table_name, col_name)]] = 1

    # metadata (used columns)
    tab_vec = np.zeros(dim_tables)

    if "table_id" in plan_params.keys():
        table_id = table_to_id_dict[plan_params["table_name"]]
        tab_vec[int(table_id)] = 1

    # parse predicates into predicate graph and fill sample vector
    predicate_idx = -1
    sample_bitmap = np.zeros(dim_bitmaps)
    if "filter" in plan_params.keys():
        filter_column = plan_params["filter"]
        plan_sample_vec = plan_params["sample_vec"]
        assert plan_sample_vec is not None

        sample_bitmap[: len(plan_sample_vec)] = plan_sample_vec
        # check if node already exists in the graph
        predicate_idx = parse_baseline_predicates(
            column_statistics,
            word_embeddings,
            feature_statistics=feature_statistics,
            filter_column=filter_column,
            wl_graph=wl_graph,
            col_id_dict=col_id_dict,
            dim_word_hash=dim_word_hash,
            dim_word_emdb=dim_word_emdb,
        )

    plan_feat_vec = np.concatenate([operation_vec, col_vec, tab_vec, sample_bitmap])
    wl_graph.plan_features.append(plan_feat_vec)
    wl_graph.plan_to_filter_mapping.append(predicate_idx)

    if parent_node_id is not None:
        assert depth > 0
        wl_graph.plan_edges.append((plan_node_id, parent_node_id))
    else:
        assert depth == 0

    # continue recursively
    if "children" in node.keys():
        for c in node["children"]:
            plan_to_graphs(
                column_statistics,
                word_embeddings,
                feature_statistics=feature_statistics,
                node=c,
                wl_graph=wl_graph,
                table_to_id_dict=table_to_id_dict,
                col_id_dict=col_id_dict,
                depth=depth + 1,
                parent_node_id=plan_node_id,
                dim_word_hash=dim_word_hash,
                dim_word_emdb=dim_word_emdb,
                dim_bitmaps=dim_bitmaps,
            )


def extract_dimensions(feature_statistics: Dict, extended=False):
    assert isinstance(feature_statistics, Dict), (
        f"Feature statistics is not a dict: {feature_statistics}"
    )
    assert "operator" in feature_statistics, (
        f"Feature statistics does not contain operator info: {feature_statistics}"
    )
    assert isinstance(feature_statistics["operator"], Dict), (
        f"Feature statistics operator info is not a dict: {feature_statistics['operator']}"
    )
    dim_pred_op = feature_statistics["operator"]["no_vals"]
    dim_ops = feature_statistics["op_name"]["no_vals"]
    if extended:
        dim_joins = feature_statistics["join"]["no_vals"]
        dim_aggs = feature_statistics["aggregation"]["no_vals"]
        return dim_ops, dim_pred_op, dim_joins, dim_aggs

    return dim_ops, dim_pred_op


def encode_one_hot(feature_name, feature_statistics, v):
    hot_idx = feature_statistics[feature_name]["value_dict"][v]
    no_feats = feature_statistics[feature_name]["no_vals"]
    one_hot_vec = np.zeros(no_feats)
    one_hot_vec[hot_idx] = 1
    return one_hot_vec


def parse_baseline_predicates(
    column_statistics,
    word_embeddings,
    feature_statistics: Dict,
    filter_column,
    wl_graph,
    col_id_dict: Dict[Tuple[str, str], int],
    depth=0,
    parent_predicate=None,
    dim_word_hash=None,
    dim_word_emdb=None,
):
    filter_node_id = len(wl_graph.predicate_depths)
    wl_graph.predicate_depths.append(depth)

    if filter_column["operator"] == str(LogicalOperator.AND):
        wl_graph.predicate_operators.append(1)
    elif filter_column["operator"] == str(LogicalOperator.OR):
        wl_graph.predicate_operators.append(-1)
    else:
        wl_graph.predicate_operators.append(0)

    if filter_column["operator"] in {str(LogicalOperator.AND), str(LogicalOperator.OR)}:
        wl_graph.predicate_features.append([0 for _ in range(10)])
    else:
        pred_feat_vec = encode_predicate(
            column_statistics,
            dim_word_emdb,
            dim_word_hash,
            feature_statistics,
            filter_column,
            word_embeddings,
            col_id_dict,
        )
        wl_graph.predicate_features.append(pred_feat_vec)

    if parent_predicate is not None:
        assert depth > 0
        wl_graph.predicate_edges.append((filter_node_id, parent_predicate))
    else:
        assert depth == 0

    if "children" in filter_column.keys():
        for c in filter_column["children"]:
            parse_baseline_predicates(
                column_statistics,
                word_embeddings,
                feature_statistics,
                c,
                wl_graph,
                col_id_dict=col_id_dict,
                depth=depth + 1,
                parent_predicate=filter_node_id,
                dim_word_hash=dim_word_hash,
                dim_word_emdb=dim_word_emdb,
            )

    return filter_node_id


def encode_predicate(
    column_statistics,
    dim_word_emdb,
    dim_word_hash,
    feature_statistics: Dict,
    filter_column,
    word_embeddings,
    col_id_dict: Dict[Tuple[str, str], int],
):
    _, dim_pred_op = extract_dimensions(feature_statistics)
    col_vec = np.zeros(len(col_id_dict))

    col_name = filter_column["col_name"]
    table_name = filter_column["table_name"]

    assert (table_name, col_name) in col_id_dict, (
        f"Column {(table_name, col_name)} not in col_id_dict {col_id_dict.keys()}"
    )
    col_id = col_id_dict[(table_name, col_name)]
    col_vec[col_id] = 1
    operation_vec = encode_one_hot(
        "operator", feature_statistics, filter_column["operator"]
    )
    # find out how to encode the column

    col_stats = column_statistics[table_name][col_name]
    literal_vec = np.zeros(dim_word_emdb + dim_word_hash)
    if filter_column["operator"] not in {"IS NULL", "IS NOT NULL"}:
        if col_stats["data_type"] in {"float", "int"}:
            assert filter_column["literal"] is not None
            # cast literal
            if col_stats["data_type"] == "float":
                filter_column["literal"] = float(filter_column["literal"])
            elif col_stats["data_type"] == "int":
                filter_column["literal"] = int(float(filter_column["literal"]))
            else:
                raise ValueError(f"Unknown column type {col_stats['data_type']}")
            try:
                val_enc = filter_column["literal"] - col_stats["min"]
            except Exception as e:
                print(
                    f"Error encoding value {filter_column['literal']} for column {col_name} with stats {col_stats}",
                    flush=True,
                )
                raise e
            if col_stats["max"] - col_stats["min"] > 0:
                val_enc /= col_stats["max"] - col_stats["min"]
            literal_vec[0] = val_enc
        elif col_stats["data_type"] in {"categorical", "misc"}:
            str_words = [filter_column["literal"]]
            if filter_column["operator"] in {"LIKE", "NOT LIKE"}:
                str_words = [
                    w.strip()
                    for w in filter_column["literal"].split("%")
                    if len(w.strip()) > 0
                ]
            elif filter_column["operator"] in {"IN"}:
                str_words = filter_column["literal"].split(",")
            str_words = [f"{col_name}_{w}" for w in str_words]

            # average of word embeddings and hashed value representation
            no_words = 0
            for w in str_words:
                if w in word_embeddings:
                    no_words += 1
                    literal_vec[:dim_word_emdb] += word_embeddings[w]
                    for t in w:
                        literal_vec[dim_word_emdb + hash(t) % dim_word_hash] += 1.0
            if no_words > 1:
                literal_vec /= no_words
        else:
            raise NotImplementedError
    return np.concatenate([col_vec, operation_vec, literal_vec])


def baseline_plan_collator(
    plans,
    feature_statistics,
    db_statistics,
    table_id_dict: Dict[str, int],  # not used but kept for interface consistency
    col_id_dict: Dict[
        Tuple[str, str], int
    ],  # not used but kept for interface consistency
    db_schema_info,
    column_statistics=None,
    word_embeddings=None,
    dim_word_hash=None,
    dim_word_emdb=None,
    dim_bitmaps=None,
):
    """
    Combines physical plans into a large graph that can be fed into ML models.
    :return:
    """

    wl_graph = WorkloadDrivenGraph()

    # get table and column id mappings
    table_name_to_id_dict, col_name_to_id_dict = get_general_table_col_to_ids_dict(
        db_statistics
    )

    query_stats_dict = collections.defaultdict(list)
    for sample_idx, p in plans:
        wl_graph.labels.append(p["plan_runtime_ms"])
        assign_additional_plan_info(p, db_statistics, db_schema_info=db_schema_info)

        # extract query stats
        extract_query_stats_for_plan(
            query_stats_dict,
            p,
            sample_idx,
        )

        plan_to_graphs(
            column_statistics,
            word_embeddings,
            feature_statistics=feature_statistics,
            node=p,
            wl_graph=wl_graph,
            table_to_id_dict=table_name_to_id_dict,
            col_id_dict=col_name_to_id_dict,
            dim_word_hash=dim_word_hash,
            dim_word_emdb=dim_word_emdb,
            dim_bitmaps=dim_bitmaps,
        )

    pred_graph, plan_pred_ids, plan_pred_ops, pred_feats = predicate_graph(wl_graph)
    features, plan_graph, plan_dict = create_plan_graph(wl_graph)

    plan_pred_mapping = []
    for plan_id, (pred_op, pred_id) in enumerate(zip(plan_pred_ops, plan_pred_ids)):
        if pred_id == -1:
            plan_pred_mapping.append([-1, -1, pred_id, pred_op])
            continue

        u_node_id, d_u = plan_dict[plan_id]
        plan_pred_mapping.append([d_u, u_node_id, pred_id, pred_op])

    features["pred"] = pred_feats

    features = tensorize_feats(features)

    # rather deal with runtimes in secs
    labels = np.array(wl_graph.labels, dtype=np.float32)
    labels /= 1000
    labels = torch.from_numpy(labels)

    plan_pred_mapping = torch.from_numpy(np.array(plan_pred_mapping, dtype=np.int64))

    return (
        labels,
        plan_graph,
        features,
        pred_graph,
        plan_pred_mapping,
        query_stats_dict,
    )


def tensorize_feats(features):
    tensor_features = dict()
    for k in features.keys():
        v = features[k]
        if len(v) == 0:
            continue
        v = np.array(v, dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0)
        v = torch.from_numpy(v)
        tensor_features[k] = v
    return tensor_features


def create_plan_graph(wl_graph):
    max_depth = max(wl_graph.plan_depths)
    plan_dict = dict()
    nodes_per_depth = collections.defaultdict(int)
    for plan_node, d in enumerate(wl_graph.plan_depths):
        plan_dict[plan_node] = (nodes_per_depth[d], d)
        nodes_per_depth[d] += 1
    # create edge and node types depending on depth in the plan
    data_dict = collections.defaultdict(list)

    # If only single plan nodes exist, create dummy graph with a self-pointing node and an empty edge type
    if len(wl_graph.plan_edges) == 0:
        for node_key, (node_id, d) in plan_dict.items():
            data_dict[(f"plan{d}", "", f"plan{d}")].append((node_id, node_id))
    else:
        for u, v in wl_graph.plan_edges:
            u_node_id, d_u = plan_dict[u]
            v_node_id, d_v = plan_dict[v]
            assert d_v < d_u
            data_dict[(f"plan{d_u}", "intra_plan", f"plan{d_v}")].append(
                (u_node_id, v_node_id)
            )
    num_nodes_dict = {f"plan{d}": nodes_per_depth[d] for d in range(max_depth + 1)}
    features = collections.defaultdict(list)
    for u, plan_feat in enumerate(wl_graph.plan_features):
        u_node_id, d_u = plan_dict[u]
        features[f"plan{d_u}"].append(plan_feat)
    assert data_dict, "data_dict for plan graph is empty"
    graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    graph.max_depth = max_depth
    return features, graph, plan_dict


def predicate_graph(wl_graph):
    pred_dict = dict()
    preds_no = collections.defaultdict(int)
    pred_feats = list()
    # to maintain mapping of predicates to plan_nodes
    for pred_id, (pred_op, d, f) in enumerate(
        zip(
            wl_graph.predicate_operators,
            wl_graph.predicate_depths,
            wl_graph.predicate_features,
        )
    ):
        pred_dict[pred_id] = (preds_no[pred_op], pred_op, d)
        preds_no[pred_op] += 1
        if pred_op == 0:
            pred_feats.append(f)
    # maintain mapping from plan to predicates
    plan_pred_ops = []
    plan_pred_ids = []
    for pred_id in wl_graph.plan_to_filter_mapping:
        if pred_id == -1:
            plan_pred_ops.append(-2)
            plan_pred_ids.append(-1)

        else:
            u_node_id, u_pred_op, _ = pred_dict[pred_id]
            plan_pred_ops.append(u_pred_op)
            plan_pred_ids.append(u_node_id)
    # create edges
    data_dict = collections.defaultdict(list)
    max_pred_depth = None
    if len(wl_graph.predicate_depths) > 0:
        max_pred_depth = max(wl_graph.predicate_depths)

    # If only single nodes exist, create dummy graph with self-pointing nodes
    if len(wl_graph.predicate_edges) == 0:
        for node_key, (node_id, pred_op, depth) in pred_dict.items():
            node_t = pred_node_type(pred_op)
            data_dict[(node_t, "", node_t)].append((node_id, node_id))

    else:
        for u, v in wl_graph.predicate_edges:
            u_node_id, u_pred_op, u_depth = pred_dict[u]
            v_node_id, v_pred_op, v_depth = pred_dict[v]
            assert v_pred_op != 0

            u_node_t = pred_node_type(u_pred_op)
            v_node_t = pred_node_type(v_pred_op)

            data_dict[(u_node_t, f"{v_node_t}_d{v_depth}", v_node_t)].append(
                (u_node_id, v_node_id)
            )

    num_nodes_dict = {
        "pred": preds_no[0],
        "and": preds_no[1],
        "or": preds_no[-1],
    }
    assert data_dict, "predicate_graph dictionary is empty"
    graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    graph.max_depth = max_pred_depth
    return graph, plan_pred_ids, plan_pred_ops, pred_feats


def pred_node_type(pred_op):
    if pred_op == 1:
        return "and"
    elif pred_op == -1:
        return "or"
    elif pred_op == 0:
        return "pred"
    else:
        raise NotImplementedError
