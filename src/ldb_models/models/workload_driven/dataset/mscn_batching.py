from collections import defaultdict
from typing import Dict

import dgl
import numpy as np
from models.white_mamba.extract_query_stats import extract_query_stats_for_plan
from models.workload_driven.dataset.plan_tree_batching import (
    encode_one_hot,
    encode_predicate,
    extract_dimensions,
    tensorize_feats,
)
from models.zeroshot.postgres_plan_batching import assign_additional_plan_info
from training.dataset.cmp_operator import LogicalOperator


def extract_predicate_features(
    n,
    pred_feats,
    column_statistics,
    db_statistics,
    dim_word_emdb,
    dim_word_hash,
    feature_statistics,
    word_embeddings,
):
    # (column one hot, operator one hot, predicate)
    if n["operator"] not in {str(LogicalOperator.AND), str(LogicalOperator.OR)}:
        pred_feat_vec = encode_predicate(
            column_statistics,
            db_statistics,
            dim_word_emdb,
            dim_word_hash,
            feature_statistics,
            n,
            word_embeddings,
        )
        pred_feats.append(pred_feat_vec)

    if "children" in n.keys():
        for c in n["children"]:
            extract_predicate_features(
                c,
                pred_feats,
                column_statistics,
                db_statistics,
                dim_word_emdb,
                dim_word_hash,
                feature_statistics,
                word_embeddings,
            )


def extract_set_features(
    plan,
    feature_statistics,
    dim_word_hash,
    dim_word_emdb,
    column_statistics,
    db_statistics,
    word_embeddings,
    dim_bitmaps,
    table_feats=None,
    pred_feats=None,
):
    if table_feats is None:
        table_feats = []
    if pred_feats is None:
        pred_feats = []

    params = plan["plan_parameters"]
    sample_bitmap = np.ones(dim_bitmaps)

    # predicate feature set
    if "filter" in params.keys():
        filter_columns = params["filter"]
        if "sample_vec" not in params.keys():
            raise ValueError("Please make sure to augment sample vectors before use")
        sample_bitmap_vec = params["sample_vec"]
        sample_bitmap[: len(sample_bitmap_vec)] = np.array(sample_bitmap_vec)
        extract_predicate_features(
            filter_columns,
            pred_feats,
            column_statistics,
            db_statistics,
            dim_word_emdb,
            dim_word_hash,
            feature_statistics,
            word_embeddings,
        )

    # table feature set#

    if "table_id" in params.keys():
        table = params["table_id"]
        _, _, dim_tables, _ = extract_dimensions(feature_statistics)
        table_vec = np.zeros(dim_tables)
        table_vec[table] = 1
        table_feat = np.concatenate([table_vec, sample_bitmap])
        table_feats.append(table_feat)

    if "children" in plan.keys():
        for c in plan["children"]:
            extract_set_features(
                c,
                feature_statistics,
                dim_word_hash,
                dim_word_emdb,
                column_statistics,
                db_statistics,
                word_embeddings,
                dim_bitmaps,
                table_feats=table_feats,
                pred_feats=pred_feats,
            )

    return table_feats, pred_feats


def get_condition(params: dict, feature_statistics: dict):
    if "join" in params.keys():
        condition = (
            params["join"]["table_name1"]
            + "."
            + params["join"]["column_name1"]
            + " "
            + params["join"]["operator"]
            + " "
            + params["join"]["table_name2"]
            + "."
            + params["join"]["column_name2"]
        )

        if condition not in feature_statistics["join"]["value_dict"].keys():
            condition = (
                params["join"]["table_name2"]
                + "."
                + params["join"]["column_name2"]
                + " "
                + params["join"]["operator"]
                + " "
                + params["join"]["table_name1"]
                + "."
                + params["join"]["column_name1"]
            )
        assert condition in feature_statistics["join"]["value_dict"].keys(), (
            f"Join condition {condition} not in feature statistics"
        )

        return condition


def get_all_join_conds(plan: dict, feature_statistics: dict):
    join_conditions = []
    params = plan["plan_parameters"]
    if "join" in params.keys():
        if params["join"]["operator"] == str(LogicalOperator.AND):
            for c in plan["children"]:
                cond = get_condition(c, feature_statistics)
                if cond:
                    join_conditions += cond
        else:
            join_condition = get_condition(params, feature_statistics)
            join_conditions.append(join_condition)
    if "children" in plan.keys():
        for c in plan["children"]:
            join_conditions += get_all_join_conds(c, feature_statistics)
    return join_conditions


def mscn_plan_collator(
    plans,
    feature_statistics=None,
    db_statistics=None,
    column_statistics=None,
    word_embeddings=None,
    dim_word_hash=None,
    dim_word_emdb=None,
    dim_bitmaps=None,
    db_schema_info: Dict = None,
):
    """
    Combines physical plans into a large graph that can be fed into ML models.
    :return:
    """
    # features pro query are sets:
    # - table set (table one hot, bitmap)
    # - join set (one hot: which tables are joined)
    # - predicate set (column one hot, operator one hot, predicate one hot)
    # - output column set (which aggregation and which column)

    # compute input dimensions for model
    db_stats = db_statistics[0]
    features = {"table": [], "pred": [], "agg": [], "join": []}
    edge_dict = {
        ("table", "table_plan", "plan"): [],
        ("pred", "pred_plan", "plan"): [],
        ("agg", "agg_plan", "plan"): [],
        ("join", "join_plan", "plan"): [],
    }
    labels = []
    sql_statements = []
    hints = []

    query_stats_dict = defaultdict(list)

    sample_idxs = []
    for plan_id, (sample_idx, p) in enumerate(plans):
        labels.append(p["plan_runtime_ms"])
        assign_additional_plan_info(p, db_statistics, db_schema_info=db_schema_info)

        # extract query stats
        extract_query_stats_for_plan(
            query_stats_dict,
            p,
            sample_idx,
        )

        curr_table_feats, curr_pred_feats = extract_set_features(
            p,
            feature_statistics,
            dim_word_hash,
            dim_word_emdb,
            column_statistics,
            db_stats,
            word_embeddings,
            dim_bitmaps,
        )

        join_conditions = get_all_join_conds(p, feature_statistics)
        curr_agg_feats, curr_join_feats = extract_joins_aggregations(
            feature_statistics, p, join_conditions
        )

        # sanity checks
        assert curr_agg_feats is not None
        # assert len(curr_join_feats) + 1 == len(curr_table_feats)3
        # ToDO: This assertion is not always correct, as some queries do not have joins but still multiple table features.
        # For instance, if a query is having to filters on the same table, then it will have two table features (?)
        # There are however plans that have two "table"-entities inside of them with different sample vector but only one
        # physical table.

        def append_edges_and_feats(features, plan_id, node_type, curr_feats, edges):
            edges += [
                (len(features[node_type]) + i, plan_id) for i in range(len(curr_feats))
            ]
            features[node_type] += curr_feats

        append_edges_and_feats(
            features,
            plan_id,
            "table",
            curr_table_feats,
            edge_dict[("table", "table_plan", "plan")],
        )
        append_edges_and_feats(
            features,
            plan_id,
            "pred",
            curr_pred_feats,
            edge_dict[("pred", "pred_plan", "plan")],
        )
        append_edges_and_feats(
            features,
            plan_id,
            "agg",
            curr_agg_feats,
            edge_dict[("agg", "agg_plan", "plan")],
        )
        if curr_join_feats:
            append_edges_and_feats(
                features,
                plan_id,
                "join",
                curr_join_feats,
                edge_dict[("join", "join_plan", "plan")],
            )

    num_nodes_dict = {k: len(v) for k, v in features.items()}
    num_nodes_dict.update({"plan": len(plans)})

    graph = dgl.heterograph(edge_dict, num_nodes_dict=num_nodes_dict)

    features = tensorize_feats(features)
    labels = np.array(labels, dtype=np.float32)
    labels /= 1000

    return (graph, features, labels, query_stats_dict)


def extract_joins_aggregations(
    feature_statistics: dict, p: dict, join_conditions: list
):
    # join feature set
    curr_join_feats = []

    for join_cond in join_conditions:
        join_vec = encode_one_hot("join", feature_statistics, join_cond)
        curr_join_feats.append(join_vec)
    # output column set
    curr_agg_feats = []
    dim_cols, _, _, _ = extract_dimensions(feature_statistics)
    for output_col in p["plan_parameters"]["output_columns"]:
        if output_col["aggregation"] is None:
            continue
        agg_vec = encode_one_hot(
            "aggregation", feature_statistics, output_col["aggregation"]
        )
        col_vec = np.zeros(dim_cols)
        for c_idx in output_col["columns"]:
            col_vec[c_idx["col_id"]] = 1
        agg_feat = np.concatenate([agg_vec, col_vec])
        curr_agg_feats.append(agg_feat)
    return curr_agg_feats, curr_join_feats
