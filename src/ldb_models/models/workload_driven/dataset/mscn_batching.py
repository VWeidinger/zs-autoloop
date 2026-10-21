from collections import defaultdict
from typing import Dict, Tuple

import dgl
import numpy as np

from ldb_models.models.white_mamba.extract_query_stats import (
    extract_query_stats_for_plan,
)
from ldb_models.models.workload_driven.dataset.plan_tree_batching import (
    encode_one_hot,
    encode_predicate,
    tensorize_feats,
)
from ldb_models.models.zeroshot.postgres_plan_batching import (
    assign_additional_plan_info,
)
from ldb_models.training.dataset.cmp_operator import LogicalOperator


def extract_predicate_features(
    n,
    pred_feats,
    column_statistics,
    dim_word_emdb,
    dim_word_hash,
    feature_statistics,
    word_embeddings,
    col_id_dict: Dict[Tuple[str, str], int],
    stable_literal_hash: bool = False,
):
    # (column one hot, operator one hot, predicate)
    if n["operator"] not in {str(LogicalOperator.AND), str(LogicalOperator.OR)}:
        pred_feat_vec = encode_predicate(
            column_statistics,
            dim_word_emdb,
            dim_word_hash,
            feature_statistics=feature_statistics,
            filter_column=n,
            word_embeddings=word_embeddings,
            col_id_dict=col_id_dict,
            stable_literal_hash=stable_literal_hash,
        )
        pred_feats.append(pred_feat_vec)

    if "children" in n.keys():
        for c in n["children"]:
            extract_predicate_features(
                n=c,
                pred_feats=pred_feats,
                column_statistics=column_statistics,
                dim_word_emdb=dim_word_emdb,
                dim_word_hash=dim_word_hash,
                feature_statistics=feature_statistics,
                word_embeddings=word_embeddings,
                col_id_dict=col_id_dict,
                stable_literal_hash=stable_literal_hash,
            )


def extract_set_features(
    plan,
    feature_statistics: Dict,
    dim_word_hash,
    dim_word_emdb,
    column_statistics,
    word_embeddings,
    dim_bitmaps,
    table_id_dict: Dict[str, int],
    col_id_dict: Dict[Tuple[str, str], int],
    table_feats=None,
    pred_feats=None,
    stable_literal_hash: bool = False,
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
            column_statistics=column_statistics,
            dim_word_emdb=dim_word_emdb,
            dim_word_hash=dim_word_hash,
            feature_statistics=feature_statistics,
            word_embeddings=word_embeddings,
            col_id_dict=col_id_dict,
            stable_literal_hash=stable_literal_hash,
        )

    # table feature set#

    if "table_id" in params.keys():
        # run_table_id = params["table_id"] # not used, but could be useful in the future if we want to match the base table more robustly (e.g. in case of self joins)
        table_name = params["table_name"]

        # lookup global table id
        table_id = table_id_dict[table_name]

        # one hot encoding
        dim_tables = len(table_id_dict)
        table_vec = np.zeros(dim_tables)
        table_vec[table_id] = 1
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
                word_embeddings,
                dim_bitmaps,
                table_id_dict=table_id_dict,
                col_id_dict=col_id_dict,
                table_feats=table_feats,
                pred_feats=pred_feats,
                stable_literal_hash=stable_literal_hash,
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
    feature_statistics: Dict,
    db_statistics: Dict,
    table_id_dict: Dict[str, int],
    col_id_dict: Dict[Tuple[str, str], int],
    column_statistics=None,
    word_embeddings=None,
    dim_word_hash=None,
    dim_word_emdb=None,
    dim_bitmaps=None,
    db_schema_info: Dict = None,
    stable_literal_hash: bool = False,
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
    features = {"table": [], "pred": [], "agg": [], "join": []}
    edge_dict = {
        ("table", "table_plan", "plan"): [],
        ("pred", "pred_plan", "plan"): [],
        ("agg", "agg_plan", "plan"): [],
        ("join", "join_plan", "plan"): [],
    }
    labels = []

    query_stats_dict = defaultdict(list)

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
            dim_word_hash=dim_word_hash,
            dim_word_emdb=dim_word_emdb,
            column_statistics=column_statistics,
            word_embeddings=word_embeddings,
            dim_bitmaps=dim_bitmaps,
            table_id_dict=table_id_dict,
            col_id_dict=col_id_dict,
            stable_literal_hash=stable_literal_hash,
        )

        join_conditions = get_all_join_conds(p, feature_statistics)
        curr_agg_feats, curr_join_feats = extract_joins_aggregations(
            feature_statistics,
            p,
            join_conditions,
            num_cols=len(col_id_dict),
            col_id_dict=col_id_dict,
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
    feature_statistics: dict,
    p: dict,
    join_conditions: list,
    num_cols: int,
    col_id_dict: Dict[Tuple[str, str], int],
):
    # join feature set
    curr_join_feats = []

    for join_cond in join_conditions:
        join_vec = encode_one_hot("join", feature_statistics, join_cond)
        curr_join_feats.append(join_vec)
    # output column set
    curr_agg_feats = []

    # dimension of column one hot
    for output_col in p["plan_parameters"]["output_columns"]:
        if output_col["aggregation"] is None:
            continue
        agg_vec = encode_one_hot(
            "aggregation", feature_statistics, output_col["aggregation"]
        )
        col_vec = np.zeros(num_cols)
        for c_idx in output_col["columns"]:
            table_name = c_idx["table_name"]
            col_name = c_idx["col_name"]
            col_vec[col_id_dict[(table_name, col_name)]] = 1
        agg_feat = np.concatenate([agg_vec, col_vec])
        curr_agg_feats.append(agg_feat)
    return curr_agg_feats, curr_join_feats
