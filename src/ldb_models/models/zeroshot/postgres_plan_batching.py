import collections
from typing import DefaultDict, Dict, List, Tuple

import dgl
import numpy as np
import torch
from models.white_mamba.extract_query_stats import extract_query_stats_for_plan
from models.zeroshot.annotate_additional_plan_info import assign_additional_plan_info
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from training.dataset.cmp_operator import Operator
from training.featurizations import Featurization
from training.preprocessing.feature_statistics import FeatureType


def encode(column, plan_params, feature_statistics):
    #  fallback in case actual cardinality is not in plan parameters
    if column == "act_card" and column not in plan_params:
        value = 0
    else:
        value = plan_params[column]
    if feature_statistics[column].get("type") == str(FeatureType.numeric):
        enc_value = value  # we do the feature scaling later on - more efficiently
    elif feature_statistics[column].get("type") == str(FeatureType.categorical):
        value_dict = feature_statistics[column]["value_dict"]
        assert len(list(value_dict.keys())) > 1, (
            f"Value dict has only one key: {value_dict} for feature {column}"
        )
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        assert str(value) in value_dict, f"{value} not in {value_dict.keys()}"
        enc_value = value_dict[str(value)]
    else:
        raise NotImplementedError
    return enc_value


def plan_to_graph(
    node,
    run_file_id: int,
    plan_depths,
    plan_features,
    subplan_labels,
    subplan_card_labels,
    subplan_est_card_labels,
    plan_to_plan_edges,
    db_statistics,
    feature_statistics,
    filter_to_plan_edges,
    predicate_col_features,
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
    parent_node_id=None,
    depth=0,
    skip_top_aggregation: bool = False,
    filter_inline_col_feats: bool = True,
    annotate_base_table_card_to_scan_op: bool = False,
    subplan_label_op_time: bool = False,
):
    plan_params = node["plan_parameters"]

    if "Aggregate" in plan_params["op_name"] and skip_top_aggregation:
        assert len(node["children"]) == 1, (
            f"Aggregation has more than one child: {node}"
        )
        plan_to_graph(
            node["children"][0],
            run_file_id,
            plan_depths,
            plan_features,
            subplan_labels,
            subplan_card_labels,
            subplan_est_card_labels,
            plan_to_plan_edges,
            db_statistics,
            feature_statistics,
            filter_to_plan_edges,
            predicate_col_features,
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
            parent_node_id=parent_node_id,
            depth=depth,
            skip_top_aggregation=False,
            filter_inline_col_feats=filter_inline_col_feats,
            annotate_base_table_card_to_scan_op=annotate_base_table_card_to_scan_op,
            subplan_label_op_time=subplan_label_op_time,
        )
        return

    plan_node_id = len(plan_depths)

    subplan_card_labels.append(plan_params.get("act_card", 1.0))
    subplan_est_card_labels.append(plan_params.get("est_card", 1.0))

    # Extract the ground-truth runtime for this subplan (plan node)
    if subplan_label_op_time:
        if "act_time" in plan_params:
            act_time = plan_params["act_time"]
            children_act_times = [
                child["plan_parameters"].get("act_time", 0)
                for child in node["children"]
            ]
            sum_child_time = sum(children_act_times) if children_act_times else 0

            op_time = act_time - sum_child_time
            # Clamp at 0, as operation time cannot be negative (due to measurement noise)
            subplan_labels.append(max(op_time, 1e-6))
        else:
            # Cannot compute op_time without act_time
            print("No act_time in plan parameters for op_time calculation")
            subplan_labels.append(1e-6)
    else:
        if "act_time" in plan_params:
            if len(node["children"]) > 0:
                children_act_times = [
                    child["plan_parameters"]["act_time"] for child in node["children"]
                ]
                sum_child_time = sum(children_act_times) if children_act_times else 0
                if plan_params["act_time"] < sum_child_time:
                    subplan_labels.append(sum_child_time)
                else:
                    subplan_labels.append(plan_params["act_time"])
            else:
                subplan_labels.append(plan_params["act_time"])
        else:
            print("No act_time in plan parameters")
            if len(node["children"]) > 0:
                children_act_times = [
                    child["plan_parameters"]["act_time"] for child in node["children"]
                ]
                print(f"Using children act_time: {children_act_times}")
                sum_child_time = sum(children_act_times) if children_act_times else 0
                subplan_labels.append(sum_child_time)
            else:
                subplan_labels.append(1e-6)

    is_scan_op = "table_id" in plan_params

    # add plan features
    curr_plan_features = []
    for feature_name in plan_featurization.PLAN_FEATURES:
        if (
            feature_name in ["est_children_card", "act_children_card"]
            and is_scan_op
            and annotate_base_table_card_to_scan_op
        ):
            # extract base table card
            base_table_card = db_statistics[run_file_id]["table_stats"][
                plan_params["table_id"]
            ]["reltuples"]

            # annotate the base table card to the scan op
            curr_plan_features.append(
                encode(
                    "base_table_card",
                    {feature_name: base_table_card},
                    feature_statistics,
                )
            )
        else:
            curr_plan_features.append(
                encode(feature_name, plan_params, feature_statistics)
            )

    plan_features.append(curr_plan_features)
    plan_depths.append(depth)

    # encode output columns which can in turn have several columns as a product in the aggregation
    output_columns = plan_params.get("output_columns")
    if output_columns is not None:
        for output_column in output_columns:
            output_column_id_list = [col["col_id"] for col in output_column["columns"]]
            output_column_node_id = output_column_idx.get(
                (output_column["aggregation"], str(output_column_id_list), run_file_id)
            )

            # if not, create
            if output_column_node_id is None:
                curr_output_column_features = [
                    encode(column, output_column, feature_statistics)
                    for column in plan_featurization.OUTPUT_COLUMN_FEATURES
                ]

                output_column_node_id = len(output_column_features)
                output_column_features.append(curr_output_column_features)
                output_column_idx[
                    (
                        output_column["aggregation"],
                        str(output_column_id_list),
                        run_file_id,
                    )
                ] = output_column_node_id

                # featurize product of columns if there are any
                run_file_column_features = db_statistics[run_file_id]["column_stats"]
                assert run_file_column_features is not None
                for column in output_column["columns"]:
                    # lookup column node id or create node if not yet existing
                    column_node_id = get_or_add_col_id(
                        column["col_id"],
                        column_idx,
                        run_file_id,
                        run_file_column_features=run_file_column_features,
                        feature_statistics=feature_statistics,
                        plan_featurization=plan_featurization,
                        column_features=column_features,
                    )
                    column_to_output_column_edges.append(
                        (column_node_id, output_column_node_id)
                    )

            # in any case add the corresponding edge
            output_column_to_plan_edges.append((output_column_node_id, plan_node_id))

    # filter_columns (we do not reference the filter columns to columns since we anyway have to create a node per filter
    #  node)
    filter = plan_params.get("filter")
    if filter is not None:
        assert "is_index_cond" in filter, (
            f"No is_index_cond in filter: {filter} / {node}"
        )
        db_column_features = db_statistics[run_file_id]["column_stats"]
        try:
            parse_predicates(
                db_column_features,
                feature_statistics,
                filter,
                filter_to_plan_edges,
                column_to_filter_edges=column_to_filter_edges,
                plan_featurization=plan_featurization,
                predicate_col_features=predicate_col_features,
                predicate_depths=predicate_depths,
                intra_predicate_edges=intra_predicate_edges,
                logical_preds=logical_preds,
                column_features=column_features,
                column_idx=column_idx,
                run_file_id=run_file_id,
                plan_node_id=plan_node_id,
                inline_col_feats=filter_inline_col_feats,
            )
        except Exception as e:
            print(f"Error in filter: {filter}", flush=True)
            raise e

    # featurize join conditions
    join = plan_params.get("join")

    # check if JOIN is a condition of the form: t1.c1 = t1.c2 (i.e. it is a filter rather than a join)
    # in this case skip this join
    if is_scan_op and join is not None:
        if join["table_name1"] == join["table_name2"]:
            # this is a filter rather than a join
            # currently we cannot featurize this
            # skip this
            join = None

    if join is not None:
        assert "is_index_cond" in join, f"No is_index_cond in join: {join} / {node}"
        run_file_column_features = db_statistics[run_file_id]["column_stats"]
        try:
            parse_predicates(
                run_file_column_features,
                feature_statistics,
                join,
                filter_to_plan_edges,
                column_to_filter_edges=column_to_filter_edges,
                plan_featurization=plan_featurization,
                predicate_col_features=predicate_col_features,
                predicate_depths=predicate_depths,
                intra_predicate_edges=intra_predicate_edges,
                logical_preds=logical_preds,
                column_features=column_features,
                column_idx=column_idx,
                run_file_id=run_file_id,
                plan_node_id=plan_node_id,
                is_join=True,
                inline_col_feats=filter_inline_col_feats,
            )
        except Exception as e:
            print(f"Error in join: {join}", flush=True)
            raise e

    assert not (filter is not None and join is not None), (
        f"Filter and join in same plan node: {node} - the parser should prevent that! In this cases messages will get averaged. If this is wanted, please add automatic creation of a logical plan node that combines both filters and joins."
    )

    # tables
    table_id = plan_params.get("table_id")
    if (
        "Scan" in plan_params["op_name"]
        and plan_params["op_name"] != "Bitmap Index Scan"
    ):
        assert table_id is not None, f"No table in scan: {plan_params}"
    if table_id is not None:
        assert isinstance(table_id, int), (
            f"Table id is not an int: {table_id} / {plan_params}"
        )
        table_node_id = table_idx.get((table_id, run_file_id))
        run_file_table_statistics = db_statistics[run_file_id]["table_stats"]

        if table_node_id is None:
            curr_table_features = [
                encode(
                    feature_name,
                    run_file_table_statistics[table_id],
                    feature_statistics,
                )
                for feature_name in plan_featurization.TABLE_FEATURES
            ]
            table_node_id = len(table_features)
            table_features.append(curr_table_features)
            table_idx[(table_id, run_file_id)] = table_node_id

        table_to_plan_edges.append((table_node_id, plan_node_id))

    # add edge to parent
    if parent_node_id is not None:
        plan_to_plan_edges.append((plan_node_id, parent_node_id))

    # continue recursively
    for i, c in enumerate(node["children"]):
        plan_to_graph(
            c,
            run_file_id,
            plan_depths,
            plan_features,
            subplan_labels,
            subplan_card_labels,
            subplan_est_card_labels,
            plan_to_plan_edges,
            db_statistics,
            feature_statistics,
            filter_to_plan_edges,
            predicate_col_features,
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
            parent_node_id=plan_node_id,
            depth=depth + 1,
            skip_top_aggregation=skip_top_aggregation,
            filter_inline_col_feats=filter_inline_col_feats,
            annotate_base_table_card_to_scan_op=annotate_base_table_card_to_scan_op,
            subplan_label_op_time=subplan_label_op_time,
        )


def print_errors_recursive(node, depth=0):
    plan_params = node["plan_parameters"]
    indent = "--" * depth
    est = int(plan_params.get("est_card", -1))
    est_pre = int(plan_params.get("est_card_pre", -1))
    act = int(plan_params.get("act_card", -1))

    est_bu = int(plan_params.get("est_card_bu", -1))
    est_td = int(plan_params.get("est_card_td", -1))

    op_name = plan_params.get("op_name", "Unknown")
    print(
        f"{indent}Op: {op_name} | Est: {est_pre} | Est_Upd: {est}| Act: {act} | Est_BU: {est_bu} | Est_TD: {est_td}",
        flush=True,
    )
    children = node.get("children", [])
    len_children = len(children)
    if len_children == 2:
        # if one is join, print it second
        op_name = plan_params.get("op_name", "").lower()
        if "join" in op_name or "nested loop" in op_name:
            print_errors_recursive(children[1], depth + 1)
            print_errors_recursive(children[0], depth + 1)
        else:
            print_errors_recursive(children[0], depth + 1)
            print_errors_recursive(children[1], depth + 1)
    elif len_children == 1:
        print_errors_recursive(children[0], depth + 1)
    else:
        for c in children:
            print_errors_recursive(c, depth + 1)


def get_operator_type(op_name):
    op_name = op_name.lower()
    # Scans (Base cases)
    if "scan" in op_name:
        # Index Scan, Seq Scan, Index Only Scan
        return "SCAN"

    # Joins (Multiplicative error propagation)
    elif "join" in op_name or "nested loop" in op_name:
        # Hash Join, Merge Join, Nested Loop Join, Index Nested Loop Join
        return "JOIN"

    # Aggregates (Cardinality reset or dependent on grouping)
    elif "aggregate" in op_name or "group" in op_name:
        if "simple" in op_name:
            # e.g., "Simple Aggregate" (COUNT(*)) -> Always returns 1 row
            return "SCALAR_AGG"
        return "GROUP_AGG"

    elif "sort" in op_name or "materialize" in op_name or "hash" in op_name:
        # Sort, Hash, Materialize in op_name:
        return "Pass-through"

    else:
        raise NotImplementedError(f"Unknown operator type for op_name: {op_name}")


def get_or_add_col_id(
    column_id: int,
    column_idx: Dict,
    run_file_id: int,
    run_file_column_features: Dict,
    feature_statistics: Dict,
    plan_featurization,
    column_features: List,
):
    column_node_id = column_idx.get((str(column_id), run_file_id))
    if column_node_id is None:
        assert column_id < len(run_file_column_features), (
            f"Column {column_id} w. run file id: {run_file_id} not in run_file_column_features: {len(run_file_column_features)} / {run_file_column_features}"
        )
        try:
            curr_column_features = [
                encode(
                    feature_name,
                    run_file_column_features[column_id],
                    feature_statistics,
                )
                for feature_name in plan_featurization.COLUMN_FEATURES
            ]
        except Exception as e:
            print(
                f"Error in column: {column_id} / {run_file_column_features[column_id]} / {str(feature_statistics)[:100]}",
                flush=True,
            )
            raise e
        column_node_id = len(column_features)
        column_features.append(curr_column_features)
        column_idx[(str(column_id), run_file_id)] = column_node_id

    return column_node_id


def parse_predicates(
    run_file_column_features,
    feature_statistics,
    filter,
    filter_to_plan_edges,
    column_to_filter_edges,
    plan_featurization,
    predicate_col_features,
    predicate_depths,
    intra_predicate_edges,
    logical_preds,
    column_features,
    column_idx: Dict,
    run_file_id: int,
    plan_node_id=None,
    parent_filter_node_id=None,
    depth=0,
    is_join: bool = False,
    inline_col_feats: bool = True,
):
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

    assert "operator" in filter, f"No operator in filter: {filter}"

    filter["is_join_cond"] = 1 if is_join else 0

    # gather features
    if filter["operator"] in {str(op) for op in list(Operator)}:
        curr_filter_features = [
            encode(feature_name, filter, feature_statistics)
            for feature_name in plan_featurization.FILTER_FEATURES
        ]

        if inline_col_feats:
            # inline the column features to the filter features - do not add a separate node for the column
            if is_join:
                filter["col_id"] = filter["col_id1"]

            if "col_name" in filter or "column_name" in filter:
                assert "col_id" in filter, f"No col_id in filter: {filter}"

            if "col_id" in filter and filter["col_id"] is not None:
                curr_filter_col_feats = [
                    encode(
                        column,
                        run_file_column_features[filter["col_id"]],
                        feature_statistics,
                    )
                    for column in plan_featurization.COLUMN_FEATURES
                ]
            # hack for cases in which we have no base filter column (e.g., in a having clause where the column is some
            # result column of a subquery/groupby). In the future, this should be replaced by some graph model that also
            # encodes the structure of this output column
            else:
                curr_filter_col_feats = [0 for _ in plan_featurization.COLUMN_FEATURES]
            curr_filter_features += curr_filter_col_feats
        else:
            if is_join:
                col_id1 = filter["col_id1"]
                col_id2 = filter["col_id2"]

                # create col nodes or get existing ones
                col_node_id1 = get_or_add_col_id(
                    col_id1,
                    column_idx=column_idx,
                    run_file_id=run_file_id,
                    run_file_column_features=run_file_column_features,
                    feature_statistics=feature_statistics,
                    plan_featurization=plan_featurization,
                    column_features=column_features,
                )
                col_node_id2 = get_or_add_col_id(
                    col_id2,
                    column_idx=column_idx,
                    run_file_id=run_file_id,
                    run_file_column_features=run_file_column_features,
                    feature_statistics=feature_statistics,
                    plan_featurization=plan_featurization,
                    column_features=column_features,
                )

                # add edge col to filter
                column_to_filter_edges.append((col_node_id1, filter_node_id))
                column_to_filter_edges.append((col_node_id2, filter_node_id))
            else:
                col_id = filter["col_id"]
                # create col nodes or get existing ones
                col_node_id = get_or_add_col_id(
                    col_id,
                    column_idx=column_idx,
                    run_file_id=run_file_id,
                    run_file_column_features=run_file_column_features,
                    feature_statistics=feature_statistics,
                    plan_featurization=plan_featurization,
                    column_features=column_features,
                )
                # add edge col to filter
                column_to_filter_edges.append((col_node_id, filter_node_id))

        logical_preds.append(False)

    else:
        curr_filter_features = [
            encode(feature_name, filter, feature_statistics)
            for feature_name in plan_featurization.FILTER_FEATURES
        ]

        # bring logical pred features to the same length as the normal filter features
        if inline_col_feats:
            # pad with 0 - there are not col features for logical preds
            curr_filter_features = curr_filter_features + [
                0 for _ in plan_featurization.COLUMN_FEATURES
            ]

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
    if "children" in filter:
        for c in filter["children"]:
            parse_predicates(
                run_file_column_features,
                feature_statistics,
                c,
                filter_to_plan_edges,
                column_to_filter_edges=column_to_filter_edges,
                plan_featurization=plan_featurization,
                predicate_col_features=predicate_col_features,
                predicate_depths=predicate_depths,
                intra_predicate_edges=intra_predicate_edges,
                logical_preds=logical_preds,
                column_features=column_features,
                column_idx=column_idx,
                run_file_id=run_file_id,
                parent_filter_node_id=filter_node_id,
                depth=depth + 1,
                is_join=is_join,
                inline_col_feats=inline_col_feats,
            )


# run the feature scaling
def apply_feature_scalers(features, feature_statistics, feature_names: List[str]):
    # convert features to numpy array
    try:
        features = np.array(features, dtype=np.float32)
    except Exception as e:
        print(len(features), flush=True)
        if len(features) > 0:
            print(len(features[0]), flush=True)

        raise e

    # Handle empty feature lists
    if features.size == 0:
        # return an empty 2D array with zero rows and correct number of columns
        return np.zeros((0, len(feature_names)), dtype=np.float32)

    # If features is 1D, it can mean two things:
    #  - a single sample row with len == n_features (shape should be (1, n_features))
    #  - a single feature column with shape (n_samples,) where n_features == 1
    # Normalize both cases to a 2D array with shape (n_samples, n_features)
    if features.ndim == 1:
        if features.shape[0] == len(feature_names):
            # single row: reshape to (1, n_features)
            features = features.reshape((1, -1))
        else:
            # assume single-column features: reshape to (n_samples, 1)
            features = features.reshape((-1, 1))

    # Validate that the number of columns matches expected feature names
    if features.shape[1] != len(feature_names):
        raise AssertionError(
            f"Features have {features.shape[1]} columns but expected {len(feature_names)} ({feature_names})"
        )

    scaled_features = []
    for i, column in enumerate(feature_names):
        # extract features of this column (always 1D)
        column_features = features[:, i]

        # apply scalers
        if feature_statistics[column].get("type") == str(FeatureType.numeric):
            try:
                scaler = feature_statistics[column].get("scaler")
                if scaler is None:
                    # Fallback: if no scaler is set, don't transform (identity)
                    scaled = column_features
                else:
                    scaled = scaler.transform(column_features.reshape((-1, 1))).reshape(
                        (-1,)
                    )
            except Exception as e:
                print(
                    f"Error applying scaler for column {column} with features {column_features}",
                    flush=True,
                )
                raise e
        else:
            scaled = column_features

        assert len(scaled) == len(column_features), (
            f"Scaled features {len(scaled)} do not match original features {len(column_features)}"
        )

        # store in list
        scaled_features.append(scaled)

    assert len(scaled_features) == len(feature_names), (
        f"Scaled features {len(scaled_features)} do not match original features {len(feature_names)}"
    )

    # concat the features
    scaled_features = np.stack(scaled_features, axis=1)

    assert scaled_features.shape == features.shape, (
        f"Scaled features shape {scaled_features.shape} does not match original shape {features.shape}"
    )

    return scaled_features


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
    scaler_type: str = "robust",
    subplan_label_op_time: bool = False,
    log_transform_card_labels: bool = False,
    use_log_input_card: bool = False,
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
    for sample_idx, p in plans:
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
        )

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

    return (
        graph,
        postprocessed_features,
        labels,
        {
            "sample_idxs": sample_idxs,
            **query_stats_dict,
            "subplan_labels": subplan_labels,
            "plan_dict": plan_dict,
            "subplan_card_labels": subplan_card_labels,
            "subplan_est_card_labels": subplan_est_card_labels,
        },
    )


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


def get_depths(plan_depths, predicate_depths):
    max_depth = max(plan_depths)
    max_pred_depth = 0
    if len(predicate_depths) > 0:
        max_pred_depth = max(predicate_depths)
    return max_depth, max_pred_depth


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


def add_numerical_scalers(feature_statistics: Dict, scaler_type: str):
    for k, v in feature_statistics.items():
        if v["type"] == str(FeatureType.numeric):
            if scaler_type == "robust":
                scaler = RobustScaler()
                scaler.center_ = v["center"]
                scaler.scale_ = v["scale"]
                feature_statistics[k]["scaler"] = scaler
            elif scaler_type == "minmax":
                feature_range = (0, 1)
                scaler = MinMaxScaler(feature_range=feature_range)
                scaler.data_min_ = np.array([v["min"]], dtype=np.float32)
                scaler.data_max_ = np.array([v["max"]], dtype=np.float32)

                range_diff = v["max"] - v["min"]
                if range_diff == 0:
                    # avoid division by zero
                    range_diff = 1.0
                scaler.data_range_ = np.array([range_diff], dtype=np.float32)

                scaler.scale_ = (
                    np.array([feature_range[1] - feature_range[0]], dtype=np.float32)
                    / scaler.data_range_
                )
                scaler.min_ = (
                    np.array([feature_range[0]], dtype=np.float32)
                    - scaler.data_min_ * scaler.scale_
                )
                scaler.n_samples_seen_ = 1  # any positive int to mark as "fit"

                feature_statistics[k]["scaler"] = scaler
            else:
                raise ValueError(
                    f"Unknown scaler type: {scaler_type}. Use 'robust' or 'minmax'."
                )


def pred_node_type_id(logical_preds, pred_dict, u):
    if logical_preds[u]:
        u_node_id, depth = pred_dict[u]
        node_type = f"logical_pred_{depth}"
    else:
        u_node_id = pred_dict[u]
        node_type = "filter_column"
    return node_type, u_node_id
