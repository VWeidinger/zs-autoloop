from typing import Dict, List

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from training.dataset.cmp_operator import Operator
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
