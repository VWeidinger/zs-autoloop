from typing import Tuple, List, Dict
import numpy as np
from models.zeroshot.postgres_plan_batching import (
    add_numerical_scalers,
    apply_feature_scalers,
    assign_additional_plan_info,
    postprocess_labels,
)
from training.preprocessing.feature_statistics import FeatureType
import torch
from collections import defaultdict
from functools import lru_cache

def augment_feature_stats_for_mamba(feature_statistics: Dict):
    feature_statistics["left_child_card"] = feature_statistics["est_card"]
    feature_statistics["right_child_card"] = feature_statistics["est_card"]
    feature_statistics["base_table_card"] = feature_statistics[
        "est_card"
    ]  # base table card is the same as est_card

    # update op_name list
    op_name_values = phys_op_scan_list + phys_op_join_list
    feature_statistics["op_name"]["no_vals"] = len(op_name_values)
    feature_statistics["op_name"]["value_dict"] = {
        name: i for i, name in enumerate(op_name_values)
    }

    # add pseudo entry for is_pk
    feature_statistics["left_is_pk"] = {
        "type": str(FeatureType.numeric),
        "min": 0,
        "max": 1,
        "avg": 0.5,
        "scale": 0.5,
        "center": 0.5,
    }
    feature_statistics["right_is_pk"] = {
        "type": str(FeatureType.numeric),
        "min": 0,
        "max": 1,
        "avg": 0.5,
        "scale": 0.5,
        "center": 0.5,
    }


def mamba_plan_collator(
    plans: List[Tuple],
    feature_statistics: Dict,
    plan_featurization,
    db_statistics: Dict,
    db_schema_info: Dict,
    output_tensors: bool = False,
):
    # edge information
    edge_ids = []  # list edges (src ndoe id, dst node id)
    edge_left_right_information = []  # Edge types: 0=left child, 1=right child

    # node information
    plan_ids = []  # plan id for each node (used for batching)
    node_depths = []  # depth of each node in the plan
    node_type_lookup = []  # stores the node type (join/scan) for each node (join=0, scan=1)
    node_type_id_lookup = []  # stores the scan/join id for the global id

    # node features
    join_features = []  # node features for joins
    scan_features = []  # node features for scans

    # plan information
    labels = []
    sql_statements = []
    hints = []

    # other test information
    selectivity_list = []
    scan_op_list = []
    join_op_list = []
    num_tables_list = []
    graph_depth_list = []
    sample_idxs = []

    # augment feature_statistics
    augment_feature_stats_for_mamba(feature_statistics)

    # prepare robust encoder for the numerical fields
    add_numerical_scalers(feature_statistics)

    for plan_id, plan in plans:
        sample_idxs.append(plan_id)  # store the sample index
        labels.append(plan["plan_runtime_ms"])
        sql_statements.append(plan["sql"])
        hints.append(plan["hint"])
        num_tables_list.append(plan["num_tables"])

        # annotate additional info
        assign_additional_plan_info(plan, db_statistics)
        selectivity_list += plan["selectivity"]
        scan_op_list += plan["scan_op"]
        join_op_list += plan["join_op"]

        num_join_nodes_before = len(join_features)
        num_scan_nodes_before = len(scan_features)

        plan_depth_start_idx = len(
            node_depths
        )  # store the start index of the plan depth

        # parse the plan
        plan_to_graph(
            plan,
            database_id=plan["database_id"],
            depth=0,
            scan_features_list=scan_features,
            join_features_list=join_features,
            join_edges=edge_ids,
            edge_left_right_information=edge_left_right_information,
            node_depths_list=node_depths,
            node_type_lookup=node_type_lookup,
            node_type_id_lookup=node_type_id_lookup,
            feature_statistics=feature_statistics,
            plan_featurization=plan_featurization,
            db_statistics=db_statistics,
            db_schema_info=db_schema_info,
            parent_node_id=None,
            node_is_left=None,
        )

        # add plan id to the features
        plan_ids.extend(
            [plan_id]
            * (
                len(join_features)
                - num_join_nodes_before
                + len(scan_features)
                - num_scan_nodes_before
            )
        )

        # assert that the number of join and scan features match the number of join and scan depths
        assert len(join_features) + len(scan_features) == len(node_depths), (
            f"Join features length {len(join_features)} + Scan features length {len(scan_features)} does not match node depths length {len(node_depths)}"
        )
        assert len(edge_ids) == len(edge_left_right_information), (
            f"Edge ids length {len(edge_ids)} does not match edge left/right information length {len(edge_left_right_information)}"
        )
        assert len(node_type_lookup) == len(node_depths), (
            f"Node type lookup length {len(node_type_lookup)} does not match node depths length {len(node_depths)}"
        )
        assert len(node_type_id_lookup) == len(node_depths), (
            f"Node type id length {len(node_type_id_lookup)} does not match node depths length {len(node_depths)}"
        )
        assert len(plan_ids) == len(node_depths), (
            f"Plan ids length {len(plan_ids)} does not match node depths length {len(node_depths)}"
        )

        graph_depth_list.append(max(node_depths[plan_depth_start_idx:], default=0))

    # add pseudo entry for filter features
    local_feature_statistics = (
        feature_statistics.copy()
    )  # make a copy to avoid modifying the original
    local_feature_statistics["filter_flat_vector"] = {
        "type": str(FeatureType.categorical)
    }
    local_feature_statistics["outcols_flat_vector"] = {
        "type": str(FeatureType.categorical)
    }

    # apply feature scaling
    if len(join_features) > 0:
        join_features = apply_feature_scalers(
            join_features,
            feature_statistics=local_feature_statistics,
            feature_names=plan_featurization.JOIN_FEATURES,
        )
    else:
        join_features = np.zeros(
            (0, len(plan_featurization.JOIN_FEATURES)), dtype=np.float32
        )

    assert len(scan_features) > 0, "There should be at least one scan feature."
    scan_features = apply_feature_scalers(
        scan_features,
        feature_statistics=local_feature_statistics,
        feature_names=expand_scan_featurization(plan_featurization),
    )

    if len(edge_ids) > 0:
        # transpose matrix: [[src,dst]] --> [[src], [dst]]
        edge_ids = np.array(edge_ids).T
        edge_ids = torch.as_tensor(edge_ids, dtype=torch.int64)
    else:
        # if there are no edges, create an empty tensor
        edge_ids = torch.empty((2, 0), dtype=torch.int64)

    edge_left_right_information = torch.as_tensor(
        edge_left_right_information, dtype=torch.int64
    )
    plan_ids = torch.as_tensor(plan_ids, dtype=torch.int64)
    node_depths = torch.as_tensor(node_depths, dtype=torch.int64)
    node_type_lookup = torch.as_tensor(node_type_lookup, dtype=torch.int64)
    node_type_id_lookup = torch.as_tensor(node_type_id_lookup, dtype=torch.int64)
    join_features = torch.from_numpy(join_features)
    scan_features = torch.from_numpy(scan_features)

    # rather deal with runtimes in secs
    labels = postprocess_labels(labels)

    if output_tensors:
        # create tensors
        labels = torch.as_tensor(labels, dtype=torch.float32)
        sample_idxs = torch.arange(len(labels), dtype=torch.int32)

    return (
        edge_ids,
        edge_left_right_information,
        plan_ids,
        node_depths,
        node_type_lookup,
        node_type_id_lookup,
        join_features,
        scan_features,
        labels,
        sample_idxs,
        sql_statements,
        hints,
        selectivity_list,
        scan_op_list,
        join_op_list,
        num_tables_list,
        graph_depth_list,
    )


def encode(column, plan_params, feature_statistics):
    #  fallback in case actual cardinality is not in plan parameters
    if column == "act_card" and column not in plan_params:
        value = 0
    else:
        value = plan_params[column]
    if feature_statistics[column].get("type") == str(FeatureType.numeric):
        enc_value = value  # we do the feature scaling later on - more efficiently
        # enc_value = feature_statistics[column]['scaler'].transform(np.array([[value]])).item()
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


cmp_operators = [
    "=",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "LIKE",
    "NOT LIKE",
    "IN",
    "IS NULL",
    "IS NOT NULL",
    "BETWEEN",
]

def construct_filter_flat_vector(filter: Dict):
    op_counts = defaultdict(int)
    op_on_index_count = defaultdict(int)

    def recurse(node):
        if "children" in node and isinstance(node["children"], list):
            for child in node["children"]:
                recurse(child)
        else:
            op = node.get("operator")
            is_index = node.get("is_index_cond")

            assert op in cmp_operators, (
                f"Operator {op} not in {cmp_operators}. Available operators: {list(node.keys())}"
            )
            op_counts[op] += 1
            if is_index:
                op_on_index_count[op] += 1

    if filter:
        recurse(filter)

    # Return as flat vectors in the order of cmp_operators
    op_count_vec = [op_counts[op] for op in cmp_operators]
    op_on_index_vec = [op_on_index_count[op] for op in cmp_operators]
    return op_count_vec + op_on_index_vec


def get_filter_flat_vector_length():
    # Each operator has two values: count and count on index
    return len(cmp_operators) * 2


col_types = ["integer", "float", "character varying"]

def construct_outcols_flat_vector(outcols: List[str], col_stats: Dict):
    total_width = 0
    dtype_counts = defaultdict(int)

    for outcol in outcols:
        for col in outcol["columns"]:
            col_id = col["col_id"]
            stats = col_stats[col_id]
            dtype_counts[stats["data_type"]] += 1
            total_width += stats["avg_width"]

    dtype_count_vec = [dtype_counts[dt] for dt in col_types]

    # scale the width down by this factor
    total_width /= 50  # apply feature scaling

    return dtype_count_vec + [total_width]


def get_outcols_flat_vector_length():
    return len(col_types) + 1


@lru_cache(maxsize=1000)
def expand_scan_featurization(featurization):
    scan_features = []

    # expand SCAN_FEATURES
    for feature_name in featurization.SCAN_FEATURES:
        if feature_name == "filter_flat_vector":
            # get the length of the filter flat vector
            dim = get_filter_flat_vector_length()

            # add the feature multiple times
            scan_features.extend(["filter_flat_vector"] * dim)
        elif feature_name == "outcols_flat_vector":
            # get the length of the outcols flat vector
            dim = get_outcols_flat_vector_length()

            # construct outcols flat vector
            scan_features.extend(["outcols_flat_vector"] * dim)
        else:
            scan_features.append(feature_name)

    return scan_features


phys_op_scan_list = ["Seq Scan", "Index Scan", "Index Only Scan"]
phys_op_join_list = ["Hash Join", "Merge Join", "Index Nested Loop", "Nested Loop"]


def process_node_features(node, feature_statistics, plan_featurization, db_statistics, db_schema_info, database_id):
    plan_params = node["plan_parameters"]
    phys_op = plan_params["op_name"]

    if phys_op in phys_op_scan_list:
        node_features = []
        for feature_name in plan_featurization.SCAN_FEATURES:
            if feature_name == "base_table_card":
                base_table_card = db_statistics[database_id]["table_stats"][plan_params["table_id"]]["reltuples"]
                enc_value = feature_statistics["est_card"]["scaler"].transform(np.array([[base_table_card]])).item()
                node_features.append(enc_value)
            elif feature_name == "filter_flat_vector":
                node_features.extend(construct_filter_flat_vector(plan_params.get("filter", {})))
            elif feature_name == "outcols_flat_vector":
                node_features.extend(construct_outcols_flat_vector(plan_params.get("outcols", []), db_statistics[database_id]["column_stats"]))
            else:
                node_features.append(encode(feature_name, plan_params, feature_statistics))
        return node_features, 0  # 0 for scan nodes

    elif phys_op in phys_op_join_list:
        node_features = []
        for feature_name in plan_featurization.JOIN_FEATURES:
            if feature_name == "left_child_card":
                left_child_card = node["children"][0]["plan_parameters"]["est_card"]
                enc_value = encode("est_card", {"est_card": left_child_card}, feature_statistics)
                node_features.append(enc_value)
            elif feature_name == "right_child_card":
                right_child_card = node["children"][1]["plan_parameters"]["est_card"]
                enc_value = encode("est_card", {"est_card": right_child_card}, feature_statistics)
                node_features.append(enc_value)
            elif feature_name in ["left_is_pk", "right_is_pk"]:
                is_pk = process_primary_key(feature_name, plan_params, db_schema_info, database_id)
                node_features.append(1 if is_pk else 0)
            else:
                node_features.append(encode(feature_name, plan_params, feature_statistics))
        return node_features, 1  # 1 for join nodes

    raise NotImplementedError(f"Physical operator {phys_op} not implemented.")


def process_primary_key(feature_name, plan_params, db_schema_info, database_id):
    if "table_name1" not in plan_params["join"] or "table_name2" not in plan_params["join"]:
        return False
    join_table = plan_params["join"]["table_name1"] if feature_name == "left_is_pk" else plan_params["join"]["table_name2"]
    join_col = plan_params["join"]["column_name1"] if feature_name == "left_is_pk" else plan_params["join"]["column_name2"]
    return db_schema_info[database_id]["table_col_info"][join_table][join_col].get("pk", False)


def plan_to_graph(
    node: Dict,
    database_id: int,
    depth: int,
    join_features_list: List[List],
    scan_features_list: List[List],
    join_edges: List[Tuple[int, int]],
    edge_left_right_information: List[int],
    node_depths_list: List[int],
    node_type_lookup: List[int],
    node_type_id_lookup: List[int],
    feature_statistics: Dict,
    plan_featurization,
    db_statistics: Dict,
    db_schema_info: Dict,
    parent_node_id: str,
    node_is_left: bool,
):
    plan_params = node["plan_parameters"]
    phys_op = plan_params["op_name"]
    children = node.get("children", None)

    # skip all these intermediate operations
    if (
        "Aggregate" in phys_op or phys_op in [
            "Sort",
            "Hash",
            "Limit",
            "Materialize",
            "Gather",
            "Gather Merge",
        ]
    ):
        # Only one child is expected
        assert children is not None and len(children) == 1, (
            f"Expected one child for {phys_op}, but got {len(children)} children."
        )
        child = children[0]
        
        return plan_to_graph(
            child,
            database_id,
            depth,
            join_features_list,
            scan_features_list,
            join_edges,
            edge_left_right_information,
            node_depths_list,
            node_type_lookup,
            node_type_id_lookup,
            feature_statistics,
            plan_featurization,
            db_statistics,
            db_schema_info,
            parent_node_id,
            node_is_left,
        )

    node_features, node_type = process_node_features(
        node, feature_statistics, plan_featurization, db_statistics, db_schema_info, database_id
    )
    if node_type == 0:
        type_local_node_id = len(scan_features_list)
        scan_features_list.append(node_features)
    else:
        type_local_node_id = len(join_features_list)
        join_features_list.append(node_features)

    node_id = len(node_depths_list)
    node_depths_list.append(depth)
    node_type_lookup.append(node_type)
    node_type_id_lookup.append(type_local_node_id)

    # add edge to parent node if it exists
    if parent_node_id is not None:
        join_edges.append((node_id, parent_node_id))
        edge_left_right_information.append(0 if node_is_left else 1)

    ###
    # process children
    ###
    if node_type == 1:
        # Join node: must have two children
        left_child, right_child = children  # Unpack directly
        # Process left and right children without repeated argument lookups
        plan_to_graph(
            left_child,
            database_id,
            depth + 1,
            join_features_list,
            scan_features_list,
            join_edges,
            edge_left_right_information,
            node_depths_list,
            node_type_lookup,
            node_type_id_lookup,
            feature_statistics,
            plan_featurization,
            db_statistics,
            db_schema_info,
            parent_node_id=node_id,
            node_is_left=True,
        )
        # process right child
        plan_to_graph(
            right_child,
            database_id,
            depth + 1,
            join_features_list,
            scan_features_list,
            join_edges,
            edge_left_right_information,
            node_depths_list,
            node_type_lookup,
            node_type_id_lookup,
            feature_statistics,
            plan_featurization,
            db_statistics,
            db_schema_info,
            parent_node_id=node_id,
            node_is_left=False,
        )
