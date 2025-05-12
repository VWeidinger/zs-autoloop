import re
from typing import Tuple

import numpy as np
import torch

from models.zeroshot.postgres_plan_batching import assign_additional_plan_info, postprocess_labels


def is_join(plan_node: dict) -> bool:
    op_name = plan_node["plan_parameters"]["op_name"]
    return op_name in {"Hash Join", "Merge Join", "Nested Loop", "Index Nested Loop"}

def is_scan(plan_node: dict) -> bool:
    op_name = plan_node["plan_parameters"]["op_name"]
    return op_name in {"Seq Scan", "Index Scan", "Bitmap Heap Scan", "Index Only Scan"}

def get_table_id(table_name: str, table_statistics: list) -> int:
    """
    Returns the table id from the table statistics.
    :return: The table id.
    """
    for index, stats in enumerate(table_statistics):
        if stats["relname"] == table_name:
            return index
    raise ValueError(f"Table {table_name} not found in table statistics")


def extract_join_recursively(sql: str, hint: str, plan: dict, table_statistics: list, join_tables: list) -> list:
    if is_join(plan):
        join_info = plan["plan_parameters"]["join"]

        # Quick fix in case of joining three tables - this happens in care cases of transitive joins
        if "children" in join_info:
            print(f"Joining three tables for {sql}")
            join_info = join_info["children"][0]

        # Left side
        table_name1 = join_info["table_name1"]
        col_id1 = join_info["col_id1"]
        col_name1 = join_info["column_name1"].replace("id_", "")

        # Right side
        table_name2 = join_info["table_name2"]
        col_id2 = join_info["col_id2"]
        col_name2 =join_info["column_name2"].replace("id_", "")

        # Check that they are joined on the same column
        #assert col_name1 == col_name2, f"Column names do not match: {join_info["column_name1"]} != {join_info["column_name2"]}"

        left_table_id = get_table_id(table_name1, table_statistics)
        right_table_id = get_table_id(table_name2, table_statistics)

        # Collect join information
        join_tables.append({"left_table": table_name1,
                            "left_col": col_id1,
                            "right_table": table_name2,
                            "right_col": col_id2})

    if "children" in plan.keys():
        for child in plan["children"]:
            join_tables = extract_join_recursively(sql=sql,
                                                   plan=child,
                                                   hint=hint,
                                                   table_statistics=table_statistics,
                                                   join_tables=join_tables)
    return join_tables

def extract_join_matrix(plan: dict, table_statistics: list) -> np.ndarray:
    """
    Extracts the join matrix from a plan, as shown in the paper
    Note that the order of joins is preserved.
    For instance, joining (A, B) and (B, C) is different from (A, C) and (B, C).

    :param plan: The plan from which to extract the join matrix.
    :param table_statistics: The table statistics to know the length of tables
    :return: The join matrix as a numpy array.
    """
    table_ids = init_table_ids(table_statistics)

    joins = []
    joins_infos = extract_join_recursively(sql=plan["sql"],
                                           hint=plan["hint"],
                                           plan=plan,
                                           table_statistics=table_statistics,
                                           join_tables=[])
    for join_info in joins_infos:
        joins.append(sorted([join_info["left_table"], join_info["right_table"]]))

    # Create join matrix
    join_matrix = np.zeros((len(table_statistics), len(table_statistics)))

    for pair in joins:
        left_index = table_ids[pair[0]]
        right_index = table_ids[pair[1]]
        join_matrix[left_index, right_index] = 1
        join_matrix[left_index, right_index] = 1

    # Flatten matrix by getting values above the main diagonal (excluding diagonal), as shown in paper
    rows, cols = np.triu_indices(join_matrix.shape[0], k=1)
    result = join_matrix[rows, cols]

    #assert result.sum() == plan["num_tables"] - 1, f"Expected {plan['num_tables'] - 1} joins, but got {result.sum()} for {plan}"
    return result

def extract_filter_columns_recursively(filter_params: dict, table_name: str, predicates: list):
    if "col_id" in filter_params:
        predicates.append({"col_id": filter_params["col_id"],
                           "col_name": filter_params["col_name"],
                           "table_name": table_name})

    if "children" in filter_params:
        for child_filter in filter_params["children"]:
            extract_filter_columns_recursively(child_filter, table_name, predicates)

def extract_predicates_recursively(plan: dict, column_statistics: dict,
                                   feature_statistics: dict, predicates: list) -> list[dict]:
    plan_params = plan.get("plan_parameters", {})
    table_name = plan_params.get("table_name")

    if is_scan(plan): # When there is a scan, but no filter, we do not add any predicates
        if "filter" in plan_params.keys():
            filter_params = plan_params["filter"]
            extract_filter_columns_recursively(filter_params, table_name, predicates)

    for child in plan.get("children", []):
        extract_predicates_recursively(child, column_statistics, feature_statistics, predicates)
    return predicates

def extract_predicate_encoding(plan: dict, column_statistics: dict, feature_statistics: dict) -> np.ndarray:
    # Extracts the predicate encoding from a plan
    predicates = extract_predicates_recursively(plan, column_statistics, feature_statistics, predicates=[])

    print(f"Predicates: {predicates}")

    # Use only column ids and sort them - we might later implement more sophisticated encoding as written in the paper
    predicate_ids = []
    for p in predicates:
        predicate_ids.append(int(p["col_id"]))
    predicate_ids = np.array(sorted(predicate_ids))

    # Use column ids as indexes in predicate encoding vector
    predicate_encodings = np.zeros(len(column_statistics))
    for p in predicate_ids:
        predicate_encodings[p] = 1
    return predicate_encodings

def extract_plan_encoding_recursively(plan: dict, table_statistics: list, table_ids: list,
                                      plan_encodings: list, depth: int, max_depth: int) -> list:

    assert depth < max_depth, f"Plan depth exceeds maximum limit of {max_depth}"
    # Initialize plan encoding feature vectors for operator type, table access, and index encoding
    operator_vector = np.zeros(2)
    table_vector = np.zeros(len(table_statistics))
    index_vector = np.zeros(len(table_statistics))

    op_name = plan["plan_parameters"]["op_name"]

    if is_join(plan) or is_scan(plan):
        if is_scan(plan):
            #table_id = plan["plan_parameters"].get("table_id")
            table_id = table_ids[plan["plan_parameters"]["table_name"]]
            table_vector[table_id] = 1
            operator_vector[:] = [0, 0]
            if op_name in {"Index Scan", "Index Only Scan"}:
                index_vector[table_id] = 1

        elif is_join(plan):
            join_params = plan["plan_parameters"]["join"]

            if "children" in join_params:
                join_params = join_params["children"][0]

            left_table_id = table_ids[join_params["table_name1"]]
            right_table_id =  table_ids[join_params["table_name2"]]
            table_vector[left_table_id] = 1
            table_vector[right_table_id] = 1
            operator_vector[:] = [1, 0] if op_name == "Hash Join" else [0, 1]

        # Store encoding within the graph for later merging
        plan_encoding = np.concatenate((operator_vector, table_vector, index_vector))
        plan["neo_feature_vector"] = plan_encoding
        plan_encodings.append(plan_encoding)

    for child in plan.get("children", []):
        extract_plan_encoding_recursively(
            plan=child,
            table_statistics=table_statistics,
            plan_encodings=plan_encodings,
            table_ids=table_ids,
            max_depth=max_depth,
            depth=depth if is_join(plan) else depth + 1)
    return plan_encodings

def _walk(curr: tuple[int], vecs: list[tuple]) -> None:
    if curr[1] == 0:
        # curr is a leaf.
        vecs.append(curr)
    else:
        vecs.append((curr[0], curr[1][0], curr[2][0]))
        _walk(curr[1], vecs)
        _walk(curr[2], vecs)


def _make_preorder_ids_tree(curr: dict, root_index: int = 1) -> tuple[tuple[int, int, int], int] | tuple[
    tuple[int, tuple[int, int, int], tuple[int, int, int]], int]:
    """Returns a tuple containing a tree of preorder positional IDs.

    Returns (tree structure, largest id under me).  The tree structure itself
    (the first slot) is a 3-tuple:

    If curr is a leaf (= scan):
      tree structure is (my id, 0, 0) (note that valid IDs start with 1)
    Else:
      tree structure is
        (my id, tree structure for LHS, tree structure for RHS).

    This function traverses each node exactly once (i.e., O(n) time complexity).
    """

    # for scans
    if len(curr["children"]) == 0:
        return (root_index, 0, 0), root_index

    # in case of aggregations, ignore and call children
    if len(curr["children"]) == 1:
        return _make_preorder_ids_tree(curr["children"][0], root_index)

    # If we are here, it has to be a join
    assert is_join(curr), f"Expected join, but got {curr['plan_parameters']['op_name']}"
    lhs, lhs_max_id = _make_preorder_ids_tree(curr=curr["children"][0], root_index=root_index + 1)
    rhs, rhs_max_id = _make_preorder_ids_tree(curr=curr["children"][1], root_index=lhs_max_id + 1)
    return (root_index, lhs, rhs), rhs_max_id


def make_indexes(plan: dict) -> np.ndarray:
    # Join(A, B)            --> preorder_ids = (1, (2, 0, 0), (3, 0, 0))
    # Join(Join(A, B), C)   --> preorder_ids = (1, (2, 3, 4), (5, 0, 0))
    preorder_ids: tuple[int, int, int]
    preorder_ids, _ = _make_preorder_ids_tree(plan)  # method returns (tree structure, largest id under me), tuple[tuple[int, int, int], int]
    vecs = []
    _walk(preorder_ids, vecs)
    vecs = np.asarray(vecs).reshape(-1, 1)
    return vecs

def merge_features(node: dict, left_vec: np.ndarray, right_vec:np.ndarray) -> np.ndarray:
    assert is_join(node)
    len_join_enc = 2
    # The relations under 'node' and their scan types.  Merging <=> summing.
    vec = left_vec + right_vec
    # Make sure the first part is correct.
    vec[:len_join_enc] = [1, 0] if node["plan_parameters"]["op_name"] == "Hash Join" else [0, 1]
    return vec


def collect_feature_vectors(node: dict, vecs: list):
    """Applies func over each subnode of 'node'."""
    if "final_feature_vector" in node.keys():
        vecs.append(node["final_feature_vector"])

    for c in node["children"]:
        collect_feature_vectors(c, vecs)

def _featurize_tree(plan: dict, maximum_depth: int) -> np.ndarray:
    def _bottom_up(curr: dict) -> np.ndarray:
        """Calls node_featurizer on each node exactly once, bottom-up."""

        if len(curr["children"]) == 1:
            return _bottom_up(curr["children"][0])

        if is_scan(curr):
            curr["final_feature_vector"] = curr["neo_feature_vector"]

        if is_join(curr):
            left_vec = _bottom_up(curr["children"][0])
            right_vec = _bottom_up(curr["children"][1])
            merged_feature_vector = merge_features(curr, left_vec, right_vec)
            curr["final_feature_vector"] = merged_feature_vector
        return curr["final_feature_vector"]
    _bottom_up(plan)
    vecs = []
    collect_feature_vectors(node=plan, vecs=vecs)

    # Add a zero-vector at index 0. --- RH: No Idea why this is done. Disabling it
    #result = np.zeros((len(vecs) + 1, vecs[0].shape[0]), dtype=np.float32)
    #result[1:] = vecs

    # pad the result to the maximum depth
    result = np.array(vecs)
    result = np.pad(result, ((0, maximum_depth - len(result)), (0, 0)), mode='constant', constant_values=0)
    return result

def pad_indexes(data):
    lens = [vec.shape[0] for vec in data]
    if len(set(lens)) == 1:
        # Common path.
        return np.asarray(data)
    xs = np.zeros((len(data), np.max(lens), data[0].shape[1]), dtype=np.float32)
    for i, vec in enumerate(data):
        xs[i, :vec.shape[0], :] = vec
    return xs

def init_table_ids(table_stats):
    # We can't use the table IDs from the database statistics, because they contain pg_ and sql_ tables
    # For this we maintain a separate list of table IDs
    table_stats = [x for x in table_stats if not re.match(r"^(pg_|sql_)", x["relname"])]
    table_ids = {}
    for index, stats in enumerate(table_stats):
        table_ids[stats["relname"]] = index
    return table_ids




def neo_collator(batch: Tuple,
                 db_statistics: dict,
                 column_statistics: dict,
                 feature_statistics: dict,
                 max_depth = 35):

    # Get plan encodings
    sample_idxs = []
    labels = []
    all_trees = []
    all_query_encodings = []
    all_indexes = []

    # In case of test mode, we need sql_statements and hints to match corresponding queries that belong
    # to the same plan
    sql_statements = []
    hints = []
    selectivity_list = []
    scan_op_list = []
    join_op_list = []
    num_tables_list = []

    column_statistics = db_statistics[0]["column_stats"]
    table_stats = db_statistics[0]["table_stats"]

    for sample_idx, p in batch:
        sample_idxs.append(sample_idx)
        sql_statements.append(p["sql"])
        hints.append(p["hint"])
        labels.append(p["plan_runtime_ms"])
        assign_additional_plan_info(p, db_statistics)
        selectivity_list += p["selectivity"]
        scan_op_list += p["scan_op"]
        join_op_list += p["join_op"]
        num_tables_list.append(p["num_tables"])

        # Do first query level encoding, consisting out of join graph and predicate encodings
        join_encoding = extract_join_matrix(plan=p, table_statistics=table_stats)

        predicate_encoding = extract_predicate_encoding(plan=p,
                                                        column_statistics=column_statistics,
                                                        feature_statistics=feature_statistics)
        query_encoding = np.concatenate([join_encoding, predicate_encoding])
        all_query_encodings.append(query_encoding)

        # Next is plan level encoding. We add first native feature vectors to each node and later combine them
        extract_plan_encoding_recursively(plan=p,
                                          table_statistics=table_stats,
                                          plan_encodings=[],
                                          table_ids=init_table_ids(table_stats),
                                          depth=0,
                                          max_depth=max_depth)

        # Create 'indexes', that reflect the trees along with their ids
        indexes = make_indexes(plan=p)
        all_indexes.append(indexes)

        # Combine features and by that create trees
        trees = _featurize_tree(plan=p, maximum_depth=max_depth)
        all_trees.append(trees)

    all_query_encodings = torch.from_numpy(np.array(all_query_encodings)).float()
    all_trees = torch.from_numpy(np.array(all_trees)).float()
    all_indexes = torch.from_numpy(pad_indexes(all_indexes)).long()
    labels = postprocess_labels(labels)
    labels = torch.from_numpy(np.array(labels))

    graph_depth = [0] * len(batch) # ToDo: Implement later
    return  (all_query_encodings, all_trees, all_indexes, labels, sample_idxs, sql_statements, hints,
             selectivity_list, scan_op_list, join_op_list, num_tables_list, graph_depth)