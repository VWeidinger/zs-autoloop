from collections import defaultdict
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
from torch.utils.data import Sampler


def compute_query_min_max_tensors(
    plans: List[Dict],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute query-level minimum and maximum runtimes for relevancy generation.
    Fully vectorized and returns tensors for efficient downstream processing.

    Args:
        plans: List of plan dictionaries, each with 'sql' and 'plan_runtime_ms' keys

    Returns:
        A tuple containing:
        - query_min_tensor: 1D tensor of shape [n_plans] with min runtime (in seconds) for each plan's query
        - query_max_tensor: 1D tensor of shape [n_plans] with max runtime (in seconds) for each plan's query
        - single_plan_mask: 1D boolean tensor of shape [n_plans] indicating single-plan queries
    """
    print("Computing query-level min/max runtimes for relevancy generation...")

    # Extract SQL queries and runtimes as arrays for vectorized operations
    sql_list = [plan["sql"] for plan in plans]
    runtimes_ms = np.array(
        [plan.get("plan_runtime_ms", 0) for plan in plans], dtype=np.float64
    )

    # Group by SQL using numpy operations
    query_to_sample_idxs = defaultdict(list)
    for i, sql in enumerate(sql_list):
        query_to_sample_idxs[sql].append(i)

    # Pre-allocate arrays
    n_plans = len(plans)
    query_min_array = np.zeros(n_plans, dtype=np.float32)
    query_max_array = np.zeros(n_plans, dtype=np.float32)
    single_plan_mask_array = np.zeros(n_plans, dtype=bool)

    # Vectorized computation of min/max per query
    num_queries = 0
    num_single_plan_queries = 0

    for sql, sample_idxs in query_to_sample_idxs.items():
        num_queries += 1
        sample_idxs_arr = np.array(sample_idxs, dtype=np.int64)
        query_runtimes = runtimes_ms[sample_idxs_arr]
        min_runtime_ms = query_runtimes.min()
        max_runtime_ms = query_runtimes.max()

        # Convert to seconds and assign to all sample indices for this query
        min_runtime_sec = min_runtime_ms / 1000.0
        max_runtime_sec = max_runtime_ms / 1000.0
        query_min_array[sample_idxs_arr] = min_runtime_sec
        query_max_array[sample_idxs_arr] = max_runtime_sec

        # Track single-plan queries
        if len(sample_idxs) == 1:
            num_single_plan_queries += 1
            single_plan_mask_array[sample_idxs_arr] = True

    print(f"Computed query_min/max for {num_queries} unique queries")
    print(f"Identified {num_single_plan_queries} single-plan queries")

    # Convert to tensors once
    query_min_tensor = torch.from_numpy(query_min_array)
    query_max_tensor = torch.from_numpy(query_max_array)
    single_plan_mask = torch.from_numpy(single_plan_mask_array)

    return query_min_tensor, query_max_tensor, single_plan_mask


def debug_sampler_metrics(
    plans: List[Dict],
    batch_size: int,
    group_size: int,
    random_percentage: float,
    key: str,
):
    """
    Unified debug print function for all samplers to compute and display metrics.

    Args:
        plans (List[Dict]): List of plan dictionaries.
        batch_size (int): Batch size used in the sampler.
        group_size (int): Group size used in the sampler.
        random_percentage (float): Random shuffle percentage.
        key (str): Key used for sorting or grouping.
    """
    print(f"--- Sampler Debug Metrics (key='{key}') ---")
    print(f"Total plans in dataset: {len(plans)}")
    # Compute stats for plans per unique SQL
    sql_list = [plan["sql"] for plan in plans]
    unique_sqls, counts = np.unique(sql_list, return_counts=True)
    print(f"Max plans for a single query: {counts.max()}")
    print(f"Min plans for a single query: {counts.min()}")
    print(f"Average plans per query: {counts.mean():.2f}")
    print(f"Median plans per query: {np.median(counts):.2f}")
    print(f"10th Percentile: {np.percentile(counts, 10):.2f}")
    print(f"90th Percentile: {np.percentile(counts, 90):.2f}")
    print(f"Total unique queries: {len(unique_sqls)}")
    del sql_list, unique_sqls, counts

    print(f"Keys in first plan: {list(plans[0].keys())}")
    print(f"Configured batch_size: {batch_size}, group_size: {group_size}")
    print(f"Random shuffle percentage per epoch: {random_percentage * 100}%")
    print("-------------------------------------------------")


class SimilarSampler(Sampler[List[int]]):
    """
    Unified sampler that sorts plans by a given key and then performs a
    partial random shuffle plus group shuffling.
    Can be used for collate functions that store the plan

    Key arguments:
        plans: list of dict-like objects
        batch_size: number of indices per yielded batch
        key: the dictionary key to sort by (e.g. 'plan_runtime_ms' or 'sql' or 'both')
        random_percentage: fraction of positions to randomly permute each epoch
        group_size: size of groups used when shuffling groups
        compute_query_min: if True, compute minimum runtime per query for relevancy generation
    """

    def __init__(
        self,
        plans: List[Dict],
        batch_size: int,
        key: str = "sql",
        random_percentage: float = 0.1,
        group_size: int = 16,
        compute_query_min: bool = False,
        compute_single_plan_mask: bool = False,
    ):
        self.batch_size = batch_size
        self.random_percentage = random_percentage
        self.group_size = group_size
        self.key = key
        self.compute_query_min = compute_query_min
        self.compute_single_plan_mask = compute_single_plan_mask

        assert key in ["sql", "plan_runtime_ms", "both"], (
            "SimilarSampler only supports 'sql' or 'plan_runtime_ms' or 'both' as key"
        )
        assert random_percentage >= 0.0 and random_percentage <= 1.0, (
            "random_percentage must be in [0.0, 1.0]"
        )
        assert group_size > 0, "group_size must be positive"
        assert batch_size % group_size == 0, (
            "batch_size must be a multiple of group_size"
        )
        if key == "both":
            # Sort by SQL first, then by plan_runtime_ms within each SQL group
            sorted_indices = sorted(
                range(len(plans)),
                key=lambda i: (plans[i]["sql"], plans[i]["plan_runtime_ms"]),
            )
        else:
            sorted_indices = sorted(
                range(len(plans)), key=lambda i: plans[i][self.key], reverse=True
            )

        num_plans = len(sorted_indices)
        missing_plans = batch_size - (num_plans % batch_size)

        # Pad by duplicating entries from the end
        if missing_plans != batch_size:
            padding = sorted_indices[-missing_plans:]
            sorted_indices += padding

        # store as numpy array for efficient per-epoch ops
        self.sorted_indices = np.array(sorted_indices, dtype=np.int64)
        self.num_plans = len(self.sorted_indices)
        self.num_batches = self.num_plans // self.batch_size

        # Compute query_min/max if requested (stores as tensors for efficiency)
        self.query_min_tensor = None
        self.query_max_tensor = None
        self.single_plan_mask = None
        if self.compute_query_min or self.compute_single_plan_mask:
            self.query_min_tensor, self.query_max_tensor, self.single_plan_mask = (
                compute_query_min_max_tensors(plans)
            )

        debug_sampler_metrics(plans, batch_size, group_size, random_percentage, key)

    def __iter__(self) -> Iterator[List[int]]:
        # Work on a copy each epoch
        indices = self.sorted_indices.copy()

        # Partial random shuffle: pick positions and shuffle their values
        num_to_shuffle = int(self.num_plans * self.random_percentage)
        if num_to_shuffle > 0:
            positions_to_shuffle = np.random.choice(
                self.num_plans, num_to_shuffle, replace=False
            )
            values_to_shuffle = indices[positions_to_shuffle]
            np.random.shuffle(values_to_shuffle)
            indices[positions_to_shuffle] = values_to_shuffle

        # Group shuffle
        total_groups = self.num_plans // self.group_size
        if total_groups > 0:
            shuffled_group_order = np.random.permutation(total_groups)
            final_order_indices = indices.reshape(total_groups, self.group_size)[
                shuffled_group_order
            ].ravel()
        else:
            final_order_indices = indices

        # Yield batches
        for i in range(0, self.num_plans, self.batch_size):
            # shuffle the plans in each group
            batch_slice = final_order_indices[i : i + self.batch_size]

            # reshape into (n_groups_per_batch, group_size) and shuffle
            # the order *within* each group while keeping groups intact
            n_groups = self.batch_size // self.group_size
            groups = batch_slice.reshape(n_groups, self.group_size)
            for g in range(groups.shape[0]):
                np.random.shuffle(groups[g])
            shuffled_batch = groups.ravel()
            yield shuffled_batch.tolist()

    def __len__(self) -> int:
        return self.num_batches


class AllPlansForQuerySampler(Sampler[List[int]]):
    """
    A sampler that yields batches, where each batch contains ALL plan indices
    for a single unique query.

    This is designed for validation/testing of models that need to see all
    candidate plans for a query at once (e.g., attention-based rankers).
    """

    def __init__(self, plans: List[dict], batch_size: int):
        grouped_indices = defaultdict(list)
        for i, plan in enumerate(plans):
            grouped_indices[plan["sql"]].append(i)

        multi_plan = []
        single_plan = []

        for idxs in grouped_indices.values():
            if len(idxs) == 1:
                single_plan.append(idxs[0])
            else:
                multi_plan.append(idxs)

        single_plan = np.array(single_plan, dtype=int)

        single_batches = []
        if len(single_plan) > 0:
            n = single_plan.shape[0]
            n_full = n // batch_size
            for i in range(n_full):
                start = i * batch_size
                end = start + batch_size
                single_batches.append(single_plan[start:end].tolist())

            rem = n % batch_size
            if rem > 0:
                single_batches.append(single_plan[-rem:].tolist())

        self.query_batches = multi_plan + single_batches

        debug_sampler_metrics(plans, batch_size, batch_size, 0.0, "sql")

    def __iter__(self) -> Iterator[List[int]]:
        for idx in range(len(self.query_batches)):
            yield self.query_batches[int(idx)]

    def __len__(self) -> int:
        return len(self.query_batches)
