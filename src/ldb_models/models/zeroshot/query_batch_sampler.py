from collections import defaultdict
from typing import Dict, Iterator, List

import numpy as np
from torch.utils.data import Sampler


class SimilarSampler(Sampler[List[int]]):
    """
    Unified sampler that sorts plans by a given key and then performs a
    partial random shuffle plus group shuffling.
    Can be used for collate functions that store the plan

    Key arguments:
        plans: list of dict-like objects
        batch_size: number of indices per yielded batch
        key: the dictionary key to sort by (e.g. 'plan_runtime_ms' or 'sql')
        random_percentage: fraction of positions to randomly permute each epoch
        group_size: size of groups used when shuffling groups
    """

    def __init__(
        self,
        plans: List[Dict],
        batch_size: int,
        key: str = "plan_runtime_ms",
        random_percentage: float = 0.1,
        group_size: int = 16,
    ):
        self.batch_size = batch_size
        self.random_percentage = random_percentage
        self.group_size = group_size
        self.key = key

        assert key in ["sql", "plan_runtime_ms"], (
            "SimilarSampler only supports 'sql' or 'plan_runtime_ms' as key."
        )
        assert random_percentage >= 0.0 and random_percentage <= 1.0, (
            "random_percentage must be in [0.0, 1.0]"
        )
        assert group_size > 0, "group_size must be positive"
        assert batch_size % group_size == 0, (
            "batch_size must be a multiple of group_size"
        )

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

        # Print helpful stats (numeric min/max only when applicable)
        print(f"--- SimilarSampler initialized (key='{self.key}') ---")
        print(f"Total plans in dataset: {len(plans)}")
        print(f"Total plans after padding: {self.num_plans}")
        print(f"Number of batches per epoch: {self.num_batches}")
        # Try numeric min/max
        try:
            vals = [plans[i][self.key] for i in range(len(plans))]
            numeric_vals = [float(v) for v in vals]
            print(f"Max {self.key}: {max(numeric_vals)}")
            print(f"Min {self.key}: {min(numeric_vals)}")
        except Exception:
            # Non-numeric key (e.g., SQL); skip numeric stats
            pass
        print(f"Keys in first plan: {list(plans[0].keys())}")
        print(
            f"Configured batch_size: {self.batch_size}, group_size: {self.group_size}"
        )
        print(f"Random shuffle percentage per epoch: {self.random_percentage * 100}%")
        print("-------------------------------------------------")

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
            yield final_order_indices[i : i + self.batch_size].tolist()

    def __len__(self) -> int:
        return self.num_batches


class BatchByQuerySampler(Sampler[List[int]]):
    """
    Custom sampler to batch plans from the same query together.

    This sampler performs several key operations:
    1. Groups plans by their SQL text.
    2. Filters out queries that have only one plan, as they are not useful for ranking.
    3. If a query group (batch) is larger than `max_batch_size`, it shuffles the
       plans within that group and splits them into smaller sub-batches.
    4. Shuffles the order of the main query groups at the start of each epoch.
    """

    def __init__(self, plans: List[dict], max_batch_size: int):
        """
        Args:
            plans (List[dict]): The list of plan dictionaries. Each plan must
                                have an 'sql' key for grouping.
            max_batch_size (int): The maximum number of plans allowed in a single
                                  batch yielded by the sampler.
            shuffle_queries (bool): If True, the order of queries (batches) is
                                    shuffled at the start of each epoch.
        """

        self.max_batch_size = max_batch_size

        # 1. Group indices by SQL query text
        grouped_indices = defaultdict(list)
        for i, plan in enumerate(plans):
            grouped_indices[plan["sql"]].append(i)
        initial_query_batches = list(grouped_indices.values())

        num_initial_queries = len(initial_query_batches)

        # 2. Filter out queries with only one plan and pre-split large groups.
        # To make iteration fast we convert every (sub-)batch to a NumPy array
        # at initialization. This avoids creating new lists each epoch.
        precomputed_batches = []
        for batch in initial_query_batches:
            if len(batch) <= 1:
                continue
            if len(batch) > self.max_batch_size:
                # Split into sub-batches of size <= max_batch_size,
                # dropping any resulting sub-batch of size 1.
                for start in range(0, len(batch), self.max_batch_size):
                    end = min(start + self.max_batch_size, len(batch))
                    if end - start > 1:
                        sub = np.array(batch[start:end], dtype=np.int64)
                        precomputed_batches.append(sub)
            else:
                precomputed_batches.append(np.array(batch, dtype=np.int64))

        self.query_batches = precomputed_batches
        num_filtered_queries = len(self.query_batches)

        # 3. Pre-calculate the total number of batches for __len__
        self._num_batches = len(self.query_batches)

        print("--- QueryBatchSampler Initialized (NumPy precomputed batches) ---")
        print(f"Total plans in dataset: {len(plans)}")
        print(f"Initial unique queries: {num_initial_queries}")
        print(
            f"Queries after filtering (more than 1 plan / sub-batches): {num_filtered_queries}"
        )
        if len(self.query_batches) > 0:
            sizes = [b.shape[0] for b in self.query_batches]
            print(f"Max plans for a single query/sub-batch: {max(sizes)}")
            print(f"Min plans for a single query/sub-batch: {min(sizes)}")
            print(f"Average plans per query/sub-batch: {np.mean(sizes)}")
        print(f"Configured max_batch_size: {self.max_batch_size}")
        print(f"Total number of batches to be yielded per epoch: {self._num_batches}")
        print("---------------------------------------------------------------")
        """
        --- QueryBatchSampler Initialized ---
        Total plans in dataset: 39442
        Initial unique queries: 682
        Queries after filtering (more than 1 plan): 512
        Max plans for a single query: 4453
        Min plans for a single query: 2
        Average plans per query: 76.703125
        Configured max_batch_size: 256
        Total number of batches to be yielded per epoch: 590
        -----------------------------------

        --- QueryBatchSampler Initialized ---
        Total plans in dataset: 100637
        Initial unique queries: 61841
        Queries after filtering (more than 1 plan): 546
        Max plans for a single query: 4436
        Min plans for a single query: 2
        Average plans per query: 72.05494505494505
        Configured max_batch_size: 256
        Total number of batches to be yielded per epoch: 627
        -----------------------------------
        """

    def __iter__(self) -> Iterator[List[int]]:
        """
        Yields batches of indices. Handles shuffling and splitting of large batches.
        """
        # Use a fast NumPy permutation of the precomputed batches.
        order = np.random.permutation(len(self.query_batches))
        for idx in order:
            batch_arr = self.query_batches[int(idx)]
            # batch_arr is a NumPy array; yield a Python list for compatibility
            yield batch_arr.tolist()

    def __len__(self) -> int:
        """
        Returns the total number of batches that will be yielded in an epoch.
        """
        return self._num_batches


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

        print("--- AllPlansForQuerySampler Initialized ---")
        print(f"Total plans in dataset: {len(plans)}")
        print(f"Total unique queries (batches): {len(self.query_batches)}")
        print(
            f"Max plans in a single query/batch: {max(len(b) for b in self.query_batches)}"
        )
        print(
            f"Min plans in a single query/batch: {min(len(b) for b in self.query_batches)}"
        )

    def __iter__(self) -> Iterator[List[int]]:
        # Simply iterate through the pre-computed query groups. No shuffling for validation.
        for batch in self.query_batches:
            yield batch

    def __len__(self) -> int:
        # The number of batches is the number of unique queries
        return len(self.query_batches)
