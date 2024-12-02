import random
from typing import List, Dict

import numpy as np
from tabulate import tabulate


def stratify_plans(plans: List[Dict], max_runtime_ms: int, output_identifier_hint: str, print_distr_stats: bool = False,
                   shuffle_before: bool = True):
    """
    Stratify the plans by runtime
    :param plans: list of plans
    :param max_runtime: maximum runtime
    :param output_identifier_hint: log
    :param print_distr_stats: log
    :param shuffle_before: shuffle before stratification
    """
    if len(plans) == 0:
        return plans

    # try to get a uniform distribution of runtimes

    orig_num_plans = len(plans)
    assert max_runtime_ms is not None
    max_runtime = float(max_runtime_ms) / 1000  # convert to seconds

    # shuffle the plans with seed 42
    if shuffle_before:
        random.Random(0).shuffle(plans)

    # get the runtimes
    runtimes_s = np.array([p['plan_runtime_ms'] / 1000 for p in plans])

    # get the number of bins
    no_bins = 30

    # get the bin edges
    bin_edges = np.linspace(0, max_runtime - max_runtime / no_bins, no_bins)

    # get the bin ids for each runtime
    bin_ids = np.digitize(runtimes_s, bin_edges, right=False) - 1
    # get the number of plans in each bin
    no_plans_per_bin = np.bincount(bin_ids, minlength=no_bins)
    # get the number of plans in the smallest bin
    min_no_plans_per_bin = np.min(no_plans_per_bin)
    max_no_plans_per_bin = np.max(no_plans_per_bin)

    # move the plans to the bins
    plans_per_bin = [[] for _ in range(no_bins)]
    for plan, bin_id in zip(plans, bin_ids):
        plans_per_bin[bin_id].append(plan)

    # keep min_no_plans and at least 10% of the plans in each bin
    no_plans_per_bin_balanced = []
    joins_perc = []  # average inside datasets
    joins_overall = []  # average across datasets
    for i in range(no_bins):
        # keep at least 10 plans
        # keep at least 10% of the plans of the largest bin
        # keep at least min_no_plans_per_bin
        num_plans_to_keep = min(max([min_no_plans_per_bin, int(0.1 * max_no_plans_per_bin), 10]),
                                len(plans_per_bin[i]))

        plans = plans_per_bin[i]

        # extract num loops and num branches
        num_joins = []
        for p in plans:
            num_joins.append(p['num_tables'] - 1)

        plans_per_bin[i] = plans[:num_plans_to_keep]
        no_plans_per_bin_balanced.append(num_plans_to_keep)

        # log the average number of joins
        joins_perc.append(np.mean(num_joins[:num_plans_to_keep]))
        joins_overall.extend(num_joins[:num_plans_to_keep])

    # get the plans
    plans = []
    for i in range(no_bins):
        plans.extend(plans_per_bin[i])

    assert len(bin_edges) == len(no_plans_per_bin) == len(
        no_plans_per_bin_balanced), f"Lengths do not match: {len(bin_edges)}, {len(no_plans_per_bin)}, {len(no_plans_per_bin_balanced)}"

    random.Random(0).shuffle(plans)

    print(f'Balancing of dataset by runtimes: {orig_num_plans} -> {len(plans)} ({output_identifier_hint})')
    if print_distr_stats:
        tabulate_data = zip(bin_edges, no_plans_per_bin, no_plans_per_bin_balanced, joins_perc)
        head = ['Bin edges', 'No plans per bin', 'No plans per bin (balanced)', 'Avg. Num Joins']
        print(tabulate(tabulate_data, headers=head))
        print(f"Overall joins: {np.mean(joins_overall):.3f}")

    return plans
