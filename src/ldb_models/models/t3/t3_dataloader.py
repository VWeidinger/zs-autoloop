from pathlib import Path
from typing import List

from classes.classes import DataLoaderOptions, T3ModelConfig
from classes.workload_runs import WorkloadRuns
from models.t3.benchmarked_query import BenchmarkedQuery
from models.t3.query_plan import QueryPlan
from training.dataset.dataset_creation import read_workload_runs


def create_t3_dataloader(workload_runs: WorkloadRuns,
                         model_config: T3ModelConfig,) -> List[BenchmarkedQuery]:
    # read files
    is_test_dataset = False
    plans, db_statistics = read_workload_runs(workload_run_paths=workload_runs.train_workload_runs,
                                              input_format=model_config.input_format,
                                              include_timeout_queries=model_config.include_timeout_queries and not is_test_dataset,
                                              limit_plans_per_file=model_config.limit_plans_per_file,
                                              limit_plans_overall=model_config.limit_plans_overall)


    benchmark_queries = []
    for parsed_plan in plans:
        # create BenchmarkQuery

        db = None
        predicted_cardinalities = None

        plan = QueryPlan(parsed_plan, db, predicted_cardinalities)
        plan.build_pipelines(None)

        # extract runtimes
        runtime = parsed_plan['plan_runtime_ms']
        sql = parsed_plan['sql']

        # create BenchmarkQuery
        benchmark_queries.append(
            BenchmarkedQuery(query_plan=plan, total_runtimes=[runtime], name=None, query_text=sql, query_category=None))

    return benchmark_queries