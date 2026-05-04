from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from ldb_models.models.t3.features import FeatureMapper
from ldb_models.models.t3.metrics import q_error
from ldb_models.models.t3.operators import OperatorType
from ldb_models.models.t3.query_plan import QueryPlan


@dataclass
class BenchmarkedQuery:
    query_plan: QueryPlan
    total_runtimes: list[float]  # in seconds
    name: str
    query_text: str
    query_category: str
    feature_matrix: Optional[np.ndarray] = None
    pipeline_runtimes: Optional[list[float]] = None

    def get_total_runtime(self) -> float:
        return np.median(self.total_runtimes)

    def get_analyze_plan_runtime(self) -> float:
        all_times = [x for p in self.query_plan.pipelines for x in (p.start, p.stop)]
        start = min(all_times)
        stop = max(all_times)
        # sometimes this will say 0, in this case we say 1 microsecond
        if start >= stop:
            return 1e-6
        return (stop - start) / 1_000_000

    def check_pipeline_overlap(self):
        # sort pipelines by their start time
        pipelines = sorted(self.query_plan.pipelines, key=lambda p: (p.start, p.stop))
        # iterate over pipelines - start with the earliest pipeline
        for i, p in enumerate(pipelines[:-1]):
            # check the following pipeline
            p2 = pipelines[i + 1]
            if not p.stop <= p2.start:
                # we have a pipeline start/stop overlap: two pipelines are running at the same time - this should not happen
                ids1 = [o.operator.op_id for o in p.operators]
                ids2 = [o.operator.op_id for o in p2.operators]
                common_ops = list(set(ids1).intersection(set(ids2)))
                if not len(common_ops) > 0:
                    pipelines_start_stop = [(p.start, p.stop) for p in pipelines]
                    raise ValueError(
                        f"pipelines overlap without common op: {pipelines_start_stop} \n{self.query_plan}"
                    )
                    print(" pipelines overlap without common op")
                common_ops = [o for o in p.operators if o.operator.op_id in common_ops]
                if (
                    len(common_ops) == 1
                    and common_ops[0].operator.type == OperatorType.SetOperation
                ):
                    # SetOperation operators do not show pipeline correctly
                    # assert common_ops[0].operator.json["operation"] != "unionall", "unionall should be fixed already"
                    p.stop = p2.start
                    p2.stop = max(p.stop, p2.stop)
                else:
                    raise ValueError(
                        f"pipelines overlap! common ops: {', '.join(o.operator.operator_name for o in common_ops)}"
                    )
                    print("pipelines overlap!")
                    print(
                        f" common ops: {', '.join(o.operator.operator_name for o in common_ops)}"
                    )

    def get_pipeline_runtimes(self, verbose: bool = False) -> list[float]:
        if self.pipeline_runtimes is None:
            total_time = self.get_total_runtime()
            analyze_plan_runtime = self.get_analyze_plan_runtime()
            # assert q_error(total_time, analyze_plan_runtime) < 2, "pipeline times seem off"
            if not q_error(total_time, analyze_plan_runtime) < 2 and total_time > 1e-5:
                if verbose:
                    print("pipeline times seem off")
            result = []
            self.check_pipeline_overlap()
            for p in self.query_plan.pipelines:
                result.append(
                    (p.stop - p.start) / (analyze_plan_runtime * 1_000_000) * total_time
                )
            pipeline_times_sum = sum(result)
            if abs(pipeline_times_sum - total_time) > max(total_time * 0.25, 0.0005):
                if verbose:
                    print(
                        f"pipeline times do not add up: {pipeline_times_sum} != {total_time}"
                    )
            if pipeline_times_sum == 0:
                result = [total_time / len(result) for _ in result]
            else:
                correction_factor = total_time / pipeline_times_sum
                result = [x * correction_factor for x in result]
            pipeline_times_sum = sum(result)
            assert abs(pipeline_times_sum - total_time) < 0.0005, (
                f"pipeline times do not add up: {pipeline_times_sum} != {total_time}"
            )
            self.pipeline_runtimes = result
        return self.pipeline_runtimes

    def get_per_tuple_pipeline_runtimes(self) -> list[float]:
        result = []
        for pipeline, runtime in zip(
            self.query_plan.pipelines, self.get_pipeline_runtimes()
        ):
            if pipeline.get_pipeline_scan_cardinality() == 0:
                result.append(runtime)
            else:
                result.append(runtime / pipeline.get_pipeline_scan_cardinality())
        return result

    def get_runtime_data(
        self, feature_mapper: FeatureMapper
    ) -> Tuple[np.ndarray, float]:
        return feature_mapper.get_single_estimation_vector(
            self.query_plan
        ), self.get_total_runtime()

    def get_pipeline_runtime_data(
        self, feature_mapper: FeatureMapper
    ) -> list[Tuple[np.ndarray, float]]:
        features = feature_mapper.get_pipeline_estimation_matrix(self.query_plan)
        targets = self.get_pipeline_runtimes()
        return list((f, t) for f, t in zip(features, targets))

    def get_per_tuple_pipeline_runtime_data(
        self, feature_mapper: FeatureMapper
    ) -> list[tuple[np.ndarray, float]]:
        features = self.get_feature_matrix(feature_mapper)
        targets = self.get_per_tuple_pipeline_runtimes()
        return list((f, t) for f, t in zip(features, targets))

    def get_feature_matrix(self, feature_mapper: FeatureMapper) -> np.ndarray:
        if self.feature_matrix is None:
            self.feature_matrix = feature_mapper.get_pipeline_estimation_matrix(
                self.query_plan
            )
        return self.feature_matrix
