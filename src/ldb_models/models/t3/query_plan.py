import math
from enum import Enum
from functools import cmp_to_key
from typing import Optional, Tuple, Dict

from models.t3.database import Database
from models.t3.operator_stages import (
    ExecutionPhase,
    Pipeline,
    OperatorStage,
    build_pipeline,
)
from models.t3.operators import Operator, OperatorType, parse_operator_type, Expressions


class CardType(Enum):
    pg = 1
    act = 2
    deepdb = 3


class QueryPlan:
    operators: dict[int, Operator]
    execution_phases: list[ExecutionPhase]
    pipelines: list[Pipeline]
    db: Database
    ius: dict[str, float]  # maps from iu name to estimated size
    predicted_cardinalities: bool

    def __init__(self, plan: dict, card_type: CardType, db_statistics: Dict):
        self.db_statistics = db_statistics
        self.json_plan = plan
        self.operators = {}
        self.ius = None  # self._parse_ius(plan["ius"]) # TODO implement
        self._parse_operator(self.json_plan, [], card_type=card_type)

    @staticmethod
    def _parse_ius(ius: list[dict]) -> dict[str, float]:
        return {
            iu["iu"]: iu["estimatedSize"] for iu in ius
        }  # TODO not available in postgres

    @staticmethod
    def _get_output_cardinality(op: dict, card_type: CardType) -> float:
        if card_type == CardType.pg:
            return op["plan_parameters"]["est_card"]
        elif card_type == CardType.act:
            return op["plan_parameters"]["act_card"]
        elif card_type == CardType.deepdb:
            return op["plan_parameters"]["deepdb_card"]
        else:
            raise ValueError(f"Unknown cardinality type {card_type}")

    @staticmethod
    def _get_left_cardinality(
        op: dict, operator_type: OperatorType, card_type: CardType
    ) -> float:
        assert operator_type.is_join_type()
        return QueryPlan._get_output_cardinality(op["left"], card_type=card_type)

    def _get_table_name(self, op: dict) -> str:
        """
        our database does not give us quotes on table names, so we add them if needed
        """
        return op["plan_parameters"]["table_name"]

    def _get_input_cardinality(
        self, op: dict, operator_type: OperatorType, card_type: CardType
    ) -> float:
        if operator_type.is_join_type():
            return self._get_left_cardinality(op, operator_type, card_type=card_type)
        elif operator_type == OperatorType.TableScan:
            return self.db_statistics["table_stats_dict"][self._get_table_name(op)][
                "reltuples"
            ]
        elif operator_type == OperatorType.IdxScan:
            # an index scan does not need to scan the whole table - we use the index selectivity
            return self._get_output_cardinality(op, card_type=card_type)
        elif operator_type in (
            OperatorType.PipelineBreakerScan,
            OperatorType.InlineTable,
        ):
            return self._get_output_cardinality(op, card_type=card_type)
            # return op["cardinality"]
        elif operator_type == OperatorType.MultiWayJoin:
            return 0
        elif operator_type == OperatorType.SetOperation:
            if op["operation"] == "unionall":
                return 0
            else:
                return self._get_output_cardinality(op, card_type=card_type)
        else:
            assert "input" in op, f"operator {op['operator']} has more than one child"
            return self._get_output_cardinality(op["input"], card_type=card_type)

    def _get_right_cardinality(
        self, op: dict, operator_type: OperatorType, card_type: CardType
    ) -> Optional[float]:
        # only in case of join operations we can compute this
        if operator_type.is_join_type():
            return QueryPlan._get_output_cardinality(op["right"], card_type=card_type)
        else:
            return None

    def _get_tuple_size(self, op: dict) -> float:
        return op["plan_parameters"][
            "est_width"
        ]  # TODO not sure why ius where used for umbra here!

    @staticmethod
    def _annotate_child(parent: Operator, child: Operator):
        if parent.type.is_join_type():
            if parent.json["left"] == child.json:
                parent.input_op = child
            elif parent.json["right"] == child.json:
                parent.right_input_op = child
            else:
                assert False, "unhandeld join child"
        elif parent.type == OperatorType.SetOperation:
            parent.input_op = child
        elif parent.type == OperatorType.PipelineBreakerScan:
            if "pipelineBreaker" in parent.json:
                assert parent.json["pipelineBreaker"] == child.json
            parent.input_op = child
        elif parent.type == OperatorType.MultiWayJoin:
            parent.input_op = child
        elif "input" in parent.json and parent.json["input"] == child.json:
            parent.input_op = child
        else:
            assert False, "unknown child"

    @staticmethod
    def _featurize_expression(
        expression: dict,
        result: Expressions,
        incoming_selectivity: float,
        expression_selectivity: float,
    ):
        # join
        if "table_name2" in expression:
            result.join_filter_count += 1
        elif expression["operator"] in (
            "<",
            "<=",
            ">",
            ">=",
            "=",
            "!=",
            "isnotnull",
            "is",
            "IS NOT NULL",
            "IS NULL",
        ):
            # null checks should be similarly cheap as comparisons
            result.compare_count += 1
            result.compare_selectivity += incoming_selectivity
        # NOT
        elif expression["operator"] in ("not", "NOT"):
            assert len(expression["children"]) == 1, (
                f"not operator with more than one child: {expression}"
            )
            QueryPlan._featurize_expression(
                expression["children"][0],
                result,
                incoming_selectivity,
                expression_selectivity,
            )
        # OR
        elif expression["operator"] in ("or", "OR"):
            result.or_expression_count += 1
            result.or_selectivity += incoming_selectivity
            n_expressions = len(expression["children"])
            outgoing_sel = incoming_selectivity * expression_selectivity
            # we assume that number of expressions reached is uniformly distributed
            single_expression_sel = (
                incoming_selectivity - outgoing_sel
            ) / n_expressions
            for i, input in enumerate(expression["children"]):
                current_incoming_sel = incoming_selectivity - i * single_expression_sel
                QueryPlan._featurize_expression(
                    input, result, current_incoming_sel, single_expression_sel
                )
        # AND
        elif expression["operator"] in ("and", "AND"):
            n_expressions = len(expression["children"])
            outgoing_sel = incoming_selectivity * expression_selectivity
            # we assume that number of expressions reached is uniformly distributed
            single_expression_sel = (
                incoming_selectivity - outgoing_sel
            ) / n_expressions
            for i, input in enumerate(expression["children"]):
                current_incoming_sel = incoming_selectivity - i * single_expression_sel
                QueryPlan._featurize_expression(
                    input, result, current_incoming_sel, single_expression_sel
                )
        # IN
        elif expression["operator"] in ("in", "IN"):
            result.in_expression_count += 1
            result.in_expression_selectivity += incoming_selectivity
        elif (
            "mode" in expression and expression["mode"] in ("[]", "[)", "(]", "()")
        ) or ("expression" in expression and expression["expression"] == "between"):
            result.between_count += 1
            result.between_selectivity += incoming_selectivity
        # LIKE
        elif expression["operator"] in ("like", "LIKE", "NOT LIKE"):
            result.like_count += 1
            result.like_selectivity += incoming_selectivity
        # startswith
        elif expression["operator"] in ("startswith", "STARTSWITH"):
            result.starts_with_count += 1
            result.starts_with_selectivity += incoming_selectivity
        elif ("mode" in expression and expression["mode"] == "false") or (
            "expression" in expression and expression["mode"] == "false"
        ):
            result.false_count = 0
        else:
            assert False, f"unhandled expression {expression}"

    @staticmethod
    def _get_expression_selectivity(expression: dict) -> float:
        if "estimatedSelectivity" in expression:
            return expression["estimatedSelectivity"]  # does not exist for postgres
        elif expression["operator"] in ("<", "<=", ">", ">="):
            return 0.5
        elif expression["operator"] == "=":
            return 0.01
        elif expression["operator"] in ("<>", "!="):
            return 0.99
        elif expression["operator"] in (
            "between",
            "isnotnull",
            "BETWEEN",
            "IS NOT NULL",
            "IS NULL",
        ):
            return 0.5
        elif expression["operator"] in (
            "in",
            "like",
            "startswith",
            "IN",
            "LIKE",
            "STARTSWITH",
        ):
            return 0.01
        elif expression["operator"] in ("NOT LIKE"):
            return 0.99
        elif expression["operator"] in ("not", "NOT"):
            return 1.0 - QueryPlan._get_expression_selectivity(expression["children"])
        elif expression["operator"] in ("and", "AND"):
            selectivities = [
                QueryPlan._get_expression_selectivity(e) for e in expression["children"]
            ]
            return math.prod(selectivities)
        elif expression["operator"] in ("or", "OR"):
            selectivities = [
                QueryPlan._get_expression_selectivity(e) for e in expression["children"]
            ]
            return min([sum(selectivities), 1.0])
        assert False, f"could not find selectivity for expression: {expression}"

    def _list_expressions(self, op: dict) -> tuple[list[dict], list[float]]:
        if "filter" in op["plan_parameters"]:
            filters = [op["plan_parameters"]["filter"]]
        else:
            filters = []

        if "join" in op["plan_parameters"]:
            joins = [op["plan_parameters"]["join"]]
        else:
            joins = []

        expressions = filters + joins
        selectivities = [self._get_expression_selectivity(e) for e in expressions]
        return expressions, selectivities

    def _parse_expressions(self, op: dict, operator_type: OperatorType) -> Expressions:
        result = Expressions()
        if operator_type == OperatorType.TableScan:
            expressions, selectivities = self._list_expressions(op)
            current_selectivity = 1.0
            for expression, selectivity in zip(expressions, selectivities):
                self._featurize_expression(
                    expression, result, current_selectivity, selectivity
                )
                current_selectivity *= selectivity

        return result

    def _parse_operator(self, op: dict, parent: list[Operator], card_type: CardType):
        assert len(parent) <= 1
        operator_type = parse_operator_type(op)
        output_cardinality = self._get_output_cardinality(op, card_type=card_type)
        input_cardinality = self._get_input_cardinality(
            op, operator_type, card_type=card_type
        )
        right_cardinality = self._get_right_cardinality(
            op, operator_type, card_type=card_type
        )
        output_tuple_size = self._get_tuple_size(op)

        expressions = self._parse_expressions(op, operator_type)

        assert "op_id" in op["plan_parameters"], f"operator {op} does not have an op_id"

        current_op = Operator(
            type=operator_type,
            operator_name=op["plan_parameters"]["op_name"],
            op_id=op["plan_parameters"]["op_id"],
            output_cardinality=output_cardinality,
            input_cardinality=input_cardinality,
            right_input_cardinality=right_cardinality,
            output_tuple_size=output_tuple_size,
            expressions=expressions,
            parents=parent,
            input_op=None,
            right_input_op=None,
            json=op,
        )

        if current_op.op_id in self.operators:
            self.operators[current_op.op_id].parents.extend(parent)
            return

        # recursively call on children
        if operator_type.is_join_type():
            self._parse_operator(op["left"], [current_op], card_type=card_type)
            self._parse_operator(op["right"], [current_op], card_type=card_type)
        elif operator_type == OperatorType.MultiWayJoin:
            for input in op["inputs"]:
                self._parse_operator(
                    input["op"], [current_op], card_type=card_type
                )  # TODO change
        elif operator_type == OperatorType.PipelineBreakerScan:
            # only one of the scanners will include the input
            if "pipelineBreaker" in op:
                self._parse_operator(
                    op["pipelineBreaker"], [current_op], card_type=card_type
                )  # TODO change
        elif operator_type in (
            OperatorType.TableScan,
            OperatorType.InlineTable,
            OperatorType.IdxScan,
        ):
            pass
        elif operator_type == OperatorType.SetOperation:
            for a in op["arguments"]:
                self._parse_operator(
                    a["input"], [current_op], card_type=card_type
                )  # TODO change
        else:
            self._parse_operator(op["input"], [current_op], card_type=card_type)

        for p in parent:
            self._annotate_child(p, current_op)

        assert current_op.op_id not in self.operators
        self.operators[current_op.op_id] = current_op

    def _get_operator_pipelines(self) -> dict[frozenset[int], Pipeline]:
        result = {}
        for pipeline in self.pipelines:
            ops = set()
            for op in pipeline.operators:
                ops.add(op.operator.op_id)
            result[frozenset(ops)] = pipeline
        return result

    @staticmethod
    def _resolve_dangling_pipelines(
        dangling_pipelines: dict[int, frozenset[int]],
        unused_pipelines: dict[frozenset, Pipeline],
        op_id_to_benchmark_id: dict[int, int],
        result: dict[Tuple[int, int], ExecutionPhase],
    ):
        for p, ops in dangling_pipelines.items():
            current_pipelines = {}
            for op in ops:
                potential_pipeline = None
                for u_ops in unused_pipelines:
                    if op in u_ops:
                        potential_pipeline = u_ops
                        break
                current_pipelines[op] = potential_pipeline

            total_set = {
                c_op
                for current_pipeline in current_pipelines.values()
                for c_op in current_pipeline
            }

            if total_set == ops:
                for op in ops:
                    containing_pipeline = unused_pipelines[current_pipelines[op]]
                    result[op_id_to_benchmark_id[op], p] = (
                        containing_pipeline.get_execution_phase(op)
                    )
                for pipeline in set(current_pipelines.values()):
                    unused_pipelines.pop(pipeline)

    def annotate_samples(
        self,
        operator_pipelines: list[Tuple[int, int]],
        benchmark_operator_names: dict[
            int, str
        ],  # names of operators by benchmark file id
        pipeline_names: dict[int, str],
    ) -> dict[Tuple[int, int], ExecutionPhase]:
        no_pipeline_names = {"No pipeline", "No pipeline running"}
        op_id_to_benchmark_id: dict[int, int] = {}
        operator_dict: dict[str, Operator] = {
            op.operator_name: op for op in self.operators.values()
        }
        result = {}
        pipeline_ops = {}

        for o, p in operator_pipelines:
            name = benchmark_operator_names[o]
            if pipeline_names[p] in no_pipeline_names:
                continue
            elif name in operator_dict:
                if p not in pipeline_ops:
                    pipeline_ops[p] = set()
                pipeline_ops[p].add(operator_dict[name].op_id)
                op_id_to_benchmark_id[operator_dict[name].op_id] = o
            else:
                print(f"unknown operator {name}")

        _operator_pipelines = self._get_operator_pipelines()
        dangling_pipelines: dict[int, frozenset[int]] = {}
        for p, ops in pipeline_ops.items():
            op_set = frozenset(ops)
            if op_set in _operator_pipelines:
                current_pipeline = _operator_pipelines.pop(frozenset(ops))
                for op in op_set:
                    result[op_id_to_benchmark_id[op], p] = (
                        current_pipeline.get_execution_phase(op)
                    )
            else:
                dangling_pipelines[p] = op_set
        self._resolve_dangling_pipelines(
            dangling_pipelines, _operator_pipelines, op_id_to_benchmark_id, result
        )
        return result

    def fix_union_all(self):
        """
        union-all pipelines are completely wrong, we need to identify the target pipeline that runs on the resulting
        data of the union all operator.
        All operators of this pipline will be appended to all pipelines that end with the union_all operator.
        """
        for op in self.operators.values():
            if (
                op.type == OperatorType.SetOperation
                and op.json["operation"] == "unionall"
            ):
                tail_pipeline: Optional[Pipeline] = None
                for pipeline in self.pipelines:
                    if len(pipeline.operators) == 0:
                        continue
                    if pipeline.operators[0].operator == op:
                        tail_pipeline = pipeline
                        break
                tail_pipeline.start = 0
                tail_pipeline.stop = 0
                tail_pipeline.operators[0].stage = OperatorStage.PassThrough
                union_cardinality = tail_pipeline.operators[
                    0
                ].operator.output_cardinality
                if union_cardinality < 1:
                    union_cardinality = 1
                for pipeline in self.pipelines:
                    if len(pipeline.operators) == 0:
                        continue
                    if pipeline.operators[-1].operator == op:
                        fraction = (
                            pipeline.operators[-2].operator.output_cardinality
                            / union_cardinality
                        )
                        # assert pipeline.operators[-1].stage == OperatorStage.PassThrough
                        pipeline.operators[-1].stage = OperatorStage.PassThrough
                        append_ops = [op.copy() for op in tail_pipeline.operators[1:]]
                        for append_op in append_ops:
                            append_op.fraction *= fraction
                            append_op.pipeline = pipeline
                        pipeline.operators += append_ops
                tail_pipeline.operators = []
                # return

    def build_pipelines(self, pipelines: list[dict]):
        """
        build pipelines using only the json plan
        """
        operator_dict: dict[int, Operator] = {
            op.json["plan_parameters"]["analyze_plan_id"]: op
            for op in self.operators.values()
        }
        result = []
        for pipeline in pipelines:
            if (
                pipeline["operators"] == [0]
                and 0 not in operator_dict
                and pipeline["duration"] == 0
            ):
                assert False, "could not assign operators to pipelines"
            ops = [operator_dict[op_id] for op_id in pipeline["operators"]]
            ops.sort(
                key=cmp_to_key(lambda b, a: a.precedes(b))
            )  # sort by precedence: inverting the order! (higher id at the begin)
            start = float(pipeline["start"])
            stop = float(pipeline["stop"])
            result.append(build_pipeline(ops, start, stop))
        self.pipelines = result
        self.fix_union_all()
