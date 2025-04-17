from enum import Enum


class Operator(Enum):
    NEQ = '!='
    EQ = '='
    LEQ = '<='
    GEQ = '>='
    LIKE = 'LIKE'
    NOT_LIKE = 'NOT LIKE'
    IS_NOT_NULL = 'IS NOT NULL'
    IS_NULL = 'IS NULL'
    IN = 'IN'
    BETWEEN = 'BETWEEN'

    def __str__(self):
        return self.value


class LogicalOperator(Enum):
    AND = 'AND'
    OR = 'OR'

    def __str__(self):
        return self.value


class PredicateOperator:
    def __init__(self, logical_op, children=None):
        self.logical_op = logical_op
        if children is None:
            children = []
        self.children = children


class Aggregator(Enum):
    AVG = 'AVG'
    SUM = 'SUM'
    COUNT = 'COUNT'

    def __str__(self):
        return self.value


class ExtendedAggregator(Enum):
    MIN = 'MIN'
    MAX = 'MAX'