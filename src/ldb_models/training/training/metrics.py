import copy
from typing import List

import numpy as np
import scipy
import sklearn
from sklearn.metrics import mean_squared_error


class Metric:
    """
    Abstract class defining a metric used to evaluate the zero-shot cost model performance (e.g., Q-error)
    """

    def __init__(
        self,
        metric_prefix="",
        metric_name="metric",
        maximize=True,
        early_stopping_metric=False,
    ):
        self.maximize = maximize
        self.default_value = -np.inf
        if not self.maximize:
            self.default_value = np.inf
        self.best_seen_value = self.default_value
        self.last_seen_value = self.default_value
        self.metric_name = metric_prefix + metric_name
        self.best_model = None
        self.early_stopping_metric = early_stopping_metric

    def evaluate(
        self,
        model=None,
        metrics_dict=None,
        logging_suffix: str = None,
        update_best_seen: bool = True,
        show_best_seen: bool = True,
        **kwargs,
    ):
        if logging_suffix is not None:
            metric_name = f"{self.metric_name}_{logging_suffix}"
        else:
            metric_name = self.metric_name

        metric = self.default_value
        try:
            metric = self.evaluate_metric(**kwargs)
        except ValueError as e:
            print(f"Observed ValueError for {metric_name} in calculation: {e}")

        metrics_dict[metric_name] = metric

        # compose log string
        log_string = f"{metric_name:<30}: {metric:<10.4f}"
        if show_best_seen:
            log_string += f" [best: {self.best_seen_value:.4f}]"
        print(log_string)

        best_seen = False
        if update_best_seen:
            self.last_seen_value = metric
            if (self.maximize and metric > self.best_seen_value) or (
                not self.maximize and metric < self.best_seen_value
            ):
                self.best_seen_value = metric
                best_seen = True
                if model is not None:
                    self.best_model = copy.deepcopy(model.state_dict())
        return best_seen


def init_metrics(loss_class_name: str) -> List[Metric]:
    if loss_class_name in ["QLoss", "QPPLoss", "DaceLoss", "QLogSqLoss"]:
        metrics: List[Metric] = [
            RMSE(),
            MAPE(),
            QError(percentile=50, early_stopping_metric=True),
            QError(percentile=95),
            QError(percentile=100),
            SpearmanRank(),
        ]

    elif loss_class_name in ["MSELoss"]:
        metrics: List[Metric] = [
            RMSE(early_stopping_metric=True),
            MAPE(),
            QError(percentile=50),
            QError(percentile=95),
            QError(percentile=100),
            SpearmanRank(),
        ]
    elif loss_class_name in ["RankLoss"]:
        metrics: List[Metric] = [
            RMSE(),
            MAPE(),
            QError(percentile=50),
            QError(percentile=95),
            QError(percentile=100),
            SpearmanRank(early_stopping_metric=True),
        ]
    else:
        raise Exception(f"Loss class not supported: {loss_class_name}")
    return metrics


class MAPE(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name="mape", maximize=False, **kwargs)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        mape = np.mean(np.abs((labels - preds) / labels))
        return mape

    def evaluate_metric(self, labels=None, preds=None):
        raise NotImplementedError


class RMSE(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name="mse", maximize=False, **kwargs)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        val_mse = np.sqrt(mean_squared_error(labels, preds))
        return val_mse


class MAPE(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name="mape", maximize=False, **kwargs)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        mape = np.mean(np.abs((labels - preds) / labels))
        return mape


class QError(Metric):
    def __init__(self, percentile=50, min_val=0.1, **kwargs):
        super().__init__(
            metric_name=f"median_q_error_{percentile}", maximize=False, **kwargs
        )
        self.percentile = percentile
        self.min_val = min_val

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        # if not np.all(labels >= self.min_val):
        #    print("WARNING: some labels are smaller than min_val")
        preds = np.abs(preds)
        # preds = np.clip(preds, self.min_val, np.inf)

        q_errors = np.maximum(labels / preds, preds / labels)
        q_errors = np.nan_to_num(q_errors, nan=np.inf)
        median_q = np.percentile(q_errors, self.percentile)
        return median_q


class SpearmanRank(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name="spearman_rank", maximize=True, **kwargs)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        value = scipy.stats.spearmanr(labels, preds).statistic
        return value


class RankAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name="rank_accuracy", maximize=True, **kwargs)

    def evaluate_metric(self, ranking_correct=None, probs=None):
        rank_accuracy = np.mean(ranking_correct)
        return rank_accuracy


class SelectedRuntime(Metric):
    """Given a list of queries, this metric reports the real runtime of the plan that is chosen according to the
    model predictions"""

    def __init__(self, display_name: str = "Selected Runtime (s)", **kwargs):
        super().__init__(metric_name="runtime", **kwargs)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        if isinstance(preds, np.ndarray):
            index_of_minimal_prediction = np.argmin(preds)
        else:
            index_of_minimal_prediction = preds.idxmin()
        runtime_of_minimal_prediction = labels[index_of_minimal_prediction]
        return runtime_of_minimal_prediction


class OptimalRuntime(Metric):
    """Given a list of queries, this metric reports the real runtime of the optimal plan"""

    def __init__(self, **kwargs):
        super().__init__(metric_name="optimal_runtime", **kwargs)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        return labels.min()


class SurpassedPlans(Metric):
    """Given a list of plans this metric reports of how optimal the chosen plan is by calculating the fraction of
    worse plans that are surpassed"""

    def __init__(self, **kwargs):
        super().__init__(metric_name="missed_plans_fraction", **kwargs)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        if isinstance(preds, np.ndarray):
            index_of_minimal_prediction = np.argmin(preds)
        else:
            index_of_minimal_prediction = preds.idxmin()
        runtime_of_minimal_prediction = labels[index_of_minimal_prediction]
        masks = labels < runtime_of_minimal_prediction
        missed_plans_fraction = 1 - masks.sum() / len(labels)
        return missed_plans_fraction * 100


class MaxUnderestimation(Metric):
    """This metric calculates the maximal factor of underestimation of the model predictions for
    a given set of queries"""

    def __init__(self, **kwargs):
        super().__init__(metric_name="max_underestimation", **kwargs)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        max_underestimation = 1 / min(preds / labels)
        # Cap lower value to 1
        max_underestimation = max(1, max_underestimation)
        return max_underestimation


class MaxOverestimation(Metric):
    """This metric calculates the maximal factor of overestmation of the model predictions for
    a given set of queries"""

    def __init__(self, **kwargs):
        super().__init__(metric_name="max_overestimation", **kwargs)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        max_overestimation = max(preds / labels)
        # Cap lower value to 1
        max_overestimation = max(1, max_overestimation)
        return max_overestimation


class PickRate(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name="pick_rate", **kwargs)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        labels = list(labels)
        preds = list(preds)
        return labels.index(min(labels)) == preds.index(min(preds))


class BalancedAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name="balanced_accuracy", maximize=True, **kwargs)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        return sklearn.metrics.balanced_accuracy_score(labels, preds)
