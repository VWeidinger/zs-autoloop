import copy
from typing import List

import numpy as np
import scipy
import sklearn
from sklearn.metrics import mean_squared_error, ndcg_score


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
    if loss_class_name in [
        "QLoss",
        "QPPLoss",
        "DaceLoss",
        "QLogSqLoss",
        "LogQLoss",
        "LogHuberQLoss",
        "LogQLossPenalty",
    ]:
        metrics: List[Metric] = [
            RMSE(),
            MAPE(),
            QError(percentile=50, early_stopping_metric=True),
            QError(percentile=95),
            QError(percentile=100),
            SpearmanRank(),
            NDCG(group_size=32),
            NDCG(group_size=16),
            NDCG(group_size=8),
            NDCG(group_size=4),
        ]
    elif loss_class_name in ["MSELoss"]:
        metrics: List[Metric] = [
            RMSE(early_stopping_metric=True),
            MAPE(),
            QError(percentile=50),
            QError(percentile=95),
            QError(percentile=100),
            SpearmanRank(),
            NDCG(group_size=32),
            NDCG(group_size=16),
            NDCG(group_size=8),
            NDCG(group_size=4),
        ]
    elif loss_class_name in ["RankLoss"]:
        metrics: List[Metric] = [
            RMSE(),
            MAPE(),
            QError(percentile=50),
            QError(percentile=95),
            QError(percentile=100),
            SpearmanRank(early_stopping_metric=True),
            NDCG(group_size=32),
            NDCG(group_size=16),
            NDCG(group_size=8),
            NDCG(group_size=4),
        ]
    elif loss_class_name in [
        "LambdaLoss",
        "NDCGLoss",
        "RankNetLoss",
        "ListNetLoss",
        "ListMLELoss",
        "LearnToRankWithRegressionLoss",
    ]:
        metrics: List[Metric] = [
            RMSE(),
            MAPE(),
            QError(percentile=50),
            QError(percentile=95),
            QError(percentile=100),
            SpearmanRank(),
            NDCG(),
            NDCG(group_size=32),
            NDCG(group_size=16, early_stopping_metric=True),
            NDCG(group_size=8),
            NDCG(group_size=4),
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


class RMSE(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name="mse", maximize=False, **kwargs)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        val_mse = np.sqrt(mean_squared_error(labels, preds))
        return val_mse


class QError(Metric):
    def __init__(self, percentile=50, min_val=1e-8, **kwargs):
        super().__init__(
            metric_name=f"median_q_error_{percentile}", maximize=False, **kwargs
        )
        self.percentile = percentile
        self.min_val = min_val

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        # if not np.all(labels >= self.min_val):
        #    print("WARNING: some labels are smaller than min_val")
        preds = np.clip(preds, self.min_val, np.inf)
        labels = np.clip(labels, self.min_val, np.inf)

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


class NDCG(Metric):
    def __init__(self, group_size=None, **kwargs):
        super().__init__(metric_name=f"ndcg_{group_size}", maximize=True, **kwargs)
        self.group_size = group_size

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        labels = np.array(labels)
        preds = np.array(preds)

        if self.group_size is not None:
            total = len(labels)
            remainder = total % self.group_size

            if total > self.group_size and remainder != 0:
                full_len = total - remainder
                needed = self.group_size - remainder

                # sample only from the fully-formed portion
                sample_idx = np.random.choice(full_len, size=needed, replace=False)

                labels = np.concatenate([labels, labels[sample_idx]])
                preds = np.concatenate([preds, preds[sample_idx]])

            if total > self.group_size:
                grouped_labels = labels.reshape(-1, self.group_size)
                grouped_preds = preds.reshape(-1, self.group_size)
            else:
                grouped_labels = labels.reshape(1, -1)
                grouped_preds = preds.reshape(1, -1)
        else:
            grouped_labels = labels.reshape(1, -1)
            grouped_preds = preds.reshape(1, -1)

        # apply the 1 / (x + 1) transformation to make ndcg focus on smaller values
        transformed_labels = 1.0 / (grouped_labels + 1.0)
        transformed_preds = 1.0 / (grouped_preds + 1.0)

        value = ndcg_score(transformed_labels, transformed_preds)
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
        # --- NumPy Array Implementation ---
        assert len(labels) == len(preds), (
            "Labels and predictions must have the same length."
        )
        if isinstance(preds, np.ndarray):
            # 1. Find the actual minimum prediction value
            min_pred_value = np.min(preds)

            # 2. Get all indices where the prediction equals the minimum value
            indices_of_all_mins = np.where(preds == min_pred_value)[0]

            # 3. Get the corresponding runtimes for all these indices
            runtimes_at_min = labels[indices_of_all_mins]

            # 4. Return the median of these runtimes
            return np.median(runtimes_at_min)

        # --- pandas Series/DataFrame Implementation ---
        else:
            # 1. Find the actual minimum prediction value
            min_pred_value = preds.min()

            # 2. Get all indices where the prediction equals the minimum value
            # This creates a boolean mask and gets the index from the filtered Series
            indices_of_all_mins = preds[preds == min_pred_value].index

            # 3. Get the corresponding runtimes for all these indices using .loc
            runtimes_at_min = labels.loc[indices_of_all_mins]

            # 4. Return the median of these runtimes
            return runtimes_at_min.median()


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

    def __init__(self, min_val=1e-8, **kwargs):
        super().__init__(metric_name="max_underestimation", **kwargs)
        self.min_val = min_val

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        preds = np.clip(preds, self.min_val, np.inf)
        max_underestimation = 1 / min(preds / labels)
        # Cap lower value to 1
        max_underestimation = max(1, max_underestimation)
        return max_underestimation


class MaxOverestimation(Metric):
    """This metric calculates the maximal factor of overestmation of the model predictions for
    a given set of queries"""

    def __init__(self, min_val=1e-8, **kwargs):
        super().__init__(metric_name="max_overestimation", **kwargs)
        self.min_val = min_val

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        preds = np.clip(preds, self.min_val, np.inf)
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
