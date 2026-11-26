"""Per-epoch metric curves (wandb history or baseline json) and the gate check."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .spec import Gate

_JO = {"job_light": "join_order_job_light_eval", "job": "join_order_job_eval",
       "job_complex": "join_order_job_complex_eval"}

# short metric name -> how to compute it from one history row (None if not present)
def _sel(row: Dict, b: str) -> Optional[float]:
    return row.get(f"{_JO[b]}-per_epoch/total_selected_runtime")


def _opt(row: Dict, b: str) -> Optional[float]:
    return row.get(f"{_JO[b]}-per_epoch/total_optimal_runtime")


def metric_value(row: Dict, metric: str) -> Optional[float]:
    if metric == "train_loss":
        return row.get("mean_train_loss")
    if metric == "val_q50":
        return row.get("epoch_val_median_q_error_50")
    if metric == "val_spearman":
        return row.get("epoch_val_spearman_rank")
    if metric == "imdb_q50":
        return row.get("workload_200k_s1.json/epoch_test_median_q_error_50")
    if metric in ("job_light_ratio", "job_ratio", "job_complex_ratio"):
        b = metric[: -len("_ratio")]
        s, o = _sel(row, b), _opt(row, b)
        return s / o if s is not None and o else None
    if metric == "job_selected_sum":
        vals = [_sel(row, b) for b in _JO]
        return sum(vals) if all(v is not None for v in vals) else None
    if metric == "job_ratio_mean":
        vals = [metric_value(row, f"{b}_ratio") for b in _JO]
        return sum(vals) / len(vals) if all(v is not None for v in vals) else None
    return row.get(metric)  # raw wandb key


HIGHER_IS_BETTER = {"val_spearman"}
METRICS = ["train_loss", "val_q50", "val_spearman", "imdb_q50", "job_light_ratio", "job_ratio",
           "job_complex_ratio", "job_selected_sum", "job_ratio_mean"]


def curve(rows: Iterable[Dict], metric: str) -> List[Tuple[int, float]]:
    out = []
    for r in rows:
        if r.get("epoch") is None:
            continue
        v = metric_value(r, metric)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            out.append((int(r["epoch"]), float(v)))
    return sorted(out)


def best_so_far(c: List[Tuple[int, float]], epoch: int, metric: str) -> Optional[float]:
    vals = [v for e, v in c if e <= epoch]
    if not vals:
        return None
    return max(vals) if metric in HIGHER_IS_BETTER else min(vals)


def load_baseline(baselines_dir: Path, name: str) -> List[Dict]:
    return json.loads((baselines_dir / f"{name}.json").read_text())["curve"]


def has_nan(rows: Iterable[Dict]) -> bool:
    for r in rows:
        v = r.get("mean_train_loss")
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True
    return False


def check_gates(rows: List[Dict], gates: List[Gate], baselines_dir: Path) -> List[Dict]:
    """Return one verdict per gate whose epoch has been reached: {gate, value, threshold, passed}."""
    verdicts = []
    max_epoch = max((int(r["epoch"]) for r in rows if r.get("epoch") is not None), default=-1)
    for g in gates:
        if max_epoch < g.epoch:
            continue
        v = best_so_far(curve(rows, g.metric), g.epoch, g.metric)
        if g.ref is not None:
            base = best_so_far(curve(load_baseline(baselines_dir, g.ref), g.metric), g.epoch, g.metric)
            thr = None if base is None else base * g.factor
        else:
            thr = g.value
        if v is None or thr is None:
            verdicts.append({"gate": g.__dict__, "value": v, "threshold": thr, "passed": None})
            continue
        passed = v <= thr if g.op == "<=" else v >= thr
        verdicts.append({"gate": g.__dict__, "value": v, "threshold": thr, "passed": passed})
    return verdicts
