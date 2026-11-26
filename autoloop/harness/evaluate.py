"""Plan-selection quality from ldb_models' <workload>_<seed>v<version>_test_pred.csv files.

For every SQL statement with more than one candidate plan, the model selects the candidate with
the lowest prediction; its true runtime (label) is compared to the best candidate's runtime.
This is computed identically for every benchmark (join order, JOB, JOB-complex, access path,
physical operator), so the numbers are comparable across runs and need no wandb.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config

csv.field_size_limit(1 << 30)  # JOB-complex SQL strings exceed the default 128 KiB


def selection_metrics(pred_csv: Path) -> Dict[str, float]:
    groups: Dict[str, List[tuple]] = {}
    with open(pred_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                label, pred = float(row["label"]), float(row["prediction"])
            except (KeyError, ValueError):
                continue
            groups.setdefault(row["sql"], []).append((pred, label))
    selected = optimal = 0.0
    n_q = n_opt = 0
    surpassed: List[float] = []
    ratios: List[float] = []
    for cands in groups.values():
        if len(cands) < 2:
            continue  # single candidate: nothing to select (same rule as ldb_models / subplex)
        n_q += 1
        pick = min(cands, key=lambda c: c[0])[1]
        best = min(c[1] for c in cands)
        selected += pick
        optimal += best
        n_opt += int(pick <= best)
        surpassed.append(sum(1 for c in cands if c[1] >= pick) / len(cands))
        if best > 0:
            ratios.append(pick / best)
    if n_q == 0:
        return {"num_queries": 0}
    return {
        "num_queries": n_q,
        "total_selected_runtime": selected,
        "total_optimal_runtime": optimal,
        "ratio": selected / optimal if optimal > 0 else math.inf,
        "pick_rate": n_opt / n_q,
        "avg_surpassed_plans": statistics.mean(surpassed),
        "geo_mean_query_ratio": math.exp(statistics.mean(math.log(r) for r in ratios)) if ratios else math.nan,
    }


def qerror_median(pred_csv: Path) -> Optional[float]:
    q = []
    with open(pred_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                l, p = float(row["label"]), float(row["prediction"])
            except (KeyError, ValueError):
                continue
            if l > 0 and p > 0:
                q.append(max(l / p, p / l))
    return statistics.median(q) if q else None


def find_pred_csv(eval_dir: Path, stem: str) -> Optional[Path]:
    hits = sorted(eval_dir.glob(f"{stem}_*_test_pred.csv"))
    # job_w_optimal... must not match job_complex...; stems are exact prefixes up to the seed part
    hits = [h for h in hits if h.name[len(stem) + 1].isdigit()]
    return hits[0] if hits else None


def evaluate_dir(cfg: Config, eval_dir: Path) -> Dict[str, object]:
    benches = cfg["objective"]["benchmarks"]
    per: Dict[str, Dict[str, float]] = {}
    for b in benches:
        p = find_pred_csv(eval_dir, b)
        if p is not None:
            per[b] = selection_metrics(p)
    wl = find_pred_csv(eval_dir, "workload_200k_s1")
    q50 = qerror_median(wl) if wl else None
    ratios = [m["ratio"] for m in per.values() if m.get("num_queries")]
    totals = [(m["total_selected_runtime"], m["total_optimal_runtime"]) for m in per.values() if m.get("num_queries")]
    score = None
    if ratios:
        if cfg["objective"]["aggregate"] == "mean_ratio":
            score = statistics.mean(ratios)
        else:
            score = sum(t[0] for t in totals) / sum(t[1] for t in totals)
    return {
        "eval_dir": str(eval_dir),
        "benchmarks": per,
        "missing_benchmarks": [b for b in benches if b not in per],
        "score": score,                       # lower is better, 1.0 = always optimal
        "score_aggregate": cfg["objective"]["aggregate"],
        "imdb_200k_median_qerror": q50,
    }


def guard_ok(cfg: Config, result: Dict[str, object], reference: Dict[str, object]) -> Optional[bool]:
    q, ref = result.get("imdb_200k_median_qerror"), reference.get("imdb_200k_median_qerror")
    if q is None or ref is None:
        return None
    return q <= ref * cfg["objective"]["qerror_guard_factor"]


def write_result(cfg: Config, exp_dir: Path, run_results: Dict[str, Dict[str, object]],
                 reference: Optional[Dict[str, object]] = None) -> Path:
    scores = [r["score"] for r in run_results.values() if r.get("score") is not None]
    out = {
        "runs": run_results,
        "score_mean_over_seeds": statistics.mean(scores) if scores else None,
        "reference": reference,
        "guard_ok": {rid: guard_ok(cfg, r, reference) for rid, r in run_results.items()} if reference else None,
    }
    path = exp_dir / "result.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    return path


def leaderboard(cfg: Config) -> str:
    rows = []
    for rp in sorted(cfg.experiments_dir.glob("e*/result.json")):
        r = json.loads(rp.read_text())
        guard = r.get("guard_ok") or {}
        rows.append((r.get("score_mean_over_seeds"), rp.parent.name,
                     all(v is not False for v in guard.values()),
                     {rid: x.get("imdb_200k_median_qerror") for rid, x in r["runs"].items()}))
    rows.sort(key=lambda x: (x[0] is None, x[0]))
    lines = ["# Leaderboard (generated by `python -m autoloop.harness leaderboard`, do not edit)", "",
             "score = mean over benchmarks of selected/optimal runtime (1.0 = optimal, lower is better)", "",
             "| rank | experiment | score | q-error guard | IMDB 200k median q-error |", "|---|---|---|---|---|"]
    for i, (s, name, ok, q) in enumerate(rows, 1):
        qs = ", ".join(f"{v:.3f}" for v in q.values() if v is not None)
        lines.append(f"| {i} | {name} | {s:.4f} |" if s is not None else f"| {i} | {name} | - |")
        lines[-1] += f" {'ok' if ok else 'VIOLATED'} | {qs} |"
    return "\n".join(lines) + "\n"
