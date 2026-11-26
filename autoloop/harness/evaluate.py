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


SHORT = {"access_path_selection": "AP", "access_path_selection_with_aggregate": "AP-agg",
         "physical_operator": "PO", "physical_operator_w_idx": "PO-idx", "join_order": "JOB-light",
         "job_w_optimal_w_idxs": "JOB", "job_complex_w_optimal_w_idxs": "JOB-cx"}
WIN_MARGIN = 0.02   # a benchmark counts as won/lost only beyond +-2 % of the reference ratio


def geo_mean(values: List[float]) -> Optional[float]:
    values = [v for v in values if v and v > 0 and math.isfinite(v)]
    return math.exp(statistics.mean(math.log(v) for v in values)) if values else None


def evaluate_dir(cfg: Config, eval_dir: Path, reference: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Per-benchmark first: every benchmark gets its own ratio and its own verdict against the reference.

    score = geometric mean of the per-benchmark ratios, so a 10 % gain counts the same on every
    benchmark (JOB-complex's 6.9x cannot drown the rest, as it would in an arithmetic mean or a
    runtime total)."""
    benches = cfg["objective"]["benchmarks"]
    per: Dict[str, Dict[str, float]] = {}
    for b in benches:
        p = find_pred_csv(eval_dir, b)
        if p is not None:
            per[b] = selection_metrics(p)
    wl = find_pred_csv(eval_dir, "workload_200k_s1")
    q50 = qerror_median(wl) if wl else None
    ratios = {b: m["ratio"] for b, m in per.items() if m.get("num_queries")}
    score = None
    if ratios:
        agg = cfg["objective"]["aggregate"]
        if agg == "geo_mean_ratio":
            score = geo_mean(list(ratios.values()))
        elif agg == "mean_ratio":
            score = statistics.mean(ratios.values())
        else:
            score = (sum(per[b]["total_selected_runtime"] for b in ratios)
                     / sum(per[b]["total_optimal_runtime"] for b in ratios))
    vs_ref: Dict[str, Dict[str, object]] = {}
    ref_ratio = (reference or {}).get("ratio") or {}
    for b, r in ratios.items():
        if b in ref_ratio:
            rel = r / ref_ratio[b]
            vs_ref[b] = {"ratio": round(r, 4), "ref_ratio": ref_ratio[b], "relative": round(rel, 4),
                         "verdict": "better" if rel < 1 - WIN_MARGIN else ("worse" if rel > 1 + WIN_MARGIN else "same")}
    return {
        "eval_dir": str(eval_dir),
        "benchmarks": per,
        "missing_benchmarks": [b for b in benches if b not in per],
        "ratios": ratios,
        "vs_reference": vs_ref,
        "wins": sum(v["verdict"] == "better" for v in vs_ref.values()),
        "losses": sum(v["verdict"] == "worse" for v in vs_ref.values()),
        "score": score,                       # lower is better, 1.0 = always optimal
        "score_aggregate": cfg["objective"]["aggregate"],
        "imdb_200k_median_qerror": q50,
    }


def guard_ok(cfg: Config, result: Dict[str, object], reference: Dict[str, object]) -> Optional[bool]:
    q, ref = result.get("imdb_200k_median_qerror"), reference.get("imdb_200k_median_qerror")
    if q is None or ref is None:
        return None
    return q <= ref * cfg["objective"]["qerror_guard_factor"]


def load_reference(cfg: Config) -> Optional[Dict[str, object]]:
    p = cfg.baselines_dir / "reference_eval.json"
    return json.loads(p.read_text())["reference"] if p.exists() else None


def write_result(cfg: Config, exp_dir: Path, run_results: Dict[str, Dict[str, object]],
                 reference: Optional[Dict[str, object]] = None) -> Path:
    scores = [r["score"] for r in run_results.values() if r.get("score") is not None]
    per_bench: Dict[str, List[float]] = {}
    for r in run_results.values():
        for b, v in (r.get("ratios") or {}).items():
            per_bench.setdefault(b, []).append(v)
    out = {
        "runs": run_results,
        "score_mean_over_seeds": statistics.mean(scores) if scores else None,
        "ratio_mean_over_seeds": {b: statistics.mean(v) for b, v in per_bench.items()},
        "reference": reference,
        "guard_ok": {rid: guard_ok(cfg, r, reference) for rid, r in run_results.items()} if reference else None,
    }
    path = exp_dir / "result.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    return path


def leaderboard(cfg: Config) -> str:
    benches = cfg["objective"]["benchmarks"]
    ref = load_reference(cfg) or {}
    ref_ratio = ref.get("ratio") or {}
    rows = []
    for rp in sorted(cfg.experiments_dir.glob("e*/result.json")):
        r = json.loads(rp.read_text())
        guard = r.get("guard_ok") or {}
        from .spec import ANALYSIS_SECTIONS, framework_errors
        explained = not framework_errors(rp.parent / "analysis.md", ANALYSIS_SECTIONS)
        rows.append((r.get("score_mean_over_seeds"), rp.parent.name, all(v is not False for v in guard.values()),
                     r.get("ratio_mean_over_seeds") or {},
                     [x.get("imdb_200k_median_qerror") for x in r["runs"].values()], explained))
    rows.sort(key=lambda x: (x[0] is None, x[0]))
    head = " | ".join(SHORT.get(b, b) for b in benches)
    lines = ["# Leaderboard (generated by `python -m autoloop.harness leaderboard`, do not edit)", "",
             "Per benchmark: selected/optimal runtime (1.0 = optimal, lower is better); arrows against the reference "
             f"({ref.get('name', '-')}, +-{int(WIN_MARGIN * 100)} %). score = geometric mean over the benchmarks.",
             "An experiment only counts as a result when its analysis.md explains it through subplex steps 1-3.", "",
             f"| rank | experiment | score | {head} | won/lost | q-error guard | IMDB 200k q50 | explained by subplex |",
             "|---" * (len(benches) + 7) + "|",
             f"| - | reference | {geo_mean(list(ref_ratio.values())) or 0:.3f} | "
             + " | ".join(f"{ref_ratio[b]:.3f}" if b in ref_ratio else "-" for b in benches)
             + f" | - | - | {ref.get('imdb_200k_median_qerror', '-')} | - |"]
    for i, (s, name, ok, ratios, qs, explained) in enumerate(rows, 1):
        cells, won, lost = [], 0, 0
        for b in benches:
            v = ratios.get(b)
            if v is None:
                cells.append("-")
                continue
            arrow = ""
            if b in ref_ratio:
                rel = v / ref_ratio[b]
                arrow = " ↓" if rel < 1 - WIN_MARGIN else (" ↑" if rel > 1 + WIN_MARGIN else "")
                won += rel < 1 - WIN_MARGIN
                lost += rel > 1 + WIN_MARGIN
            cells.append(f"{v:.3f}{arrow}")
        q = ", ".join(f"{v:.3f}" for v in qs if v is not None) or "-"
        lines.append(f"| {i} | {name} | {s:.3f} | " if s is not None else f"| {i} | {name} | - | ")
        lines[-1] += " | ".join(cells) + (f" | {won}/{lost} | {'ok' if ok else 'VIOLATED'} | {q} | "
                                          f"{'yes' if explained else 'NO - not a result yet'} |")
    return "\n".join(lines) + "\n"
