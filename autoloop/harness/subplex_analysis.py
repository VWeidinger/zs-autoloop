"""Run the subplex framework (~/subplex) on a finished run: plain Python, called by the driver.

Output: <agent_root>/subplex/<run_id>/
    report/summary.json, overview.csv          per benchmark: slowdown, wrong picks, diagnoses, vs reference
    report/critical_<b>.csv                    the wrong picks that make up 80 % of the lost runtime
    report/flips_<b>.csv                       statement by statement: better / worse than the reference
    report/compare_<b>_worst<k>_step<n>.csv    node-level selected vs optimal plan of the worst picks
    annotate_subplans/<run_id>/<dataset>.json  the model's prediction on every IMDB subplan (step 2)
    tmp/step4/<run_id>/<dataset>/...           step 4 error curve / diagnosis of the subplan predictions
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from . import command
from .config import Config
from .spec import Spec


def out_root(cfg: Config, run_id: str) -> Path:
    return cfg.agent_root / "subplex" / run_id


def is_ppv50(spec: Spec) -> bool:
    return "parsed_plans50" in spec.plans_dir or "parsed_plans_version50" in spec.model_type


def analysis_command(cfg: Config, spec: Spec, seed: int, stages: Optional[List[str]] = None) -> List[str]:
    s = cfg["subplex"]
    rid = f"{spec.id}-s{seed}"
    ppv50 = is_ppv50(spec)
    cmd = [s["uv"], "run", "--frozen", "--project", s["repo"], "python", "-m", "subplex.external_model",
           "--name", rid, "--architecture", spec.model_type, "--seed", str(seed),
           "--model-dir", str(cfg.agent_root / "models" / rid),
           "--eval-dir", str(cfg.agent_root / "evaluation" / rid),
           "--code-src", str(command.run_dir(cfg, rid) / "code" / "src"),
           "--feature-statistics", str(cfg.resolve_data(spec.statistics_file)),
           "--out-root", str(out_root(cfg, rid)),
           "--reference", s["reference_ppv50"] if ppv50 else s["reference_pp3"],
           "--subplan-datasets", *s["subplan_datasets"]]
    if ppv50:
        cmd.append("--ppv50")
    if "reproducible_message_passing" not in spec.model_type:
        cmd.append("--batch-independent")
    if stages:
        cmd += ["--stages", *stages]
    return cmd


def run_analysis(cfg: Config, spec: Spec, seed: int, gpu: str, dry_run: bool = False) -> Dict[str, object]:
    rid = f"{spec.id}-s{seed}"
    root = out_root(cfg, rid)
    cmd = analysis_command(cfg, spec, seed) + ["--gpu", gpu] + (["--dry-run"] if dry_run else [])
    if dry_run:
        return {"command": cmd}
    root.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "SUBPLEX_TMP_DIR": str(root / "tmp"), "CUDA_VISIBLE_DEVICES": gpu}
    with open(root / "analysis.log", "w") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=cfg["subplex"]["repo"],
                              timeout=cfg["subplex"]["timeout_min"] * 60).returncode
    summary = root / "report" / "summary.json"
    return {"returncode": code, "summary": str(summary) if summary.exists() else None,
            "log": str(root / "analysis.log")}
