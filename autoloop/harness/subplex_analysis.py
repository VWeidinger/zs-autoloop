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
    if stages is None:
        stages = ["annotate-benchmarks", "annotate-subplans", "report", "step4"] + (["step3"] if s.get("step3") else [])
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


# --- reference diagnosis: steps 1-2 of the reference models, from their existing annotations ----------

LDB = Path("/mnt/labstore/vweidinger/subplex/learned_db")
REFERENCE_MODELS = {
    # short name: (registry name, checkpoint, architecture, ppv50, batch_independent)
    "ref-zs-pg_opt": ("pm-zeroshot-pg_opt",
                      "/mnt/labstore/SIGs/ML/learned_db/models/pm-zeroshot-pg_opt/imdb/pm-zeroshot-pg_opt_9.pt",
                      "pm-zeroshot-include_pg_opt_workloads-join_repr_modesum", False, True),
    "ref-zs-ppv50-jit": ("pm-zeroshot-pg_opt-RMP-epochs200-limitpf10050-PPV50-JIT",
                         str(LDB / "models/pm-zeroshot-pg_opt-RMP-epochs200-limitpf10050-PPV50-JIT/imdb/"
                                   "pm-zeroshot-pg_opt-RMP-epochs200-limitpf10050-PPV50-JIT_9.pt"),
                         "pm-zeroshot-include_pg_opt_workloads-reproducible_message_passing-epochs200"
                         "-limit_plans_per_file10050-parsed_plans_version50-use_jit_features", True, False),
    "ref-sumo-ppv50-jit": ("pm-zs_sumo-hi-PostgresEstLoopsIdxCond-hidden_dim256-subpW10-RMP-epochs20-limitpf10050-PPV50-JIT",
                           str(LDB / "models/pm-zs_sumo-hi-PostgresEstLoopsIdxCond-hidden_dim256-subpW10-RMP-epochs20"
                                     "-limitpf10050-PPV50-JIT/imdb/pm-zs_sumo-hi-PostgresEstLoopsIdxCond-hidden_dim256"
                                     "-subpW10-RMP-epochs20-limitpf10050-PPV50-JIT_9.pt"),
                           "pm-zs_sumo-include_hint_idx_workloads-PostgresEstLoopsIdxCond-hidden_dim256-subplan_weight10"
                           "-reproducible_message_passing-epochs20-limit_plans_per_file10050-parsed_plans_version50"
                           "-use_jit_features", True, False),
}


def reference_diagnosis(cfg: Config, dry_run: bool = False) -> Dict[str, str]:
    """Steps 1-2 (CPU) for the reference models into <agent_root>/subplex/_reference/<short>/.

    Their existing annotations are linked (read-only), nothing is re-predicted. Step 3 of the
    reference Zero-Shot exists already in ~/subplex/tmp_with_jit/step6/ (REPORT.md, counterfactual/)."""
    results = {}
    for short, (registry, ckpt, arch, ppv50, bi) in REFERENCE_MODELS.items():
        root = cfg.agent_root / "subplex" / "_reference" / short
        src = LDB / "annotate_subplans" / registry
        dst = root / "annotate_subplans" / short
        cmd = [cfg["subplex"]["uv"], "run", "--frozen", "--project", cfg["subplex"]["repo"], "python", "-m",
               "subplex.external_model", "--name", short, "--architecture", arch, "--checkpoint", ckpt,
               "--out-root", str(root), "--stages", "report", "step4",
               "--subplan-datasets", *cfg["subplex"]["subplan_datasets"]]
        if registry != cfg["subplex"]["reference_pp3"]:
            cmd += ["--reference", cfg["subplex"]["reference_pp3"]]
        if ppv50:
            cmd.append("--ppv50")
        if bi:
            cmd.append("--batch-independent")
        if dry_run:
            results[short] = " ".join(cmd)
            continue
        (dst / "benchmarks").mkdir(parents=True, exist_ok=True)
        for f in list((src / "benchmarks").glob("*.candidates.csv.gz")) + list((src / "benchmarks").glob("subplex_*.json")):
            link = dst / "benchmarks" / f.name
            if not link.exists():
                link.symlink_to(f)
        for ds in cfg["subplex"]["subplan_datasets"]:
            link = dst / f"{ds}.json"
            if not link.exists() and (src / f"{ds}.json").exists():
                link.symlink_to(src / f"{ds}.json")
        env = {**os.environ, "SUBPLEX_TMP_DIR": str(root / "tmp"), "CUDA_VISIBLE_DEVICES": ""}
        with open(root / "analysis.log", "w") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                  cwd=cfg["subplex"]["repo"]).returncode
        results[short] = f"exit {code}: {root}"
    return results
