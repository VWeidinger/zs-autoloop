"""The loop. Plain Python: it waits for free (no tokens) and calls the agent only to PLAN and ANALYSE.

    round r:  PLAN (agent writes <= N specs)  ->  submit to free GPUs  ->  watch (gates, no LLM)
              -> every finished/killed experiment: evaluate (harness) + ANALYSE (agent)
              -> when all experiments of the round are analysed: next round
    after `rounds_before_review` rounds: AWAITING_REVIEW, notify, exit. You review, edit
    autoloop/REVIEW.md, then `python -m autoloop.harness resume` and start the driver again.
    `touch <agent_root>/state/STOP` (or `python -m autoloop.harness pause`) stops the driver at
    the next tick; trainings keep running and are picked up when the driver starts again.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import agent, evaluate, launcher, ledger, notify, subplex_analysis, watcher
from .remote import probe_host
from .config import Config
from .remote import Runner
from .spec import Spec, load_spec, validate_spec



def log(cfg: Config, msg: str) -> None:
    """Timestamped driver event, to stderr and to <agent_root>/state/driver.log (survives without tee)."""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, file=sys.stderr, flush=True)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.state_dir / "driver.log", "a") as f:
        f.write(line + "\n")


def _state_path(cfg: Config) -> Path:
    return cfg.state_dir / "driver.json"


def load_state(cfg: Config) -> Dict:
    p = _state_path(cfg)
    if p.exists():
        return json.loads(p.read_text())
    return {"phase": "RUNNING", "round": 1, "rounds_since_review": 0, "rounds": {}, "review_started": None}


def save_state(cfg: Config, st: Dict) -> None:
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    _state_path(cfg).write_text(json.dumps(st, indent=1))


def _round(st: Dict) -> Dict:
    return st["rounds"].setdefault(str(st["round"]), {"planned": False, "experiments": [], "analysed": []})


def ready_specs(cfg: Config, exclude: List[str]) -> List[Spec]:
    out = []
    for p in sorted(cfg.experiments_dir.glob("e*/spec.yaml")):
        s = load_spec(p)
        if s.status == "ready" and s.id not in exclude:
            out.append(s)
    return out


def next_experiment_number(cfg: Config) -> int:
    nums = [int(m.group(1)) for p in cfg.experiments_dir.glob("e*")
            if (m := re.match(r"e(\d{3})-", p.name))]
    return max(nums, default=0) + 1


def _status_text(cfg: Config) -> str:
    lines = []
    for rid, s in launcher.all_states(cfg).items():
        lines.append(f"- {rid}: {s.get('status')} host={s.get('host')} gpu={s.get('gpu')} "
                     f"epoch={s.get('last_epoch')} {s.get('kill_reason', '')}")
    return "\n".join(lines) or "(no runs yet)"


def _experiment_runs_ended(cfg: Config, exp: str) -> bool:
    spec = load_spec(cfg.experiments_dir / exp / "spec.yaml")
    sts = [launcher.read_state(cfg, rid).get("status") for rid in spec.run_ids()]
    return all(s in ("finished", "failed", "killed", "launch_failed") for s in sts)


def _commit_generated(cfg: Config, dry_run: bool = False) -> None:
    """Regenerate LEADERBOARD.md / FRAMEWORK_REPORT.md and commit them with the framework usage log."""
    from . import framework_trace
    if dry_run:
        return
    (cfg.autoloop_dir / "LEADERBOARD.md").write_text(evaluate.leaderboard(cfg))
    (cfg.autoloop_dir / "FRAMEWORK_REPORT.md").write_text(framework_trace.report(cfg))
    paths = ["autoloop/LEADERBOARD.md", "autoloop/FRAMEWORK_REPORT.md", "autoloop/framework_usage"]
    git = ["git", "-C", str(cfg.repo)]
    subprocess.run(git + ["add", "--", *paths], capture_output=True)
    if subprocess.run(git + ["diff", "--cached", "--quiet"]).returncode != 0:
        subprocess.run(git + ["commit", "-q", "-m", "autoloop: leaderboard, framework report, framework usage log"],
                       capture_output=True)


def push_progress(cfg: Config, dry_run: bool = False) -> Optional[str]:
    """Push the agent's local commits (main + agent/* branches) to origin after every phase.

    Runs here, outside the sandbox, with your git credentials; the agent itself cannot push.
    Never forced: if origin diverged (someone pushed), the push fails and is only logged."""
    if dry_run or not cfg.get("git", {}).get("auto_push", False):
        return None
    cmd = ["git", "-C", str(cfg.repo), "push", "--porcelain", "origin", "main",
           "refs/heads/agent/*:refs/heads/agent/*"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    msg = (p.stdout + p.stderr).strip().splitlines()[-1:] or [""]
    log(cfg, f"push: rc={p.returncode} {msg[0]}")
    return None if p.returncode == 0 else msg[0]


def _local_gpu(cfg: Config, run: Runner) -> str:
    """A free GPU on the driver's host for the subplex annotation, '' (= CPU) if none is idle."""
    host = cfg["subplex"]["local_host"]
    busy = {s.get("gpu") for s in launcher.active_runs(cfg).values() if s.get("host") == host}
    for g in probe_host(run, host):
        if g.idle(cfg["cluster"]["free_mem_mib"]) and g.index not in busy:
            return str(g.index)
    return ""


def _run_subplex(cfg: Config, spec: Spec, run: Runner, dry_run: bool) -> None:
    for seed in spec.seeds:
        rid = f"{spec.id}-s{seed}"
        st = launcher.read_state(cfg, rid)
        # killed runs are analysed too when a checkpoint exists: it shows why the run went wrong
        has_ckpt = any((cfg.agent_root / "models" / rid).rglob(f"*_{seed}.pt"))
        if st.get("subplex") or st.get("status") not in ("finished", "killed") or not (has_ckpt or dry_run):
            continue
        log(cfg, f"subplex analysis {rid} started")
        res = subplex_analysis.run_analysis(cfg, spec, seed, _local_gpu(cfg, run), dry_run=dry_run)
        log(cfg, f"subplex analysis {rid} done: {res.get('returncode')}")
        launcher.write_state(cfg, rid, subplex=res)


def _write_result(cfg: Config, exp: str) -> None:
    spec = load_spec(cfg.experiments_dir / exp / "spec.yaml")
    runs = {}
    for rid in spec.run_ids():
        p = cfg.runs_dir / rid / "eval.json"
        st = launcher.read_state(cfg, rid)
        runs[rid] = json.loads(p.read_text()) if p.exists() else {"status": st.get("status"),
                                                                  "kill_reason": st.get("kill_reason"),
                                                                  "score": None}
        runs[rid]["subplex"] = st.get("subplex")
    evaluate.write_result(cfg, cfg.experiments_dir / exp, runs, evaluate.load_reference(cfg))
    (cfg.autoloop_dir / "LEADERBOARD.md").write_text(evaluate.leaderboard(cfg))


def step(cfg: Config, run: Runner, dry_run: bool = False, history=watcher.wandb_history) -> str:
    if (cfg.state_dir / "STOP").exists():
        return "paused"
    st = load_state(cfg)
    if st["phase"] == "AWAITING_REVIEW":
        return "awaiting_review"
    rnd = _round(st)

    # 1. watch (no LLM)
    if not dry_run:
        for ev in watcher.tick(cfg, run, history):
            log(cfg, f"run event: {ev}")

    # 2. analyse experiments whose runs all ended
    for exp in rnd["experiments"]:
        if exp in rnd["analysed"] or not _experiment_runs_ended(cfg, exp):
            continue
        _run_subplex(cfg, load_spec(cfg.experiments_dir / exp / "spec.yaml"), run, dry_run)
        _write_result(cfg, exp)
        prompt = agent.render_prompt("analyse", experiment=exp, round=st["round"],
                                     subplex_root=cfg.agent_root / "subplex",
                                     status=_status_text(cfg), budget_left=_budget_left(cfg))
        log(cfg, f"agent ANALYSE {exp} started")
        r = agent.run_phase(cfg, f"analyse-{exp}", prompt, dry_run=dry_run)
        log(cfg, f"agent ANALYSE {exp} done: rc={r.get('returncode')} usd={r.get('usd')}")
        _commit_generated(cfg, dry_run)
        push_progress(cfg, dry_run)
        rnd["analysed"].append(exp)
        save_state(cfg, st)

    # 3. plan the round
    if not rnd["planned"]:
        rf = cfg.autoloop_dir / "REVIEW.md"
        review = rf.read_text() if rf.exists() else "(no review yet)"
        prompt = agent.render_prompt(
            "plan", round=st["round"], max_runs=cfg["loop"]["max_runs_per_round"],
            max_full=cfg["loop"]["max_full_runs_per_round"], next_id=f"e{next_experiment_number(cfg):03d}",
            status=_status_text(cfg), budget_left=_budget_left(cfg), review=review,
            subplex_root=cfg.agent_root / "subplex")
        log(cfg, f"agent PLAN round {st['round']} started")
        r = agent.run_phase(cfg, f"plan-r{st['round']}", prompt, dry_run=dry_run)
        log(cfg, f"agent PLAN round {st['round']} done: rc={r.get('returncode')} usd={r.get('usd')}")
        _commit_generated(cfg, dry_run)
        push_progress(cfg, dry_run)
        done_before = [e for r in st["rounds"].values() for e in r["experiments"]]
        specs = ready_specs(cfg, exclude=done_before)
        specs = _enforce_round_caps(cfg, specs)
        rnd["experiments"] = [s.id for s in specs]
        rnd["planned"] = True
        save_state(cfg, st)
        if not specs:
            st["phase"] = "AWAITING_REVIEW"
            st["review_started"] = time.time()
            save_state(cfg, st)
            notify.notify(cfg, f"{cfg['name']}: agent planned nothing",
                          f"Round {st['round']}: no valid ready spec. Review needed.", dry_run)
            return "awaiting_review"

    # 4. submit what fits
    specs = [load_spec(cfg.experiments_dir / e / "spec.yaml") for e in rnd["experiments"]]
    started = launcher.submit(cfg, specs, run, dry_run=dry_run)
    if started:
        log(cfg, f"submitted runs: {started}")

    # 5. round complete?
    if rnd["experiments"] and all(e in rnd["analysed"] for e in rnd["experiments"]):
        st["rounds_since_review"] += 1
        st["round"] += 1
        if st["rounds_since_review"] >= cfg["loop"]["rounds_before_review"]:
            st["phase"] = "AWAITING_REVIEW"
            st["review_started"] = time.time()
            save_state(cfg, st)
            notify.notify(cfg, f"{cfg['name']}: review needed",
                          f"{st['rounds_since_review']} rounds done, spent ${ledger.spent(cfg):.2f}. "
                          f"Read autoloop/NOTEBOOK.md and LEADERBOARD.md, write autoloop/REVIEW.md, then resume.",
                          dry_run)
            return "awaiting_review"
        save_state(cfg, st)
        return "round_done"
    save_state(cfg, st)
    return "waiting"


def _enforce_round_caps(cfg: Config, specs: List[Spec]) -> List[Spec]:
    out, runs, full = [], 0, 0
    for s in specs:
        errs = validate_spec(s, cfg)
        if errs:
            log(cfg, f"rejected spec {s.id}: {errs}")
            continue
        n = len(s.seeds)
        if runs + n > cfg["loop"]["max_runs_per_round"]:
            continue
        if s.tier == "full":
            if full + n > cfg["loop"]["max_full_runs_per_round"]:
                continue
            full += n
        runs += n
        out.append(s)
    return out


def _budget_left(cfg: Config) -> str:
    return f"{cfg['budget']['total_usd'] - ledger.spent(cfg):.2f} USD"


def run_loop(cfg: Config, run: Runner, dry_run: bool = False, max_steps: Optional[int] = None) -> str:
    n = 0
    while True:
        r = step(cfg, run, dry_run=dry_run)
        n += 1
        if r in ("paused", "awaiting_review") or (max_steps and n >= max_steps):
            return r
        if r != "round_done":
            time.sleep(0 if dry_run else cfg["loop"]["poll_interval_s"])


def resume(cfg: Config, force: bool = False) -> str:
    st = load_state(cfg)
    (cfg.state_dir / "STOP").unlink(missing_ok=True)
    if st["phase"] == "AWAITING_REVIEW":
        rf = cfg.autoloop_dir / "REVIEW.md"
        reviewed = rf.exists() and st.get("review_started") and rf.stat().st_mtime > st["review_started"]
        if not (reviewed or force):
            return "write autoloop/REVIEW.md first (or --force)"
        st["phase"] = "RUNNING"
        st["rounds_since_review"] = 0
        save_state(cfg, st)
    return "resumed"
