"""Submit, track and cancel runs. Run state lives on the shared labstore: runs/<run_id>/state.json."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import command
from .config import Config, load_secrets
from .remote import Runner, free_slots
from .spec import Spec, SpecError, validate_spec

ACTIVE = ("submitted", "running")


def read_state(cfg: Config, run_id: str) -> Dict:
    p = command.run_dir(cfg, run_id) / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def write_state(cfg: Config, run_id: str, **upd) -> Dict:
    rd = command.run_dir(cfg, run_id)
    rd.mkdir(parents=True, exist_ok=True)
    st = read_state(cfg, run_id)
    st.update(upd, updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (rd / "state.json").write_text(json.dumps(st, indent=1))
    return st


def all_states(cfg: Config) -> Dict[str, Dict]:
    out = {}
    if cfg.runs_dir.exists():
        for p in sorted(cfg.runs_dir.glob("*/state.json")):
            out[p.parent.name] = json.loads(p.read_text())
    return out


def active_runs(cfg: Config) -> Dict[str, Dict]:
    return {k: v for k, v in all_states(cfg).items() if v.get("status") in ACTIVE}


def snapshot_code(cfg: Config, commit: str, dest: Path) -> None:
    """git archive of exactly the experiment commit -> dest (shared FS, visible to all nodes)."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(["git", "-C", str(cfg.repo), "archive", commit], capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, check=True)


def resolve_commit(cfg: Config, spec: Spec) -> str:
    ref = spec.commit or spec.branch
    if not ref:
        raise SpecError(f"{spec.id}: spec needs a branch (or commit) holding its code")
    return subprocess.run(["git", "-C", str(cfg.repo), "rev-parse", "--verify", ref + "^{commit}"],
                          capture_output=True, text=True, check=True).stdout.strip()


def install_wandb_secret(cfg: Config) -> None:
    key = load_secrets(cfg).get("WANDB_API_KEY")
    if not key:
        return
    d = cfg.agent_root / ".secrets"
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    f = d / "wandb.env"
    f.write_text(f"export WANDB_API_KEY={shlex.quote(key)}\n")
    os.chmod(f, 0o600)


def submit(cfg: Config, specs: List[Spec], run: Runner, dry_run: bool = False) -> List[str]:
    """Place as many runs as caps and free GPUs allow. Returns the run ids started."""
    cap = cfg["cluster"]["max_concurrent"]
    active = active_runs(cfg)
    busy = {(s["host"], s["gpu"]) for s in active.values() if "host" in s}
    budget = cap - len(active)
    if budget <= 0:
        return []
    pending: List[Tuple[Spec, int]] = []
    for spec in specs:
        errs = validate_spec(spec, cfg, check_files=not dry_run)
        if errs:
            raise SpecError(f"{spec.id}: " + "; ".join(errs))
        for seed in spec.seeds:
            rid = f"{spec.id}-s{seed}"
            if read_state(cfg, rid).get("status") is None:
                pending.append((spec, seed))
    if not pending:
        return []
    slots = free_slots(cfg, run, busy)[:budget]
    started = []
    if not dry_run:
        install_wandb_secret(cfg)
    for (spec, seed), (host, gpu) in zip(pending, slots):
        rid = f"{spec.id}-s{seed}"
        rd = command.run_dir(cfg, rid)
        commit = "DRYRUN" if dry_run else resolve_commit(cfg, spec)
        if not dry_run:
            snapshot_code(cfg, commit, rd / "code")
            venv = cfg.agent_root / "venvs" / command.lock_hash(rd / "code")
        else:
            rd.mkdir(parents=True, exist_ok=True)
            venv = cfg.agent_root / "venvs" / "dryrun"
        (rd / "inner.sh").write_text(command.inner_script(cfg, spec, seed, venv))
        (rd / "outer.sh").write_text(command.outer_script(cfg, spec, seed, gpu))
        for f in ("inner.sh", "outer.sh"):
            os.chmod(rd / f, 0o750)
        rc, _ = run(host, f"screen -dmS zsal-{rid} bash {rd}/outer.sh")
        if rc != 0:
            write_state(cfg, rid, status="launch_failed", host=host, gpu=gpu)
            continue
        write_state(cfg, rid, status="submitted", experiment=spec.id, seed=seed, tier=spec.tier,
                    host=host, gpu=gpu, commit=commit, wandb_name=command.wandb_name(cfg, spec, seed),
                    submitted_at=time.time(), dry_run=dry_run)
        started.append(rid)
    return started


def cancel(cfg: Config, run_id: str, run: Runner, reason: str) -> None:
    st = read_state(cfg, run_id)
    host = st.get("host")
    rd = command.run_dir(cfg, run_id)
    if host:
        # the run directory path is unique to this run and appears in every process' command line
        run(host, f"screen -S zsal-{run_id} -X quit; pkill -TERM -f {shlex.quote(str(rd))}/ || true")
    write_state(cfg, run_id, status="killed", kill_reason=reason)
