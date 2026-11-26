"""One watcher tick: update every active run from files on labstore and wandb, enforce gates.

No LLM is involved here. A run ends as finished / failed / killed; finished runs are evaluated.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import command, curves, evaluate, launcher
from .config import Config
from .remote import Runner
from .spec import load_spec

History = Callable[[Config, str], List[Dict]]  # (cfg, wandb_name) -> rows


def wandb_history(cfg: Config, name: str) -> List[Dict]:
    import wandb
    api = wandb.Api(timeout=60)
    w = cfg["wandb"]
    runs = api.runs(f"{w['entity']}/{w['project']}", filters={"display_name": name, "group": w["group"]})
    for r in runs:
        return [row for row in r.scan_history(page_size=1000) if row.get("epoch") is not None]
    return []


def _spec_for(cfg: Config, experiment: str):
    return load_spec(cfg.experiments_dir / experiment / "spec.yaml")


def tick(cfg: Config, run: Runner, history: History = wandb_history) -> List[Dict]:
    """Returns events (dicts with type finished/failed/killed) for the driver."""
    events = []
    stale_s = cfg["loop"]["stale_log_minutes"] * 60
    for rid, st in launcher.active_runs(cfg).items():
        rd = command.run_dir(cfg, rid)
        ec = rd / "exit_code"
        if ec.exists():
            code = int(ec.read_text().strip() or 1)
            status = "finished" if code == 0 else "failed"
            launcher.write_state(cfg, rid, status=status, exit_code=code, ended_at=time.time())
            ev = {"type": status, "run": rid, "experiment": st["experiment"]}
            if status == "finished":
                res = evaluate.evaluate_dir(cfg, cfg.agent_root / "evaluation" / rid)
                (rd / "eval.json").write_text(json.dumps(res, indent=1, default=str))
                ev["score"] = res["score"]
            events.append(ev)
            continue
        if (rd / "started").exists() and st["status"] == "submitted":
            launcher.write_state(cfg, rid, status="running")
        log = rd / "train.log"
        if log.exists() and time.time() - log.stat().st_mtime > stale_s:
            launcher.cancel(cfg, rid, run, f"log unchanged for {cfg['loop']['stale_log_minutes']} min")
            events.append({"type": "killed", "run": rid, "experiment": st["experiment"], "reason": "stale"})
            continue
        try:
            rows = history(cfg, st["wandb_name"])
        except Exception as e:  # wandb hiccup: try again next tick
            launcher.write_state(cfg, rid, last_history_error=str(e)[:200])
            continue
        if not rows:
            continue
        if curves.has_nan(rows):
            launcher.cancel(cfg, rid, run, "NaN/inf training loss")
            events.append({"type": "killed", "run": rid, "experiment": st["experiment"], "reason": "nan"})
            continue
        spec = _spec_for(cfg, st["experiment"])
        verdicts = curves.check_gates(rows, spec.gates, cfg.baselines_dir)
        last_epoch = max(int(r["epoch"]) for r in rows)
        launcher.write_state(cfg, rid, last_epoch=last_epoch, gates=verdicts)
        failed = [v for v in verdicts if v["passed"] is False]
        if failed:
            g = failed[0]
            reason = (f"gate {g['gate']['metric']} {g['gate']['op']} {g['threshold']:.4g} at epoch "
                      f"{g['gate']['epoch']} missed (value {g['value']:.4g})")
            launcher.cancel(cfg, rid, run, reason)
            events.append({"type": "killed", "run": rid, "experiment": st["experiment"], "reason": reason})
    return events
