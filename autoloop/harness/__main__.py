"""python -m autoloop.harness <command>

Agent-safe (offline) commands:  validate, status, leaderboard, evaluate, baseline-check
Operator commands (you / the driver): probe, submit, cancel, driver, pause, resume
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import driver, evaluate, launcher, ledger
from .config import load_config, load_secrets
from .remote import dry_runner, ensure_ssh_agent, free_slots, probe_host, ssh_runner
from .spec import load_spec, validate_spec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="autoloop.harness")
    ap.add_argument("--config", type=Path, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="check a spec.yaml (offline)")
    v.add_argument("spec", type=Path)
    sub.add_parser("status", help="runs, driver state, budget (offline)")
    sub.add_parser("leaderboard", help="regenerate autoloop/LEADERBOARD.md (offline)")
    sub.add_parser("framework-report", help="regenerate autoloop/FRAMEWORK_REPORT.md (offline)")
    e = sub.add_parser("evaluate", help="selection metrics of an evaluation dir (offline)")
    e.add_argument("eval_dir", type=Path)
    p = sub.add_parser("probe", help="show GPU state on all nodes (ssh)")
    p.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("submit", help="submit ready specs to free GPUs (ssh)")
    s.add_argument("specs", type=Path, nargs="+")
    s.add_argument("--dry-run", action="store_true")
    c = sub.add_parser("cancel")
    c.add_argument("run_id")
    c.add_argument("--reason", default="manual")
    d = sub.add_parser("driver", help="run the loop until review / pause")
    d.add_argument("--dry-run", action="store_true", help="no ssh, no agent, no wandb; print prompts")
    d.add_argument("--max-steps", type=int, default=None)
    sub.add_parser("pause", help="stop the driver at its next tick (trainings continue)")
    rd = sub.add_parser("reference-diagnosis", help="subplex steps 1-2 of the reference models (CPU, existing annotations)")
    rd.add_argument("--dry-run", action="store_true")
    r = sub.add_parser("resume")
    r.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)

    if a.cmd == "validate":
        errs = validate_spec(load_spec(a.spec), cfg)
        print("OK" if not errs else "\n".join(f"- {x}" for x in errs))
        return 1 if errs else 0
    if a.cmd == "status":
        print(json.dumps({"driver": driver.load_state(cfg), "runs": launcher.all_states(cfg),
                          "spent_usd": round(ledger.spent(cfg), 2),
                          "budget_usd": cfg["budget"]["total_usd"]}, indent=1, default=str))
        return 0
    if a.cmd == "leaderboard":
        (cfg.autoloop_dir / "LEADERBOARD.md").write_text(evaluate.leaderboard(cfg))
        return 0
    if a.cmd == "framework-report":
        from . import framework_trace
        (cfg.autoloop_dir / "FRAMEWORK_REPORT.md").write_text(framework_trace.report(cfg))
        return 0
    if a.cmd == "evaluate":
        print(json.dumps(evaluate.evaluate_dir(cfg, a.eval_dir), indent=1, default=str))
        return 0
    if a.cmd == "pause":
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        (cfg.state_dir / "STOP").touch()
        print("driver will stop at its next tick; trainings keep running")
        return 0
    if a.cmd == "reference-diagnosis":
        from . import subplex_analysis
        for k, v in subplex_analysis.reference_diagnosis(cfg, dry_run=a.dry_run).items():
            print(k, v)
        return 0
    if a.cmd == "resume":
        print(driver.resume(cfg, force=a.force))
        return 0

    dry = getattr(a, "dry_run", False)
    if dry:
        # dry runs keep their own state, so a later real driver never sees phantom runs
        cfg.raw["paths"]["agent_root"] = str(cfg.agent_root / "_dryrun")
        run = dry_runner(log=None, free={n: list(range(cfg["cluster"]["gpus_per_node"]))
                                         for n in cfg["cluster"]["nodes"]})
    else:
        sec = load_secrets(cfg)
        ensure_ssh_agent(Path(cfg["cluster"]["ssh_key"]).expanduser(), sec.get("SSH_KEY_PASSPHRASE"))
        run = ssh_runner(cfg)

    if a.cmd == "probe":
        for host in cfg["cluster"]["nodes"]:
            gpus = probe_host(run, host)
            print(host, "unreachable" if not gpus else
                  ", ".join(f"gpu{g.index}: {g.mem_used}MiB {g.util}% procs={g.n_procs}" for g in gpus))
        return 0
    if a.cmd == "submit":
        print(launcher.submit(cfg, [load_spec(p) for p in a.specs], run, dry_run=dry))
        return 0
    if a.cmd == "cancel":
        launcher.cancel(cfg, a.run_id, run, a.reason)
        return 0
    if a.cmd == "driver":
        print(driver.run_loop(cfg, run, dry_run=dry, max_steps=a.max_steps))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
