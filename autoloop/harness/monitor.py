"""Human views of the loop: `watch` (what the driver and the agent are doing) and `costs` (the ledger).

    python3 -m autoloop.harness watch            # status + driver log + the current/latest agent phase
    python3 -m autoloop.harness watch --follow   # ... and keep streaming the agent's events live
    python3 -m autoloop.harness watch --phase plan-r1        # a specific past phase
    python3 -m autoloop.harness costs            # every agent call: tokens x prices = USD
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from . import launcher, ledger
from .config import Config


def _short(text: str, n: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + " ..."


def render_event(e: dict, verbose: bool = False) -> Optional[str]:
    t, item = e.get("type", ""), e.get("item") or {}
    kind = item.get("type")
    if t == "turn.completed":
        u = e.get("usage") or {}
        return f"   [tokens] in {u.get('input_tokens', 0):,} (cached {u.get('cached_input_tokens', 0):,}), out {u.get('output_tokens', 0):,}"
    if kind == "agent_message" and t == "item.completed":
        return f"AGENT  {_short(item.get('text', ''), 2000)}"
    if kind == "reasoning" and t == "item.completed":
        return f"think  {_short(item.get('text', ''), 400)}"
    if kind == "command_execution" and t == "item.started":
        return f"$ {_short(item.get('command', ''), 300)}"
    if kind == "command_execution" and t == "item.completed":
        out = (item.get("aggregated_output") or "").strip().splitlines()
        shown = out if verbose else out[:3]
        tail = "" if verbose or len(out) <= 3 else f"  ... (+{len(out) - 3} lines)"
        body = "".join(f"\n     | {_short(l, 200)}" for l in shown)
        return f"   -> exit {item.get('exit_code')}{tail}{body}"
    if kind == "file_change" and t == "item.completed":
        changes = item.get("changes") or []
        return "EDIT   " + ", ".join(f"{c.get('kind', '')} {c.get('path', '')}" for c in changes)
    if kind == "todo_list" and t in ("item.started", "item.updated", "item.completed"):
        items = item.get("items") or []
        return "TODO   " + " | ".join(("[x] " if i.get("completed") else "[ ] ") + _short(i.get("text", ""), 60) for i in items)
    if t in ("error", "turn.failed"):
        return f"ERROR  {_short(json.dumps(e), 500)}"
    return None


def _logs(cfg: Config) -> Path:
    return cfg.state_dir / "agent_logs"


def _phase_file(cfg: Config, phase: Optional[str]) -> Optional[Path]:
    logs = _logs(cfg)
    if not logs.exists():
        return None
    if phase:
        hits = sorted(logs.glob(f"*{phase}*.out.jsonl"))
        return hits[-1] if hits else None
    cur = (logs / "CURRENT").read_text().strip() if (logs / "CURRENT").exists() else ""
    if cur:
        return logs / f"{cur}.out.jsonl"
    hits = sorted(logs.glob("*.out.jsonl"))
    return hits[-1] if hits else None


def status_text(cfg: Config, driver_lines: int = 15) -> str:
    from .driver import load_state
    st = load_state(cfg)
    lines = [f"== driver: phase {st['phase']}, round {st['round']}, rounds since review {st['rounds_since_review']}"
             f" | spent ${ledger.spent(cfg):.2f} of ${cfg['budget']['total_usd']}"]
    if (cfg.state_dir / "STOP").exists():
        lines.append("   (STOP file present: driver pauses at its next tick)")
    runs = launcher.all_states(cfg)
    for rid, s in runs.items():
        if s.get("status") in ("submitted", "running") or time.time() - (s.get("ended_at") or 0) < 86400:
            lines.append(f"   {rid:42s} {s.get('status', ''):10s} {s.get('host', '')}:{s.get('gpu', '')} "
                         f"epoch {s.get('last_epoch', '-')} {s.get('kill_reason', '')}")
    dl = cfg.state_dir / "driver.log"
    if dl.exists():
        lines.append(f"== driver log (last {driver_lines})")
        lines += ["   " + l for l in dl.read_text().splitlines()[-driver_lines:]]
    return "\n".join(lines)


def watch(cfg: Config, follow: bool = False, phase: Optional[str] = None, verbose: bool = False) -> None:
    print(status_text(cfg))
    f = _phase_file(cfg, phase)
    if f is None or not f.exists():
        print("== no agent phase yet")
        return
    live = (_logs(cfg) / "CURRENT").exists() and (_logs(cfg) / "CURRENT").read_text().strip() == f.name[:-len(".out.jsonl")]
    print(f"== agent phase {f.name[:-len('.out.jsonl')]} ({'RUNNING' if live else 'finished'})"
          f"  prompt: {f.with_name(f.name.replace('.out.jsonl', '.prompt.md'))}")
    pos = 0
    while True:
        with open(f) as h:
            h.seek(pos)
            for line in h:
                if not line.endswith("\n"):
                    break
                pos += len(line.encode())
                try:
                    r = render_event(json.loads(line), verbose)
                except json.JSONDecodeError:
                    continue
                if r:
                    print(r, flush=True)
        if not follow:
            return
        cur = (_logs(cfg) / "CURRENT").read_text().strip() if (_logs(cfg) / "CURRENT").exists() else ""
        if cur and cur != f.name[:-len(".out.jsonl")]:          # a new phase started: switch to it
            f, pos = _logs(cfg) / f"{cur}.out.jsonl", 0
            print(f"\n== agent phase {cur} (RUNNING)", flush=True)
        time.sleep(2)


def costs(cfg: Config) -> str:
    p = ledger.ledger_path(cfg)
    rows = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
    prices = cfg["budget"]["prices"]
    out = ["USD = (input - cached) x price_in + cached x price_cached + output x price_out, per 1M tokens",
           "prices: " + "; ".join(f"{m} {v['input']}/{v['cached_input']}/{v['output']}" for m, v in prices.items()), "",
           f"{'time':19s}  {'phase':28s} {'model':14s} {'input':>10s} {'cached':>10s} {'output':>8s} {'USD':>8s} {'min':>5s}"]
    by_model: dict = {}
    for r in rows:
        usd = r.get("usd")
        out.append(f"{r['ts']:19s}  {r['phase'][:28]:28s} {r['model'][:14]:14s} {r['input']:>10,} {r['cached_input']:>10,} "
                   f"{r['output']:>8,} {usd:>8.3f} {r.get('seconds', 0) / 60:>5.1f}")
        by_model[r["model"]] = by_model.get(r["model"], 0.0) + (usd if usd == usd else 0.0)
    out += ["", "total: " + ", ".join(f"{m} ${v:.2f}" for m, v in by_model.items())
            + f"  |  ${ledger.spent(cfg):.2f} of ${cfg['budget']['total_usd']} budget",
            f"ledger file: {p}"]
    return "\n".join(out)


# --- audit: is the agent using everything as intended? ------------------------------------------

READ_CHECKS = {
    "brief": r"AGENT_BRIEF",
    "subplex guide": r"SUBPLEX_GUIDE",
    "notebook": r"NOTEBOOK",
    "reference diagnosis": r"_reference",
    "own run's subplex output": r"agent_zs-autoloop/subplex/e\d{3}",
}


def audit(cfg: Config) -> str:
    import re
    import subprocess
    from collections import Counter
    from . import framework_trace
    from .spec import ANALYSIS_SECTIONS, HYPOTHESIS_SECTIONS, framework_errors, load_spec, validate_spec

    led = {}
    lp = ledger.ledger_path(cfg)
    if lp.exists():
        for l in lp.read_text().splitlines():
            r = json.loads(l)
            led.setdefault(r["phase"], []).append(r)
    out = ["# Audit: is the agent using everything as intended?", ""]
    for f in sorted(_logs(cfg).glob("*.out.jsonl")) if _logs(cfg).exists() else []:
        name = f.name[:-len(".out.jsonl")]
        phase = name.split("-", 2)[-1]
        if "smoke" in phase:
            continue
        cmds, failed, edits, denied, msgs = [], 0, 0, 0, 0
        for line in f.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            it = e.get("item") or {}
            if e.get("type") != "item.completed":
                continue
            if it.get("type") == "command_execution":
                cmds.append(it.get("command", ""))
                failed += it.get("exit_code") not in (0, None)
                o = it.get("aggregated_output") or ""
                denied += bool(re.search(r"Permission denied|Operation not permitted", o))
            elif it.get("type") == "file_change":
                edits += len(it.get("changes") or [])
            elif it.get("type") == "agent_message":
                msgs += 1
        alltext = "\n".join(cmds)
        steps = Counter(r["step"] for r in framework_trace.extract(f.read_text()))
        cost = sum(r["usd"] for r in led.get(phase, []) if r.get("usd") == r.get("usd"))
        model = ", ".join(sorted({r["model"] for r in led.get(phase, [])})) or "-"
        reads = " ".join(f"{'✓' if re.search(p, alltext) else '✗'} {k}" for k, p in READ_CHECKS.items())
        smoke = "✓" if re.search(r"main\.py", alltext) else "✗"
        out += [f"## {name}  ({model}, ${cost:.2f})",
                f"- commands {len(cmds)} (failed {failed}), file edits {edits}, messages {msgs}, sandbox denials {denied}",
                f"- read: {reads}",
                f"- subplex use: " + (", ".join(f"{k} {v}" for k, v in steps.items()) or "NONE"),
                f"- ran ldb_models (smoke test): {smoke}" if phase.startswith("plan") else "", ""]
    out += ["## Experiments", "", "| experiment | hypothesis sections | spec valid | code branch | analysis sections |",
            "|---|---|---|---|---|"]
    for exp in sorted(p for p in cfg.experiments_dir.glob("e*") if p.is_dir()):
        base = exp.name.startswith(("e000-", "e001-"))
        hyp = "baseline" if base else ("ok" if not framework_errors(exp / "hypothesis.md", HYPOTHESIS_SECTIONS) else "MISSING")
        try:
            spec = load_spec(exp / "spec.yaml")
            errs = validate_spec(spec, cfg)
            valid = "ok" if not errs else f"NO ({len(errs)})"
            br = spec.branch or "-"
            has = subprocess.run(["git", "-C", str(cfg.repo), "rev-parse", "--verify", "-q", br],
                                 capture_output=True).returncode == 0
            branch = f"{br} {'✓' if has else '✗'}"
        except Exception as e:  # noqa: BLE001 - report, do not crash the audit
            valid, branch = f"NO ({e.__class__.__name__})", "-"
        ana = "-" if not (exp / "analysis.md").exists() else \
            ("ok" if not framework_errors(exp / "analysis.md", ANALYSIS_SECTIONS) else "INCOMPLETE")
        out.append(f"| {exp.name} | {hyp} | {valid} | {branch} | {ana} |")
    fb = cfg.autoloop_dir / "SUBPLEX_FEEDBACK.md"
    n_fb = fb.read_text().count("\n## ") if fb.exists() else 0
    out += ["", f"subplex feedback entries: {n_fb}  |  spent ${ledger.spent(cfg):.2f} of ${cfg['budget']['total_usd']}"]
    return "\n".join(l for l in out if l is not None) + "\n"
