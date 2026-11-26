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
