"""Token/cost ledger built from the agent CLI's own JSON output. The only spend record we have."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

from .config import Config


def ledger_path(cfg: Config) -> Path:
    return cfg.state_dir / "ledger.jsonl"


def _walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)


def usage_from_output(text: str) -> Dict[str, int]:
    """Sum token usage over JSON(L) output of `codex exec --json` or `claude -p --output-format json`.

    codex: events like {"type": "turn.completed", "usage": {"input_tokens", "cached_input_tokens",
    "output_tokens"}}; claude: one object with "usage": {"input_tokens", "cache_read_input_tokens",
    "output_tokens", ...}. Only top-level `usage` objects of completed turns/results are counted so
    that nested per-message copies are not double counted.
    """
    tot = {"input": 0, "cached_input": 0, "output": 0}
    objs = []
    text = text.strip()
    try:
        objs = [json.loads(text)]
    except json.JSONDecodeError:
        for line in text.splitlines():
            try:
                objs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    for o in objs:
        if not isinstance(o, dict):
            continue
        t = str(o.get("type", ""))
        u = o.get("usage")
        if not isinstance(u, dict):
            continue
        if t and not any(x in t for x in ("completed", "result")):
            continue
        cached = int(u.get("cached_input_tokens", 0) or u.get("cache_read_input_tokens", 0) or 0)
        inp = int(u.get("input_tokens", 0) or 0)
        tot["input"] += inp
        tot["cached_input"] += cached
        tot["output"] += int(u.get("output_tokens", 0) or 0)
    return tot


def price_known(cfg: Config, model: str) -> bool:
    p = (cfg["budget"]["prices"] or {}).get(model) or {}
    return all(isinstance(p.get(k), (int, float)) for k in ("input", "cached_input", "output"))


def cost_usd(cfg: Config, model: str, usage: Dict[str, int]) -> float:
    p = cfg["budget"]["prices"][model]
    # codex reports cached tokens as a subset of input tokens
    uncached = max(usage["input"] - usage["cached_input"], 0)
    return (uncached * p["input"] + usage["cached_input"] * p["cached_input"] + usage["output"] * p["output"]) / 1e6


def record(cfg: Config, phase: str, model: str, usage: Dict[str, int], seconds: float, note: str = "") -> float:
    c = cost_usd(cfg, model, usage) if price_known(cfg, model) else float("nan")
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    with ledger_path(cfg).open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "phase": phase, "model": model,
                            **usage, "usd": c, "seconds": round(seconds), "note": note}) + "\n")
    return c


def spent(cfg: Config) -> float:
    p = ledger_path(cfg)
    if not p.exists():
        return 0.0
    tot = 0.0
    for line in p.read_text().splitlines():
        u = json.loads(line).get("usd")
        if isinstance(u, (int, float)) and u == u:
            tot += u
    return tot


def phase_limit(cfg: Config, phase: str = "plan") -> float:
    """budget.max_usd_per_phase: a number, or {plan: x, analyse: y} by phase kind."""
    lim = cfg["budget"]["max_usd_per_phase"]
    if isinstance(lim, dict):
        kind = phase.split("-", 1)[0]
        return float(lim.get(kind, max(lim.values())))
    return float(lim)


def can_afford_phase(cfg: Config, phase: str = "plan") -> bool:
    return spent(cfg) + phase_limit(cfg, phase) <= cfg["budget"]["total_usd"]
