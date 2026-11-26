"""Call the coding agent (Codex CLI or Claude Code) for one phase, headless, with a fresh context."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

from . import ledger
from .config import Config, load_secrets

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


def render_prompt(name: str, **values: str) -> str:
    text = (PROMPTS / f"{name}.md").read_text()
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", str(v))
    return text


def agent_env(cfg: Config) -> Dict[str, str]:
    """Only the agent's own credential is passed; wandb/ssh secrets stay with the harness."""
    sec = load_secrets(cfg)
    env = {k: v for k, v in os.environ.items()
           if k not in ("WANDB_API_KEY", "SSH_AUTH_SOCK", "AUTOLOOP_SSH_PASS")}
    if cfg["agent"]["cli"] == "codex" and sec.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = env["CODEX_API_KEY"] = sec["OPENAI_API_KEY"]
    if cfg["agent"]["cli"] == "claude" and sec.get("ANTHROPIC_API_KEY"):
        env["ANTHROPIC_API_KEY"] = sec["ANTHROPIC_API_KEY"]
    env["AUTOLOOP_CONFIG"] = str(cfg.path)
    env["PYTHONPATH"] = str(cfg.repo)  # the sandbox launcher is autoloop.harness.sandbox
    # ~/.gitconfig is outside the sandbox: the agent commits under its own identity
    gitconfig = cfg.agent_root / "agent.gitconfig"
    if not gitconfig.exists():
        gitconfig.parent.mkdir(parents=True, exist_ok=True)
        gitconfig.write_text("[user]\n\tname = zs-autoloop agent\n\temail = zs-autoloop-agent@localhost\n"
                             "[core]\n\texcludesFile = /dev/null\n"
                             f"[safe]\n\tdirectory = {cfg.repo}\n")
    env["GIT_CONFIG_GLOBAL"] = str(gitconfig)
    # ~/.config is outside the sandbox: tools that read XDG config (git, uv, ...) look here instead
    xdg = cfg.agent_root / "xdg_config"
    xdg.mkdir(parents=True, exist_ok=True)
    env["XDG_CONFIG_HOME"] = str(xdg)
    env["MPLCONFIGDIR"] = str(cfg.agent_root / "mpl_cache")   # matplotlib (subplex) caches here
    return env


def phase_model(cfg: Config, phase: str) -> str:
    """plan-r<n> -> agent.phase_models.plan, analyse-<exp> -> agent.phase_models.analyse."""
    kind = phase.split("-", 1)[0]
    return (cfg["agent"].get("phase_models") or {}).get(kind) or cfg["agent"]["model"]


def run_phase(cfg: Config, phase: str, prompt: str, model: Optional[str] = None,
              dry_run: bool = False) -> Dict[str, object]:
    model = model or phase_model(cfg, phase)
    cli = cfg["agent"]["cli"]
    logs = cfg.state_dir / "agent_logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (logs / f"{stamp}-{phase}.prompt.md").write_text(prompt)
    if dry_run:
        return {"dry_run": True, "prompt_file": str(logs / f"{stamp}-{phase}.prompt.md")}
    if not ledger.price_known(cfg, model):
        raise RuntimeError(f"no price for {model} in config budget.prices - refusing to run the agent")
    if not ledger.can_afford_phase(cfg):
        raise RuntimeError(f"budget exhausted: spent {ledger.spent(cfg):.2f} of {cfg['budget']['total_usd']}")
    tmpl = cfg["agent"]["commands"][cli]
    argv = [a.format(model=model, repo=cfg.repo, agent_root=cfg.agent_root,
                     effort=cfg["agent"].get("reasoning_effort", "high")) for a in tmpl]
    if cli == "claude":
        argv.append(prompt)
        stdin = None
    else:
        stdin = prompt
    t0 = time.time()
    try:
        p = subprocess.run(argv, input=stdin, capture_output=True, text=True, cwd=cfg.repo,
                           env=agent_env(cfg), timeout=cfg["agent"]["phase_timeout_min"] * 60)
        out, rc = p.stdout, p.returncode
        err = p.stderr
    except subprocess.TimeoutExpired as e:
        out, rc, err = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), -1, "timeout"
    (logs / f"{stamp}-{phase}.out.jsonl").write_text(out)
    (logs / f"{stamp}-{phase}.err.txt").write_text(err or "")
    usage = ledger.usage_from_output(out)
    usd = ledger.record(cfg, phase, model, usage, time.time() - t0, note=f"rc={rc}")
    from . import framework_trace
    framework_trace.record_phase(cfg, phase, stamp, out)
    return {"returncode": rc, "usage": usage, "usd": usd, "log": str(logs / f"{stamp}-{phase}.out.jsonl")}
