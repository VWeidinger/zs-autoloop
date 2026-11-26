"""How the agent used the subplex framework - recorded automatically, for the thesis.

After every agent phase the Codex JSON log is scanned for commands that touched subplex (its code,
the guide, or subplex outputs). Each is classified by framework step and appended to
autoloop/framework_usage/<stamp>-<phase>.jsonl (committed and pushed with the agent's work).
`python -m autoloop.harness framework-report` joins that with the experiments' hypotheses, analyses
and results into autoloop/FRAMEWORK_REPORT.md: which subplex finding motivated each experiment,
whether the predicted framework change happened, and what only subplex showed.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config

# first match wins; order from specific to general
STEP_PATTERNS = [
    ("step3-training (step7)", r"step7|training_evidence|training_index|training_set|evidence_for|find_twins|noise_ladder|ladder"),
    ("step3-why (step5/6)", r"counterfactual|step6|surrogate|step5|input_gap|model_input|attribution|REPORT\.md"),
    ("step2-what (step4)", r"step4|curve|emm\b|recalibration|diagnose"),
    ("step1-where", r"benchmark_selection|summary\.json|decisive_|flips_|critical_|compare_|overview\.csv|external_model"),
    ("guide", r"SUBPLEX_GUIDE|subplex/README|DECISIONS\.md|AGENT_BRIEF"),
]
SUBPLEX_MARK = re.compile(r"subplex|SUBPLEX|/_reference/|decisive_|flips_|step[1-7]")


def classify(command: str) -> Optional[str]:
    if not SUBPLEX_MARK.search(command):
        return None
    for step, pat in STEP_PATTERNS:
        if re.search(pat, command):
            return step
    return "other subplex"


def extract(codex_jsonl: str) -> List[Dict[str, object]]:
    rows = []
    for line in codex_jsonl.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = ev.get("item") or {}
        if ev.get("type") != "item.completed" or item.get("type") != "command_execution":
            continue
        cmd = str(item.get("command", ""))
        step = classify(cmd)
        if step:
            rows.append({"step": step, "command": cmd[:400], "exit_code": item.get("exit_code")})
    return rows


def usage_dir(cfg: Config) -> Path:
    return cfg.autoloop_dir / "framework_usage"


def record_phase(cfg: Config, phase: str, stamp: str, codex_jsonl: str) -> Path:
    rows = extract(codex_jsonl)
    d = usage_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{stamp}-{phase}.jsonl"
    path.write_text("".join(json.dumps({"phase": phase, **r}) + "\n" for r in rows))
    return path


def _section(text: str, head: str, max_chars: int = 400) -> str:
    pos = text.find(head)
    if pos < 0:
        return "-"
    rest = text[pos + len(head):]
    nxt = rest.find("\n## ")
    body = " ".join((rest[:nxt] if nxt >= 0 else rest).split())
    body = body.lstrip("-–: ").strip()
    return (body[:max_chars] + " ...") if len(body) > max_chars else body or "-"


def report(cfg: Config) -> str:
    usage: Dict[str, Counter] = defaultdict(Counter)
    for f in sorted(usage_dir(cfg).glob("*.jsonl")) if usage_dir(cfg).exists() else []:
        for line in f.read_text().splitlines():
            r = json.loads(line)
            usage[r["phase"]][r["step"]] += 1
    lines = ["# How the subplex framework drove the experiments (generated, do not edit)", "",
             "Built by `python -m autoloop.harness framework-report` from the hypotheses, analyses, results "
             "and the automatically recorded subplex commands of every agent phase.", "",
             "## Framework use per agent phase", "",
             "| phase | step 1 where | step 2 what | step 3 why | step 3 training (step 7) | guide/other |",
             "|---|---|---|---|---|---|"]
    for phase, c in usage.items():
        lines.append(f"| {phase} | {c['step1-where']} | {c['step2-what (step4)']} | {c['step3-why (step5/6)']} | "
                     f"{c['step3-training (step7)']} | {c['guide'] + c['other subplex']} |")
    lines += ["", "## Experiments", ""]
    for exp in sorted(p for p in cfg.experiments_dir.glob("e*") if p.is_dir()):
        hyp = (exp / "hypothesis.md").read_text() if (exp / "hypothesis.md").exists() else ""
        ana = (exp / "analysis.md").read_text() if (exp / "analysis.md").exists() else ""
        res = json.loads((exp / "result.json").read_text()) if (exp / "result.json").exists() else {}
        lines += [f"### {exp.name}", "",
                  f"- **targeted (step 1):** {_section(hyp, '## Step 1')}",
                  f"- **behaviour (step 2):** {_section(hyp, '## Step 2')}",
                  f"- **cause (step 3):** {_section(hyp, '## Step 3')}",
                  f"- **change:** {_section(hyp, '## Change')}",
                  f"- **score:** {res.get('score_mean_over_seeds', '-')}",
                  f"- **verdict:** {_section(ana, '## Verdict')}",
                  f"- **what only subplex showed:** {_section(ana, '## Framework value')}", ""]
    return "\n".join(lines) + "\n"
