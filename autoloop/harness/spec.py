"""Experiment specs: what the agent writes, what the harness checks before anything runs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .config import Config

ID_RE = re.compile(r"^e\d{3}-[a-z0-9][a-z0-9-]{2,40}$")
EPOCHS_RE = re.compile(r"-epochs(\d+)")
# flags the harness sets itself; a spec may not override them
RESERVED_ARGS = {
    "--mode", "--model_type", "--workload_runs", "--test_workload_runs", "--statistics_file",
    "--model_dir", "--target_dir", "--device", "--wandb_project", "--wandb_entity", "--wandb_name",
    "--wandb_group", "--seed", "--version", "--qo_eval_after_each_epoch", "--wandb_resume_id",
}
GATE_OPS = {"<=", ">="}


class SpecError(ValueError):
    pass


# The research goal is to improve the model THROUGH the subplex framework: every experiment names the
# framework finding it targets (step 1 where, step 2 what, step 3 why) and the change it predicts there.
HYPOTHESIS_SECTIONS = ["## Step 1", "## Step 2", "## Step 3", "## Change", "## Predicted framework change"]
ANALYSIS_SECTIONS = ["## Verdict", "## Step 1", "## Step 2", "## Step 3", "## Explanation", "## Framework value"]


def framework_errors(path: Path, sections: List[str], min_chars: int = 40) -> List[str]:
    """Missing or empty framework sections in hypothesis.md / analysis.md."""
    if not path.exists():
        return [f"{path.name} is missing"]
    text = path.read_text()
    errs = []
    for i, head in enumerate(sections):
        pos = text.find(head)
        if pos < 0:
            errs.append(f"{path.name}: section '{head}' missing")
            continue
        rest = text[pos + len(head):]
        nxt = min([rest.find(h) for h in sections if rest.find(h) > 0] + [len(rest)])
        if len(rest[:nxt].strip()) < min_chars:
            errs.append(f"{path.name}: section '{head}' is empty")
    return errs


@dataclass
class Gate:
    epoch: int
    metric: str
    op: str
    value: Optional[float] = None       # absolute threshold
    ref: Optional[str] = None           # baseline name -> threshold = baseline(best-so-far at epoch) * factor
    factor: float = 1.0


@dataclass
class Spec:
    id: str
    tier: str
    hypothesis: str
    model_type: str
    statistics_file: str
    train_workloads: List[str] = field(default_factory=list)   # explicit files (relative to data_root)
    workload_names: List[str] = field(default_factory=list)    # expanded over all non-IMDB databases
    plans_dir: str = "runs/parsed_plans3"                      # where workload_names are looked up
    seeds: List[int] = field(default_factory=lambda: [9])
    version: int = 5
    extra_args: List[str] = field(default_factory=list)
    gates: List[Gate] = field(default_factory=list)
    branch: Optional[str] = None
    commit: Optional[str] = None
    parent: Optional[str] = None         # experiment this one builds on / promotes
    status: str = "draft"                # draft | ready | submitted
    path: Optional[Path] = None

    @property
    def epochs(self) -> Optional[int]:
        m = EPOCHS_RE.search(self.model_type)
        return int(m.group(1)) if m else None

    def expanded_train_workloads(self, cfg: Config) -> List[str]:
        """explicit files + every <plans_dir>/<db>/<name>.json that exists, IMDB excluded."""
        out = list(self.train_workloads)
        root = cfg.resolve_data(self.plans_dir)
        for name in self.workload_names:
            for db_dir in sorted(root.glob("*_scaled*")):
                if db_dir.name.startswith("imdb"):
                    continue
                f = db_dir / f"{name}.json"
                if f.exists():
                    out.append(str(f))
        return out

    def run_ids(self) -> List[str]:
        return [f"{self.id}-s{s}" for s in self.seeds]


def load_spec(path: Path) -> Spec:
    raw: Dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    gates = [Gate(**g) for g in raw.pop("gates", []) or []]
    known = {f for f in Spec.__dataclass_fields__ if f not in ("gates", "path")}
    unknown = set(raw) - known
    if unknown:
        raise SpecError(f"{path}: unknown keys {sorted(unknown)}")
    try:
        spec = Spec(**raw, gates=gates, path=Path(path))
    except TypeError as e:
        raise SpecError(f"{path}: {e}") from e
    return spec


def _under(p: Path, roots: List[Path]) -> bool:
    p = p.resolve() if p.exists() else p
    return any(str(p).startswith(str(r)) for r in roots)


def validate_spec(spec: Spec, cfg: Config, check_files: bool = True) -> List[str]:
    """Return a list of problems (empty = valid). The harness refuses to submit invalid specs."""
    errs: List[str] = []
    if not ID_RE.match(spec.id):
        errs.append(f"id {spec.id!r} must look like e001-short-name")
    tiers = cfg["tiers"]
    if spec.tier not in tiers:
        errs.append(f"tier {spec.tier!r} not in {list(tiers)}")
    else:
        t = tiers[spec.tier]
        if spec.epochs is None:
            errs.append("model_type must contain -epochs<N> so the tier cap can be checked")
        elif spec.epochs > t["max_epochs"]:
            errs.append(f"{spec.epochs} epochs > {spec.tier} cap {t['max_epochs']}")
        if not 1 <= len(spec.seeds) <= t["max_seeds"]:
            errs.append(f"{len(spec.seeds)} seeds, tier {spec.tier} allows 1..{t['max_seeds']}")
    if not spec.hypothesis or len(spec.hypothesis.strip()) < 20:
        errs.append("hypothesis must state the finding it tests and the predicted effect")
    train = spec.expanded_train_workloads(cfg) if check_files else list(spec.train_workloads)
    if check_files and not train:
        errs.append("no training workloads (train_workloads / workload_names expand to nothing)")

    readable = cfg.read_only_data + [cfg.agent_root]
    for w in train + ([spec.plans_dir] if spec.workload_names else []):
        p = cfg.resolve_data(w)
        # leakage guard: the target database never enters training
        if "imdb" in str(p).lower():
            errs.append(f"train workload {w} touches IMDB (test-only database)")
        if not _under(p, readable):
            errs.append(f"train workload {w} is outside the allowed data roots")
        if check_files and not p.exists():
            errs.append(f"train workload {w} does not exist ({p})")
    sp = cfg.resolve_data(spec.statistics_file)
    if not _under(sp, readable):
        errs.append(f"statistics_file {spec.statistics_file} outside allowed roots")
    if check_files and not sp.exists():
        errs.append(f"statistics_file does not exist ({sp})")
    if "imdb" in sp.name.lower():
        errs.append("statistics_file must not be IMDB-specific")

    for a in spec.extra_args:
        if a.split("=")[0] in RESERVED_ARGS:
            errs.append(f"extra_args may not set {a} (the harness sets it)")

    for g in spec.gates:
        if g.op not in GATE_OPS:
            errs.append(f"gate op {g.op!r} not in {GATE_OPS}")
        if (g.value is None) == (g.ref is None):
            errs.append(f"gate at epoch {g.epoch} needs exactly one of value/ref")
        if g.ref is not None and not (cfg.baselines_dir / f"{g.ref}.json").exists():
            errs.append(f"gate ref {g.ref!r} has no baselines/{g.ref}.json")
        # gates may use the per-epoch IMDB benchmark curves: they only ever ABORT a run,
        # they never select a checkpoint or a winner, so they do not leak the test set.
    if spec.path is not None and spec.id.startswith("e") and not spec.id.startswith(("e000-", "e001-")):
        errs += framework_errors(spec.path.parent / "hypothesis.md", HYPOTHESIS_SECTIONS)
    if spec.tier == "full" and not spec.parent:
        errs.append("a full run must name the proxy experiment it promotes (parent)")
    return errs
