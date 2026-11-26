"""Configuration and secrets for the autoloop harness."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "autoloop.yaml"


@dataclass
class Config:
    raw: Dict[str, Any]
    path: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # paths -----------------------------------------------------------------
    @property
    def repo(self) -> Path:
        return Path(self.raw["paths"]["repo"]).expanduser()

    @property
    def agent_root(self) -> Path:
        return Path(self.raw["paths"]["agent_root"]).expanduser()

    @property
    def data_root(self) -> Path:
        return Path(self.raw["paths"]["data_root"]).expanduser()

    @property
    def read_only_data(self) -> list[Path]:
        return [Path(p).expanduser() for p in self.raw["paths"]["read_only_data"]]

    @property
    def experiments_dir(self) -> Path:
        p = self.raw["paths"].get("experiments")
        return Path(p).expanduser() if p else Path(__file__).resolve().parents[1] / "experiments"

    @property
    def autoloop_dir(self) -> Path:
        """Folder holding NOTEBOOK.md, LEADERBOARD.md, REVIEW.md (parent of experiments/)."""
        return self.experiments_dir.parent

    @property
    def baselines_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "baselines"

    @property
    def runs_dir(self) -> Path:
        return self.agent_root / "runs"

    @property
    def state_dir(self) -> Path:
        return self.agent_root / "state"

    def resolve_data(self, p: str) -> Path:
        path = Path(p).expanduser()
        return path if path.is_absolute() else self.data_root / path


def load_config(path: Optional[Path] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    path = Path(path or os.environ.get("AUTOLOOP_CONFIG", DEFAULT_CONFIG))
    raw = yaml.safe_load(path.read_text())
    if overrides:
        _deep_update(raw, overrides)
    return Config(raw=raw, path=path)


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> None:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def load_secrets(cfg: Config) -> Dict[str, str]:
    """KEY=VALUE lines (optionally prefixed with 'export'). Never logged, never passed to the agent."""
    p = Path(cfg["paths"]["secrets_file"]).expanduser()
    out: Dict[str, str] = {}
    if not p.exists():
        return out
    mode = p.stat().st_mode & 0o777
    if mode & 0o077:
        raise PermissionError(f"{p} must be chmod 600 (is {oct(mode)})")
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip("'\"")
    return out
