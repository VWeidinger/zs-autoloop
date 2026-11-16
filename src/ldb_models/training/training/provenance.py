"""
Record how a model was trained, next to and inside its checkpoint.

A checkpoint alone does not say which data it was trained on: two copies of a workload
can share a file name (e.g. a JIT-filtered copy saved as ``parsed_plans3``), so the paths in
a run script are not enough. ``collect_training_info`` stores the command line, the parsed
arguments, the resolved model config, the code version and a fingerprint of every input
file (resolved path, size, mtime and a hash of its first and last MiB - cheap even for
multi-GB workloads, and enough to tell two copies of a file apart).

It is written once as ``{model_name}_training_info.json`` when training starts (also for
tabular models, which never go through ``save_checkpoint``) and stored under
``checkpoint["training_info"]`` in every ``.pt`` checkpoint. The dict is a few tens of KB.

Which plan was trained on and which was only validated on depends on the random state at
the moment of the split, so it cannot be read off the files. ``create_datasets`` therefore
hands the split to ``record_split`` and ``main.py`` writes it as
``{model_name}_training_split.json.gz``: per split a list of ``[file, position]``, where
``file`` indexes the ``files`` list of the same document and ``position`` is the plan's
index in that file's plan list (``run_file_pos``, counted before any filtering). The
training info only gets the path, hash and sizes of that file (``info["split"]``), so the
checkpoints do not grow by it.

Inspect a model with ``python -m ldb_models.training.training.provenance <.pt or .json>``.
"""

import getpass
import gzip
import hashlib
import inspect
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import attrs

FINGERPRINT_BYTES = 1 << 20  # hash the first and the last MiB of every input file

# PYTHONHASHSEED as the interpreter saw it at start-up. Setting it later (as use_model does)
# does not change hash(), so this is what decides whether string hashing (e.g. MSCN's
# literal encoding) was reproducible.
PYTHONHASHSEED_AT_START = os.environ.get("PYTHONHASHSEED")


def file_fingerprint(path) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    path = Path(path)
    info: Dict[str, Any] = {"path": str(path), "realpath": os.path.realpath(path)}
    if not path.is_file():
        info["exists"] = False
        return info
    stat = path.stat()
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read(FINGERPRINT_BYTES))
        if stat.st_size > FINGERPRINT_BYTES:
            handle.seek(max(FINGERPRINT_BYTES, stat.st_size - FINGERPRINT_BYTES))
            digest.update(handle.read(FINGERPRINT_BYTES))
    info.update(
        exists=True,
        size_bytes=stat.st_size,
        mtime=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        head_tail_sha256=digest.hexdigest(),
    )
    return info


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if inspect.isclass(value) or callable(value):
        return f"{getattr(value, '__module__', '?')}.{getattr(value, '__qualname__', repr(value))}"
    return repr(value)


def config_to_dict(config) -> Dict[str, Any]:
    """The resolved model config; classes and functions are stored by their import path."""
    return _jsonable(attrs.asdict(config, recurse=False))


def code_version() -> Dict[str, Any]:
    package_dir = Path(__file__).resolve().parents[2]
    version: Dict[str, Any] = {"package_dir": str(package_dir)}
    try:
        commit = subprocess.run(
            ["git", "-C", str(package_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(package_dir), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        version.update(git_commit=commit, git_dirty=bool(dirty))
    except Exception:
        # installed as a package (e.g. by uv from git): the commit is in direct_url.json
        try:
            direct_url = json.loads(metadata.distribution("ldb_models").read_text("direct_url.json") or "{}")
            vcs = direct_url.get("vcs_info", {})
            version.update(
                git_commit=vcs.get("commit_id"),
                git_revision=vcs.get("requested_revision"),
                url=direct_url.get("url", "").split("@")[-1],  # never store credentials
            )
        except Exception:
            pass
    return version


def collect_training_info(
    cli_args: Dict[str, Any],
    config,
    train_workloads: Iterable,
    test_workloads: Iterable,
    other_files: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "user": getpass.getuser(),
        "cwd": os.getcwd(),
        "command": " ".join(shlex.quote(a) for a in [sys.executable, *sys.argv]),
        "cli_args": _jsonable(cli_args),
        "python_hash_seed_at_start": PYTHONHASHSEED_AT_START,
        "code": code_version(),
        "model_config": config_to_dict(config),
        "train_workloads": [file_fingerprint(p) for p in train_workloads],
        "test_workloads": [file_fingerprint(p) for p in test_workloads],
        "files": {name: file_fingerprint(p) for name, p in other_files.items()},
    }


def training_info_path(model_dir, model_name: str) -> Path:
    return Path(model_dir) / f"{model_name}_training_info.json"


def write_training_info(info: Dict[str, Any], model_dir, model_name: str) -> Path:
    path = training_info_path(model_dir, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(info, handle, indent=1)
    return path


def read_training_info(path) -> Optional[Dict[str, Any]]:
    """From a ``_training_info.json`` or from a ``.pt`` checkpoint (None for older ones)."""
    path = Path(path)
    if path.suffix == ".json":
        with open(path) as handle:
            return json.load(handle)
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint.get("training_info")


# ---------------------------------------------------------------------------
# the train / validation split
# ---------------------------------------------------------------------------

# The split of the training data of this process, set by create_datasets and taken by
# main.py once the loaders are built. A module global, because create_datasets is called
# from every model's own dataloader factory and none of them knows the model directory.
_RECORDED_SPLIT: Optional[Dict[str, Any]] = None


def _positions(plans, indexes) -> List[List[int]]:
    """``[file, position]`` per entry of ``indexes``, in the order the dataset holds them.
    -1 marks a plan that carries no position (plans not read by read_workload_runs)."""
    if indexes is None:
        return []
    out = []
    for i in indexes:
        plan = plans[i]
        file_id, pos = plan.get("run_file_id"), plan.get("run_file_pos")
        out.append([-1 if file_id is None else int(file_id), -1 if pos is None else int(pos)])
    return out


def record_split(workload_run_paths, plans, train_indexes, val_indexes) -> None:
    """Remember which plans are in the training and in the validation set.

    ``plans`` and the index lists are what ``split_plans_in_train_val`` worked on - one
    list, or one list per file (online hard example mining; there the lists are the pools
    each epoch draws from). With ``train_on_subplans`` a subplan carries the position of
    the query it was cut from, so the record names queries, not subplans.
    """
    global _RECORDED_SPLIT
    per_file = bool(plans) and isinstance(plans[0], list)
    if per_file:
        val_indexes = val_indexes or [None] * len(plans)
        train = [p for pl, idx in zip(plans, train_indexes) for p in _positions(pl, idx)]
        val = [p for pl, idx in zip(plans, val_indexes) for p in _positions(pl, idx)]
    else:
        train, val = _positions(plans, train_indexes), _positions(plans, val_indexes)
    _RECORDED_SPLIT = {
        "files": [str(p) for p in workload_run_paths],
        "per_file_pools": per_file,
        "train": train,
        "val": val,
    }


def take_split() -> Optional[Dict[str, Any]]:
    """The recorded split, once (None if no training data was split in this process)."""
    global _RECORDED_SPLIT
    split, _RECORDED_SPLIT = _RECORDED_SPLIT, None
    return split


def training_split_path(model_dir, model_name: str) -> Path:
    return Path(model_dir) / f"{model_name}_training_split.json.gz"


def write_training_split(split: Dict[str, Any], model_dir, model_name: str) -> Dict[str, Any]:
    """Write the split next to the checkpoint; returns what goes into ``info["split"]``."""
    path = training_split_path(model_dir, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(split, separators=(",", ":")).encode()
    # mtime=0 and no file name in the gzip header: the same split gives the same bytes
    with open(path, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as handle:
        handle.write(payload)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n_train": len(split["train"]),
        "n_val": len(split["val"]),
        "per_file_pools": split.get("per_file_pools", False),
    }


def read_training_split(path) -> Dict[str, Any]:
    """``{"files": [...], "train": [[file, position], ...], "val": [...]}``."""
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def _summary(info: Dict[str, Any]) -> str:
    lines = [
        f"trained {info['created_at']} on {info['host']} by {info['user']}",
        f"code    {info['code'].get('git_commit')}" + (" (dirty)" if info["code"].get("git_dirty") else ""),
        f"hash    PYTHONHASHSEED at start = {info['python_hash_seed_at_start']}",
        f"command {info['command']}",
        "train workloads:",
    ]
    for f in info["train_workloads"]:
        lines.append(f"  {f.get('realpath')}  {f.get('size_bytes', 'missing')} B  {f.get('head_tail_sha256', '')[:12]}")
    lines.append("test workloads:")
    for f in info["test_workloads"]:
        lines.append(f"  {f.get('realpath')}  {f.get('size_bytes', 'missing')} B")
    split = info.get("split")
    if split:
        lines.append(f"split   {split['n_train']} train / {split['n_val']} val plans in {split['path']}")
    else:
        lines.append("split   not recorded (training predates it)")
    return "\n".join(lines)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        found = read_training_info(arg)
        print(f"== {arg}\n" + (_summary(found) if found else "no training info (checkpoint predates it)"))
