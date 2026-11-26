"""ssh to the GPU nodes and the GPU free-slot probe. Every remote call can be dry-run."""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .config import Config

Runner = Callable[[str, str], Tuple[int, str]]  # (host, command) -> (returncode, stdout)


def ssh_runner(cfg: Config) -> Runner:
    timeout = cfg["cluster"]["ssh_connect_timeout_s"]

    def run(host: str, command: str) -> Tuple[int, str]:
        if host in ("localhost", os.uname().nodename.split(".")[0]):
            p = subprocess.run(["bash", "-lc", command], capture_output=True, text=True, timeout=120)
        else:
            p = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}", host, command],
                capture_output=True, text=True, timeout=120)
        return p.returncode, p.stdout

    return run


def dry_runner(log: Optional[List[str]] = None, free: Optional[Dict[str, List[int]]] = None) -> Runner:
    """Pretends every listed GPU is idle; records commands instead of running them."""
    free = free or {}

    def run(host: str, command: str) -> Tuple[int, str]:
        if log is not None:
            log.append(f"[{host}] {command}")
        if "nvidia-smi --query-gpu" in command:
            gpus = free.get(host, [])
            return 0, "".join(f"{g}, GPU-{host}-{g}, 3, 0\n" for g in gpus)
        if "nvidia-smi --query-compute-apps" in command:
            return 0, ""
        return 0, ""

    return run


@dataclass
class GpuState:
    host: str
    index: int
    uuid: str
    mem_used: int
    util: int
    n_procs: int

    def idle(self, max_mem: int) -> bool:
        return self.mem_used < max_mem and self.n_procs == 0


PROBE_GPU = "nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits"
PROBE_APPS = "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader"


def probe_host(run: Runner, host: str) -> List[GpuState]:
    rc, out = run(host, PROBE_GPU)
    if rc != 0:
        return []
    _, apps = run(host, PROBE_APPS)
    procs: Dict[str, int] = {}
    for line in apps.splitlines():
        if line.strip():
            u = line.split(",")[0].strip()
            procs[u] = procs.get(u, 0) + 1
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx, uuid, mem, util = parts[:4]
        gpus.append(GpuState(host, int(idx), uuid, int(float(mem)), int(float(util)), procs.get(uuid, 0)))
    return gpus


def free_slots(cfg: Config, run: Runner, busy: set[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """GPUs idle in every one of `free_probes` probes and not occupied by one of our own runs."""
    import time
    c = cfg["cluster"]
    candidates: Optional[set] = None
    for i in range(c["free_probes"]):
        now = set()
        for host in c["nodes"]:
            for g in probe_host(run, host):
                if g.index < c["gpus_per_node"] and g.idle(c["free_mem_mib"]):
                    now.add((host, g.index))
        candidates = now if candidates is None else candidates & now
        if i + 1 < c["free_probes"] and c["probe_interval_s"] > 0:
            time.sleep(c["probe_interval_s"])
    return sorted((candidates or set()) - busy)


def ensure_ssh_agent(key: Path, passphrase: Optional[str]) -> None:
    """Start an ssh-agent for this process tree and add the key, using the passphrase from secrets."""
    if os.environ.get("SSH_AUTH_SOCK") and subprocess.run(["ssh-add", "-l"], capture_output=True).returncode == 0:
        return
    out = subprocess.run(["ssh-agent", "-s"], capture_output=True, text=True, check=True).stdout
    for part in out.replace("\n", ";").split(";"):
        if "=" in part and part.strip().startswith(("SSH_AUTH_SOCK", "SSH_AGENT_PID")):
            k, v = part.strip().split("=", 1)
            os.environ[k] = v
    if passphrase is None:
        subprocess.run(["ssh-add", str(key)], check=True)
        return
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as f:
        f.write("#!/bin/sh\nprintf '%s\\n' \"$AUTOLOOP_SSH_PASS\"\n")
        askpass = f.name
    os.chmod(askpass, 0o700)
    try:
        env = dict(os.environ, SSH_ASKPASS=askpass, SSH_ASKPASS_REQUIRE="force",
                   AUTOLOOP_SSH_PASS=passphrase, DISPLAY=os.environ.get("DISPLAY", ":0"))
        subprocess.run(["ssh-add", str(key)], env=env, check=True, stdin=subprocess.DEVNULL,
                       capture_output=True)
    finally:
        os.unlink(askpass)
