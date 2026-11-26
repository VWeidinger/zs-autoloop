"""Landlock sandbox for the agent CLI. No root, no user namespaces (Ubuntu's AppArmor blocks
bubblewrap, so Codex's own sandbox cannot start on the lab machines).

Landlock is applied to this process and inherited by every child; it cannot be lifted from inside.
    files    read-only / read-write allow-lists; everything else (~/.ssh, ~/.config, credentials,
             other users' data) is invisible
    network  outgoing TCP only to the listed ports (443 for the model API); no listening sockets

    python3 -m autoloop.harness.sandbox --ro /usr --rw /tmp --connect 443 -- cmd args...
    python3 -m autoloop.harness.sandbox --profile agent -- codex exec ...     (paths from the config)
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path
from typing import Dict, List

SYS_CREATE_RULESET, SYS_ADD_RULE, SYS_RESTRICT_SELF = 444, 445, 446   # x86_64
RULE_PATH_BENEATH, RULE_NET_PORT = 1, 2
PR_SET_NO_NEW_PRIVS = 38

FS = {name: 1 << i for i, name in enumerate([
    "EXECUTE", "WRITE_FILE", "READ_FILE", "READ_DIR", "REMOVE_DIR", "REMOVE_FILE", "MAKE_CHAR", "MAKE_DIR",
    "MAKE_REG", "MAKE_SOCK", "MAKE_FIFO", "MAKE_BLOCK", "MAKE_SYM", "REFER", "TRUNCATE"])}
FILE_ONLY = FS["EXECUTE"] | FS["WRITE_FILE"] | FS["READ_FILE"] | FS["TRUNCATE"]
NET_BIND, NET_CONNECT = 1, 2


class RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64)]


class PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class NetPortAttr(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("port", ctypes.c_uint64)]


_libc = ctypes.CDLL(None, use_errno=True)


def _check(ret: int, what: str) -> int:
    if ret < 0:
        err = ctypes.get_errno()
        raise OSError(err, f"landlock {what}: {os.strerror(err)}")
    return ret


def abi_version() -> int:
    return _libc.syscall(SYS_CREATE_RULESET, None, 0, 1)


def _fs_masks(abi: int) -> Dict[str, int]:
    handled = sum(v for k, v in FS.items()
                  if not (k == "REFER" and abi < 2) and not (k == "TRUNCATE" and abi < 3))
    ro = FS["EXECUTE"] | FS["READ_FILE"] | FS["READ_DIR"]
    return {"handled": handled, "ro": ro, "rw": handled}


def restrict(ro: List[str], rw: List[str], connect_ports: List[int]) -> None:
    abi = abi_version()
    if abi < 1:
        raise OSError("landlock is not available on this kernel")
    masks = _fs_masks(abi)
    attr = RulesetAttr(masks["handled"], (NET_BIND | NET_CONNECT) if abi >= 4 else 0)
    size = ctypes.sizeof(RulesetAttr) if abi >= 4 else 8
    fd = _check(_libc.syscall(SYS_CREATE_RULESET, ctypes.byref(attr), size, 0), "create_ruleset")
    for paths, access in ((ro, masks["ro"]), (rw, masks["rw"])):
        for p in paths:
            p = os.path.expanduser(p)
            if not os.path.exists(p):
                continue
            pfd = os.open(p, os.O_PATH | os.O_CLOEXEC)
            try:
                allowed = access & FILE_ONLY if not os.path.isdir(p) else access
                rule = PathBeneathAttr(allowed, pfd)
                _check(_libc.syscall(SYS_ADD_RULE, fd, RULE_PATH_BENEATH, ctypes.byref(rule), 0), f"add_rule {p}")
            finally:
                os.close(pfd)
    if abi >= 4:
        for port in connect_ports:
            rule = NetPortAttr(NET_CONNECT, port)
            _check(_libc.syscall(SYS_ADD_RULE, fd, RULE_NET_PORT, ctypes.byref(rule), 0), f"net port {port}")
    _check(_libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), "no_new_privs")
    _check(_libc.syscall(SYS_RESTRICT_SELF, fd, 0), "restrict_self")
    os.close(fd)


def agent_profile(cfg) -> Dict[str, List]:
    """What the agent may touch: code, its data area, the read-only training data, subplex."""
    home = Path.home()
    sb = cfg["agent"]["sandbox"]
    repo = Path(cfg.repo)
    (repo / "worktrees").mkdir(exist_ok=True)
    ro = ["/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/opt", "/run", "/proc", "/sys", "/snap",
          str(home / ".local/bin"), str(home / ".local/share/uv"), str(home / ".local/share/codex-package"),
          str(repo),                      # the whole repo readable: harness, config, prompts, brief stay READ-ONLY
          cfg["subplex"]["repo"], *[str(p) for p in cfg.read_only_data]] + list(sb.get("extra_ro", []))
    # writable in the repo: only what the agent owns. Model code changes happen in worktrees/.
    agent_owned = [repo / ".git", repo / "worktrees", cfg.experiments_dir,
                   cfg.autoloop_dir / "NOTEBOOK.md", cfg.autoloop_dir / "SUBPLEX_FEEDBACK.md"]
    rw = ["/tmp", "/dev", "/var/tmp", *[str(p) for p in agent_owned], str(cfg.agent_root), str(home / ".codex"),
          str(home / ".cache/uv")] + list(sb.get("extra_rw", []))
    return {"ro": ro, "rw": rw, "connect": list(sb.get("connect_ports", [443]))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="autoloop.harness.sandbox")
    ap.add_argument("--ro", action="append", default=[])
    ap.add_argument("--rw", action="append", default=[])
    ap.add_argument("--connect", action="append", type=int, default=[])
    ap.add_argument("--profile", choices=["agent"])
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command")
    ro, rw, ports = list(a.ro), list(a.rw), list(a.connect)
    if a.profile == "agent":
        from .config import load_config
        prof = agent_profile(load_config())
        ro += prof["ro"]
        rw += prof["rw"]
        ports += prof["connect"]
    restrict(ro, rw, ports)
    os.execvp(cmd[0], cmd)
    return 127


if __name__ == "__main__":
    sys.exit(main())
