"""Notifications via wandb alerts (e-mail/Slack per your wandb account settings)."""

from __future__ import annotations

import sys

from .config import Config, load_secrets


def notify(cfg: Config, title: str, text: str, dry_run: bool = False) -> None:
    print(f"[notify] {title}: {text}", file=sys.stderr)
    if dry_run or not cfg["notify"].get("wandb_alert"):
        return
    try:
        import os
        import wandb
        key = load_secrets(cfg).get("WANDB_API_KEY")
        if key:
            os.environ.setdefault("WANDB_API_KEY", key)
        w = cfg["wandb"]
        run = wandb.init(entity=w["entity"], project=w["project"], group=w["group"], job_type="driver",
                         name=f"{cfg['name']}-driver", id=f"{cfg['name']}-driver", resume="allow",
                         settings=wandb.Settings(silent=True))
        run.alert(title=title[:64], text=text[:1000])
        run.finish()
    except Exception as e:  # a failed notification must never stop the loop
        print(f"[notify] wandb alert failed: {e}", file=sys.stderr)
