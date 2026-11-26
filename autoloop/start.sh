#!/usr/bin/env bash
# Start the zs-autoloop driver in a detached tmux session (survives ssh disconnects).
#   ./autoloop/start.sh            start
#   tmux attach -t autoloop        look at the driver window (leave with Ctrl-b d, NOT Ctrl-c)
#   python3 -m autoloop.harness watch -f    follow the agent live (Ctrl-c here only stops watching)
#   python3 -m autoloop.harness pause       stop the driver cleanly at its next step
set -euo pipefail
cd "$(dirname "$0")/.."
if tmux has-session -t autoloop 2>/dev/null; then
  echo "already running - look at it with: tmux attach -t autoloop"; exit 1
fi
rm -f /mnt/labstore/vweidinger/agent_zs-autoloop/state/STOP
tmux new-session -d -s autoloop \
  "python3 -m autoloop.harness driver 2>&1 | tee -i -a ~/autoloop-driver.log; echo; echo 'driver ended - press Enter to close'; read"
echo "started in tmux session 'autoloop'."
echo "  follow the agent:   cd ~/zs-autoloop && python3 -m autoloop.harness watch -f"
echo "  status:             python3 -m autoloop.harness watch"
echo "  stop cleanly:       python3 -m autoloop.harness pause"
