# zs-autoloop: operator guide

An agent (Codex CLI by default; Claude Code also works) improves Zero-Shot's plan selection in
rounds. A plain-Python driver does all the waiting, so **no tokens are used while GPUs train**.

```
driver (tmux on c02, plain Python)
  PLAN    -> agent writes autoloop/experiments/<id>/{hypothesis.md,spec.yaml}, code on agent/<id>
  submit  -> harness: validate spec, probe c01-c05, code snapshot -> labstore,
             ssh node "screen ... run_exp -m 'zs-autoloop <run>' -n <gpu> -t <hh:mm> -- inner.sh"
  watch   -> every 10 min: exit_code file, log staleness, wandb curves vs the spec's gates
             (NaN / stale / missed gate -> pkill). No LLM.
  subplex -> harness runs `python -m subplex.external_model` on the run (c02 GPU, minutes):
             step 1 benchmark report (diagnoses, flips vs reference, node-level worst picks),
             step 2 subplan annotation, step 4 error curve -> <agent_root>/subplex/<run>/
  ANALYSE -> harness writes result.json (per-benchmark ratios + verdicts); agent reads the subplex
             report, writes analysis.md + NOTEBOOK
  after 2 rounds -> AWAITING_REVIEW, wandb alert (e-mail), driver exits
```

| What | Where |
|---|---|
| config (caps, nodes, tiers, model, budget, prices) | `autoloop/config/autoloop.yaml` |
| agent instructions | `AGENTS.md` (+ `CLAUDE.md`), `autoloop/AGENT_BRIEF.md`, `autoloop/prompts/` |
| experiments (git) | `autoloop/experiments/<id>/`, `NOTEBOOK.md`, `LEADERBOARD.md`, `REVIEW.md` |
| runs, models, evaluations, venvs, state, ledger | `/mnt/labstore/vweidinger/agent_zs-autoloop/` |
| secrets (wandb, ssh passphrase, agent API key) | `~/.config/zs-autoloop/secrets.env` (chmod 600, never in git) |

## Commands

```bash
cd ~/zs-autoloop
python3 -m pytest autoloop/tests -q                       # offline tests
python3 -m autoloop.harness validate autoloop/experiments/<id>/spec.yaml
python3 -m autoloop.harness probe                         # GPU state on c01-c05 (ssh)
python3 -m autoloop.harness status                        # runs, round, spent USD
python3 -m autoloop.harness driver --dry-run --max-steps 1  # no ssh/agent/wandb; writes prompts to state/agent_logs
tmux new -s autoloop 'python3 -m autoloop.harness driver 2>&1 | tee -a ~/autoloop-driver.log'
python3 -m autoloop.harness pause                         # stop at next tick; trainings continue
python3 -m autoloop.harness cancel <run_id> --reason "..."
python3 -m autoloop.harness resume                        # after writing autoloop/REVIEW.md
```

## Review cycle

1. The driver stops in `AWAITING_REVIEW` after `loop.rounds_before_review` rounds and sends a wandb alert.
2. Read `NOTEBOOK.md`, `LEADERBOARD.md` and the experiments' `analysis.md`. Check `git log --all --oneline` for the agent's branches.
3. Write your feedback into `autoloop/REVIEW.md` (what to drop, what to pursue, budget changes).
4. Run `python3 -m autoloop.harness resume`, then start the driver again in tmux.

## Spend

- **LLM:** `state/ledger.jsonl` records one line per agent call, with tokens taken from the CLI's own JSON output, multiplied by the prices in the config.
  - Before each call, the driver checks `spent + max_usd_per_phase <= total_usd`.
  - It refuses to call the agent while the model's prices are unset.
- **GPU:** at most 6 concurrent runs, one per GPU, and only on GPUs that are idle in 2 probes in a row. Every job is registered with `run_exp` (shared).

## Setup state (2026-11-26)

- [x] Repository pushed to `VWeidinger/zs-autoloop`. Lab tokens were removed from the whole history; uv reads `ldb_schema` access from `~/.git-credentials`, on c02 only.
- [x] The shared venv is at `<agent_root>/venvs/3fedfb092158`, on a labstore interpreter. Nodes only reuse it and never download.
- [x] The Codex CLI package (0.172.0, with bundled bubblewrap) is in `~/.local/share/codex-package`, linked from `~/.local/bin/codex`, and logged in with the API key.
- [x] **Agent sandbox without root.** Codex's own sandbox needs bubblewrap with user namespaces, which AppArmor blocks here, and there is no sudo. So Codex runs with its sandbox off inside our Landlock sandbox (`autoloop/harness/sandbox.py`, kernel Landlock ABI 4, no root).
  - **Tested:** it can write only the repo, the agent root, `/tmp`, `~/.codex` and `~/.cache/uv`, and connect only to TCP 443.
  - **Tested:** `~/.ssh`, `~/.config` (secrets), `~/.git-credentials` and `~/.netrc` are unreachable, so ssh, push and listening ports are blocked.
  - **Tested:** git worktree and commit work, under the identity "zs-autoloop agent".
  - **Smoke call:** one real Codex call inside the sandbox cost $0.03.
  - **Check it yourself:** `PYTHONPATH=. python3 -m autoloop.harness.sandbox --profile agent -- cat ~/.ssh/vw_encrypted` must fail.
- [x] c01–c05 host keys are in `known_hosts`.
- [x] The subplex reference diagnosis (steps 1–2 of the three reference models) is in `<agent_root>/subplex/_reference/`. Rebuild it with `python3 -m autoloop.harness reference-diagnosis`.
- [ ] First real start: `tmux new -s autoloop 'python3 -m autoloop.harness driver 2>&1 | tee -a ~/autoloop-driver.log'`.
- [ ] In the wandb settings, enable "Scriptable run alerts", e-mail.

## Watching the loop (also after an ssh disconnect)

The driver runs in tmux and keeps going when your ssh connection drops.

```bash
tmux attach -t autoloop                      # back into the driver session (Ctrl-b d to leave it running)
cd ~/zs-autoloop
python3 -m autoloop.harness watch            # status, runs, driver log, the agent's current/latest phase
python3 -m autoloop.harness watch -f         # ... and stream the agent live (Ctrl-c to stop watching)
python3 -m autoloop.harness watch --phase plan-r1 -v   # a past phase with full command outputs
python3 -m autoloop.harness costs            # every agent call: tokens x prices = USD
```

Everything is on labstore under `/mnt/labstore/vweidinger/agent_zs-autoloop/state/`:

| file | content |
|---|---|
| `driver.log` | timestamped driver events: submits, run ends, kills and why, subplex analyses, agent phases, pushes |
| `agent_logs/<time>-<phase>.prompt.md` | the exact prompt of each agent phase |
| `agent_logs/<time>-<phase>.out.jsonl` | the agent's complete event stream, written live: messages, commands with output and exit code, file edits, tokens |
| `ledger.jsonl` | tokens and USD per agent call (OpenAI shows you nothing; this is the record) |
| `driver.json`, `runs/<run>/state.json` | the loop's state (round, phase) and each run's state (node, GPU, epoch, gates) |

Codex additionally keeps its own session records, including reasoning summaries, in `~/.codex/sessions/`.
In git you find what the agent decided: hypotheses, analyses, `NOTEBOOK.md`, `FRAMEWORK_REPORT.md` and
`framework_usage/`.

## Review cycle with Claude Code

When the driver stops in `AWAITING_REVIEW`:
1. Open Claude Code in `~/zs-autoloop` and discuss `NOTEBOOK.md`, `LEADERBOARD.md`, the `analysis.md` files and the subplex reports.
2. Write `autoloop/REVIEW.md` together.
3. Run `python3 -m autoloop.harness resume` and start the driver again in tmux.

You never start Codex by hand: the driver calls it for each phase.
