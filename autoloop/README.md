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

- [x] Repository pushed to `VWeidinger/zs-autoloop`. Lab tokens were removed from the whole history; uv reads `ldb_schema` access from `~/.git-credentials`.
- [x] Codex CLI 0.172.0 is in `~/.local/bin/codex` and logged in with the API key (`~/.codex/auth.json`, mode 600).
- [x] The shared venv is at `<agent_root>/venvs/3fedfb092158`, on a labstore interpreter (`<agent_root>/python`), so every node can use it.
- [x] Prices: conservative defaults in the config. The API key cannot report prices or usage.
- [x] subplex can be called for any checkpoint: `python -m subplex.external_model` (in `~/subplex`).
- [ ] c04/c05: the host keys are missing from `~/.ssh/known_hosts`. Check the fingerprints, then run `ssh-keyscan -H c04.lab.tuda.systems c05.lab.tuda.systems >> ~/.ssh/known_hosts`.
- [ ] First real start: `tmux new -s autoloop 'python3 -m autoloop.harness driver 2>&1 | tee -a ~/autoloop-driver.log'`.
- [ ] In the wandb settings, enable "Scriptable run alerts", e-mail.

## Review cycle with Claude Code

When the driver stops in `AWAITING_REVIEW`:
1. Open Claude Code in `~/zs-autoloop` and discuss `NOTEBOOK.md`, `LEADERBOARD.md`, the `analysis.md` files and the subplex reports.
2. Write `autoloop/REVIEW.md` together.
3. Run `python3 -m autoloop.harness resume` and start the driver again in tmux.

You never start Codex by hand: the driver calls it for each phase.
