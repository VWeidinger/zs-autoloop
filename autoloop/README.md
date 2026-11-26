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
  ANALYSE -> harness writes result.json from *_test_pred.csv; agent writes analysis.md + NOTEBOOK
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

## One-time setup before the first real run

- Fill in `budget.prices` in the config from the provider's pricing page.
- Install the agent CLI on c02. Check `codex exec --help` against `agent.commands.codex`.
- Check that ssh from c02 to c01 and c03–c05 works: `python3 -m autoloop.harness probe`.
- Build the shared venv once and smoke-test a 1-epoch run (see the open decisions in the setup summary).
- In the wandb settings, enable "Scriptable run alerts", e-mail.
