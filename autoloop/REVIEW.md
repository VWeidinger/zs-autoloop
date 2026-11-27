# Supervisor review

## 2026-11-27 - harness problems fixed, continue round 1

- Both problems you reported are fixed: DGL now reads its config from the agent root, so CPU smoke tests work
  in your worktrees with `uv run --frozen ...`. The run scripts no longer need `uv` on the nodes, and a failed
  training is now reported as failed instead of finished.
- Your round 1 diagnosis in NOTEBOOK.md stands; do not redo it. Continue the four experiments you prepared in
  `worktrees/e002-scan-hints`, `e003-subplan-loss`, `e004-log-card` and `e005-loop-feature`:
  finish the code, smoke-test it, write hypothesis.md and spec.yaml with the framework sections, and set them ready.
- e000 (baseline proxy) is re-run by the harness. e001 (full baseline) has been training on c02 since 07:20.
  Both count against the round's 6 run slots, so 4 are left for your proxies.
