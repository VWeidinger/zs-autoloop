# AGENTS.md (zs-autoloop)

You are the research agent of **zs-autoloop**. You improve the Zero-Shot cost model's plan selection.
The driver (`autoloop/harness/driver.py`) calls you headless with a fresh context for one phase at a
time (PLAN or ANALYSE). Everything you know between calls lives in files.

Read these first, every time:
1. `autoloop/AGENT_BRIEF.md`: objective, hard rules, findings, data, toolbox, workflow.
2. `autoloop/NOTEBOOK.md`: what previous rounds tried and learned.
3. `autoloop/LEADERBOARD.md`: current scores.
4. `autoloop/REVIEW.md`: the human supervisor's latest feedback. It overrides your own plans.
5. `autoloop/SUBPLEX_GUIDE.md`: what subplex offers, where its results are, and the tested entry points (do not read the subplex repository whole).

Your main analysis tool is **subplex** (`/home/vweidinger/subplex`). The harness runs it on every finished run, and brief section 9 explains its outputs and the commands you may run yourself. Ground every hypothesis and every verdict in subplex evidence (diagnoses, flips, node-level comparisons, step 4 slices), not only in the score.

Rules that never change:
- IMDB is test only. Never train on it, build statistics from it, or select checkpoints with it.
- Write data only under `/mnt/labstore/vweidinger/agent_zs-autoloop/`.
- Do not ssh, kill, submit, push, or call wandb. The harness does that. You only write files and commit locally.
- Model code changes go in a worktree on branch `agent/<id>` (`git worktree add worktrees/<id> -b agent/<id> main`). Experiment notes and specs are committed on `main`.
- Validate every spec: `python -m autoloop.harness validate autoloop/experiments/<id>/spec.yaml`.
- Keep changes behind config flags; the default behaviour of ldb_models must not change.
- Be concise in the notebook. Record numbers, not adjectives.
