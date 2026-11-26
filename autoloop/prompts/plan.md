PHASE: PLAN, round {{round}}

Read AGENTS.md and the files it lists. Then plan this round.

Constraints for this round:
- at most {{max_runs}} runs in total (a run = one seed of one experiment), at most {{max_full}} of them `full`
- the next free experiment id is {{next_id}} (count up from there: e00X-short-name)
- budget left for you (LLM cost): {{budget_left}}

Current runs:
{{status}}

Supervisor review (overrides your own plans):
{{review}}

Do this:
1. Decide which experiments give the most information per GPU hour, given the notebook and the
   leaderboard. Prefer proxies; promote to `full` only with evidence (a proxy that beat the baseline
   proxy). Each experiment tests ONE hypothesis tied to a finding in the brief.
2. For each experiment: write hypothesis.md, implement it on branch agent/<id> in a worktree,
   smoke-test it on CPU (≤ 10 min, tiny data), write spec.yaml with `status: ready` and honest
   gates, and validate it with the harness.
3. If something in an experiment would not validate or the smoke test fails, do not mark it ready.
4. Commit the experiment folders on main. Append a short "Round {{round}} plan" entry to
   autoloop/NOTEBOOK.md: which experiments, why these and not others.

Stop when the specs are ready. Do not wait for training; the driver will call you again.
