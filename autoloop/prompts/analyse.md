PHASE: ANALYSE experiment {{experiment}} (round {{round}})

Read AGENTS.md, autoloop/experiments/{{experiment}}/hypothesis.md, spec.yaml and result.json.

Runs:
{{status}}

Budget left for you (LLM cost): {{budget_left}}

Do this:
1. Compare result.json with the reference and with the experiment's parent or baseline proxy. Use the
   per-benchmark ratios, pick rate and surpassed plans, and the IMDB 200k q-error guard. If the run
   was killed, explain from the gate verdicts in the run state, under
   /mnt/labstore/vweidinger/agent_zs-autoloop/runs/<run>/state.json and train.log, whether the idea
   or the gate was wrong.
2. Where useful, look deeper offline: the *_test_pred.csv files in the run's evaluation folder show
   which statements flipped. You may use subplex
   (`/home/vweidinger/subplex/src/subplex/step1/benchmark_selection.py`) read-only.
3. Write autoloop/experiments/{{experiment}}/analysis.md with a verdict (confirmed / refuted /
   inconclusive), the evidence, and what this changes about the next steps.
4. Append ≤ 8 lines to autoloop/NOTEBOOK.md and commit on main.

Do not start new experiments in this phase.
