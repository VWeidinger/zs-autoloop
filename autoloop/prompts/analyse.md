PHASE: ANALYSE experiment {{experiment}} (round {{round}})

Read AGENTS.md, autoloop/experiments/{{experiment}}/hypothesis.md, spec.yaml and result.json.

Runs:
{{status}}

Budget left for you (LLM cost): {{budget_left}}

The harness has already run all three subplex steps on every run of this experiment. The output is
in {{subplex_root}}/<run_id>/, described in brief section 9:
- step 1: report/summary.json, decisive_<b>.csv, flips_<b>.csv, compare_*.csv
- step 2: tmp/step4/<run_id>/subplex_<benchmark>/ and the complex-workload curve
- step 3: tmp/step6/counterfactual/<run_id>/<dataset>/
The reference models are in {{subplex_root}}/_reference/. The parent run, if any, is in its own folder.

Do this:
1. For every quantity in the hypothesis's "Predicted framework change" table, read the value after the
   change and compare it with the value before (reference or parent) and with the prediction. If you
   need more, run further subplex analyses yourself, CPU only, on this run's outputs (commands in brief
   section 9). Keep `SUBPLEX_TMP_DIR` under this run's folder.
2. Check the per-benchmark ratios against the reference and the IMDB 200k q-error guard.
3. If the run was killed, use the gate verdicts in runs/<run>/state.json and train.log to decide
   whether the idea or the gate was wrong. Its subplex output exists if a checkpoint was written.
4. Write autoloop/experiments/{{experiment}}/analysis.md from the template, with all sections: Verdict,
   Step 1, Step 2, Step 3, Explanation.
   - The explanation must say whether the change acted through the cause you named in step 3. A score
     gain without the predicted framework change is "inconclusive", not "confirmed".
   - The leaderboard marks an experiment without a complete analysis.md as not explained.
5. Append ≤ 8 lines to autoloop/NOTEBOOK.md: the framework finding, what moved, and what that means
   for the next round. Commit on main.

Do not start new experiments in this phase.
