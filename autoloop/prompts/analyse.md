PHASE: ANALYSE experiment {{experiment}} (round {{round}})

Read AGENTS.md, autoloop/experiments/{{experiment}}/hypothesis.md, spec.yaml and result.json.

Runs:
{{status}}

Budget left for you (LLM cost): {{budget_left}}

The harness has already run the subplex framework on every run of this experiment. Its output is in
{{subplex_root}}/<run_id>/ (see "Using subplex" in the brief). Start from there:
- report/summary.json: per benchmark, the slowdown, the wrong picks, *why* they were wrong
  (selected plan under-estimated / optimal plan over-estimated / too close), and better/worse
  statement counts against the reference model.
- report/flips_<benchmark>.csv: which statements got better or worse than the reference, and by how much.
- report/compare_<benchmark>_worst<k>_step<n>.csv: node by node, the selected vs the optimal plan of
  the worst picks. Look for the operator, estimated cardinality or prediction that decides the pick.
- tmp/step4/...: where on the IMDB subplans the error sits (error curve, slices).

Do this:
1. Decide whether the hypothesis is confirmed, refuted, or inconclusive.
   - Compare the per-benchmark ratios with the reference and with the parent or baseline proxy.
   - Check the IMDB 200k q-error guard.
   - Check the subplex flips and diagnoses: did the change fix the failure mode it targeted?
   - If the run was killed, read the gate verdicts in runs/<run>/state.json and train.log, and decide
     whether the idea or the gate was wrong.
2. If you need more, run further subplex analyses yourself, on CPU only, on this run's annotations.
   The brief lists the commands. Set `SUBPLEX_TMP_DIR` to a folder under
   {{subplex_root}}/<run_id>/tmp, and never write anywhere else.
3. Write autoloop/experiments/{{experiment}}/analysis.md:
   - the verdict and the evidence, with numbers;
   - the failure mode that remains, in subplex terms (which plans and which nodes);
   - what this changes about the next steps.
4. Append ≤ 8 lines to autoloop/NOTEBOOK.md and commit on main.

Do not start new experiments in this phase.
