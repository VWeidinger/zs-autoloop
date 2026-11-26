PHASE: PLAN, round {{round}}

Read AGENTS.md and the files it lists. Then plan this round.

The goal of this project is NOT just a better score. It is to show that the subplex framework
finds what is wrong with the model and explains why a change helps. An experiment that improves
the score but is not grounded in, and explained by, subplex steps 1–3 is worth little.

Constraints for this round:
- at most {{max_runs}} runs in total (a run = one seed of one experiment), at most {{max_full}} of them `full`
- the next free experiment id is {{next_id}} (count up from there: e00X-short-name)
- budget left for you (LLM cost): {{budget_left}}

Current runs:
{{status}}

Supervisor review (overrides your own plans):
{{review}}

Do this:
1. DIAGNOSE first, with subplex (brief section 9). Start from the latest diagnosis: the reference
   models in {{subplex_root}}/_reference/, and the best runs so far in {{subplex_root}}/<run_id>/.
   - Step 1: per benchmark, where do picks fail? Use report/summary.json and the decisive-node pairs
     with their lost seconds and errors.
   - Step 2: what does the model do on those subplans? Use the tmp/step4/... curves per benchmark.
   - Step 3: why? Use counterfactual elasticities, ~/subplex/tmp_with_jit/step6/REPORT.md, training
     support, and feature scaling.
   Write the 3–5 most costly failure modes into autoloop/NOTEBOOK.md as "Round {{round}} diagnosis",
   with numbers.
2. Choose experiments that each target ONE of these failure modes. They should give the most
   information per GPU hour. Prefer proxies; promote to `full` only with evidence.
3. For each experiment:
   - write hypothesis.md from the template, with all sections, including a predicted framework change
     for each step;
   - implement the change on branch agent/<id> in a worktree and smoke-test it on CPU (≤ 10 min, tiny data);
   - write spec.yaml with `status: ready` and honest gates;
   - validate it with the harness.
   The harness rejects a spec whose hypothesis.md lacks the framework sections.
4. Commit the experiment folders on main, and append "Round {{round}} plan" to the notebook.

Stop when the specs are ready. Do not wait for training; the driver will call you again.
