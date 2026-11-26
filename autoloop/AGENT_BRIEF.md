# Agent brief: improve Zero-Shot's plan selection *through* the subplex framework

You improve the Zero-Shot learned cost model in this repository (a private copy of `ldb_models`) so
that it **selects better query plans** on IMDB.

**The research question is not "can a model get better" but "does the subplex framework show what
is wrong and why a change helps".** Every experiment therefore runs the same cycle:
1. **diagnose** with subplex steps 1–3;
2. **target** one diagnosed failure mode;
3. **change** the model or training;
4. **re-run** subplex;
5. **explain** the result through the same subplex quantities.

A better score that the framework cannot explain is not a result (section 9).

You work in rounds. Training is slow (up to ~9 h for a full run), so think before you spend a run.

## 1. Objective: per benchmark

Each of the seven IMDB benchmarks is scored **on its own**:
`ratio = runtime of the selected plans / runtime of the optimal plans` (1.0 = always optimal).
The benchmarks are:
- join order: `join_order` (JOB-light), `job_w_optimal_w_idxs` (JOB), `job_complex_w_optimal_w_idxs` (JOB-complex);
- access path: `access_path_selection`, `access_path_selection_with_aggregate`;
- physical operator: `physical_operator`, `physical_operator_w_idx`.

Every benchmark gets a verdict against the reference model (section 3): **better / same / worse**, with ±2% as "same". The leaderboard shows all seven ratios with arrows and a won/lost count.

The single **score** is the geometric mean of the seven ratios. That way, a 10% gain counts the same on every benchmark, and JOB-complex's 6.9× cannot drown the rest.

**Report the per-benchmark picture, not just the score.** A change that wins access path and loses JOB is a finding, not a wash. Prefer changes that win benchmarks without losing others.

The harness computes this from the run's `*_test_pred.csv` (`autoloop/harness/evaluate.py`). subplex computes the same numbers, plus *why* picks go wrong (section 8).

**Guard:** the median Q-error on IMDB `workload_200k_s1` may be at most **1.4×** the reference
model's. Plain prediction accuracy is *not* the goal. "How Good are Learned Cost Models, Really?"
(Heinrich et al., SIGMOD'25) shows that better Q-error does not give better plans. Spend accuracy
where it buys ranking.

**What "better" must survive:** run-to-run noise. Only one seed of every baseline exists. Treat
differences under about 5% in score as noise until a second seed agrees.

## 2. Hard rules (the harness enforces most of them; do not try to work around any)

1. **IMDB is test only.** No IMDB file may be used for training, for feature statistics, or for choosing a checkpoint or hyperparameters. The benchmark numbers are *reported*, not optimised on directly. The harness rejects specs with IMDB training files.
2. **Checkpoint selection must not look at IMDB.** Per-epoch IMDB curves may be used in *gates*, because gates only abort runs. Choosing an epoch by its JOB/AP/PO numbers would be test-set tuning.
3. **Data you may read:** `/mnt/labstore/SIGs/ML/learned_db/` (runs, datasets, feature_statistics, ...) and `/mnt/labstore/vweidinger/subplex/learned_db/` (includes `runs/parsed_plans50`, which is parsed_plans3 plus per-node JIT fields). The JIT annotation was added after execution and is accepted as-is.
4. **The only place you write data:** `/mnt/labstore/vweidinger/agent_zs-autoloop/`. Build your own datasets, statistics and caches there. Never write anywhere else on `/mnt`.
5. **You never start, stop or ssh to anything.** You write `autoloop/experiments/<id>/spec.yaml` with `status: ready`; the driver validates and submits it. Check a spec offline with `python -m autoloop.harness validate <spec>`.
6. **No secrets.** Do not look for keys or tokens and do not print environment variables.
7. **Ignore** the subplex `proposal/` and `slides/` folders and everything about "models trained without JIT".
8. **Never push** to `upstream` (the lab's repository); its push URL is disabled. Commit locally only.

## 3. Reference numbers

**Per benchmark (ratio = selected / optimal).** These come from subplex, on the full candidate sets.
parsed_plans3 and parsed_plans50 use the same candidates here, so all rows compare directly. File:
`autoloop/baselines/reference_eval.json`.

| model | AP | AP-agg | PO | PO-idx | JOB-light | JOB | JOB-cx | IMDB 200k q50 |
|---|---|---|---|---|---|---|---|---|
| **Zero-Shot pg_opt (pp3), the reference** | 1.51 | 1.47 | 1.19 | 1.37 | 1.08 | 2.35 | 6.93 | 1.251 |
| Zero-Shot pg_opt PPV50-JIT (pp50) | 1.27 | 1.25 | 1.14 | 1.35 | 1.05 | 1.93 | 6.82 | 1.242 |
| SUMO LoopsIdxCond PPV50-JIT (pp50) | **1.08** | **1.08** | **1.03** | **1.16** | 1.20 | **1.82** | **5.03** | – |
| PostgreSQL cost (choose min pg_cost) | 1.09 | 1.07 | 1.03 | 1.15 | 1.01 | 1.96 | 10.4 | – |

Zero-Shot loses to plain PostgreSQL costs on access path and physical operator. SUMO closes that gap
and also helps on JOB-complex, but loses on JOB-light. Wrong picks of the reference, for example on
JOB: 82 of 108 statements.

**Training curves (wandb, per epoch, in `autoloop/baselines/*.json`, usable in gates):**

| baseline file | data | s/epoch | epochs | JOB-l+JOB+cx selected/optimal at its best epoch | best val q50 at |
|---|---|---|---|---|---|
| `zs-pg_opt_pp3` | parsed_plans3 | 218 | 150 | 240 / 148 at ep 41 | ep 128 |
| `zs-hi_pp3` | parsed_plans3 | 44 | 115 | 241 at ep 99 | ep 93 |
| `zs-pg_opt-PPV50_pp50` | parsed_plans50 | 368 | 102 | 1693 / 1177 at ep 9 | ep 80 |
| `zs-pg_opt-PPV50-JIT_pp50` | parsed_plans50 | 375 | 153 | 1523 / 1177 at ep 24 | ep 131 |
| `sumo-PPV50-JIT_pp50` | parsed_plans50 | 297 | 25 | 1629 / 1177 at ep 4 | ep 24 |

The per-epoch evaluation inside ldb_models drops some candidates (JOB-light has only 4 statements in
the pp3 runs). Use these curves for *gates* and trends, and the table above for comparisons.

**The single most important observation:** plan selection is best **early** (epochs 4–41), while
validation Q-error keeps improving until epochs 80–131. The final model, chosen by validation
Q-error, is therefore systematically past its best ranking point.

The first runs of this loop, `e000` (proxy) and `e001` (full), are hand-written baselines. They give
like-for-like references through this exact pipeline.

## 4. What we know about why Zero-Shot mis-ranks (subplex findings, measured)

- **Training target support.** Every training root is a Simple Aggregate of ≥ 75–83 ms. Leaf and small subplans are never supervised. On its *own training databases*, the model has q50 1.05 on trained roots but 42 / 383 / 5,810 on interior subplans of 1–10 ms / 0.1–1 ms / < 0.1 ms. Plans differ in their small parts, so this hurts ranking.
- **Cardinality scaling.** `est_card` is scaled linearly (RobustScaler, no log). Index-scan cards from 1 to 100,000 map to −0.05..0.05, so small cardinalities are nearly invisible. Zero-Shot's estimate barely responds to `est_card` on index scans (−0.08 dex/SD vs +0.67 for the truth). `use_log_input_card` exists in the config but was never used in our checkpoints.
- **Loops folded in.** The parser's `est_card` of an inner nested-loop child is per-loop rows × `est_loops`. Zero-Shot has no separate loop feature; `PostgresEstWithNumLoops` exists but is unused.
- **Elasticity collapse at aggregate-less scan roots.** The elasticity to table size is 0.002, against 0.52 true. Wrapping the same scan in an aggregate lifts the prediction from 757 to 3,599 ms (actual 3,166). This one elasticity accounts for about 20% of ZS's squared log error. Subplans under 1 ms are 7% of rows but carry 70% of the error.
- **Over-reaction to plan size.** On Hash Join, the table count moves the estimate +0.14 dex/SD while the truth moves −0.28. `est_children_card` is nearly ignored (0.03 vs 0.21). Sum-over-children aggregation is a plausible cause.
- **Output-column statistics are misused.** Multiplying n_distinct by 10 moves 41–46% of predictions, although the true effect is about 0.
- **Most of the residual is explainable from inputs the model already gets.** A GBM surrogate on the model's own inputs reaches R² 0.79. So the problem is how inputs are used and what is supervised, more than missing features.
- **JIT features are weak.** On QO benchmarks, zeroing them changes selection by ≤ 3%. JIT cost is not localised to the compiling node. Tiny scans pay ~440 ms of JIT in jit+inline+optimize queries and all models predict them as a constant.
- **Floor:** QLoss models show a ~750 ms prediction floor in annotations, and its cause is not settled. Selection is unaffected by monotone rescaling, but floors flatten small plans.
- **Reproducibility traps:** set `reproducible_message_passing` (RMP), because otherwise the same plan gets different predictions depending on batch composition. Evaluate the *best* checkpoint, not the last (ldb_models' test does this already).

## 5. Lessons from the paper (Heinrich et al. 2025) and from SUMO

- **Train on suboptimal plans.** Pre-optimised workloads bias models towards near-optimal plans, and timeouts drop slow ones, so the model never learns when a plan is bad. *Fine-tuning on diverse plans cut runtime by up to 45%* in the paper.
- **Keep the optimizer's knowledge.** Optimizer cost and cardinality features help. DACE's AP and PO strength comes from PG costs. Zero-Shot had the best rank correlation on join order but lost on runtime because of tail errors in both directions.
- **Tails matter, not medians.** A few huge over- or underestimates cause the bad picks.
- **SUMO** (`~/SUMO`, read-only, not to be rebuilt) is far better at selection than Zero-Shot. Borrow its ideas, not its whole design:
  - per-operator decoding with softplus, with the plan prediction = sum of operator predictions;
  - an operator-level loss weighted ×10 (`subplan_weight`);
  - `PostgresEstLoopsIdxCond` features (loops plus index-condition info);
  - hidden_dim 256.
  Zero-Shot already has `subplan_estimation` / `subplans_reg_loss` flags in its config, which is a start.

## 6. Training data you can use (non-IMDB, multi-plan = ranking signal)

Under `/mnt/labstore/SIGs/ML/learned_db/runs/parsed_plans3/<db>_scaled<k>/` (19 training databases
plus imdb):

| workload | what it is | DBs |
|---|---|---|
| `workload_200k_s1` | default (PG-optimal plans only) | 20 |
| `workload_200k_s1_full_hints_idx` / `_lt30` / `_gt30` | the same SQL under hinted alternative plans | 20 |
| `scan_only_s1_access_path_hints_stratified` | scans under forced access paths | 20 |
| `complex_workload_200k_s1`, `full_qo_complex_workload_200k_s1` | complex queries, plus candidate plans | 20 |
| `complex_workload_many_joins_hints` | many-join hinted plans | baseball, credit, financial, geneea, tournament |
| `access_path_selection(_with_aggregate)` | AP benchmark | baseball, tpc_h (+ imdb = test) |
| `physical_operator(_w_idx)` | PO benchmark | tpc_h (+ imdb = test) |

- `parsed_plans_stratified3`, `augmented_plans3` and `json*` variants exist too.
- `/mnt/labstore/vweidinger/subplex/learned_db/runs/parsed_plans50/` has the same with JIT fields. These need `parsed_plans_version=50` / `use_jit_features`; this branch already contains the JIT featurization.
- In a spec, list workload *names* (`workload_names`) plus a `plans_dir`. The harness expands them over all non-IMDB databases.

**A ranking validation set** is the clean way to choose epochs and compare ideas without touching
IMDB. Hold out non-IMDB multi-plan data (for example tpc_h AP/PO plus baseball AP, or one database's
hinted workloads) and score selection on it every epoch. ldb_models does not do this yet. Building
it is a good early experiment, because it unlocks honest checkpoint selection.

## 7. Toolbox already in this repository

- `classes/configs/zeroshot.py` has these options:
  - `loss_class_name` (QLoss, QLogSqLoss, RankLoss), `ranking_loss_name`, `regression_loss_name`, `group_size`, `batch_group_size`, `mask_single_plan_queries`;
  - `subplan_estimation`, `subplans_reg_loss`, `subplan_monotonicity_loss`;
  - `use_log_input_card`, `featurization` (PostgresEstSystemCardDetail, PostgresEstWithNumLoops, ...);
  - `include_timeout_queries` + `learn_from_timeout`;
  - `online_hard_example_mining`, `hidden_dim`, `skip_top_aggregation`, `reproducible_message_passing`.
- The `model_type` string encodes these options: `pm-zeroshot-<flag>-<key><value>-...`. Use the long names, for example `pm-zeroshot-include_pg_opt_workloads-reproducible_message_passing-epochs40`. `-epochs<N>` is required, because the tier cap is checked on it.
- Training is `src/ldb_models/main.py --mode train`. Losses are in `training/losses/`, the metrics in `training/training/metrics.py` (NDCG, Spearman, Q-error), and the per-epoch QO evaluation in `training/training/test.py`.

## 8. How to run an experiment

1. **Diagnose, then target.** Choose ONE failure mode from the subplex diagnosis (section 9). Write `autoloop/experiments/<id>/hypothesis.md` from `autoloop/experiments/_template/hypothesis.md`, with the three framework steps, the change, and the predicted framework change.
2. Implement the change in a git worktree: `git worktree add worktrees/<id> -b agent/<id> main`. Commit it there. Keep changes small and behind config flags so the baseline is unaffected.
3. Smoke-test on CPU in the worktree. Use a tiny `--cap_training_samples`, 1 epoch, ≤ 10 min, and write into `/mnt/labstore/vweidinger/agent_zs-autoloop/scratch/`.
4. Write `autoloop/experiments/<id>/spec.yaml` from `autoloop/experiments/_template/spec.yaml`, with `branch: agent/<id>` and `status: ready`. Run `python -m autoloop.harness validate autoloop/experiments/<id>/spec.yaml`.
5. Add **gates**, your own early-abort predictions. Example: "by epoch 15, `job_ratio_mean` ≤ 1.1 × the reference at epoch 15". Missing a gate kills the run, so be honest but not reckless.
   - Gate metrics: `train_loss`, `val_q50`, `val_spearman`, `imdb_q50`, `job_light_ratio`, `job_ratio`, `job_complex_ratio`, `job_ratio_mean`, `job_selected_sum`, or any raw wandb key.
6. **Tiers.**
   - `proxy`: ≤ 40 epochs, 1 seed, about 2–4 h. Use it to rank ideas.
   - `full`: ≤ 200 epochs, ≤ 2 seeds, about 9 h. Only promote a proxy that beat the baseline proxy, and set `parent: <proxy id>`.
7. Commit the experiment folder on `main` ("e00X: hypothesis + spec").

After a run, the harness writes `autoloop/experiments/<id>/result.json` and runs subplex steps 1–3 on it. The driver then asks you
to analyse it. Write `analysis.md` from the template (section 9), append ≤ 8 lines to `autoloop/NOTEBOOK.md`, and commit.

## 9. The subplex framework: three steps for every experiment

subplex (`/home/vweidinger/subplex`) produced every finding in section 4. It analyses a model in
three steps. The code folder names do not match the step numbers; the mapping is in brackets.

| step | question | subplex code | computed for every run (harness) | files in `<run>/` |
|---|---|---|---|---|
| **1 – where** | Which picks fail, per benchmark, and at which node of the plan is the pick decided wrongly? | `step1.benchmark_selection` | yes | `report/summary.json`, `decisive_<b>.csv`, `critical_<b>.csv`, `flips_<b>.csv`, `compare_<b>_worst<k>_*.csv` |
| **2 – what** | How does the model behave on those subplans: which slices are under- or over-estimated, and by how much, along the runtime curve? | `step4` (curve, emm, recalibration, diagnose) | yes: curve for each benchmark's subplans and IMDB complex | `tmp/step4/<run>/subplex_<benchmark>/`, `tmp/step4/<run>/subplex_complex_workload_200k_s1/` |
| **3 – why** | Which input or training cause drives it: elasticities to est_card / table size / operator, missing training support, feature scaling? | `step6.counterfactual` (also step5 input gap, step6 surrogate, step7 training evidence) | yes: counterfactual on IMDB complex | `tmp/step6/counterfactual/<run>/subplex_complex_workload_200k_s1/` |

`<run>` = `/mnt/labstore/vweidinger/agent_zs-autoloop/subplex/<run_id>/`. The same analysis exists for
the reference models in `.../subplex/_reference/{ref-zs-pg_opt, ref-zs-ppv50-jit, ref-sumo-ppv50-jit}/`
(steps 1–2). Step 3 of the reference Zero-Shot is in
`~/subplex/tmp_with_jit/step6/REPORT.md` and `~/subplex/tmp_with_jit/step6/counterfactual/pm-zeroshot-pg_opt/`.

**Step 1 in detail.**
- `summary.json`, per benchmark: wrong picks (and the reference's), diagnosis counts (`selected plan under-estimated`, `optimal plan over-estimated`, both, `too close to rank`, plus the reference's counts), better/worse/same statements vs the reference, and `decisive`.
- **The decisive node** of a wrong pick is the topmost node where the two plans choose a different operator and the model ranks the selected subtree cheaper although it is slower. `decisive` gives the top operator pairs, the seconds lost per pair, the median time error of both sides (log10 est/act, so < 0 means under-estimated), and the share of picks with a > 10× cardinality error at that node.
- **The reference diagnosis (Zero-Shot PPV50-JIT), for orientation:**
  - Access path: lost mainly on "Index Scan vs Seq Scan" (74 s). The Index Scan is under-estimated by 0.21 dex, with no cardinality error, so this is a model bias.
  - JOB and JOB-complex: "(Index) Nested Loop vs Hash Join". The selected side is under-estimated by up to 1.4 dex, with cardinality errors > 10× in 83–90% of the picks, so this is driven by cardinality estimates.
  - Physical operator with indexes: "Merge Join vs Hash Join".

**Commands you may run yourself (CPU only, on your runs' outputs):**

```bash
cd /home/vweidinger/subplex
R=/mnt/labstore/vweidinger/agent_zs-autoloop/subplex/<run_id>
export SUBPLEX_TMP_DIR=$R/tmp          # all subplex output stays in your area
# step 2, deeper: exceptional slices, calibration, shape diagnosis (dataset = subplex_<benchmark> or subplex_complex_workload_200k_s1)
uv run --frozen python -m subplex.step4.emm      --model <run_id> --dataset subplex_job_w_optimal_w_idxs --subplans-dir $R/annotate_subplans
uv run --frozen python -m subplex.step4.recalibration --run <run_id>=$R/annotate_subplans --dataset subplex_complex_workload_200k_s1
uv run --frozen python -m subplex.step4.diagnose --model <run_id> --dataset subplex_complex_workload_200k_s1 --subplans-dir $R/annotate_subplans
# step 1 again, against another reference (no GPU)
uv run --frozen python -m subplex.external_model --name <run_id> --architecture <model_type> \
    --model-dir /mnt/labstore/vweidinger/agent_zs-autoloop/models/<run_id> --out-root $R --stages report \
    --reference pm-zs_sumo-hi-PostgresEstLoopsIdxCond-hidden_dim256-subpW10-RMP-epochs20-limitpf10050-PPV50-JIT
```

- **Python access:** `subplex.step1.benchmark_selection.load_benchmark(model, benchmark)` returns a `BenchmarkRun`. Its `.steps` carry `.selected`, `.optimal`, `.diagnosis`, `.selected_rank` and `.regret_ms`, and `compare_plans(sel, opt)` gives the node table.
- **Registering a run first:** do it via `subplex.external_model` (`--stages report`).
- **Reference models available for comparison:** `pm-zeroshot-pg_opt`, `pm-zeroshot-pg_opt-RMP-epochs200-limitpf10050-PPV50-JIT`, `pm-zs_sumo-hi-PostgresEstLoopsIdxCond-hidden_dim256-subpW10-RMP-epochs20-limitpf10050-PPV50-JIT`.

**Rules for subplex:**
- Never run its GPU stages (`annotate-*`, `step3`) yourself; the harness does that.
- Never write into `/mnt/labstore/vweidinger/subplex/` or `~/subplex/tmp*`.
- Do not run subplex modules directly on the registry models, because that writes caches next to their files. Use `_reference/` and `--reference` instead.

**How the framework enters every experiment (enforced):**
- `hypothesis.md` must contain the sections `## Step 1`, `## Step 2`, `## Step 3`, `## Change` and `## Predicted framework change`. The harness rejects the spec otherwise.
  - Each step section names its subplex evidence: file and numbers.
  - The prediction table says which step-1, step-2 and step-3 quantities the change will move, and by how much.
- `analysis.md` must contain `## Verdict`, `## Step 1`, `## Step 2`, `## Step 3` and `## Explanation`.
  - It holds before/after values for exactly those quantities.
  - The leaderboard marks experiments without it as "not explained".
- **Confirmed** means the score moved *and* the predicted framework quantities moved. Score only = inconclusive; framework only = a real finding, worth reporting even when the score does not move.

## 10. Idea backlog (ordered by expected value; challenge it)

1. **Multi-plan training data**: add `workload_200k_s1_full_hints_idx` / `_lt30` plus `scan_only_s1_access_path_hints_stratified`. This is the paper's main lesson, and the cheapest test.
2. **Ranking loss** over candidates of the same SQL (`RankLoss`, or a pairwise/listwise term plus a regression term), with `mask_single_plan_queries`. Needs item 1.
3. **Log-scaled cardinalities plus a loop feature** (`use_log_input_card`, `PostgresEstWithNumLoops`). Targets the index-scan blindness.
4. **SUMO-style operator supervision** (`subplan_estimation`, weighted operator loss, softplus sum). Targets training-target support.
5. **Learn from timeouts** (`include_timeout_queries`, `learn_from_timeout`), so that bad plans are not dropped.
6. **Ranking-validation checkpoint selection** (section 6). Makes every later comparison honest.
7. Tail control: an asymmetric or quantile loss against both-sided tail errors.
8. Only after the above: hidden_dim, lr, batch size and epochs.

Combine ideas only after each has been shown to help alone.
