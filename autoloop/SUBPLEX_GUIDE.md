# subplex guide for the agent (read this instead of the subplex repository)

subplex is at `/home/vweidinger/subplex` and is read-only for you. Its README has about 1,400 lines and
DECISIONS.md about 600; do not read them whole. This guide says what exists, where results already
are, and which entry points you can call. Open a README or DECISIONS section only when you need the
detail behind a number.

## The framework: three questions, code folders in brackets

| step | question | code | already computed for the reference models |
|---|---|---|---|
| **1 where** | Which picks fail per benchmark; at which node is the pick decided wrongly? | `step1/benchmark_selection.py` | `/mnt/labstore/vweidinger/agent_zs-autoloop/subplex/_reference/<ref>/report/` |
| **2 what** | How does the model behave on those subplans: error curve over runtime, slices, shapes? | `step4/` (curve, emm, recalibration, diagnose, shapes, watchlist) | `_reference/<ref>/tmp/step4/`, and `~/subplex/tmp_with_jit/step4/<model>/` (200k + complex) |
| **3 why: inputs** | Which inputs drive it: elasticities, input gap, surrogate attribution? | `step6/counterfactual.py`, `step6/surrogate_shap.py`, `step6/model_attribution.py`, `step5/input_gap.py` | `~/subplex/tmp_with_jit/step6/REPORT.md` (verified), `.../step6/counterfactual/pm-zeroshot-pg_opt/`, `.../step6/jit_models/` |
| **3 why: training data** | Did training contain such plans? Labelled or only inside queries? Do the training runtimes agree with IMDB? Is it noise? | `step7/` (training_set, training_index, training_evidence, noise_ladder, raw_identity) | `~/subplex/tmp_with_jit/step7/pm-zeroshot-pg_opt-RMP-epochs200-limitpf10050-PPV50-JIT/<dataset>/` |

The harness runs steps 1–3 (counterfactual) on every run you train; see AGENT_BRIEF section 9.
Step 7 is not run per run, because building the training index takes about 40 GPU minutes. Use the
reference model's index for your questions about the data. It describes the *training files*, which
change only when you change the workloads.

## Step 3 on the training data (step 7): from an IMDB failure back to the training set

This is the tool for **tuning the training data**: which kinds of subplans have no labelled
example, where the training databases disagree with IMDB, and which workloads would add support.

```bash
cd /home/vweidinger/subplex
export SUBPLEX_TMP_DIR=/home/vweidinger/subplex/tmp_with_jit          # read existing results (read-only)
export SUBPLEX_SUBPLANS_DIR=/mnt/labstore/vweidinger/subplex/learned_db/annotate_subplans
export SUBPLEX_UNPARSED_MAP_DIR=/mnt/labstore/vweidinger/subplex/learned_db/subplex/unparsed_map_jit
uv run --frozen --offline python - <<'EOF'
from subplex.step7.training_evidence import evidence_for, find_twins, load_test_frame
from subplex.step7.training_index import load_training_index
M = "pm-zeroshot-pg_opt-RMP-epochs200-limitpf10050-PPV50-JIT"
index = load_training_index(M)                                   # 1.49 M training subplans, ~5 s
test = load_test_frame(M, "subplex_complex_workload_200k_s1")    # IMDB complex subplans, ~30 s
ids = test[test["op_name"].eq("Index Only Scan")].index[:200]    # e.g. the subplans of a failure mode
print(evidence_for(list(ids), test, index))                      # support per matching level
EOF
```

Tested inside your sandbox. Example result: for 200 IMDB Index Only Scans there is no labelled
training twin at any level. Unlabelled twins (inside training queries) run 47 ms against 42 ms in IMDB,
while the model predicts 137 ms. So the model was never asked for this runtime, but the data
contains it. That is a supervision/data gap, not missing information. The fix candidates are subplan
supervision or scan-level workloads.

- Matching levels: T0 exact model input; T1 e2/e1 input within 0.01 / 0.1 scaled units; T2 d4/d3/d2 operators + estimate bin + table size + index/join-key roles; T3 operators + estimate bin only.
- `evidence_for` columns: `with_twin`, `with_labelled_twin`, `twins_median`, `databases_median`, `test_act_ms`, `model_est_ms`, `train_any_ms`, `train_label_ms`.
- `find_twins(test.loc[subplan_id], "T2 d3", index)` lists the training subplans behind one test subplan, with their database and file. These are the workloads to add or reweight.
- Existing reports: `training_evidence.txt` (support, generalisation, database offsets, transfer, verdicts) and `ladder.txt` (noise vs. missing information) in the step 7 folder above.
- To select the subplans of a failure mode, use the step-1/step-2 outputs. The decisive nodes' operators and tables are in `decisive_<b>.csv`; the slices are in `tmp/step4/.../curve.csv`.

## Where to read the detail (only when needed)

| topic | file:section |
|---|---|
| step 7 method, levels, verdicts | `~/subplex/README.md` "# step 7" (around line 1200); `DECISIONS.md` "2026-11-16" |
| benchmark selection, decisive picks | `DECISIONS.md` "2026-11-09" |
| JIT models and what JIT does | `DECISIONS.md` "2026-11-05"; `tmp_with_jit/step6/jit_models/` |
| step 6 causes (elasticities, surrogate, attribution) | `tmp_with_jit/step6/REPORT.md` (verified numbers; prefer it to agent summaries) |
| step 2 behaviour shapes and the step 2 → 3 link | `README.md` "# step 4"; `tmp_with_jit/link/REPORT.md` |
| database statistics as model input | `DECISIONS.md` "2026-11-02" |
| batch effect, best-checkpoint traps | `DECISIONS.md` "2026-10-19" |

## Rules

- Write only below `/mnt/labstore/vweidinger/agent_zs-autoloop/`. For your own subplex calls, set `SUBPLEX_TMP_DIR` there. Reading `~/subplex/tmp_with_jit` is fine, writing to it is not (the sandbox blocks it anyway).
- GPU stages (annotation, counterfactual, training index) are run by the harness, not by you.

## Improving subplex

If subplex is missing something you needed (a question it cannot answer, a slow or awkward entry
point, a wrong result), write it into `autoloop/SUBPLEX_FEEDBACK.md`: what you tried, what was
missing, and a concrete proposal. Do not change subplex itself. The supervisor reviews these; they are
part of the research.
