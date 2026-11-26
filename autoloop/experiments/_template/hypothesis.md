# e00X-short-name

Every section is required (the harness rejects the spec otherwise). Cite subplex files and numbers.

## Step 1 - where selection fails (subplex step 1)
Benchmark(s), wrong picks, diagnosis counts, and the decisive node you target, with its lost seconds
and errors. From report/summary.json, decisive_<b>.csv and compare_*.csv of the reference or parent run.

## Step 2 - what the model does there (subplex step 2 = step4 code)
The behaviour on those subplans: the error-curve slice and its direction, from tmp/step4/<model>/subplex_<benchmark>/
curve.txt and curve_findings.csv, or the complex-workload curve.

## Step 3 - why: which input or training cause (subplex step 3)
The cause you believe in and its evidence: a counterfactual elasticity (tmp/step6/counterfactual/...),
the step 6 REPORT, training support (step 7), feature scaling, ... State which brief finding this is.

## Change
Code and config (branch agent/e00X-...), and why it acts on the step-3 cause.

## Predicted framework change
| step | quantity (file) | now | predicted | refuted if |
|---|---|---|---|---|
| 1 | e.g. AP lost s on decisive "Index Scan vs Seq Scan" (summary.json) | 74 | < 40 | > 60 |
| 2 | e.g. under-estimation slice of Index Scan (curve.txt) | -0.21 dex | > -0.1 dex | unchanged |
| 3 | e.g. elasticity of est to est_card on Index Scan (counterfactual) | 0.0x | > 0.3 | < 0.1 |
| - | per-benchmark ratios / guard | ... | ... | ... |

## Smoke test
The command and its result.
