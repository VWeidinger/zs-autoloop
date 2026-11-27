# Lab notebook (append-only; newest at the bottom)

Format per entry: `## <date> <round/experiment>` then ≤ 8 lines: what, numbers, verdict, next.

## 2026-11-26 setup (human)
- Baselines pulled from wandb into autoloop/baselines/. Selection peaks early (ep 4-41), val q-error late (ep 80-131).
- e000 (proxy) and e001 (full) baseline specs written by hand: they produce the like-for-like references incl. AP/PO.

## 2026-11-26 Round 1 diagnosis
- AP: PPV50-JIT loses 74.17 s on 81 Index Scan vs Seq Scan picks (95/222 wrong), selected error -0.212 dex, 0% >10x card error; pp3 is flat ~756 ms (slope 0.001, 71% at 741 ms).
- JOB: PPV50-JIT loses 58.78+41.45 s on (Index) NL vs Hash; 71/108 wrong and 83.1% decisive-node >10x card error. INL slope is 0.147, Index Scan 0.001.
- JOB-complex: PPV50-JIT loses 175.42+51.94+47.37 s on loop alternatives; selected error -1.383 dex and 90.0% >10x card error (ratio 6.82).
- PO/PO-idx: PPV50-JIT loses 21.14 s on INL vs Merge/Hash and 52.41 s on Merge vs Hash; selected errors -0.169/-0.062 dex but only 0/3.9% >10x card error, hence model bias.
- Cause: scan-root table elasticity is 0.002 vs 0.519 true; <1 ms subplans are 7.1% but 69.8% of SSE (+4.31 dex). At T2-d4, 37.7% have only unlabelled twins carrying 73.7% SSE; labelled roots are 1.7%.

## Harness problem
- CPU smoke test failed before collection: `uv run --frozen pytest -q src/ldb_models/tests/test_zeroshot_subplan_loss.py` in `worktrees/e003-subplan-loss`.
- Error: DGL could not read `/home/vweidinger/.dgl/config.json` (`PermissionError: [Errno 13] Permission denied`). Per policy, no environment workaround was attempted; no experiment specs were marked ready.
