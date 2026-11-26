# e00X-short-name: analysis

Every section is required. An experiment without the subplex explanation does not count as a result.

## Verdict
confirmed / refuted / inconclusive, in one sentence, and the per-benchmark won/lost against the reference.

## Step 1 - where selection changed
Before vs after for the predicted step-1 quantities: wrong picks, diagnoses, decisive-node pairs and lost
seconds (report/summary.json, decisive_*.csv), and the statements that flipped (flips_*.csv).

## Step 2 - how the model's behaviour changed
Before vs after for the step-4 slice or curve you targeted (tmp/step4/...).

## Step 3 - whether the cause moved
Before vs after for the step-3 quantity (counterfactual elasticities, ...). Did the change act on the
cause you named, or did the score move for another reason?

## Explanation
Why the change helped or did not, told through steps 1 to 3, plus side effects on other benchmarks
and what remains wrong.

## Framework value
What subplex showed that the score alone would not have: the failure mode, the cause, a side effect, or
a reason the change worked differently than expected. Also note where subplex fell short: what you would
have needed and could not get. This section is collected for the thesis.
