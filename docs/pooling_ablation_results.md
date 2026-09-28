# Matched pooling ablation: measured results

Single-task click prediction with identical features, history rules, candidate sets and head dimensions. See [protocol](pooling_ablation.md). No calibration is applied.

Three seeds: 2026, 2027, 2028. Values are mean ± sample standard deviation across seeds, not confidence intervals.

Validation-selected pooling (mean NDCG@20): **attention**.

| Pooling | Split | NDCG@20 (%) | Click log loss | ROC-AUC |
| --- | --- | ---: | ---: | ---: |
| none | validation | 1.8934 ± 0.1283 | 0.6014 ± 0.0003 | 0.7341 ± 0.0005 |
| none | test | 1.8796 ± 0.1414 | 0.6167 ± 0.0011 | 0.7190 ± 0.0006 |
| mean | validation | 1.9675 ± 0.0316 | 0.5990 ± 0.0000 | 0.7362 ± 0.0003 |
| mean | test | 1.9840 ± 0.0707 | 0.6141 ± 0.0010 | 0.7213 ± 0.0003 |
| sum | validation | 1.9255 ± 0.1391 | 0.5978 ± 0.0008 | 0.7376 ± 0.0005 |
| sum | test | 1.9244 ± 0.1216 | 0.6122 ± 0.0011 | 0.7229 ± 0.0003 |
| attention | validation | 2.0030 ± 0.0235 | 0.5979 ± 0.0003 | 0.7372 ± 0.0004 |
| attention | test | 1.9893 ± 0.0549 | 0.6127 ± 0.0014 | 0.7228 ± 0.0005 |

## Individual runs

| Seed | Pooling | Best epoch | Epochs run | Validation NDCG@20 (%) | Test NDCG@20 (%) | Training seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2026 | none | 2 | 4 | 1.7454 | 1.7291 | 10.85 |
| 2026 | mean | 1 | 3 | 2.0032 | 2.0004 | 8.31 |
| 2026 | sum | 2 | 4 | 1.7660 | 1.8039 | 10.49 |
| 2026 | attention | 1 | 3 | 2.0128 | 2.0153 | 37.64 |
| 2027 | none | 1 | 3 | 1.9743 | 2.0096 | 7.90 |
| 2027 | mean | 1 | 3 | 1.9565 | 2.0451 | 7.74 |
| 2027 | sum | 1 | 3 | 2.0214 | 2.0471 | 8.19 |
| 2027 | attention | 1 | 3 | 2.0200 | 2.0263 | 40.95 |
| 2028 | none | 1 | 3 | 1.9606 | 1.8999 | 8.47 |
| 2028 | mean | 1 | 3 | 1.9429 | 1.9066 | 8.50 |
| 2028 | sum | 1 | 3 | 1.9891 | 1.9221 | 8.34 |
| 2028 | attention | 1 | 3 | 1.9762 | 1.9263 | 40.90 |

## Paired seed differences

NDCG@20 differences in percentage points, first method minus second. Each pair uses the same seed.

| Comparison | Split | Mean difference (pp) | Individual seed differences (pp) |
| --- | --- | ---: | --- |
| mean − none | validation | +0.0741 | +0.2577, -0.0177, -0.0176 |
| mean − none | test | +0.1045 | +0.2712, +0.0354, +0.0067 |
| sum − none | validation | +0.0321 | +0.0206, +0.0471, +0.0285 |
| sum − none | test | +0.0448 | +0.0748, +0.0374, +0.0222 |
| attention − mean | validation | +0.0354 | +0.0096, +0.0634, +0.0332 |
| attention − mean | test | +0.0053 | +0.0149, -0.0187, +0.0196 |
| attention − sum | validation | +0.0775 | +0.2468, -0.0014, -0.0129 |
| attention − sum | test | +0.0649 | +0.2114, -0.0207, +0.0042 |

## Practical reading

Attention minus mean pooling on test NDCG@20 is +0.0053 percentage points on average; the individual seed differences above show whether that gain is consistent.
Mean training time is 8.18s for mean pooling and 39.83s for attention (4.87 times as long). These are local training-plus-validation times, not serving latency benchmarks.
Use the validation-selected method as the experiment choice, and retain mean pooling as the simpler comparison. A small observed test difference does not demonstrate a substantial attention benefit.

## Verification and interpretation limits

- Every run passed candidate membership/original-column, finite-score, complete-rank, unchanged recall/hit-rate/coverage at 100, and exact checkpoint reload checks.
- Shared parameters have identical initial values within seed. Attention parameters remain allocated but unused in other modes; effective capacity is not identical.
- The history window and prediction head are matched. Early stopping is matched as a rule, not as an identical number of updates; checkpoints are selected by validation log loss.
- Three seeds measure some optimization variability, not population uncertainty. Do not infer significance or online lift.
- The test cohort has been examined in earlier project work. No new holdout was collected, and test scores did not select the pooling method.
- Saved per-run checkpoints, encoders, histories, candidate scores and metrics reside under `artifacts/pooling_ablation/`. Input/source hashes and complete configuration are in `results.json`.

Regenerate with `python -m kuaiflow.pooling_report`.
