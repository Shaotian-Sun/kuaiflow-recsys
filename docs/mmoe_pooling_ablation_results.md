# MMoE pooling and ranking strategies: measured results

Seeds 2026, 2027, 2028. Every ranking uses the same Week 2 top-100 candidates. See [protocol and interpretation limits](mmoe_pooling_ablation.md). Utility weights are fixed, and no calibration is applied.

Values are mean ± sample standard deviation across seeds, not confidence intervals. NDCG is shown as a percentage.

## Validation: click ranking

| Strategy | Click NDCG@20 (%) | Click Recall@20 (%) | Coverage@20 (%) |
| --- | ---: | ---: | ---: |
| single_mean_click | 1.9675 ± 0.0316 | 3.5945 ± 0.0781 | 36.9594 ± 0.3449 |
| mean_mmoe_click | 1.8009 ± 0.0529 | 3.5526 ± 0.0827 | 38.3966 ± 0.3501 |
| mean_mmoe_composite | 1.7410 ± 0.0636 | 3.5808 ± 0.1501 | 40.3378 ± 0.1288 |
| attention_mmoe_click | 1.8159 ± 0.0427 | 3.5869 ± 0.1074 | 39.2014 ± 0.5179 |
| attention_mmoe_composite | 1.7228 ± 0.0340 | 3.5281 ± 0.0187 | 40.8906 ± 0.3424 |

## Validation: other outcomes under those same rankings

| Strategy | Long-view NDCG@20 (%) | Like NDCG@20 (%) | Profile-entry NDCG@20 (%) | Hate NDCG@20 (%) ↓ |
| --- | ---: | ---: | ---: | ---: |
| single_mean_click | 1.9643 ± 0.0233 | 0.8486 ± 0.1142 | 1.6237 ± 0.1168 | 0.0000 ± 0.0000 |
| mean_mmoe_click | 1.8674 ± 0.0569 | 0.5609 ± 0.0928 | 1.5763 ± 0.1008 | 1.2330 ± 0.0294 |
| mean_mmoe_composite | 1.8093 ± 0.0467 | 0.6537 ± 0.1342 | 1.8472 ± 0.1899 | 1.4170 ± 0.0794 |
| attention_mmoe_click | 1.8752 ± 0.0419 | 0.5141 ± 0.0667 | 1.4192 ± 0.1155 | 0.8164 ± 0.7074 |
| attention_mmoe_composite | 1.7671 ± 0.0401 | 0.6195 ± 0.0565 | 1.6299 ± 0.0510 | 1.2569 ± 0.1198 |

## Test: click ranking

| Strategy | Click NDCG@20 (%) | Click Recall@20 (%) | Coverage@20 (%) |
| --- | ---: | ---: | ---: |
| single_mean_click | 1.9840 ± 0.0707 | 3.9509 ± 0.1677 | 37.2601 ± 0.2057 |
| mean_mmoe_click | 1.9092 ± 0.0505 | 3.8814 ± 0.1160 | 38.3081 ± 0.3648 |
| mean_mmoe_composite | 1.7988 ± 0.0646 | 3.7646 ± 0.1971 | 40.6341 ± 0.0870 |
| attention_mmoe_click | 1.9321 ± 0.0602 | 3.8416 ± 0.1824 | 39.2146 ± 0.4526 |
| attention_mmoe_composite | 1.8271 ± 0.0468 | 3.8055 ± 0.1187 | 41.2974 ± 0.4481 |

## Test: other outcomes under those same rankings

| Strategy | Long-view NDCG@20 (%) | Like NDCG@20 (%) | Profile-entry NDCG@20 (%) | Hate NDCG@20 (%) ↓ |
| --- | ---: | ---: | ---: | ---: |
| single_mean_click | 1.8288 ± 0.0743 | 1.0408 ± 0.0608 | 1.1719 ± 0.0834 | 1.2265 ± 0.1881 |
| mean_mmoe_click | 1.8593 ± 0.0571 | 0.7634 ± 0.0464 | 1.0460 ± 0.0785 | 0.0000 ± 0.0000 |
| mean_mmoe_composite | 1.6935 ± 0.0562 | 0.9578 ± 0.1298 | 1.5666 ± 0.1594 | 0.0000 ± 0.0000 |
| attention_mmoe_click | 1.8785 ± 0.0255 | 0.7236 ± 0.1126 | 1.0558 ± 0.0715 | 0.0000 ± 0.0000 |
| attention_mmoe_composite | 1.7297 ± 0.0397 | 0.8632 ± 0.1342 | 1.6518 ± 0.1159 | 0.0000 ± 0.0000 |

## Paired click NDCG differences

Percentage points; first strategy minus second, paired by seed.

| Comparison | Split | Mean difference (pp) | Per-seed differences (pp) |
| --- | --- | ---: | --- |
| mean_mmoe_click − single_mean_click | validation | -0.1666 | -0.2157, -0.2005, -0.0837 |
| mean_mmoe_click − single_mean_click | test | -0.0748 | -0.0737, -0.1928, +0.0421 |
| mean_mmoe_composite − mean_mmoe_click | validation | -0.0599 | -0.1151, -0.0033, -0.0612 |
| mean_mmoe_composite − mean_mmoe_click | test | -0.1105 | -0.1500, -0.1042, -0.0772 |
| mean_mmoe_click − attention_mmoe_click | validation | -0.0150 | -0.0517, -0.0105, +0.0173 |
| mean_mmoe_click − attention_mmoe_click | test | -0.0229 | -0.0575, -0.0139, +0.0028 |
| mean_mmoe_composite − attention_mmoe_composite | validation | +0.0183 | -0.0418, +0.0589, +0.0378 |
| mean_mmoe_composite − attention_mmoe_composite | test | -0.0283 | -0.0855, -0.0259, +0.0263 |

## MMoE pointwise predictions and training cost

| Pooling | Split | Click log loss | Click ROC-AUC | Watch-time MAE (seconds) | Completion MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| mean | validation | 0.6050 ± 0.0006 | 0.7288 ± 0.0016 | 22.8175 ± 0.1382 | 0.2591 ± 0.0027 |
| mean | test | 0.6190 ± 0.0012 | 0.7148 ± 0.0019 | 23.8913 ± 0.1886 | 0.2656 ± 0.0029 |
| attention | validation | 0.6030 ± 0.0005 | 0.7304 ± 0.0006 | 22.3884 ± 0.1194 | 0.2569 ± 0.0028 |
| attention | test | 0.6164 ± 0.0015 | 0.7169 ± 0.0009 | 23.4935 ± 0.1302 | 0.2636 ± 0.0028 |

Mean pooling — average training-plus-validation time: 55.06 seconds. Best epochs: 2, 2, 2.


Attention pooling — average training-plus-validation time: 99.89 seconds. Best epochs: 2, 2, 2.


## Outcome cohort sizes

| Target | Validation users | Test users |
| --- | ---: | ---: |
| click | 5000 | 5000 |
| like | 412 | 424 |
| follow | 44 | 40 |
| comment | 70 | 84 |
| forward | 28 | 33 |
| long_view | 4384 | 4389 |
| profile_enter | 595 | 623 |
| hate | 20 | 25 |

## Limits and reproducibility

- No utility-weight tuning or test-based selection was performed. Rankings for different outcomes are not combined into a post-hoc winner.
- Single-task versus MMoE differs in architecture, loss, batch size and checkpoint criterion. Mean versus attention MMoE changes only pooling.
- Rare outcomes have small eligible cohorts; observed zeros do not establish safety or absence of negative feedback. Outcomes have different eligible users.
- Reported ranking metrics use logged feedback and do not identify online policy value. Candidate reranking cannot recover retrieval misses.
- All six runs passed original-candidate preservation, finite-score, complete-rank, recall@100 invariance, exact reload and independent click metric checks.
- Input/source/configuration hashes and per-run metrics are in `artifacts/mmoe_pooling_ablation/results.json`; all strategy/outcome/seed rows are in `strategy_metrics.csv`.

Regenerate: `python -m kuaiflow.mmoe_pooling_report`.
