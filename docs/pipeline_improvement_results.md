# Retrieval-aware pipeline improvement

The search selected the existing **ItemCF-only ordering**, with no two-tower contribution and no neural reranking. This improves the recently assembled pipeline by removing harmful stages; it is not a new model outperforming ItemCF.

The search used 2,500 validation users. The choice was saved before evaluating the other 2,500 validation users or test. Earlier project work had already examined these periods; they are not newly collected untouched data.

## Selected policy

- Candidate budget: 100.
- ItemCF weight: 1.0; neural weight: 0.0.
- ItemCF trained on training positives only, with 100 item neighbors. Train-clicked items are excluded; cold users receive its training-popularity fallback.
- Selected serving path skips loading and running the two-tower and neural models.

## validation_confirmation

| Strategy | NDCG@20 (%) | Recall@20 (%) | HitRate@20 (%) | Coverage@20 (%) |
| --- | ---: | ---: | ---: | ---: |
| mixed_retrieval | 4.386 | 8.521 | 22.000 | 67.312 |
| mixed_neural | 3.363 | 6.625 | 19.560 | 17.100 |
| itemcf_only | 6.856 | 11.785 | 29.280 | 34.160 |
| tower_only | 1.173 | 2.208 | 6.560 | 73.773 |
| selected | 6.856 | 11.785 | 29.280 | 34.160 |

## validation_all

| Strategy | NDCG@20 (%) | Recall@20 (%) | HitRate@20 (%) | Coverage@20 (%) |
| --- | ---: | ---: | ---: | ---: |
| mixed_retrieval | 4.358 | 8.488 | 21.860 | 78.841 |
| mixed_neural | 3.506 | 7.075 | 19.900 | 20.960 |
| itemcf_only | 6.760 | 11.585 | 28.260 | 47.015 |
| tower_only | 1.181 | 2.205 | 6.700 | 82.873 |
| selected | 6.760 | 11.585 | 28.260 | 47.015 |

## test

| Strategy | NDCG@20 (%) | Recall@20 (%) | HitRate@20 (%) | Coverage@20 (%) |
| --- | ---: | ---: | ---: | ---: |
| mixed_retrieval | 3.885 | 7.690 | 20.720 | 79.464 |
| mixed_neural | 3.253 | 6.594 | 18.840 | 20.509 |
| itemcf_only | 6.141 | 10.781 | 27.780 | 45.476 |
| tower_only | 1.279 | 2.510 | 7.520 | 84.770 |
| selected | 6.141 | 10.781 | 27.780 | 45.476 |

## Paired uncertainty

Selected minus baseline, in NDCG@20 percentage points. Paired user bootstrap, 1,000 replicates. Intervals condition on the fitted models and selected policy; they exclude training/selection uncertainty.

| Cohort | Baseline | Difference (pp) | 95% percentile interval (pp) |
| --- | --- | ---: | --- |
| validation_confirmation | mixed_retrieval | +2.470 | [+2.146, +2.761] |
| validation_confirmation | itemcf_only | +0.000 | [+0.000, +0.000] |
| test | mixed_retrieval | +2.256 | [+2.071, +2.452] |
| test | itemcf_only | +0.000 | [+0.000, +0.000] |

## Search and limitations

- Fixed grid: interleaving plus weighted reciprocal-rank fusion (ItemCF weights 0, 0.25, 0.5, 0.75, 1; rank constant 60), crossed with neural rank-percentile weights 0, 0.05, 0.1, 0.25, 0.5, 1. Total 36 policies.
- Source fusion first chooses 100 candidates. A blend of source-rank percentile and neural-rank percentile then determines the final ordering. Both zero-neural and pure-source endpoints are included.
- Single frozen neural model (seed 2026), corrected causal two-tower and training-only ItemCF. No outcomes are model inputs and no unexposed candidates are added as training negatives.
- Aggregate catalog coverage declines versus the mixed retrieval order. This is a measured relevance/coverage tradeoff, not a claim of universal superiority or within-user diversity.
- These are offline logged-feedback metrics. No online lift, unbiased policy value, or superiority of ItemCF on every recommendation task is established.

## Serving and reproduction

Reloaded optimized serving output exactly matches saved top 20 for 256 test users. Median amortized time: 0.210 ms/user (three 256-user batches after warmup; models loaded, no disk I/O; not online p95).

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m kuaiflow.pipeline_improvement
.venv/bin/python -m kuaiflow.pipeline_report
.venv/bin/python -m kuaiflow.recommend --users 1 2 --k 20
```

The experiment refuses existing output directories. Change `output_dir` in a copied configuration to repeat. `RecommendationPipeline.load("artifacts/pipeline_improvement/selected_pipeline.json").recommend(user_ids, k=20)` is the reusable inference entry point. Numeric user IDs above are examples.

Artifacts include all tuning metrics, fixed user partitions, source/checkpoint hashes, selected policy, cached candidate logits, ranked outputs and serving verification. The original evaluated pipeline source is preserved alongside the optimized serving source hash. Earlier experiments remain unchanged.
