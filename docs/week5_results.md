# Week 5: diversity reranking and local serving

The five-week offline industry-style prototype is implemented: chronological data and baselines, two-tower/FAISS retrieval, DeepFM/DIN/MMoE ranking, exposure calibration, diversity reranking, and a working local recommendation API. ItemCF remains the selected relevance baseline. The complete two-tower + ItemCF → DIN attention → MMR path is retained as a separate runnable profile. **Diversity reranking is opt-in.** The selected ItemCF setting missed the 98% relevance-retention target on confirmation and test. After that check, the serving default was conservatively left at zero reranking strength; the experimental choice and results remain unchanged.

## Frozen protocol

- Same training cutoff, warm catalog, novel-positive ground truth, and 5,000 validation/test users as the preceding experiments.
- Frozen seed-2026 DIN attention checkpoint; corrected strict-history two-tower; training-only ItemCF. No retraining or test-based hyperparameter selection.
- Reuse the preceding 2,500 tuning / 2,500 confirmation validation-user partition. These users and periods have been examined before; confirmation is not a fresh independent holdout.
- Each profile starts with 100 unique unseen candidates. Reranking chooses 20 only from those candidates, so it cannot improve candidate Recall@100.
- The hybrid profile interleaves two-tower/FAISS and ItemCF candidates, then orders them with DIN attention click logits. The ItemCF profile retains its source ordering.
- MMR greedily maximizes `(1-strength) * relevance - strength * max_similarity_to_selected`. Relevance is the base-rank percentile; similarity is half same-author indicator plus half tag-set Jaccard similarity. Ties retain base order.
- Missing tags/authors contribute no matching similarity. Reported tag diversity excludes unknown-tag pairs and separately reports metadata coverage.
- Search strengths: 0, 0.05, 0.1, 0.2, 0.4. Maximize tuning-user mean tag diversity subject to NDCG@20 at least 98% of that profile's no-reranking control. The 2% tolerance is an illustrative offline product constraint, not a proven business requirement.
- Both selected strengths are written to the serving manifest before confirmation/test metrics are computed. A zero-strength control is always eligible.

## Validation selection

| Profile | Strength | NDCG@20 (%) | Tag diversity | Unique-author fraction |
| --- | ---: | ---: | ---: | ---: |
| itemcf | 0.0 | 6.664 | 0.8060 | 0.9951 |
| itemcf | 0.05 | 6.758 | 0.8155 | 0.9956 |
| itemcf | 0.1 | 6.826 | 0.8278 | 0.9964 |
| itemcf | 0.2 | 6.763 | 0.8547 | 0.9983 |
| itemcf **selected** | 0.4 | 6.597 | 0.9131 | 0.9994 |
| hybrid_din | 0.0 | 3.633 | 0.8009 | 0.9938 |
| hybrid_din | 0.05 | 3.622 | 0.8085 | 0.9948 |
| hybrid_din | 0.1 | 3.649 | 0.8190 | 0.9960 |
| hybrid_din | 0.2 | 3.658 | 0.8447 | 0.9980 |
| hybrid_din **selected** | 0.4 | 3.644 | 0.9061 | 0.9993 |

## validation_confirmation

| Profile | NDCG@20 (%) | Recall@20 (%) | Coverage@20 (%) | Tag diversity | Unique-author fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| itemcf_baseline | 6.856 | 11.785 | 34.160 | 0.8044 | 0.9956 |
| itemcf_reranked | 6.618 | 11.394 | 35.460 | 0.9128 | 0.9994 |
| hybrid_din_baseline | 3.553 | 6.770 | 18.002 | 0.8004 | 0.9939 |
| hybrid_din_reranked | 3.547 | 6.924 | 22.221 | 0.9036 | 0.9993 |

Tag diversity is mean within-list `1 - Jaccard(tags)` over known-tag pairs; author diversity is distinct known authors divided by known-author slots. These are metadata proxies, not user satisfaction. Catalog coverage is a separate population-level metric.

- itemcf: reranked minus baseline NDCG@20 **-0.238 pp**, paired 95% interval [-0.439, -0.036] pp.
  Observed NDCG retention: 96.53%; the 98% tuning constraint does not hold on this cohort.
- hybrid_din: reranked minus baseline NDCG@20 **-0.006 pp**, paired 95% interval [-0.176, +0.154] pp.
  Observed NDCG retention: 99.83%; the 98% tuning constraint holds on this cohort.
- Minimum known-tag slot fraction across these lists: 99.14%.

## test

| Profile | NDCG@20 (%) | Recall@20 (%) | Coverage@20 (%) | Tag diversity | Unique-author fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| itemcf_baseline | 6.141 | 10.781 | 45.476 | 0.8058 | 0.9955 |
| itemcf_reranked | 5.804 | 10.266 | 47.639 | 0.9132 | 0.9995 |
| hybrid_din_baseline | 3.386 | 6.797 | 21.611 | 0.7991 | 0.9942 |
| hybrid_din_reranked | 3.365 | 6.851 | 26.784 | 0.9026 | 0.9994 |

Tag diversity is mean within-list `1 - Jaccard(tags)` over known-tag pairs; author diversity is distinct known authors divided by known-author slots. These are metadata proxies, not user satisfaction. Catalog coverage is a separate population-level metric.

- itemcf: reranked minus baseline NDCG@20 **-0.337 pp**, paired 95% interval [-0.475, -0.204] pp.
  Observed NDCG retention: 94.51%; the 98% tuning constraint does not hold on this cohort.
- hybrid_din: reranked minus baseline NDCG@20 **-0.021 pp**, paired 95% interval [-0.144, +0.105] pp.
  Observed NDCG retention: 99.38%; the 98% tuning constraint holds on this cohort.
- Minimum known-tag slot fraction across these lists: 99.10%.

Intervals use 1,000 paired user-bootstrap replicates and condition on fixed models and chosen strengths. They do not include training or policy-selection uncertainty.

## Serving verification

The service binds only to `127.0.0.1`. Both profiles verify checkpoint/metadata hashes at startup. The hybrid profile builds its IVF-100/10 index from the corrected saved embeddings; ItemCF skips the two-tower and DIN. The `--diversity` flag enables the frozen tuning-selected setting; default requests preserve the baseline order. Inference consumes user IDs and frozen training state, never evaluation outcomes.

| Profile | Startup (s) | Single request median (ms) | Single request p95 (ms) | Batch amortized (ms/user) | Exact replay users |
| --- | ---: | ---: | ---: | ---: | ---: |
| itemcf_baseline | 0.206 | 2.747 | 3.026 | 0.769 | 128 |
| itemcf_diversity | 0.203 | 2.755 | 2.926 | 0.789 | 128 |
| hybrid_din_baseline | 1.057 | 16.077 | 16.517 | 1.795 | 128 |
| hybrid_din_diversity | 1.176 | 16.282 | 16.573 | 1.803 | 128 |

Latency includes loopback HTTP, JSON serialization, and the in-memory recommendation path. Single-request values use 32 sequential one-user requests after five warmups; batching uses three 64-user requests. These are local CPU observations, not a production p95 or load-test SLA. Startup is excluded from request timings.

Replay checks verify exact saved top-20 order, top-5 prefix consistency, duplicate user handling, unknown-user fallback, catalog membership, and invalid-input rejection. Temporary verification servers are stopped after testing.

## Five-week status

1. **Week 1 complete:** chronological preprocessing and Popularity/ItemCF/BPR controls. [Report](week1_results.md).
2. **Week 2 complete, with correction:** learned retrieval, FAISS comparisons, and strict tied-timestamp history exclusion. Earlier saved retrieval metrics remain historical. [Corrected retrieval report](retrieval_budget_results.md).
3. **Week 3 complete:** DeepFM, DIN, DeepFM+MMoE, DIN+MMoE, matched pooling and fixed-utility comparisons. Neural stages have not beaten ItemCF on the current novel-item click benchmark. [Ranking report](week3_results.md), [pooling](pooling_ablation_results.md), [MMoE](mmoe_pooling_ablation_results.md).
4. **Week 4 complete:** frozen standard/random exposure calibration diagnostics. Calibrators remain exposure-specific; they are not silently applied to the serving policy. A positive-slope single-task calibration cannot change item ordering. [Report](week4_results.md).
5. **Week 5 complete:** measured relevance/diversity selection, local API with ItemCF and complete neural profiles, replay checks, latency measurements, and this final report.

## Scope and remaining work

This completes the five-week **offline prototype**, not a deployed industrial recommender. No live feature ingestion, impression logging, online learning, authenticated public endpoint, concurrent load test, A/B test, or causal policy lift is claimed. Serving uses training-cutoff histories and requires local data/checkpoints. Month-aggregated video engagement statistics are excluded. MMoE remains a measured offline alternative rather than the default API scorer.

Future model work should investigate training/evaluation mismatch and retrieval-aware ranking against the ItemCF control. It is not a prerequisite for finishing the pipeline or a reason to conceal the baseline result. The original serving source captured during evaluation is preserved as `serving_at_evaluation.py`; HTTP replay records the updated serving-source hash after adding the explicit opt-in flag.

## Reproduction

See [Week 5 operations guide](week5.md) for build order, API examples, artifacts, and restart instructions. The evaluator refuses to overwrite an existing output directory. Change `output_dir` in a copied YAML to repeat; verification and report generation accept `--directory` for the new result directory.
