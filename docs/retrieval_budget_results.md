# Retrieval budget, oracle ceiling and tuning results

**Corrected causal retrieval histories:** equal-time training clicks are excluded. These are new results; historical Week 2 scores used the earlier tie-handling implementation.

Seed 2026. Same fixed evaluation users, training catalog and frozen mean-pooling ranker throughout. No calibration or ranking retraining. See [protocol](retrieval_budget.md).

Validation-selected budget: **100**. Validation-selected retrieval: **ivf_itemcf_interleave**.

## Validation: fixed-model candidate budget

| Retrieval | K | Candidate recall (%) | Candidate hit rate (%) | Oracle NDCG@20 (%) | Achieved NDCG@20 (%) | Achieved/oracle (%) | Pipeline ms/user |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ivf_baseline | 100 | 10.143 | 24.920 | 12.943 | 2.136 | 16.50 | 0.494 |
| ivf_baseline | 200 | 17.624 | 38.620 | 21.846 | 2.126 | 9.73 | 0.824 |
| ivf_baseline | 500 | 33.797 | 60.320 | 39.776 | 2.014 | 5.06 | 1.655 |
| exact_baseline | 100 | 9.919 | 24.200 | 12.621 | 2.106 | 16.69 | 0.557 |

## Test: fixed-model candidate budget

| Retrieval | K | Candidate recall (%) | Candidate hit rate (%) | Oracle NDCG@20 (%) | Achieved NDCG@20 (%) | Achieved/oracle (%) | Pipeline ms/user |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ivf_baseline | 100 | 10.133 | 25.360 | 12.995 | 1.924 | 14.80 | 0.493 |
| ivf_baseline | 200 | 17.290 | 38.660 | 21.565 | 1.864 | 8.64 | 0.763 |
| ivf_baseline | 500 | 32.176 | 59.200 | 38.218 | 1.806 | 4.72 | 1.640 |
| exact_baseline | 100 | 9.834 | 24.620 | 12.619 | 1.977 | 15.66 | 0.562 |

## Validation retrieval tuning at the selected budget

| Retrieval | Candidate recall (%) | Oracle NDCG@20 (%) | Final NDCG@20 (%) | Pipeline ms/user |
| --- | ---: | ---: | ---: | ---: |
| ivf_baseline | 10.143 | 12.943 | 2.136 | 0.494 |
| temperature_0.04 | 9.975 | 12.785 | 2.055 | 0.493 |
| temperature_0.1 | 10.257 | 13.097 | 2.156 | 0.506 |
| ivf_itemcf_interleave | 24.373 | 29.723 | 3.506 | 0.643 |

## Budget interpretation

Validation K=100 → 500 changes candidate recall from 10.143% to 33.797%, oracle NDCG@20 from 12.943% to 39.776%, and achieved NDCG@20 from 2.136% to 2.014%.
The fixed ranker does not turn the larger candidate pool into better top-20 ranking here. Improve candidate discrimination or training/evaluation alignment before increasing the serving budget solely on recall.
Separate saved integrity checks verify that both larger IVF budgets preserve the original top-100 entries and ranks, with identical logits for every shared candidate on both splits.

## Selected configuration: test result

ivf_itemcf_interleave, K=100: candidate recall 23.453%, final click NDCG@20 3.253%, oracle NDCG@20 28.817%, pipeline 0.637 ms/user.

Only the validation-selected new retrieval option is evaluated on test. The fixed baseline budget curve and exact reference are diagnostic comparisons, not a test-set selection rule.

## Interpretation limits

- Oracle scores use held-out labels and represent only the candidate-set ceiling. They are not attainable deployment results or training targets.
- A gap between achieved and oracle NDCG indicates ranking headroom; missing candidate positives indicate retrieval headroom. Larger recall alone need not improve the final top 20.
- Training-timestamp tie correction changes the retrieval baseline; do not attribute differences from historical Week 2 solely to candidate budget or temperature.
- The ranker was trained on logged impressions and is frozen. Larger or mixed candidate pools may change its scoring distribution; these results reflect that fixed ranker.
- Latency is median amortized batch time on 256 users with three repeats, not online-request p95. Artifacts include the individual repeats.
- ItemCF-only with the same ranker was not tested: the hybrid improvement does not isolate complementary contributions from its two sources.
- Single-seed validation search and a previously examined test cohort do not establish statistical significance, online lift, or an optimal hyperparameter setting.

## Training cost

| Retrieval fit | Seconds | Final training loss |
| --- | ---: | ---: |
| baseline | 145.94 | 5.03675 |
| temperature_0.04 | 146.97 | 5.04249 |
| temperature_0.1 | 147.28 | 5.04759 |

Full candidate lists, logits, checkpoint caches, frozen embeddings, provenance and metrics are saved under `artifacts/retrieval_budget_causal/`. Regenerate with `python -m kuaiflow.retrieval_budget_report`.
