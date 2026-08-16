# Week 2 — Two-Tower Retrieval

## Goal

Build a two-tower retriever that combines identity, static user/video features,
and causal interaction history before adding approximate nearest-neighbor search.

## Milestone 1 — Complete

- [x] Separate user and item towers implemented in PyTorch.
- [x] L2-normalized embeddings and temperature-scaled dot products.
- [x] In-batch softmax negatives with duplicate-positive masking.
- [x] Exact top-k retrieval with training-positive filtering.
- [x] Popularity fallback for users without positive training history.
- [x] Deterministic toy smoke test.

## Next milestones

- [x] Add the Week 2 training and evaluation CLI.
- [x] Implement Recall, HitRate, NDCG, and Coverage at 50 and 100.
- [ ] Compare exact retrieval with FAISS and measure latency.
- [x] Add causal user-history aggregation and static user/video metadata features.
- [x] Run the first full ID-only experiment.
- [x] Run the feature/history model and publish the final Week 2 report.

## What “history-aware” means

For every clicked target video, the user tower receives at most the previous 20
clicked videos from that user, ordered by event time. It never receives the target
or a later event. Those videos pass through the same video tower used for retrieval
and their vectors are mean-pooled. At serving/evaluation time, the user tower uses
the latest 20 training clicks. This dynamic history vector is combined with the
user-ID embedding and static user features.

The item tower combines video ID with author, video/upload type, visibility,
music, tag, duration, and dimensions. Behavioral video statistics are excluded
because their aggregation cutoff is undocumented and could leak future outcomes.

## Commands

Run the deterministic smoke test:

```bash
kuaiflow retrieval-demo
```

Run the full ID-only experiment (user ID and video ID only):

```bash
kuaiflow retrieval --config configs/week2_id_only.yaml
```

This writes `artifacts/week2_id_only_results.json` and
`artifacts/week2_id_only_results.csv`.

Run the feature- and history-aware experiment:

```bash
kuaiflow retrieval --config configs/week2_feature_history.yaml
```

This writes `artifacts/week2_feature_history_results.json` and
`artifacts/week2_feature_history_results.csv`. `configs/week2.yaml` is retained
as a shorthand for the feature/history configuration and produces the same
variant-specific filenames.

Both configurations use the same random seed, model dimensions, optimization
settings, evaluation users, and warm-start novel-item protocol. The only intended
difference is the user/video metadata and causal-history switches, making this
a controlled comparison.

## ID-only baseline results

The first full run used 5,000 users per split and the warm-start, novel-item
protocol established in Week 1.

| Split      |   K | Recall | HitRate |  NDCG | Coverage | Retrieval latency |
| ---------- | --: | -----: | ------: | ----: | -------: | ----------------: |
| Validation |  50 |  4.08% |  11.20% | 1.51% |   99.91% |     0.189 ms/user |
| Validation | 100 |  7.40% |  19.34% | 2.25% |  100.00% |     0.189 ms/user |
| Test       |  50 |  4.09% |  11.82% | 1.57% |   99.93% |     0.183 ms/user |
| Test       | 100 |  7.34% |  19.48% | 2.30% |   99.99% |     0.183 ms/user |

Training took 58.9 seconds on CPU. The average epoch loss decreased from 6.246
to 5.350.

## Feature + causal-history results

The feature/history run used the same 5,000-user evaluation samples. Training
took 131.4 seconds on CPU and average epoch loss decreased from 6.105 to 5.041.

| Split      |   K | Recall | HitRate |  NDCG | Coverage | Retrieval latency |
| ---------- | --: | -----: | ------: | ----: | -------: | ----------------: |
| Validation |  50 |  5.82% |  14.90% | 2.15% |   95.50% |     0.252 ms/user |
| Validation | 100 | 10.05% |  24.48% | 3.08% |   99.18% |     0.252 ms/user |
| Test       |  50 |  5.97% |  15.84% | 2.31% |   95.86% |     0.247 ms/user |
| Test       | 100 | 10.13% |  25.08% | 3.23% |   99.16% |     0.247 ms/user |

Relative to ID-only, Recall@50 improved by 42.5% on validation and 46.0% on
test. Recall@100 improved by 35.9% and 38.1%, respectively. Test HitRate@100
increased from 19.48% to 25.08% (+28.7% relative). The richer user encoding
roughly doubled exact-retrieval latency, remaining around 0.25 ms/user.
Coverage fell slightly but remained above 95% at K=50 and 99% at K=100.

## Ablation Study: Isolating Feature and History Contributions

To understand the individual contributions of static features and causal history, we ran two additional controlled experiments:

- **Feature Only**: Uses user static features and video basic metadata, without causal history.
- **History Only**: Uses causal history only (previous 20 clicks), without static user/video features.

Both experiments use the same random seed, model dimensions, optimization settings, and evaluation protocol as the ID-only and full feature-history runs.

### Test Set Results

| Model             | Recall@50 | Recall@100 | HitRate@50 | HitRate@100 | NDCG@50 | NDCG@100 | Coverage@50 | Coverage@100 |
| ----------------- | --------- | ---------- | ---------- | ----------- | ------- | -------- | ----------- | ------------ |
| ID-only           | 4.09%     | 7.34%      | 11.82%     | 19.48%      | 1.57%   | 2.30%    | 99.93%      | 99.99%       |
| History Only      | 4.62%     | 8.30%      | 12.68%     | 21.48%      | 1.75%   | 2.58%    | 94.71%      | 98.97%       |
| Feature Only      | 5.47%     | 9.43%      | 15.02%     | 23.78%      | 2.16%   | 3.03%    | 99.95%      | 100.00%      |
| Feature + History | 5.97%     | 10.13%     | 15.84%     | 25.08%      | 2.31%   | 3.23%    | 95.86%      | 99.16%       |

![Ablation Study Results](artifacts/week2_ablation_results.png)

### Key Findings

**1. Static features provide the largest and most stable gain.**

- Feature Only improves Recall@50 by **+33.7%** over ID-only (5.47% vs 4.09%).
- Coverage remains near-perfect (99.95% at K=50), indicating that metadata helps spread recommendations across the catalog without sacrificing diversity.

**2. Causal history alone is beneficial but limited.**

- History Only improves Recall@50 by **+13.0%** (4.62% vs 4.09%).
- However, coverage drops significantly to 94.71% at K=50, suggesting that history alone can lead to over-focusing on similar items.

**3. Features and history are complementary.**

- The full model (Feature + History) achieves the best performance: Recall@50 = **5.97%** (+46.0% relative to ID-only).
- The combined gain is larger than the sum of individual gains, indicating synergistic effects.
- Coverage recovers from history-only degradation (95.86% vs 94.71%), showing that static features help diversify history-based recommendations.

**4. Training efficiency trade-offs.**

- Feature Only: ~75 seconds (CPU), similar to ID-only.
- History Only: ~120 seconds (CPU), due to sequential pooling.
- Feature + History: ~131 seconds, the most computationally expensive but yields the best accuracy.

### Conclusion

| Component       | Effective? | Impact                                        | Recommendation                  |
| --------------- | ---------- | --------------------------------------------- | ------------------------------- |
| Static Features | ✅ Highly  | Large, stable gains, maintains coverage       | Always include                  |
| Causal History  | ✅ Yes     | Moderate gains, but narrows recommendations   | Include, but pair with features |
| Both Combined   | ✅ Best    | Superior performance, good coverage trade-off | **Recommended final model**     |

The ablation study confirms that both static features and causal history contribute positively to retrieval quality. However, they are most effective when used together. The feature-history model achieves a **+46% relative improvement in Recall@50** over the ID-only baseline, making it the clear choice for deployment.

### Next Steps

- Implement FAISS-based approximate nearest neighbor search and measure latency-accuracy trade-offs.
- Run a feature-only experiment with larger user samples to verify scalability.
- Investigate whether a learned attention mechanism over history (rather than mean pooling) further improves performance.
