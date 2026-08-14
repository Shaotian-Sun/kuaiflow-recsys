# Week 2 — Two-Tower Retrieval

## Goal

Build an ID-only two-tower retrieval baseline before adding content features or
approximate nearest-neighbor search. This isolates the value of the retrieval
objective from the value of additional features.

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
- [ ] Add user-history aggregation and video metadata features.
- [x] Run the first full ID-only experiment.
- [ ] Add history features, rerun, and publish the final Week 2 report.

## Commands

Run the deterministic smoke test:

```bash
kuaiflow retrieval-demo
```

Run the full ID-only experiment:

```bash
kuaiflow retrieval --config configs/week2.yaml
```

Results are written to `artifacts/week2_results.json` and
`artifacts/week2_results.csv`.

## ID-only baseline results

The first full run used 5,000 users per split and the warm-start, novel-item
protocol established in Week 1.

| Split | K | Recall | HitRate | NDCG | Coverage | Retrieval latency |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 50 | 4.32% | 11.56% | 1.68% | 99.89% | 0.133 ms/user |
| Validation | 100 | 7.69% | 19.78% | 2.43% | 100.00% | 0.133 ms/user |
| Test | 50 | 4.36% | 12.38% | 1.72% | 99.93% | 0.131 ms/user |
| Test | 100 | 7.73% | 20.20% | 2.48% | 99.99% | 0.131 ms/user |

Training took 70.3 seconds on CPU. The average epoch loss decreased from 6.246
to 5.350.

### Interpretation

The ID-only model is a functioning retrieval baseline, but not yet a strong
recommender. Its near-total coverage shows that it spreads recommendations over
the catalog, while its accuracy indicates that a single learned user-ID vector
does not capture enough short-video interest structure. The next experiment
will aggregate each user's positive history in the user tower. Metadata and
FAISS should be added only after that accuracy improvement is measured.
