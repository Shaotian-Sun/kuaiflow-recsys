# KuaiFlow

KuaiFlow is a reproducible, multi-stage short-video recommendation project built
on the KuaiRand dataset. The project starts with trustworthy implicit-feedback
baselines, then develops toward two-tower retrieval, multi-task ranking, and an
exposure-bias audit using randomized recommendations.

## Why this project

Many portfolio recommenders stop at a MovieLens notebook. KuaiFlow is organized
around the components and evaluation questions of a production recommender:

1. candidate retrieval;
2. multi-objective ranking;
3. diversity-aware reranking;
4. robustness under a different exposure policy.

Week one establishes the data and evaluation foundation with Popularity, ItemCF,
and Bayesian Personalized Ranking (BPR) baselines.

## Current scope

- Strict chronological train/validation/test split.
- Random-exposure log reserved for a later bias audit.
- Popularity and ItemCF baselines.
- BPR matrix factorization implemented from scratch in NumPy.
- Recall@K, HitRate@K, NDCG@K, and catalog coverage.
- Explicit novel-item, warm-start evaluation rather than mixing in impossible
  cold-start targets.
- Deterministic toy demo and unit tests.

See [the week-one plan](docs/week1.md) for the experimental checklist.

## Week 1 benchmark results

All models are evaluated at \(K=20\) on 5,000 users. The evaluation uses novel warm-start positives and a training catalog containing 7,538 videos. Users with no positive training interactions are retained and reported separately as the `zero_positive` group.

Higher values are better for every metric.

|   Split    | Model      | Recall@20 ↑ | HitRate@20 ↑ | NDCG@20 ↑ | Coverage@20 ↑ | Unique Items |
| :--------: | :--------- | ----------: | -----------: | --------: | ------------: | -----------: |
| Validation | Popularity |       7.95% |       20.18% |     4.45% |         0.56% |           42 |
| Validation | BPR        |      10.19% |       25.78% |     5.78% |        16.26% |        1,226 |
| Validation | **ItemCF** |  **11.58%** |   **28.26%** | **6.76%** |    **47.05%** |    **3,547** |
|    Test    | Popularity |       7.20% |       20.02% |     4.15% |         0.53% |           40 |
|    Test    | BPR        |       9.99% |       25.52% |     5.45% |        18.16% |        1,369 |
|    Test    | **ItemCF** |  **10.78%** |   **27.78%** | **6.14%** |    **45.52%** |    **3,431** |

See the [full Week 1 experimental report](docs/week1_results.md).

![KuaiFlow Week 1 Results](docs/kuaiflow_week1_portfolio.svg)

### Key findings

- ItemCF performs best across all ranking and coverage metrics.
- On the test set, ItemCF improves Recall@20 by 49.7% and HitRate@20
  by 38.8% relative to the popularity baseline.
- BPR consistently improves over popularity, but remains behind ItemCF under
  the initial untuned configuration.
- Popularity recommends only 40 distinct test items, while ItemCF reaches
  3,431, demonstrating the importance of personalization for catalog coverage.
- Validation and test results are similar, suggesting that the model comparison
  is reasonably stable across the two future time periods.

## Repository layout

```text
configs/                 Experiment configuration
data/raw/                Downloaded KuaiRand-Pure files (not tracked)
data/processed/          Chronological data splits (not tracked)
artifacts/               Benchmark results (not tracked)
src/kuaiflow/            Data, models, metrics, and CLI
tests/                    Unit and smoke tests
```

## Quick start

Create a Python 3.10+ environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Run the synthetic end-to-end demo and test suite:

```bash
kuaiflow demo
python -m unittest discover -s tests -v
```

Download and verify KuaiRand-Pure from the dataset's official Zenodo record:

```bash
kuaiflow download --config configs/week1.yaml
```

Prepare the strict chronological splits:

```bash
kuaiflow prepare --config configs/week1.yaml
```

Run every week-one baseline:

```bash
kuaiflow benchmark --config configs/week1.yaml
```

Results are written to `artifacts/week1_results.json` and
`artifacts/week1_results.csv`.

To iterate more quickly, run only selected models:

```bash
kuaiflow benchmark --config configs/week1.yaml --models popularity itemcf
```

## Evaluation notes

The raw standard-policy files contain a small timestamp overlap at their
boundary. Preprocessing conservatively removes 47 future rows whose timestamps
are not later than the final training timestamp, preserving a strictly
chronological evaluation.

The default positive signal is `is_click`. In KuaiRand this represents a click
for the two-column interface and a valid play for the single-column interface.
The later randomized-exposure log is intentionally excluded from training and
model selection.

The main benchmark removes training-seen positives from each user's ground truth
and restricts evaluation to items present in the training catalog. This protocol
measures novel-item recommendation under a warm-start setting; cold-start items
will be evaluated separately after content features are added.

Randomized exposures will first be used as a robustness and calibration audit.
The project will not claim unbiased off-policy evaluation without explicitly
accounting for the relevant logging propensities.

## Roadmap

- **Week 1 — Complete:** data pipeline, chronological evaluation, Popularity,
  ItemCF, and BPR.
- **Week 2 — Ongoing:** ID-only two-tower retrieval and exact top-k evaluation
  are complete; history features and FAISS are next.
- **Week 3:** Shared-Bottom and MMoE ranking for click, long-view, and like.
- **Week 4:** random-exposure bias audit and calibrated evaluation.
- **Week 5:** diversity-aware reranking, FAISS serving, and final report.

## Dataset

KuaiRand is released by Gao et al. under CC BY-SA 4.0. This repository does not
redistribute the dataset. Download instructions, field definitions, and citation
information are available from the
[official KuaiRand repository](https://github.com/chongminggao/KuaiRand).

## License

Code in this repository is released under the MIT License. The KuaiRand dataset
has its own CC BY-SA 4.0 license and attribution requirements.
