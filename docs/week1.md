# Week 1: Data, Evaluation, and Baselines

## Goal

Produce a leakage-aware benchmark for implicit short-video recommendation before
introducing neural retrieval or multi-task ranking.

## Checklist

- [x] Define one configuration file and deterministic seed.
- [x] Download and verify KuaiRand-Pure.
- [x] Use the earlier standard-policy period only for training.
- [x] Split the later standard-policy period chronologically into validation and test.
- [x] Preserve randomized exposures for the later bias audit.
- [x] Implement Popularity, ItemCF, and BPR from a shared interface.
- [x] Evaluate Recall@K, HitRate@K, NDCG@K, and catalog coverage.
- [x] Restrict the main benchmark to novel, warm-start positives.
- [x] Add unit tests and a synthetic end-to-end demo.
- [x] Run the full benchmark and record the first empirical results.
- [x] Inspect results by user activity and item popularity.

## Experimental protocol

The positive signal defaults to `is_click`, which KuaiRand defines as a click in
the two-column UI and a valid play in the single-column UI. The signal can be
changed in `configs/week1.yaml` for a controlled sensitivity analysis.

The randomized log is not used for model selection in week one. It is held out
for the exposure-bias audit planned after the main retrieval and ranking models
are established.

The main top-k benchmark excludes positives already observed during training and
items absent from the training catalog. This makes the reported numbers a
novel-item, warm-start evaluation. Cold-start performance will be reported as a
separate experiment once content features are introduced.

## Questions to answer in the first results note

1. How much does personalization improve over popularity?
2. Does ItemCF outperform BPR at the selected interaction threshold?
3. How much catalog coverage is lost when optimizing only top-k accuracy?
4. Are validation and test results stable across users with different activity levels?
