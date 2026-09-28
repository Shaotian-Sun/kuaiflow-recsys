# Matched DIN history-pooling ablation

This experiment isolates the history aggregation operator in the single-task DIN ranker. It compares `none` (zero interest), masked `mean`, masked `sum`, and target-conditioned `attention`. All modes keep seven categorical fields, three numeric fields, 16-dimensional embeddings, the 176-dimensional concatenation, and the same 128/64 ReLU prediction head. User and target-video embeddings enter the head in every mode.

The history is the latest 30 positive training clicks. Training excludes current/tied timestamps; evaluation freezes training history. IDs 0 and 1 are masked. Mean pooling divides by the valid count, clamped to one; empty histories produce zeros in every mode. Sum pooling helps distinguish learned attention from the normalization/history-length effect of mean pooling. None keeps a zero 16-dimensional history slot rather than shrinking the head.

## Fixed experimental protocol

- Seeds: 2026, 2027, 2028, chosen before running.
- Base settings: `configs/week3_din.yaml`; same chronological logged-impression splits, static features and fixed Week 2 top-100 candidates.
- AdamW, learning rate 0.001, weight decay 1e-6, batch size 2048, at most five epochs and patience two. Each run selects its checkpoint by validation click log loss.
- Shared embeddings/head parameters start identically within each seed. The attention module remains allocated in every mode to preserve initialization and checkpoint structure; only attention mode executes it. Non-attention modes have fewer participating parameters. Allocated parameter count is not equal effective capacity; the zero history slot also leaves its head connections unused.
- Prediction-head dropout draws and batch orders are matched within seed. History operators have no random draws.
- The comparison selects a pooling method by mean validation candidate NDCG@20. Test results are descriptive and never select the method, seed, epoch or hyperparameters. The test set was already examined in prior project work and is not a newly untouched holdout.
- Means and sample standard deviations describe variability across three seeds; they are not confidence intervals or evidence of statistical significance.
- Unexposed candidates are scored, not used as negative training examples. These are offline logged-feedback metrics, not online or causal policy effects.

## Reproduce

From the repository root:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m kuaiflow.pooling_ablation
```

The runner caches input data in memory, records configuration/runtime/input and source hashes, verifies original candidate columns and complete ranks, checks unchanged recall/hit rate/coverage at 100, and verifies exact predictions after reloading each model, encoder, and history index. It saves every run in `artifacts/pooling_ablation/seed_<seed>/<mode>/`, separate from Week 3/4 outputs. A completed run writes `results.json` and `summary.csv`; incremental completed-run results are in `progress.json`. Existing run directories cause an error rather than silent overwrite; choose a new output directory for a repeat.

Run tests with `OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v`.

## Results

See [measured three-seed results](pooling_ablation_results.md). Regenerate the results page with `python -m kuaiflow.pooling_report` after the experiment completes.
