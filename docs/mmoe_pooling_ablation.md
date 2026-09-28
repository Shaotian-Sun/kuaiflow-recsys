# Matched pooling inside MMoE and ranking-strategy comparison

This experiment compares mean pooling and target-conditioned attention inside the same ten-task DINMMoE architecture, for seeds 2026, 2027 and 2028. It evaluates five fixed strategies:

1. Single-task mean pooling, sorted by click probability (reuses the verified previous ablation).
2. Mean-pooling MMoE, sorted by click probability.
3. Mean-pooling MMoE, sorted by the unchanged composite utility.
4. Attention MMoE, sorted by click probability.
5. Attention MMoE, sorted by the unchanged composite utility.

Every strategy is evaluated against the same per-target ground truth and candidate users. The click ordering is also evaluated against like, long-view and other labels; it is not replaced by the corresponding task head when evaluating a different outcome. Hate metrics measure concentration of recorded negative feedback, so lower is better. Sparse outcome metrics are descriptive; they are not online policy-value estimates.

## Controlled settings

Both MMoE variants use identical user/video metadata, causal latest-30 training-click histories, embeddings, four experts, ten gates/towers, task losses, loss normalization, batch size 4096, five-epoch budget and early-stopping patience two. Only the history pooling operator changes. Shared parameters start identically within each seed; an unused attention module remains allocated for mean pooling to preserve initialization. Effective capacity differs. Mean pooling excludes history IDs 0 and 1 and returns zeros for empty histories.

Each MMoE checkpoint is selected by validation normalized ten-task loss. The single-task reference previously selected by validation click loss, uses batch size 2048 and a single MLP head. Thus single-task versus MMoE is a system comparison, not a pure loss-function ablation. Mean versus attention inside MMoE is the matched pooling ablation. Click-only versus composite uses the exact same model and predictions, changing only the scoring rule.

Utility weights remain fixed at +1 for positive signals and -1 for hate. Watch-time and completion transformations are unchanged. No calibration, utility tuning, new negative sampling or model selection using test metrics is introduced. The test cohort was inspected in earlier project work; it is not a new untouched holdout. Three-seed sample standard deviations are not confidence intervals or statistical significance claims. There is no single cross-outcome winner unless an objective is specified in advance.

## Reproduction and verification

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m kuaiflow.mmoe_pooling_ablation
.venv/bin/python -m kuaiflow.mmoe_pooling_report
```

Settings are in `configs/mmoe_pooling_ablation.yaml`. Previous single-task input/source hashes, features and seeds are checked before reuse. The runner saves model/encoder/history/duration-curve artifacts and complete candidate scores in a separate `artifacts/mmoe_pooling_ablation` directory. It refuses an existing output directory rather than overwriting earlier results. `progress.json` records completed runs and `results.json` records the final experiment.

Each run verifies unchanged candidate rows and retrieval fields, complete ranks, finite scores, unchanged recall/hit rate/coverage at 100, and exact predictions after checkpoint reload. Independently computed click-only/composite click metrics must match the existing pipeline's metrics.

See [measured results](mmoe_pooling_ablation_results.md). Full binary/continuous prediction metrics and per-run candidate metrics are retained in the JSON artifact. No existing Week 3/4 or single-task ablation artifacts are overwritten.
