# Week 3 — DIN and DIN + MMoE

The Week 3 model family now includes DeepFM, DIN, DeepFM + MMoE, and DIN +
MMoE. See the [measured comparison](week3_results.md) for results. All four
train on logged impressions and score the same frozen Week 2 candidates.

## Architecture

DIN shares the target video's embedding table with its history. For target
embedding `q` and a history embedding `h`, a local activation MLP consumes
`[q, h, q-h, q*h]`. A [64, 32] network with PReLU activations produces one
unnormalized scalar weight per history item. Padding, missing, and unknown
history IDs are masked; an empty history produces exactly the zero vector.
The weighted sum is concatenated with the static field embeddings.

Standalone DIN applies a [128, 64] MLP and a click logit. DIN + MMoE sends
that same representation to four shared [128, 64] experts and one softmax
gate per task. Each task's mixture feeds a 32-unit tower and its own bias.
The ten targets, train-only loss normalization, checkpoint-selection criterion,
duration adjustment, and neutral utility weights are identical to the existing
DeepFM + MMoE pipeline.

DIN variants do not retain DeepFM's first-order or explicit FM branches. This
comparison therefore changes both the interaction model and available history;
it does not isolate the contribution of attention. PReLU replaces the original
paper's Dice activation, and AdamW replaces its mini-batch-aware embedding
regularization. These are DIN-style models rather than exact paper reproductions.

## History and leakage contract

- History contains the latest 30 positive `is_click` video IDs from training.
- For a training row, every history event has `time_ms < row.time_ms`.
  The current event and all timestamp ties are excluded even if another row
  happens to precede it in the input file. Events tied with each other are
  ordered deterministically by video ID when they become available later.
- Non-clicked videos never enter positive history. Repeated clicks remain
  repeated events; older interactions can legitimately match the target video.
- Validation, test, and all candidate rows use history frozen at the training
  cutoff. No validation or test outcomes are added, including for pointwise
  evaluation. Candidate rows have no request timestamp, so this is an explicit
  static evaluation snapshot rather than a simulated online history update.
- Training can shuffle only after its causal histories have been computed.
- Histories use the exact train-fitted video vocabulary; unseen users have
  empty histories and unknown/missing videos are masked.
- Current outcomes, retrieval scores, and retrieval ranks remain excluded from
  static features. The historical click signal is the explicitly permitted
  behavior input, not the current impression's label.

## Run and reproduce

```bash
OMP_NUM_THREADS=1 kuaiflow din --config configs/week3_din.yaml
OMP_NUM_THREADS=1 kuaiflow din-mmoe --config configs/week3_din_mmoe.yaml
make week3-comparison
```

The original `deepfm` and `mmoe` commands and default artifact names remain
compatible. `model.architecture` selects `din` / `din_mmoe` in the shared Python
training functions (`run_deepfm_ranking` / `run_mmoe_ranking`). New files use
`week3_din_*` and `week3_din_mmoe_*` prefixes. DIN's score/rank columns are
`din_score` / `din_rank`. Both multi-task models use the common `mmoe_score` /
`mmoe_rank` and per-head probability schema; their result JSON identifies the
architecture explicitly.

The loaders `load_deepfm_artifacts` and `load_mmoe_artifacts` restore either
family using checkpoint metadata. The DIN model receives a `history_index`
attribute containing the persisted training events. For scoring new pairs:

```python
model, encoder = load_deepfm_artifacts(
    "artifacts/week3_din_model.pt", "artifacts/week3_din_encoder.json"
)
categorical, numeric = encoder.transform(static_features)
history = model.history_index.transform(static_features)
with torch.no_grad():
    logits = model(
        torch.as_tensor(categorical, dtype=torch.long),
        torch.as_tensor(numeric, dtype=torch.float32),
        torch.as_tensor(history, dtype=torch.long),
    )
```

`static_features` must contain the configured joined request-time fields and
`user_id`. To reconstruct histories for training-time rows, use
`history_index.transform(frame, causal=True)` with `time_ms`. Deployment beyond
this offline snapshot would need a separate event-update and availability
contract; the saved index deliberately does not ingest new evaluation events.

## Validation

The DIN tests cover strict timestamp ties, history truncation, negative-event
exclusion, unknown users, frozen validation/test history, invalid split ordering,
target-dependent attention, padding invariance, empty histories, attention and
embedding gradients, multi-task gates, both complete toy pipelines, candidate
membership, candidate chunking, and exact checkpoint/history reloads.

References: [Deep Interest Network](https://arxiv.org/abs/1706.06978) and
[Multi-gate Mixture-of-Experts](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/).
