# Week 3 — DeepFM Click Reranking

## What this stage does

Week 2 retrieves 100 plausible videos for each user. DeepFM does not retrieve
new videos; it assigns a new click score to each of those 100 and changes their
order.

```text
all videos -> Week 2 retrieval -> 100 candidates -> DeepFM -> reordered 100
```

The model is trained on logged impressions, where `is_click` is observed. The
candidate table is not a training-label table: most candidate pairs were never
shown, so an absent click must not be treated as a negative example.

## The three parts of DeepFM

For feature fields `x`, the model produces one click logit:

```text
logit = linear term + FM interaction term + deep-network term
```

### 1. Linear term

The linear branch learns the independent effect of a feature value. Examples
include a particular video's general click tendency or the average tendency of
a user activity bucket.

### 2. FM term

The factorization-machine branch learns every pairwise interaction through
embedding dot products. For example, a certain user and video or a user-active
degree and video type can be useful together even when neither feature is
strong alone.

The implementation uses the efficient identity:

```text
0.5 * ((sum of embeddings)^2 - sum of squared embeddings)
```

This computes all field pairs without an explicit nested loop.

### 3. Deep term

The same field embeddings are flattened and passed through an MLP. This branch
learns higher-order combinations that cannot be expressed as a sum of pairwise
dot products.

The FM and deep branches share embeddings. Consequently, the single click loss
updates first-order weights, embeddings, and MLP weights in one backward pass.

## Feature and leakage contract

The strict first version uses user/video IDs and stable video metadata available
for both logged impressions and retrieved candidates. Vocabulary and numerical
scaling statistics are fitted on training data only. Index 0 is reserved for an
unseen category and is fixed to a zero embedding; index 1 represents a missing
value and can be learned.

The untimestamped user-profile snapshot is excluded from the strict baseline.
Activity status and follower/friend counts can change over time, so using one
snapshot across all chronological splits could leak future information. The
multi-valued `tag` field is also deferred until it has a proper split-and-pool
encoder instead of treating a whole tag list as one category.

Post-impression outcomes such as play time, long-view, like, follow, comment,
and profile dwell are forbidden as inputs. Request-time fields are also omitted
because the current candidate table represents one list per user and split,
not one timestamped request. `retrieval_rank` is retained for comparison and
tie-breaking but is not learned as a feature.

## Optimization

The target is `is_click` and the loss is ordinary binary cross-entropy with
logits. Clicks make up about 46% of training impressions, so no positive-class
weight is applied. AdamW supplies parameter regularization, dropout regularizes
the deep branch, and validation log loss selects the best epoch.

## Evaluation

Two different evaluations answer different questions:

- Pointwise ROC-AUC, PR-AUC, and log loss are computed only on later logged
  impressions with observed labels.
- Recall, HitRate, NDCG, and coverage compare the original Week 2 order with the
  DeepFM order over exactly the same candidate membership.

At K=100, reranking cannot improve recall because it cannot introduce an item
that Week 2 failed to retrieve. It can improve NDCG@100 and all ranking metrics
at smaller cutoffs by moving clicked items upward.

## Run

First generate the selected Week 2 top-100 table, then train DeepFM:

```bash
OMP_NUM_THREADS=1 kuaiflow retrieval \
  --config configs/week2_faiss_ivf.yaml --mode faiss
OMP_NUM_THREADS=1 kuaiflow deepfm --config configs/week3_deepfm.yaml
```

Generated pipeline data:

- `data/processed/candidates/week2_feature_history_faiss_ivf_top100.csv.gz`
- `data/processed/ranking/week3_deepfm_top100.csv.gz`

Model and evaluation artifacts:

- `artifacts/week3_deepfm_model.pt`
- `artifacts/week3_deepfm_encoder.json`
- `artifacts/week3_deepfm_results.json`
- `artifacts/week3_deepfm_ranking_metrics.csv`

## First full-data result

With seed 2026, training stopped after epoch 4 and restored epoch 2, which had
the lowest validation log loss. The retained model achieved validation ROC-AUC
0.7348 and test ROC-AUC 0.7195. Test log loss was 0.6168.

On the fixed test candidate sets, DeepFM changed the K=20 metrics as follows:

- Recall: 0.0260 to 0.0350 (about 34.7% relative improvement).
- HitRate: 0.0742 to 0.1040 (about 40.2% relative improvement).
- NDCG: 0.01415 to 0.01687 (about 19.2% relative improvement).
- Catalog coverage: 0.8408 to 0.3866.

The accuracy gains therefore come with a substantial concentration cost: the
model promotes a much smaller set of videos into the first 20 positions. That
trade-off should remain visible rather than being hidden behind one headline
metric. At K=100, recall and HitRate are unchanged, confirming that DeepFM only
reordered the Week 2 candidates.
