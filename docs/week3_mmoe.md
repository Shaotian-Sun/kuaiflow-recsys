# Week 3 — DeepFM + MMoE Multi-Task Ranking

## What changed from the DeepFM baseline

The single-task DeepFM baseline predicts only click. This model keeps DeepFM's
linear and factorization-machine branches, but replaces its one deep MLP with a
multi-gate mixture-of-experts (MMoE) branch.

```text
logged impressions ── train labels ───────────────────────────────┐
                                                                  │
request-time feature fields                                      │
        │                                                         │
 shared embeddings                                                │
   ┌────┴──────────────┐                                          │
   │                   │                                          │
FM pair interactions   four shared MLP experts                    │
   │                   │                                          │
task-specific scale    one softmax gate per task                   │
   │                   │                                          │
   └──────────┬────────┘                                          │
      task-specific linear term + task tower ◄────────────────────┘
                         │
             ten jointly trained logits
                         │
Week 2 top 100 ── score only; never create labels ── reorder same items
```

For task \(k\), the final logit is

\[
z_k = \operatorname{Linear}_k(x)
    + \alpha_k\operatorname{FM}(x)
    + \operatorname{Tower}_k\!\left(
        \sum_e g_{k,e}(x)\operatorname{Expert}_e(x)
      \right).
\]

The gate weights \(g_{k,e}\) use a softmax, so they are non-negative and sum to
one for each example and task. Click may rely on one mixture of experts while
follow or watch time relies on another. The shared embeddings, shared FM
interaction, and experts let dense tasks teach useful structure to sparse
tasks. Task-specific first-order terms, FM scales, gates, and towers reduce
unwanted interference.

This is a DeepFM + MMoE model. DIN is not included yet. DIN needs a causal,
time-ordered sequence of previously watched items and a target-item attention
mask; it should be added only after this multi-task baseline is sound.

## Targets

Eight clean per-impression binary outcomes are trained with ordinary
binary-cross-entropy-with-logits:

- click: `is_click`
- like: `is_like`
- follow: `is_follow`
- comment: `is_comment`
- forward/share: `is_forward`
- long view: `long_view`
- profile entry: `is_profile_enter`
- negative feedback: `is_hate`

Their sigmoid outputs are conditional rates such as
\(P(\text{like}=1\mid x)\). They are targets, not historical-rate input
features.

There is no per-impression collection/save label in KuaiRand-Pure. The separate
video-statistics file contains month-aggregated `collect_cnt` values without a
documented as-of timestamp and overlaps the evaluation period. Treating those
counts as a label or feature would leak future information. The pipeline
therefore rejects `is_collect` and aggregate count/rate features. A collection
head can be added later without changing the architecture once a timestamped
per-impression label exists.

## Watch-time head

Let \(t\) be actual watch time in seconds and let \(\tau=1\) second. The
proposed expression \((1+t)/t\) cannot be a cross-entropy target: it is greater
than one for every positive \(t\), and it is undefined at zero.

The implemented weighted-logistic target is

\[
a=\frac{t}{\tau},\qquad y_t=\frac{a}{1+a},\qquad w_t=1+a,
\]

with loss

\[
L_t =
\frac{\sum_i w_{t,i}\,
\operatorname{BCEWithLogits}(z_{t,i},y_{t,i})}
{\sum_i w_{t,i}}.
\]

At the optimum for examples with the same features,

\[
e^{z_t}=E[t/\tau\mid x],
\qquad
\hat t=\tau e^{z_t}.
\]

The weight \(1+t/\tau\) is essential to that expected-time identity. Without
it, exponentiating the logit would not estimate mean watch time. The exported
serving estimate has a configurable 1,200-second safety cap; training uses the
uncapped objective.

## Completion-percentage head

For play time \(t\) and video duration \(d\), the soft target is

\[
r=\operatorname{clip}(t/d,0,1).
\]

Clipping matters because replayed videos can have play time larger than their
duration. Rows with `duration_ms == 0` use KuaiRand's missing-duration sentinel
and are masked from only this loss; they still train every other applicable
head. The completion logit uses soft-label binary cross-entropy.

Video length strongly affects completion. The pipeline fits a decreasing
isotonic curve on training rows only:

\[
f(d)\approx E[r\mid d].
\]

For each candidate it exports

\[
\text{completion lift}=\frac{\hat r}{f(d)}.
\]

This ratio can exceed one, so it is not called a probability and is never used
as a cross-entropy target. It means “predicted completion relative to videos of
this duration.” The bounded composite component is
\(\hat r/(\hat r+f(d))\).

## How the ten losses are combined

Raw binary losses naturally have very different magnitudes: click occurs often,
while follow, forward, and hate occur around one per thousand impressions or
less. Positive-class weighting would make sigmoid outputs poorly calibrated.
Instead, each task's raw loss is divided by the loss of its best constant
training-set predictor, then the configured task weights are averaged:

\[
L=\frac{1}{\sum_k\lambda_k}
  \sum_k\lambda_k\frac{L_k}{L_k^{\text{constant}}}.
\]

This rescales gradients so sparse tasks remain visible without changing the
optimum of any individual likelihood. All ten default loss weights are one.
Task biases are initialized to the corresponding constant-predictor logits.

One AdamW step jointly updates 1,376,200 parameters:

- 758,864 shared feature-embedding parameters;
- 115,456 shared expert parameters;
- 6,440 task-gate parameters;
- 21,130 task-tower parameters;
- 474,310 task-specific first-order, bias, and FM-scale parameters.

AdamW weight decay is \(10^{-6}\), dropout is 0.1, and the checkpoint with the
lowest normalized validation loss is restored. The current run stopped after
epoch 4 and restored epoch 2.

## Candidate score

Every one of the one million Week 2 candidate rows receives all eight action
probabilities, expected watch seconds, completion fraction, duration baseline,
and completion lift. The default composite is an explicit neutral starting
policy:

- add each positive action probability;
- subtract predicted hate probability;
- add log-scaled watch time, bounded by the training 95th-percentile reference;
- add bounded duration-adjusted completion;
- divide by the sum of absolute policy weights.

The utility weights live in `configs/week3_mmoe.yaml`. They are a serving-policy
choice, not labels learned from the dataset. `retrieval_rank` is used only as a
deterministic tie-breaker. Candidate membership is unchanged.

## Leakage contract

- Fit vocabularies, numeric transformations, null-loss scales, duration curve,
  and watch-time reference on the early training period only.
- Use the later validation split only for checkpoint selection and the final
  later test split only for reporting.
- Never train on the Week 2 candidate file and never turn an unexposed candidate
  into a negative example.
- Exclude all action labels, dwell times, request timestamps, policy fields,
  retrieval rank/score, and month-aggregated behavior statistics from features.
- Use stable basic-video metadata for candidate duration. Missing duration falls
  back to the training completion mean.

## Current full-data result

This run uses the reviewed fixed-denominator watch objective and the current
source. Training completed four epochs in 60.9 seconds and restored epoch 2,
which had the lowest normalized chronological-validation loss.

The retained model's test pointwise results include:

- click: ROC-AUC 0.7150, PR-AUC 0.6432, log loss 0.6172;
- like: ROC-AUC 0.8448, PR-AUC 0.1505;
- follow: ROC-AUC 0.7570, PR-AUC 0.0187;
- comment: ROC-AUC 0.7460, PR-AUC 0.0135;
- forward: ROC-AUC 0.7085, PR-AUC 0.0065;
- long view: ROC-AUC 0.7212, PR-AUC 0.5162;
- profile entry: ROC-AUC 0.7321, PR-AUC 0.0604;
- hate: ROC-AUC 0.7330, PR-AUC 0.0176;
- watch time: 23.66-second MAE and 39.72-second RMSE;
- completion: 0.2653 MAE and 0.3307 RMSE on valid-duration rows.

Over the unchanged test top-100 candidates, retrieval-order click NDCG@20 is
0.01415. The MMoE composite reaches 0.01799 and the click head alone reaches
0.01947. The earlier single-task DeepFM reaches 0.01687. Relative to retrieval,
the composite improves click Recall@20 by 41.8%, HitRate@20 by 43.4%, and
NDCG@20 by 27.1%. Relative to single-task DeepFM, its NDCG@20 is 6.6% higher.

The composite also improves like, long-view, and profile-entry ranking, but it
does not improve every rare action under the neutral weights: follow, comment,
and forward have only 40, 84, and 33 evaluated candidate users respectively.
Those estimates have high variance, and the composite weights should not be
presented as “optimal.”

The single-task DeepFM remains slightly better on pointwise click alone: its
test ROC-AUC is 0.7195 versus 0.7150 and its log loss is 0.6168 versus 0.6172.
This is a useful distinction: the current MMoE improves ordering inside the
retrieved candidate sets while trading a small amount of global click
classification quality for multi-objective learning.

At K=100, recall and hit rate are unchanged for each ordering because reranking
cannot recover an item absent from Week 2. NDCG can still change because the 100
items are reordered.

## Run and outputs

```bash
OMP_NUM_THREADS=1 kuaiflow mmoe --config configs/week3_mmoe.yaml
```

Generated candidate handoff:

- `data/processed/ranking/week3_mmoe_top100.csv.gz`

Reproducibility artifacts:

- `artifacts/week3_mmoe_model.pt`
- `artifacts/week3_mmoe_encoder.json`
- `artifacts/week3_mmoe_duration_curve.json`
- `artifacts/week3_mmoe_results.json`
- `artifacts/week3_mmoe_ranking_metrics.csv`

The checkpoint loader restores the model, exact task order, feature encoder,
watch-time constants, utility policy, and train-fitted duration curve.

## Saved snapshot and next step

The current local snapshot contains:

- the DeepFM + MMoE model, objective utilities, pipeline, CLI, configuration,
  toy multi-task data, and focused tests;
- the current-code checkpoint from epoch 2 selected by chronological validation
  loss;
- the exact encoder and train-only duration-completion curve;
- pointwise and candidate-ranking metrics;
- all one million scored Week 2 candidates with the ten outputs kept separate.

All 34 repository tests pass. The million-row candidate artifact contains only
finite numeric values, preserves every `(split, user_id, video_id)` pair from
Week 2, gives every user exactly ranks 1 through 100, and preserves Recall@100
and HitRate@100 under every ordering. Reloading the saved checkpoint reproduces
stored click probabilities within \(9\times10^{-8}\).

Before changing the serving policy, review the separate head metrics and choose
a business utility policy. Changing only the ten utility weights does not require model
retraining because every head prediction is already present in
`week3_mmoe_top100.csv.gz`. The current equal-magnitude formula should be kept
as a baseline, not silently promoted to an optimized policy.

The next modeling step is DIN: build a causal user-history sequence using only
events earlier than each impression, attend that sequence to the candidate
video, and concatenate the DIN interest vector with the shared MMoE input. It
should be evaluated against this frozen DeepFM + MMoE snapshot rather than
replacing it without an ablation.
