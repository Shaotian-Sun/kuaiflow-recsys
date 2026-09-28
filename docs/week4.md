# Week 4 — Exposure robustness and probability calibration

Week 4 takes the four saved Week 3 rankers—DeepFM, DIN, DeepFM+MMoE,
and DIN+MMoE—and measures whether their probabilities remain useful under
randomized exposure. It fits small post-hoc calibration maps while keeping the
trained models and feature/history state fixed. See the generated
[experimental report](week4_results.md) for measured results.

## Data contract

The audit reuses the prepared chronological standard training, validation,
and test logs, plus the saved randomized log. A row is an exposed video with
observed feedback; unexposed candidate videos are not added as negative labels.

1. **Training:** keep the Week 3 model weights and train-fitted vocabularies,
   static feature preprocessing, and DIN history state. Random labels never
   train or select the base models.
2. **Standard calibration:** reuse the original standard validation period.
   This period already selected training epochs, so it is not independent
   of base-model selection. It can fit a calibrator; its calibration fit loss
   is not a final generalization estimate.
3. **Random calibration:** use randomized impressions strictly after the
   training cutoff and at or before the standard validation cutoff. This is
   explicit adaptation to earlier randomized labels. The whole random log
   therefore ceases to be an untouched audit set.
4. **Final tests:** use standard and randomized impressions after that cutoff
   within their common observed time window. Freeze all decisions before
   evaluating these final cohorts. Matching time boundaries does not match
   the distribution of users, items, or scenarios.

The pipeline checks strict time separation, checkpoint history cutoffs, static
join alignment, and finite score vectors. It retains unseen user/item rows and
reports their frequency; train-fitted encoders handle unknown categories.
No random outcomes enter DIN histories. The audit records checkpoint/data
hashes, runtime versions, the full configuration, and cohort membership so
saved score arrays can be aligned exactly with impressions.

## What the labels mean

KuaiRand's `is_click` is a click in the two-column interface and valid play in
the single-column interface. `tab` is a recommendation-scenario identifier;
the audit does not interpret a numeric `tab` value as a verified interface.
KuaiRand-Pure contains only the selected item pool and incomplete histories.
These constraints come from the [official dataset descriptions](https://kuairand.com/).

The single-task models calibrate click only. Both multi-task models calibrate
their eight binary heads: click, like, follow, comment, forward, long-view,
profile-entry, and hate. Watch time and completion are continuous predictions
and are not passed through a binary calibrator.

## Calibration

For a frozen model logit `z`, fit

\[
p_{a,b}(z)=\sigma(az+b), \qquad
(\hat a,\hat b)=\arg\min_{a,b}\frac1n\sum_i
\left[\log(1+e^{az_i+b})-y_i(az_i+b)\right].
\]

The implementation uses stable binary cross-entropy and L-BFGS-B with
`1e-6 <= a <= 100` and `-100 <= b <= 100`. Each binary head gets two maps:
one fitted on standard calibration labels and another fitted on early random
labels. Failed optimization raises an error. A one-class fit cannot identify a
useful slope, so it falls back to a zero-slope constant with Jeffreys-smoothed
probability `(positive_count + 0.5) / (row_count + 1)` and records that reason.

A positive slope preserves the ordering of one head in exact arithmetic;
finite precision can introduce ties. The zero-slope fallback collapses the
ordering. Calibrating several heads can change a weighted composite utility,
so this week deliberately does not recalculate composite utility, rerank
candidates, or claim improved NDCG. The calibration family is fixed in advance;
the final test does not choose a variant. The
[probability calibration documentation](https://scikit-learn.org/1.4/modules/calibration.html)
provides background on sigmoid maps and reliability diagrams; the bounded
positive-slope fit here is the project's explicit implementation choice.

Raw scores and both maps are evaluated on both final cohorts. Two additional
controls predict the smoothed click prevalence of each calibration cohort
for every impression. They separate a simple prevalence correction from the
benefit of retaining model discrimination.

## Metrics and uncertainty

Log loss, Brier score, and equal-width expected calibration error (ECE) measure
probability quality. The report also shows average predicted probability,
observed prevalence, ROC-AUC, and average precision. The artifact key `pr_auc`
means average precision, not trapezoidal area under a PR curve. Empty and
one-class cohorts produce null class-dependent discrimination metrics rather
than fabricated zeros. Reliability bins include both counts and averages;
an empty bin carries null averages.

ECE uses the impression-weighted absolute gap between mean prediction and
observed label rate in each bin. It is sensitive to the number of bins and is
interpreted alongside proper scoring rules and reliability diagrams. Binning
is equal-width on `[0,1]`, left-closed/right-open, except the final bin includes
one.

For click log loss and Brier score, the pipeline computes paired differences
`calibrated - raw`. It bootstraps users with replacement, retaining each user's
entire impression cluster, and recomputes the impression-weighted mean
difference. The default is 500 replicates, seed 2026, with percentile 95%
intervals. This conditions on the frozen models and fitted calibrators; it
does not cover retraining/calibrator-fitting uncertainty, cross-user
dependence, or correction for comparing several variants. No interval is
claimed for rare-action per-head rows or scenario subsets.

## Exposure diagnostics and limits

The audit reports normalized user, item, and `tab` frequency changes using
total variation distance and Jensen–Shannon divergence in bits. It also reports
unknown-category frequencies and click metrics for training-defined activity
and item-popularity segments. The shared `tab=1` analysis is a scenario
sensitivity check; it does not isolate the causal effect of the exposure
policy. Training-only segment definitions prevent test outcomes from defining
the cohorts.

Randomized versus standard prediction differences can combine policy, item,
user, and scenario mix. No logged per-impression propensities are provided to
this implementation; it performs no inverse propensity scoring or doubly
robust off-policy evaluation. Calibration improvements do not establish a
better recommendation policy, online lift, or an unbiased population metric.

## Run and outputs

After the four Week 3 checkpoints and prepared data are available:

```bash
OMP_NUM_THREADS=1 make week4
make week4-report
python -m unittest discover -s tests -v
```

`configs/week4.yaml` controls input/output paths, batch size, calibration bins,
and bootstrap repeats. The audit saves:

- `artifacts/week4_results.json`: metrics, boundaries, cohort summaries,
  calibrators, exposure diagnostics, uncertainty, and provenance.
- `artifacts/week4_metrics.csv` and `artifacts/week4_reliability.csv`: complete
  metric and reliability-bin rows.
- `artifacts/week4_calibrators.json`: portable post-hoc fit parameters.
- `artifacts/week4_cohort_membership.csv.gz`: impression order and identifiers
  for the score arrays.
- `artifacts/week4_<model>_logits.npz`: frozen logits when `save_logits` is true.
- `docs/week4_results.md`: report regenerated from measured artifacts.
- `figures/Week_4_calibration.svg` and `figures/Week_4_reliability.svg`: editable
  scientific figures, with PNG copies when `rsvg-convert` is available.

`make week4-report` regenerates only documentation and figures from the saved
JSON; it does not refit calibrators or score the models again.

To reload the random-fitted DIN click map and apply it to the saved final raw
logits:

```python
import json
from pathlib import Path
import numpy as np
from kuaiflow.calibration import apply_calibrator

fits = json.loads(Path("artifacts/week4_calibrators.json").read_text())
fit = fits["din"]["click"]["random_platt"]
with np.load("artifacts/week4_din_logits.npz") as scores:
    raw_logits = scores["random_test__click"]
    probabilities = apply_calibrator(raw_logits, fit)
```

The input must be uncalibrated logits from the same frozen DIN model and click
head, not probabilities or composite utilities. `random_platt` records a map
fitted to early randomized impressions; its held-out performance does not
establish that it transfers to standard-policy exposures. `source_cohort` and
`cutoff_time_ms` in the fit record preserve that calibration context.
