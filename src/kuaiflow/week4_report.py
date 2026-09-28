"""Build the Week 4 report and editable figure from persisted audit results.

This module uses the standard library only.  Optional PNG rendering reuses the
existing rsvg-convert installation; it is never required to obtain a report.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from .week3_figures import BLUE, GREEN, GRID, MUTED, NAVY, SVG, _render_png


LABELS = {
    "deepfm": "DeepFM", "din": "DIN", "mmoe": "DeepFM+MMoE",
    "deepfm_mmoe": "DeepFM+MMoE", "din_mmoe": "DIN+MMoE",
    "din-mmoe": "DIN+MMoE",
}
VARIANTS = ("raw", "standard_platt", "random_platt")
VARIANT_LABELS = {
    "raw": "Raw", "standard_platt": "Standard Platt",
    "random_platt": "Random Platt", "standard_constant": "Standard constant",
    "random_constant": "Random constant",
}
COHORT_LABELS = {
    "train": "Standard training", "standard_calibration": "Standard calibration",
    "random_calibration": "Random calibration", "standard_test": "Standard test",
    "random_test": "Random test",
}


def _number(value: Any, digits: int = 4, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def _percent(value: Any) -> str:
    if value is None:
        return "—"
    return f"{100 * float(value):.2f}%"


def _integer(value: Any) -> str:
    return "—" if value is None else f"{int(value):,}"


def _time(value: Any) -> str:
    if value is None:
        return "—"
    return datetime.fromtimestamp(float(value) / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _model_label(model: str) -> str:
    return LABELS.get(model, model)


def _models(results: dict[str, Any]) -> list[str]:
    found = list(dict.fromkeys(row["model"] for row in results["metrics"]))
    order = {name: i for i, name in enumerate(("DeepFM", "DIN", "DeepFM+MMoE", "DIN+MMoE"))}
    return sorted(found, key=lambda name: order.get(_model_label(name), 99))


def _metric_index(results: dict[str, Any]) -> dict[tuple[str, ...], dict[str, Any]]:
    return {
        (row["model"], row["target"], row["cohort"], row["variant"], row["segment"]): row
        for row in results["metrics"]
    }


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    # Escape data cells so scenario names cannot accidentally change Markdown.
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(cell(value) for value in row) + " |" for row in rows),
        "",
    ]


def _calibration_figure(results: dict[str, Any], output: Path) -> list[Path]:
    models = _models(results)
    if len(models) > 4:
        raise ValueError("The Week 4 comparison figure supports the four frozen Week 3 models.")
    index = _metric_index(results)
    values = [
        float(row["log_loss"]) for row in results["metrics"]
        if row["target"] == "click" and row["segment"] == "all"
        and row["cohort"] in ("standard_test", "random_test")
        and row["variant"] in VARIANTS and row.get("log_loss") is not None
        and math.isfinite(float(row["log_loss"]))
    ]
    ymax = max(0.2, math.ceil(max(values, default=0.8) * 1.15 * 10) / 10)
    svg = SVG(1280, 1000, title="Week 4: calibration under exposure shift",
              description=f"{len(models)} frozen Week 3 models, evaluated on time-aligned standard and randomized final impressions. Shared-axis click log-loss bars compare raw scores with standard and randomized Platt calibration; lower is better.")
    svg.rect(0, 0, 1280, 1000, fill="#F5F7FB", radius=0)
    svg.text(48, 58, "Week 4 · Calibration under exposure shift", size=31, weight=800)
    svg.text(48, 90, "Frozen Week 3 models · final time holdout · click log loss (nats / impression) ↓", size=18, color=MUTED)
    for i, (variant, color) in enumerate(zip(VARIANTS, (NAVY, BLUE, GREEN))):
        x = 52 + i * 220
        svg.rect(x, 113, 16, 16, fill=color, radius=3)
        svg.text(x + 25, 127, VARIANT_LABELS[variant], size=16)
    for i, model in enumerate(models):
        x, y = 38 + (i % 2) * 620, 159 + (i // 2) * 388
        svg.rect(x, y, 604, 370, stroke=GRID)
        svg.text(x + 26, y + 35, _model_label(model), size=22, weight=750)
        left, top, width, height = x + 66, y + 66, 503, 230
        svg.add(f'<text x="{left-47}" y="{top+height/2}" transform="rotate(-90 {left-47} {top+height/2})" text-anchor="middle" font-size="12" fill="{MUTED}">Click log loss (nats / impression)</text>')
        for tick in range(5):
            value = ymax * tick / 4
            ty = top + height * (1 - value / ymax)
            svg.line(left, ty, left + width, ty, stroke=GRID, width=1)
            svg.text(left - 10, ty + 5, f"{value:.2f}", size=12, color=MUTED, anchor="end")
        for c, cohort in enumerate(("standard_test", "random_test")):
            start = left + 33 + c * 253
            for j, (variant, color) in enumerate(zip(VARIANTS, (NAVY, BLUE, GREEN))):
                row = index.get((model, "click", cohort, variant, "all"), {})
                value = row.get("log_loss")
                bx = start + j * 58
                if value is None or not math.isfinite(float(value)):
                    svg.text(bx + 22, top + height - 10, "n/a", size=12, anchor="middle", color=MUTED)
                    continue
                bar_height = float(value) / ymax * height
                svg.rect(bx, top + height - bar_height, 44, bar_height, fill=color, radius=2)
                svg.text(bx + 22, top + height - bar_height - 8, f"{float(value):.3f}", size=12, anchor="middle", weight=600)
            svg.text(start + 80, top + height + 24, "Standard test" if c == 0 else "Random test", size=15, anchor="middle")
            n = results.get("cohorts", {}).get(cohort, {}).get("rows")
            svg.text(start + 80, top + height + 44, f"n = {_integer(n)}", size=12, anchor="middle", color=MUTED)
    svg.text(48, 957, "Random Platt uses earlier randomized labels; both final cohorts remain held out. Same y-axis in every panel.", size=16, color=MUTED)
    protocol = results.get("protocol", {})
    test_dates = f"{_time(protocol.get('test_start_ms')).split(' ')[0]} to {_time(protocol.get('test_end_ms')).split(' ')[0]} UTC"
    svg.text(48, 981, f"Source: artifacts/week4_results.json · Final test window: {test_dates}", size=15, color=MUTED)
    svg.save(output)
    paths = [output]
    png = _render_png(output, 1280, 1000)
    if png is not None:
        paths.append(png)
    return paths


def _reliability_figure(results: dict[str, Any], output: Path) -> list[Path]:
    """Four random-holdout panels with shared probability axes and bin counts."""
    svg = SVG(1280, 1080, title="Week 4: randomized holdout reliability",
              description="Observed click/valid-play rate against mean predicted probability within equal-width bins. Every model uses the same zero-to-one axes. Marker area scales with the number of impressions in each bin; the dashed diagonal is perfect calibration.")
    svg.rect(0, 0, 1280, 1080, fill="#F5F7FB", radius=0)
    svg.text(48, 58, "Random final holdout · reliability", size=31, weight=800)
    svg.text(48, 90, "Click/valid play · equal-width probability bins · the diagonal denotes perfect calibration", size=17, color=MUTED)
    for i, (variant, color) in enumerate(zip(VARIANTS, (NAVY, BLUE, GREEN))):
        x = 52 + i * 220
        svg.circle(x + 8, 122, 6, fill=color)
        svg.text(x + 25, 127, VARIANT_LABELS[variant], size=16)
    bins = [row for row in results.get("reliability", []) if row["target"] == "click"
            and row["cohort"] == "random_test" and row["variant"] in VARIANTS
            and row["count"] > 0]
    max_count = max((row["count"] for row in bins), default=1)
    for i, model in enumerate(_models(results)):
        x, y = 38 + (i % 2) * 620, 156 + (i // 2) * 421
        svg.rect(x, y, 604, 405, stroke=GRID)
        svg.text(x + 26, y + 35, _model_label(model), size=22, weight=750)
        left, top, side = x + 157, y + 64, 282
        for tick in range(6):
            fraction = tick / 5
            tx, ty = left + side * fraction, top + side * (1 - fraction)
            svg.line(tx, top, tx, top + side, stroke=GRID, width=1)
            svg.line(left, ty, left + side, ty, stroke=GRID, width=1)
            svg.text(tx, top + side + 21, f"{fraction:.1f}", size=12, color=MUTED, anchor="middle")
            svg.text(left - 12, ty + 4, f"{fraction:.1f}", size=12, color=MUTED, anchor="end")
        svg.line(left, top + side, left + side, top, stroke="#8190A5", width=1.5, dash="5 5")
        svg.text(left + side / 2, top + side + 46, "Mean predicted probability", size=14, anchor="middle", color=MUTED)
        svg.add(f'<text x="{left-53}" y="{top+side/2}" transform="rotate(-90 {left-53} {top+side/2})" text-anchor="middle" font-size="14" fill="{MUTED}">Observed positive rate</text>')
        for variant, color in zip(VARIANTS, (NAVY, BLUE, GREEN)):
            selected = sorted((row for row in bins if row["model"] == model and row["variant"] == variant), key=lambda row: row["lower"])
            points = [(left + side * row["mean_prediction"], top + side * (1 - row["positive_rate"])) for row in selected]
            if points:
                encoded = " ".join(f"{a:.3f},{b:.3f}" for a, b in points)
                svg.add(f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="2" opacity="0.8"/>')
            for (cx, cy), row in zip(points, selected):
                radius = 1.5 + 8 * math.sqrt(row["count"] / max_count)
                svg.circle(cx, cy, radius, fill=color, stroke="white", stroke_width=1)
    svg.text(48, 1030, "Larger markers represent more impressions; empty bins are omitted. Thin lines connect bin averages, not fitted curves.", size=16, color=MUTED)
    protocol = results.get("protocol", {})
    test_dates = f"{_time(protocol.get('test_start_ms')).split(' ')[0]} to {_time(protocol.get('test_end_ms')).split(' ')[0]} UTC"
    svg.text(48, 1062, f"Source: artifacts/week4_results.json · Final test window: {test_dates}", size=15, color=MUTED)
    svg.save(output)
    paths = [output]
    png = _render_png(output, 1280, 1080)
    if png is not None:
        paths.append(png)
    return paths


def generate_report(results: dict[str, Any], root: Path) -> list[Path]:
    """Write a source-derived report and figures, returning created paths."""
    root = Path(root)
    paths = _calibration_figure(results, root / "figures" / "Week_4_calibration.svg")
    paths.extend(_reliability_figure(results, root / "figures" / "Week_4_reliability.svg"))
    models, index = _models(results), _metric_index(results)
    protocol, cohorts = results["protocol"], results["cohorts"]
    leading = []
    improved_random, harmed_standard = 0, 0
    for model in models:
        for cohort in ("standard_test", "random_test"):
            raw = index.get((model, "click", cohort, "raw", "all"), {}).get("log_loss")
            calibrated = index.get((model, "click", cohort, "random_platt", "all"), {}).get("log_loss")
            if raw is not None and calibrated is not None:
                improved_random += int(cohort == "random_test" and calibrated < raw)
                harmed_standard += int(cohort == "standard_test" and calibrated > raw)
    leading.append(f"- Random-fitted calibration lowers random-test click log loss for **{improved_random} of {len(models)} models**, while increasing standard-test log loss for **{harmed_standard} of {len(models)} models**. The calibration map depends on the exposure distribution.")
    example = "din" if "din" in models else (models[0] if models else None)
    if example is not None:
        lookup = lambda c, v: index.get((example, "click", c, v, "all"), {}).get("log_loss")
        leading.append(f"- **{_model_label(example)}**: random-test log loss changes from {_number(lookup('random_test', 'raw'))} raw to {_number(lookup('random_test', 'random_platt'))} with random calibration; standard calibration changes standard-test log loss from {_number(lookup('standard_test', 'raw'))} to {_number(lookup('standard_test', 'standard_platt'))}.")
    if all(cohorts.get(c, {}).get("rows") for c in ("standard_test", "random_test")):
        share = lambda c: cohorts[c].get("tab_distribution", {}).get("1", 0) / cohorts[c]["rows"]
        leading.append(f"- Scenario mix differs: `tab=1` represents **{_percent(share('standard_test'))}** of standard final impressions and **{_percent(share('random_test'))}** of random final impressions. The shared-scenario analysis below checks whether the broad calibration result persists within that scenario.")
    lines = [
        "# KuaiFlow — Week 4: exposure robustness and probability calibration", "",
        f"Week 4 evaluates {len(models)} frozen Week 3 models ({', '.join(_model_label(model) for model in models)}) on time-aligned standard-policy and randomized impressions. Standard-log Platt calibration is compared with adaptation using earlier randomized labels; later randomized impressions provide the final holdout. Every reported value is generated from the saved audit artifact.", "",
        *leading, "",
        "![Week 4 calibration comparison](../figures/Week_4_calibration.svg)", "",
        "## Protocol and data boundaries", "",
        f"- Training ends at **{_time(protocol.get('train_cutoff_ms'))}**.",
        f"- Both calibration cohorts use the cutoff **{_time(protocol.get('calibration_cutoff_ms'))}**; their last observed timestamps can differ.",
        f"- The common final test window is **{_time(protocol.get('test_start_ms'))}** through **{_time(protocol.get('test_end_ms'))}**; final rows are strictly later than the calibration cutoff.",
        "- Base-model weights, encoders, and DIN training histories remain frozen. The original standard validation period already selected training epochs and now fits standard calibration; it is not an untouched validation set. Neither final cohort fits calibrators or selects a variant.",
        "- Early randomized labels fit the random calibrator. This is explicit target-policy adaptation followed by a future holdout, so the entire random log is no longer an untouched audit set.",
        "- Each Platt map uses the model's binary score only. A positive slope preserves single-head score ordering, apart from numerical clipping/ties; a one-class fallback is a constant and can collapse that ordering. Composite utility and candidate rankings are not recalculated.",
        f"- Random rows excluded at or before the training cutoff: **{_integer(protocol.get('dropped_random_before_or_at_training'))}**. Rows outside the common test window: `{json.dumps(protocol.get('dropped_outside_common_test', {}), sort_keys=True)}`.", "",
    ]
    cohort_rows = []
    for name in COHORT_LABELS:
        if name not in cohorts:
            continue
        row = cohorts[name]
        cohort_rows.append([COHORT_LABELS[name], _integer(row.get("rows")), _integer(row.get("users")),
                            _integer(row.get("items")), _percent(row.get("positive_rate")),
                            _time(row.get("start_time_ms")), _time(row.get("end_time_ms"))])
    lines += _table(["Cohort", "Rows", "Users", "Videos", "Click rate", "First impression", "Last impression"], cohort_rows)
    lines += [
        "`is_click` represents a click in the two-column interface and valid play in the single-column interface. `tab` identifies a recommendation scenario and is not a verified UI label. Pure contains a restricted video pool and incomplete interaction histories. See the [official KuaiRand field definitions](https://kuairand.com/).", "",
        "## Final click prediction", "",
        "Log loss, Brier score, and ECE are lower-is-better; ROC-AUC and average precision are higher-is-better. `PR-AUC` below means average precision, not trapezoidal PR integration. ECE is the count-weighted absolute probability gap in equal-width bins; it depends on the chosen binning. Full reliability-bin counts are saved with the audit.", "",
    ]
    for cohort in ("standard_test", "random_test"):
        lines += [f"### {COHORT_LABELS[cohort]}", ""]
        rows = []
        for model in models:
            for variant in VARIANTS:
                row = index.get((model, "click", cohort, variant, "all"))
                if row is not None:
                    rows.append([_model_label(model), VARIANT_LABELS[variant], _number(row.get("roc_auc")),
                                 _number(row.get("pr_auc")), _number(row.get("log_loss")),
                                 _number(row.get("brier")), _number(row.get("ece")),
                                 _percent(row.get("mean_prediction"))])
        lines += _table(["Model", "Calibration", "ROC-AUC", "PR-AUC", "Log loss ↓", "Brier ↓", "ECE ↓", "Mean prediction"], rows)
    lines += ["### Constant-prediction controls", "",
              "Each constant is fitted on its named calibration cohort, including any smoothing used by the implementation. These controls show how much improvement is available from a prevalence adjustment alone. They are shared across model families; the table reads the first model's identical controls.", ""]
    constant_rows = []
    if models:
        for cohort in ("standard_test", "random_test"):
            for variant in ("standard_constant", "random_constant"):
                row = index.get((models[0], "click", cohort, variant, "all"))
                if row is not None:
                    constant_rows.append([COHORT_LABELS[cohort], VARIANT_LABELS[variant],
                                          _percent(row.get("mean_prediction")), _number(row.get("log_loss")),
                                          _number(row.get("brier")), _number(row.get("ece"))])
    lines += _table(["Test cohort", "Control", "Prediction", "Log loss ↓", "Brier ↓", "ECE ↓"], constant_rows)
    lines += ["### Randomized holdout reliability", "",
              "![Week 4 randomized reliability](../figures/Week_4_reliability.svg)", "",
              "Every panel uses the same probability axes. Bin averages describe the held-out observations; connected points do not represent an additional fitted calibration model. Very small bins can have unstable observed rates, visible as tiny markers at the extremes.", ""]
    lines += ["## Paired uncertainty for click calibration", "",
              "The estimates are **calibrated minus raw** on the same impressions. Negative differences improve the loss. The bootstrap resamples users with replacement, retaining their impression clusters and pairing raw/calibrated scores. Intervals condition on the fitted models and calibrators; they do not include retraining uncertainty, cross-user dependence, or multiple-comparison adjustment.", ""]
    uncertainty_rows = []
    uncertainty = results.get("uncertainty", [])
    for row in uncertainty:
        loss, brier = row.get("log_loss", {}), row.get("brier", {})
        uncertainty_rows.append([
            _model_label(row["model"]), COHORT_LABELS.get(row["cohort"], row["cohort"]),
            VARIANT_LABELS.get(row["variant"], row["variant"]),
            _number(loss.get("estimate"), signed=True),
            f"[{_number(loss.get('ci_lower'), signed=True)}, {_number(loss.get('ci_upper'), signed=True)}]",
            _number(brier.get("estimate"), signed=True),
            f"[{_number(brier.get('ci_lower'), signed=True)}, {_number(brier.get('ci_upper'), signed=True)}]",
        ])
    if uncertainty:
        first = uncertainty[0]
        lines += [f"Bootstrap settings: {_integer(first.get('repeats'))} replicates, confidence level {_percent(first.get('confidence_level'))}, seed {first.get('seed', '—')}. Full replicate counts, cohort sizes, and method names are in the saved JSON.", ""]
    lines += _table(["Model", "Cohort", "Calibration", "Δ log loss", "Confidence interval", "Δ Brier", "Confidence interval"], uncertainty_rows)
    lines += ["## Multi-task binary heads on the random final holdout", "",
              "Only binary action heads receive probability calibration. Watch time and completion are continuous targets and remain outside this calibration analysis. The table includes rare-event prevalence and average precision because ROC-AUC alone can obscure sparse positives. Missing class-dependent metrics are undefined, not zero.", ""]
    head_rows = []
    for model in models:
        targets = list(dict.fromkeys(row["target"] for row in results["metrics"] if row["model"] == model))
        if len(targets) < 2:
            continue
        for target in targets:
            raw = index.get((model, target, "random_test", "raw", "all"), {})
            standard = index.get((model, target, "random_test", "standard_platt", "all"), {})
            random = index.get((model, target, "random_test", "random_platt", "all"), {})
            head_rows.append([_model_label(model), target, _percent(raw.get("positive_rate")),
                              _number(raw.get("pr_auc")), _number(raw.get("log_loss")),
                              _number(standard.get("log_loss")), _number(random.get("log_loss")),
                              _number(random.get("ece"), digits=6)])
    lines += _table(["Model", "Target", "Positive rate", "Raw PR-AUC", "Raw log loss", "Standard Platt log loss", "Random Platt log loss", "Random Platt ECE"], head_rows)
    lines += ["## Shared scenario and exposure composition", "",
              "The `tab=1` subset below compares the same recorded scenario, without interpreting that number as a specific UI. It does not hold user/video mix fixed or identify a causal exposure-policy effect. Full click subgroups are retained in the metric artifact.", ""]
    segment_rows = []
    for model in models:
        for cohort in ("standard_test", "random_test"):
            raw = index.get((model, "click", cohort, "raw", "tab=1"))
            if raw is None:
                continue
            standard = index.get((model, "click", cohort, "standard_platt", "tab=1"), {})
            random = index.get((model, "click", cohort, "random_platt", "tab=1"), {})
            segment_rows.append([_model_label(model), COHORT_LABELS[cohort], _integer(raw.get("rows")),
                                 _percent(raw.get("positive_rate")), _number(raw.get("log_loss")),
                                 _number(standard.get("log_loss")), _number(random.get("log_loss"))])
    lines += _table(["Model", "Cohort, tab=1", "Rows", "Click rate", "Raw log loss", "Standard Platt log loss", "Random Platt log loss"], segment_rows)
    exposure = results.get("exposure_shift", {})
    if exposure:
        lines += ["Exposure comparisons below are descriptive. Total variation distance is half the sum of absolute differences between normalized marginal frequencies; it is not a propensity estimate.", ""]
        shift_rows = []
        for column, label in (("user_id", "User frequencies"), ("video_id", "Video frequencies"), ("tab", "Scenario frequencies")):
            row = exposure.get(column, {})
            if row:
                shift_rows.append([label, _number(row.get("total_variation")),
                                   _number(row.get("jensen_shannon_bits")), _integer(row.get("common_values"))])
        lines += _table(["Exposure marginal", "Total variation", "Jensen–Shannon, bits", "Shared IDs"], shift_rows)
        composition_rows = []
        for cohort in ("standard_test", "random_test"):
            row = exposure.get("training_segments", {}).get(cohort, {})
            total = row.get("all", 0)
            if total:
                composition_rows.append([COHORT_LABELS[cohort], _percent(row.get("tab=1", 0) / total),
                                         _percent(row.get("item=head", 0) / total),
                                         _percent(row.get("item=tail", 0) / total),
                                         _percent(row.get("item=unseen", 0) / total)])
        if composition_rows:
            lines += _table(["Cohort", "tab=1 share", "Training head-item share", "Training tail-item share", "Unseen item share"], composition_rows)
            lines += ["Head items are the top 20% of training items by exposure count, with item-ID tie breaking; tail items are the remaining known training items. Percentages weight impressions, and neither segment uses final labels.", ""]
    unknown_rows = []
    for cohort in ("standard_test", "random_test"):
        row = cohorts.get(cohort, {})
        if "train_unknown_user_rate" in row or "train_unknown_item_rate" in row:
            unknown_rows.append([COHORT_LABELS[cohort], _percent(row.get("train_unknown_user_rate")), _percent(row.get("train_unknown_item_rate"))])
    if unknown_rows:
        lines += _table(["Cohort", "Unseen training-user row rate", "Unseen training-video row rate"], unknown_rows)
    lines += ["## Calibrator fits and interpretation", ""]
    fit_rows = []
    for model, targets in results.get("calibrators", {}).items():
        for target, variants in targets.items():
            for variant, fit in variants.items():
                if target == "click" or float(fit.get("slope", 1.0)) == 0.0 or fit.get("converged") is False:
                    status = fit.get("status") or (f"{fit.get('method', 'fit')}: converged" if fit.get("converged") else "unavailable")
                    if fit.get("reason"):
                        status = f"{fit.get('method', 'fallback')}: {fit['reason']}"
                    fit_rows.append([_model_label(model), target, VARIANT_LABELS.get(variant, variant),
                                     _number(fit.get("slope")), _number(fit.get("intercept")), status])
    lines += _table(["Model", "Head", "Calibration", "Slope", "Intercept", "Fit status"], fit_rows)
    lines += ["Click fits and any nonstandard fit status are shown above; all head parameters are persisted. A one-class calibration sample uses a smoothed constant fallback. A calibrated probability can improve proper scoring rules without improving ranking, and a strictly increasing map cannot fix discrimination errors.", ""]
    findings = []
    for model in models:
        raw = index.get((model, "click", "random_test", "raw", "all"), {})
        standard = index.get((model, "click", "random_test", "standard_platt", "all"), {})
        random = index.get((model, "click", "random_test", "random_platt", "all"), {})
        if all(row.get("log_loss") is not None for row in (raw, standard, random)):
            findings.append(f"- **{_model_label(model)}**, random test: raw log loss {_number(raw['log_loss'])}; standard Platt {_number(standard['log_loss'])}; random Platt {_number(random['log_loss'])} (random-minus-raw {_number(float(random['log_loss']) - float(raw['log_loss']), signed=True)}).")
    lines += findings + ["",
        "These are fixed-variant observations on the final holdout, not a procedure for choosing a new production calibrator. Comparing the two exposure cohorts combines policy, user/video composition, and scenario effects even with time alignment. No logged per-impression propensities are supplied to this pipeline, so it performs no IPS/DR off-policy estimation and makes no unbiased policy-value or online-lift claim.", "",
        "## Reproduce", "", "```bash", "OMP_NUM_THREADS=1 make week4", "make week4-report", "python -m unittest discover -s tests -v", "```", "",
        "The audit reads the selected saved Week 3 checkpoints and configuration paths in `configs/week4.yaml`. It saves metrics, reliability bins, calibrator parameters, cohort diagnostics, and paired uncertainty in `artifacts/week4_results.json`; the report builder reads that artifact to regenerate this page and the calibration/reliability SVG figures. Matching PNGs are produced when `rsvg-convert` is installed. See [Week 4 implementation notes](week4.md) for the equations and constraints.", "",
        "References: [KuaiRand dataset and fields](https://kuairand.com/), [probability calibration background](https://scikit-learn.org/1.4/modules/calibration.html).", "",
    ]
    report = root / "docs" / "week4_results.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    paths.insert(0, report)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--results", type=Path, default=Path("artifacts/week4_results.json"))
    args = parser.parse_args()
    path = args.results if args.results.is_absolute() else args.root / args.results
    results = json.loads(path.read_text(encoding="utf-8"))
    for output in generate_report(results, args.root):
        print(output)


if __name__ == "__main__":
    main()
