"""Probability calibration and paired uncertainty for held-out binary outcomes.

Fit on validation impressions and evaluate on a later, untouched test set. The
positive Platt slope preserves score ordering (apart from floating-point ties).
The explicit single-class fallback is a smoothed constant and has no such
ordering guarantee. ``pr_auc`` denotes average precision, the stepwise area
under the precision-recall curve; ROC AUC is undefined for a single class.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Integral
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata


_EPS = np.finfo(np.float64).eps


def _vector(values: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite one-dimensional numeric array") from exc
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite one-dimensional numeric array")
    return result


def _labels(values: Any) -> np.ndarray:
    result = _vector(values, "labels")
    if not np.isin(result, [0.0, 1.0]).all():
        raise ValueError("labels must contain only binary 0/1 values")
    return result


def _probabilities(values: Any) -> np.ndarray:
    result = _vector(values, "probabilities")
    if np.any((result < 0.0) | (result > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    return result


def _paired(labels: Any, probabilities: Any) -> tuple[np.ndarray, np.ndarray]:
    y, p = _labels(labels), _probabilities(probabilities)
    if len(y) != len(p):
        raise ValueError("labels and probabilities must have the same length")
    return y, p


def _bin_count(n_bins: int) -> int:
    if isinstance(n_bins, bool) or not isinstance(n_bins, Integral) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    return int(n_bins)


def _losses(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Endpoints remain valid probabilities. Clipping only defines finite log
    # loss at exact 0/1; Brier score and calibration bins use the original p.
    clipped = np.clip(p, _EPS, 1.0 - _EPS)
    log_loss = -(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped))
    return log_loss, np.square(p - y)


def fit_calibrator(logits: Sequence[float], labels: Sequence[int]) -> dict[str, Any]:
    """Fit ``sigmoid(slope * logits + intercept)`` by validation log loss.

    L-BFGS-B constrains slope to [1e-6, 100] and intercept to [-100, 100].
    Single-class labels cannot identify a useful slope: return the Jeffreys
    smoothed constant ``(positive_count + 0.5) / (row_count + 1)`` instead.
    Empty inputs or failed optimization raise an error rather than creating an
    apparently fitted model. The result contains only JSON-compatible values.
    """
    z, y = _vector(logits, "logits"), _labels(labels)
    if len(z) != len(y):
        raise ValueError("logits and labels must have the same length")
    if not len(y):
        raise ValueError("Cannot fit a calibrator on empty inputs")
    positive_count = int(y.sum())
    common: dict[str, Any] = {
        "fit_rows": int(len(y)),
        "positive_rate": float(y.mean()),
    }
    if positive_count in (0, len(y)):
        probability = (positive_count + 0.5) / (len(y) + 1.0)
        intercept = math.log(probability) - math.log1p(-probability)
        return {
            **common,
            "method": "smoothed_constant",
            "reason": "single_class",
            "slope": 0.0,
            "intercept": float(intercept),
            "constant_probability": float(probability),
            "objective": float(np.logaddexp(0.0, (1.0 - 2.0 * y) * intercept).mean()),
            "converged": True,
            "n_iter": 0,
        }

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        transformed = parameters[0] * z + parameters[1]
        # This equivalent BCE expression avoids subtracting two large terms.
        loss = np.logaddexp(0.0, (1.0 - 2.0 * y) * transformed).mean()
        residual = expit(transformed) - y
        gradient = np.array([(residual * z).mean(), residual.mean()])
        return float(loss), gradient

    result = minimize(
        objective,
        x0=np.array([1.0, 0.0]),
        jac=True,
        method="L-BFGS-B",
        bounds=[(1e-6, 100.0), (-100.0, 100.0)],
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Platt calibration optimization failed: {result.message}")
    return {
        **common,
        "method": "positive_platt",
        "slope": float(result.x[0]),
        "intercept": float(result.x[1]),
        "objective": float(result.fun),
        "converged": bool(result.success),
        "n_iter": int(result.nit),
        "slope_bounds": [1e-6, 100.0],
        "intercept_bounds": [-100.0, 100.0],
    }


def apply_calibrator(
    logits: Sequence[float], calibrator: Mapping[str, Any]
) -> np.ndarray:
    """Apply a JSON-loaded calibrator; empty logits produce an empty vector."""
    z = _vector(logits, "logits")
    try:
        slope, intercept = float(calibrator["slope"]), float(calibrator["intercept"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("calibrator must contain finite slope and intercept") from exc
    if not math.isfinite(slope) or not math.isfinite(intercept) or slope < 0.0:
        raise ValueError("calibrator slope must be nonnegative and parameters finite")
    if slope == 0.0 and calibrator.get("method") != "smoothed_constant":
        raise ValueError("Only a smoothed_constant calibrator may have zero slope")
    with np.errstate(over="ignore"):
        return expit(slope * z + intercept)


def reliability_bins(
    labels: Sequence[int], probabilities: Sequence[float], n_bins: int = 10
) -> list[dict[str, Any]]:
    """Equal-width bins: left closed/right open, with 1 in the final bin.

    Empty bins (including all bins of an empty input) carry null averages.
    """
    y, p = _paired(labels, probabilities)
    n_bins = _bin_count(n_bins)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    counts = np.bincount(assignments, minlength=n_bins)
    predictions = np.bincount(assignments, weights=p, minlength=n_bins)
    positives = np.bincount(assignments, weights=y, minlength=n_bins)
    return [
        {
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": int(counts[index]),
            "mean_prediction": float(predictions[index] / counts[index]) if counts[index] else None,
            "positive_rate": float(positives[index] / counts[index]) if counts[index] else None,
        }
        for index in range(n_bins)
    ]


def binary_metrics(
    labels: Sequence[int], probabilities: Sequence[float], n_bins: int = 10
) -> dict[str, Any]:
    """Compute impression-weighted metrics without silently dropping rows.

    Empty inputs return rows=0 and null statistics. For a single label class,
    ROC AUC and PR AUC are null because discrimination cannot be assessed.
    ECE uses equal-width bins and each bin's fraction of all impressions.
    """
    y, p = _paired(labels, probabilities)
    bins = reliability_bins(y, p, n_bins=n_bins)
    metrics: dict[str, Any] = {
        "rows": int(len(y)),
        "positive_rate": None,
        "mean_prediction": None,
        "roc_auc": None,
        "pr_auc": None,
        "log_loss": None,
        "brier": None,
        "ece": None,
    }
    if not len(y):
        return metrics
    log_losses, brier_losses = _losses(y, p)
    metrics.update(
        positive_rate=float(y.mean()),
        mean_prediction=float(p.mean()),
        log_loss=float(log_losses.mean()),
        brier=float(brier_losses.mean()),
        ece=float(sum(
            entry["count"] * abs(entry["mean_prediction"] - entry["positive_rate"])
            for entry in bins if entry["count"]
        ) / len(y)),
    )
    positive_count = int(y.sum())
    negative_count = len(y) - positive_count
    if positive_count and negative_count:
        ranks = rankdata(p, method="average")
        metrics["roc_auc"] = float(
            (ranks[y == 1.0].sum() - positive_count * (positive_count + 1.0) / 2.0)
            / (positive_count * negative_count)
        )
        order = np.argsort(-p, kind="stable")
        ordered_y, ordered_p = y[order], p[order]
        group_ends = np.r_[np.flatnonzero(np.diff(ordered_p) != 0.0), len(y) - 1]
        cumulative_true = np.cumsum(ordered_y)[group_ends]
        precision = cumulative_true / (group_ends + 1.0)
        recall_increments = np.diff(np.r_[0.0, cumulative_true]) / positive_count
        metrics["pr_auc"] = float(np.sum(recall_increments * precision))
    return metrics


def paired_cluster_bootstrap(
    labels: Sequence[int],
    raw_p: Sequence[float],
    calibrated_p: Sequence[float],
    user_ids: Sequence[Any],
    repeats: int = 500,
    seed: int = 2026,
) -> dict[str, Any]:
    """95% paired user-cluster percentile CIs for calibrated-minus-raw losses.

    Sample users with replacement and retain all their impressions each time.
    Each replicate divides the sampled loss sums by its sampled row count,
    preserving impression weighting when users have unequal activity. The same
    sampled users are used for both predictors and both loss metrics. These CIs
    condition on the fitted model and calibrator; they do not include refitting
    uncertainty. Empty input is an error; one user gives a degenerate interval.
    """
    y, raw = _paired(labels, raw_p)
    calibrated = _probabilities(calibrated_p)
    users = np.asarray(user_ids, dtype=object)
    if users.ndim != 1 or len(users) != len(y) or len(calibrated) != len(y):
        raise ValueError("labels, predictions, and user_ids must have matching 1D lengths")
    if not len(y):
        raise ValueError("Cannot bootstrap empty inputs")
    if isinstance(repeats, bool) or not isinstance(repeats, Integral) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    user_index: dict[Any, int] = {}
    groups = np.empty(len(users), dtype=np.int64)
    for index, user in enumerate(users):
        if user is None or isinstance(user, (float, np.floating)) and not math.isfinite(user):
            raise ValueError("user_ids must be nonmissing, finite, hashable identifiers")
        try:
            groups[index] = user_index.setdefault(user, len(user_index))
        except TypeError as exc:
            raise ValueError("user_ids must be hashable identifiers") from exc
    n_users = len(user_index)
    counts = np.bincount(groups, minlength=n_users)
    raw_log, raw_brier = _losses(y, raw)
    cal_log, cal_brier = _losses(y, calibrated)
    deltas = np.stack([cal_log - raw_log, cal_brier - raw_brier], axis=1)
    user_sums = np.stack([
        np.bincount(groups, weights=deltas[:, index], minlength=n_users)
        for index in range(2)
    ], axis=1)
    rng = np.random.default_rng(seed)
    samples = np.empty((int(repeats), 2), dtype=np.float64)
    for index in range(int(repeats)):
        selected = rng.integers(0, n_users, size=n_users)
        samples[index] = user_sums[selected].sum(axis=0) / counts[selected].sum()
    limits = np.quantile(samples, [0.025, 0.975], axis=0)
    result: dict[str, Any] = {
        "method": "paired_user_cluster_percentile",
        "rows": int(len(y)),
        "users": int(n_users),
        "repeats": int(repeats),
        "seed": int(seed),
        "confidence_level": 0.95,
    }
    for index, name in enumerate(["log_loss", "brier"]):
        result[name] = {
            "estimate": float(deltas[:, index].mean()),
            "ci_lower": float(limits[0, index]),
            "ci_upper": float(limits[1, index]),
        }
    return result
