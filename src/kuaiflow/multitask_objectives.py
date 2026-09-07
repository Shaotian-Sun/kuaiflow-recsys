"""Targets and losses used by the multi-task ranking model.

The transformations in this module are deliberately independent of the model so
that the label semantics are explicit and can be reused at evaluation time.  In
particular, :class:`DurationCompletionCurve` must be fitted on the training split
only; callers then reuse the frozen curve for validation, test, and candidate
scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from torch.nn import functional as F


# These are impression outcomes, not input features.  KuaiRand does not expose a
# per-impression collection/save label, so collection is intentionally absent.
DEFAULT_BINARY_TASKS: dict[str, str] = {
    "click": "is_click",
    "like": "is_like",
    "follow": "is_follow",
    "comment": "is_comment",
    "forward": "is_forward",
    "long_view": "long_view",
    "profile_enter": "is_profile_enter",
    "hate": "is_hate",
}


def _coerce_1d(values: Iterable[Any] | np.ndarray | pd.Series, name: str) -> np.ndarray:
    """Coerce one-dimensional, possibly nullable/string values to float64."""
    raw = np.asarray(values)
    if raw.ndim == 0:
        raw = raw.reshape(1)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {raw.shape}")
    return pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(
        dtype=np.float64, copy=True
    )


def _same_length(left: np.ndarray, right: np.ndarray, names: tuple[str, str]) -> None:
    if len(left) != len(right):
        raise ValueError(
            f"{names[0]} and {names[1]} must have the same length, "
            f"got {len(left)} and {len(right)}"
        )


def watch_time_targets(
    play_time_ms: Iterable[Any] | np.ndarray | pd.Series,
    tau_seconds: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build weighted-logistic targets for expected watch time.

    For non-negative watch time ``t`` and scale ``tau``, the returned arrays are

    ``target = (t / tau) / (1 + t / tau)`` and ``weight = 1 + t / tau``.

    Therefore, the weighted BCE is algebraically equivalent (up to a constant)
    to a likelihood whose optimal logit satisfies ``exp(logit) = E[t / tau]``.
    Missing, non-finite, or negative play times are safely treated as zero.

    Returns:
        ``(soft_target, weight, watch_seconds)``, all finite float32 arrays.
    """
    tau = float(tau_seconds)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau_seconds must be finite and greater than zero")

    milliseconds = _coerce_1d(play_time_ms, "play_time_ms")
    milliseconds = np.where(
        np.isfinite(milliseconds) & (milliseconds > 0.0), milliseconds, 0.0
    )

    # Bound the public float32 outputs to finite values.  Real KuaiRand watch
    # times are many orders of magnitude below this guard; it only protects bad
    # input from turning a whole training batch into NaNs.
    float32_max = float(np.finfo(np.float32).max)
    watch_seconds64 = np.minimum(milliseconds / 1000.0, float32_max)
    scaled = np.minimum(watch_seconds64 / tau, float32_max - 1.0)
    soft_target = 1.0 - 1.0 / (1.0 + scaled)
    weight = 1.0 + scaled
    return (
        soft_target.astype(np.float32),
        weight.astype(np.float32),
        watch_seconds64.astype(np.float32),
    )


def completion_targets(
    play_time_ms: Iterable[Any] | np.ndarray | pd.Series,
    duration_ms: Iterable[Any] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Return clipped watched fractions and a valid-duration mask.

    Completion is ``clip(play_time / duration, 0, 1)``.  Replays therefore stay
    at one, and rows with a missing, non-finite, or non-positive duration receive
    a target of zero and ``False`` in the returned mask.  Invalid or negative
    play times are treated as zero.
    """
    play = _coerce_1d(play_time_ms, "play_time_ms")
    duration = _coerce_1d(duration_ms, "duration_ms")
    _same_length(play, duration, ("play_time_ms", "duration_ms"))

    play = np.where(np.isfinite(play) & (play > 0.0), play, 0.0)
    mask = np.isfinite(duration) & (duration > 0.0)
    ratio = np.zeros(len(play), dtype=np.float64)
    np.divide(play, duration, out=ratio, where=mask)
    ratio = np.clip(ratio, 0.0, 1.0)
    return ratio.astype(np.float32), mask.astype(bool, copy=False)


def _validate_soft_bce_inputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits and targets must have the same shape, got "
            f"{tuple(logits.shape)} and {tuple(targets.shape)}"
        )
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    if not torch.isfinite(targets).all():
        raise ValueError("targets must be finite")
    if torch.any((targets < 0.0) | (targets > 1.0)):
        raise ValueError("soft BCE targets must lie in [0, 1]")
    return targets


def weighted_watch_time_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    mean_weight: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Stable weighted BCE for the weighted-logistic watch-time objective.

    ``mean_weight`` should be the mean weight over the full training split when
    this function is called on mini-batches.  The resulting
    ``mean(batch_weight * loss) / train_mean_weight`` is an unbiased stochastic
    estimate of the documented global weighted likelihood.  Omitting it computes
    the exact weighted mean over the tensors passed to this call.
    """
    targets = _validate_soft_bce_inputs(logits, targets)
    if logits.shape != weights.shape:
        raise ValueError(
            f"logits and weights must have the same shape, got "
            f"{tuple(logits.shape)} and {tuple(weights.shape)}"
        )
    weights = weights.to(device=logits.device, dtype=logits.dtype)
    if not torch.isfinite(weights).all() or torch.any(weights < 0.0):
        raise ValueError("weights must be finite and non-negative")
    if weights.numel() == 0 or not torch.any(weights > 0.0):
        raise ValueError("at least one watch-time weight must be positive")

    per_example = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    if mean_weight is None:
        denominator = weights.mean()
    else:
        denominator = torch.as_tensor(
            mean_weight, device=logits.device, dtype=logits.dtype
        )
        if denominator.numel() != 1:
            raise ValueError("mean_weight must be a scalar")
        if not torch.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("mean_weight must be finite and positive")

    # Scaling both numerator and denominator by the same detached value avoids
    # overflow without changing mean(weight * loss) / mean_weight.
    scale = weights.detach().max()
    normalized_weights = weights / scale
    return (normalized_weights * per_example).mean() / (denominator / scale)


def masked_soft_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean soft-label BCE over valid rows, or differentiable zero if none.

    Returning ``logits.sum() * 0`` for an empty mask lets random mini-batches with
    no valid video duration participate safely in a larger multi-task loss.
    """
    targets = _validate_soft_bce_inputs(logits, targets)
    if logits.shape != mask.shape:
        raise ValueError(
            f"logits and mask must have the same shape, got "
            f"{tuple(logits.shape)} and {tuple(mask.shape)}"
        )
    mask = mask.to(device=logits.device, dtype=torch.bool)
    if not torch.any(mask):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[mask], targets[mask])


# Short alias for callers that already make the logits convention clear.
masked_soft_bce = masked_soft_bce_with_logits


@dataclass
class DurationCompletionCurve:
    """Train-fitted baseline completion as a decreasing function of duration.

    The fit first summarizes ``log1p(duration_ms)`` into approximately equal-size
    quantile bins.  It then fits a decreasing isotonic regression to the bin
    means, weighted by their sample counts.  This gives short and long videos a
    smooth, monotone comparison baseline without allowing validation/test labels
    into the transformation.
    """

    n_bins: int = 50
    min_baseline: float = 1e-4
    log_duration_knots: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64), repr=False
    )
    completion_knots: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64), repr=False
    )
    bin_counts: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64), repr=False
    )
    global_mean: float | None = None
    training_rows: int = 0

    def __post_init__(self) -> None:
        if int(self.n_bins) < 1:
            raise ValueError("n_bins must be at least one")
        self.n_bins = int(self.n_bins)
        self.min_baseline = float(self.min_baseline)
        if not np.isfinite(self.min_baseline) or not 0.0 < self.min_baseline <= 1.0:
            raise ValueError("min_baseline must lie in (0, 1]")
        self.log_duration_knots = np.asarray(
            self.log_duration_knots, dtype=np.float64
        )
        self.completion_knots = np.asarray(
            self.completion_knots, dtype=np.float64
        )
        self.bin_counts = np.asarray(self.bin_counts, dtype=np.int64)
        if self.log_duration_knots.ndim != 1 or self.completion_knots.ndim != 1:
            raise ValueError("curve knots must be one-dimensional")
        if len(self.log_duration_knots) != len(self.completion_knots):
            raise ValueError("duration and completion knot counts must match")
        if self.bin_counts.ndim != 1:
            raise ValueError("bin_counts must be one-dimensional")

    @property
    def is_fitted(self) -> bool:
        return (
            self.global_mean is not None
            and self.training_rows > 0
            and len(self.log_duration_knots) > 0
        )

    def fit(
        self,
        duration_ms: Iterable[Any] | np.ndarray | pd.Series,
        completion: Iterable[Any] | np.ndarray | pd.Series,
        mask: Iterable[Any] | np.ndarray | pd.Series | None = None,
    ) -> "DurationCompletionCurve":
        """Fit the curve; the caller must pass training-split rows only."""
        duration = _coerce_1d(duration_ms, "duration_ms")
        target = _coerce_1d(completion, "completion")
        _same_length(duration, target, ("duration_ms", "completion"))

        valid = (
            np.isfinite(duration)
            & (duration > 0.0)
            & np.isfinite(target)
        )
        if mask is not None:
            requested = np.asarray(mask)
            if requested.ndim == 0:
                requested = requested.reshape(1)
            if requested.ndim != 1 or len(requested) != len(duration):
                raise ValueError("mask must be one-dimensional and match duration_ms")
            valid &= requested.astype(bool)
        if not valid.any():
            raise ValueError("cannot fit duration curve without any valid rows")

        x = np.log1p(duration[valid])
        y = np.clip(target[valid], 0.0, 1.0)
        self.training_rows = int(len(x))
        self.global_mean = float(np.clip(y.mean(), self.min_baseline, 1.0))

        # Repeated video durations make quantile edges duplicate.  Removing those
        # edges keeps every populated bin well-defined and ordered.
        quantiles = np.linspace(0.0, 1.0, min(self.n_bins, len(x)) + 1)
        edges = np.unique(np.quantile(x, quantiles))
        if len(edges) == 1:
            self.log_duration_knots = edges.astype(np.float64)
            self.completion_knots = np.asarray([self.global_mean], dtype=np.float64)
            self.bin_counts = np.asarray([len(x)], dtype=np.int64)
            return self

        assignments = np.searchsorted(edges[1:-1], x, side="right")
        counts = np.bincount(assignments).astype(np.int64)
        populated = counts > 0
        bin_x = np.bincount(assignments, weights=x)[populated] / counts[populated]
        bin_y = np.bincount(assignments, weights=y)[populated] / counts[populated]
        counts = counts[populated]

        if len(bin_x) == 1:
            fitted_x = bin_x
            fitted_y = np.asarray([self.global_mean], dtype=np.float64)
        else:
            isotonic = IsotonicRegression(
                increasing=False,
                out_of_bounds="clip",
                y_min=self.min_baseline,
                y_max=1.0,
            ).fit(bin_x, bin_y, sample_weight=counts)
            fitted_x = np.asarray(isotonic.X_thresholds_, dtype=np.float64)
            fitted_y = np.asarray(isotonic.y_thresholds_, dtype=np.float64)

        self.log_duration_knots = fitted_x
        self.completion_knots = np.clip(
            fitted_y, self.min_baseline, 1.0
        ).astype(np.float64)
        # Counts describe the pre-isotonic quantile summaries and are persisted
        # as fit provenance; threshold pooling may make their lengths differ.
        self.bin_counts = counts
        return self

    def predict(
        self,
        duration_ms: Iterable[Any] | np.ndarray | pd.Series,
    ) -> np.ndarray:
        """Predict a positive completion baseline, returning float32 values."""
        if not self.is_fitted:
            raise RuntimeError("DurationCompletionCurve must be fitted before predict")
        duration = _coerce_1d(duration_ms, "duration_ms")
        valid = np.isfinite(duration) & (duration > 0.0)
        prediction = np.full(len(duration), float(self.global_mean), dtype=np.float64)
        if valid.any():
            log_duration = np.log1p(duration[valid])
            prediction[valid] = np.interp(
                log_duration,
                self.log_duration_knots,
                self.completion_knots,
                left=self.completion_knots[0],
                right=self.completion_knots[-1],
            )
        return np.clip(prediction, self.min_baseline, 1.0).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of a fitted curve."""
        if not self.is_fitted:
            raise RuntimeError("cannot serialize an unfitted DurationCompletionCurve")
        return {
            "version": 1,
            "duration_unit": "milliseconds",
            "n_bins": self.n_bins,
            "min_baseline": self.min_baseline,
            "log_duration_knots": self.log_duration_knots.tolist(),
            "completion_knots": self.completion_knots.tolist(),
            "bin_counts": self.bin_counts.tolist(),
            "global_mean": float(self.global_mean),
            "training_rows": self.training_rows,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DurationCompletionCurve":
        """Restore a curve produced by :meth:`to_dict`."""
        if int(payload.get("version", 1)) != 1:
            raise ValueError(f"unsupported duration curve version: {payload['version']}")
        if payload.get("duration_unit", "milliseconds") != "milliseconds":
            raise ValueError("DurationCompletionCurve expects millisecond durations")
        curve = cls(
            n_bins=int(payload["n_bins"]),
            min_baseline=float(payload["min_baseline"]),
            log_duration_knots=np.asarray(
                payload["log_duration_knots"], dtype=np.float64
            ),
            completion_knots=np.asarray(payload["completion_knots"], dtype=np.float64),
            bin_counts=np.asarray(payload.get("bin_counts", []), dtype=np.int64),
            global_mean=float(payload["global_mean"]),
            training_rows=int(payload["training_rows"]),
        )
        if not curve.is_fitted:
            raise ValueError("serialized duration curve is not fitted")
        if not np.all(np.diff(curve.log_duration_knots) >= 0.0):
            raise ValueError("duration curve knots must be sorted")
        if not np.all(np.diff(curve.completion_knots) <= 1e-12):
            raise ValueError("duration curve must be non-increasing")
        if not np.all(
            (curve.completion_knots >= curve.min_baseline)
            & (curve.completion_knots <= 1.0)
        ):
            raise ValueError("duration curve completion knots are out of range")
        return curve


__all__ = [
    "DEFAULT_BINARY_TASKS",
    "DurationCompletionCurve",
    "completion_targets",
    "masked_soft_bce",
    "masked_soft_bce_with_logits",
    "watch_time_targets",
    "weighted_watch_time_bce",
]
