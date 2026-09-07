"""DeepFM + MMoE multi-task training and Week 2 candidate reranking.

The logged impression table supplies labels.  The Week 2 top-100 table is used
only after training, when the frozen model scores the same candidate pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import csv
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from kuaiflow.data import Week1Splits, load_kuairand_features
from kuaiflow.metrics import build_ground_truth, evaluate_recommendations
from kuaiflow.models.mmoe import DeepFMMMoE
from kuaiflow.multitask_objectives import (
    DEFAULT_BINARY_TASKS,
    DurationCompletionCurve,
    completion_targets,
    masked_soft_bce_with_logits,
    watch_time_targets,
    weighted_watch_time_bce,
)
from kuaiflow.ranking import (
    DeepFMFeatures,
    _attach_static_features,
    _classification_metrics,
    _validate_candidates,
)


WATCH_TASK = "watch_time"
COMPLETION_TASK = "completion"
SPECIAL_TASKS = (WATCH_TASK, COMPLETION_TASK)

# None of these values exists when an unexposed Week 2 candidate is scored.
POST_EXPOSURE_OR_POLICY_COLUMNS = {
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "play_time_ms",
    "duration_ms",
    "profile_stay_time",
    "comment_stay_time",
    "is_profile_enter",
    "retrieval_rank",
    "retrieval_score",
    "split",
    "date",
    "hourmin",
    "time_ms",
    "is_rand",
    "tab",
}

# The KuaiRand video-statistics table aggregates a future-overlapping month and
# has no as-of timestamp.  It must not be smuggled in as a request-time feature.
UNSAFE_AGGREGATE_FEATURES = {
    "show_cnt",
    "show_user_num",
    "play_cnt",
    "play_user_num",
    "like_cnt",
    "like_user_num",
    "follow_cnt",
    "follow_user_num",
    "comment_cnt",
    "comment_user_num",
    "forward_cnt",
    "forward_user_num",
    "collect_cnt",
    "collect_user_num",
    "cancel_like_cnt",
    "cancel_collect_cnt",
}

# This first strict baseline intentionally accepts only immutable/basic item
# fields.  New features must be reviewed and added explicitly with an as-of-time
# contract instead of slipping through a never-complete blacklist.
STRICT_FEATURE_ALLOWLIST = {
    "user_id",
    "video_id",
    "author_id",
    "video_type",
    "upload_type",
    "music_id",
    "music_type",
    "video_duration",
    "server_width",
    "server_height",
}


@dataclass(frozen=True)
class MultiTaskTargets:
    """All labels derived from one logged-impression split."""

    binary: dict[str, np.ndarray]
    watch_soft: np.ndarray
    watch_weight: np.ndarray
    watch_seconds: np.ndarray
    completion: np.ndarray
    completion_mask: np.ndarray


@dataclass
class MMoERun:
    results: dict[str, Any]
    model: DeepFMMMoE
    encoder: DeepFMFeatures
    duration_curve: DurationCompletionCurve
    reranked_candidates: pd.DataFrame
    metadata: dict[str, Any]


def _parse_binary_tasks(config: dict[str, Any]) -> dict[str, str]:
    configured = config.get("tasks", {}).get("binary", DEFAULT_BINARY_TASKS)
    if not isinstance(configured, dict) or not configured:
        raise ValueError("tasks.binary must map task names to label columns")
    tasks = {str(name): str(column) for name, column in configured.items()}
    if len(tasks) != len(set(tasks.values())):
        raise ValueError("Every binary task must use a distinct label column")
    collection_terms = (
        "collect",
        "collection",
        "save",
        "favorite",
        "favourite",
        "bookmark",
    )
    requested_collection = [
        name
        for name, column in tasks.items()
        if any(
            term in "".join(
                character
                for character in f"{name} {column}".lower()
                if character.isalnum()
            )
            for term in collection_terms
        )
    ]
    if requested_collection:
        raise ValueError(
            "KuaiRand-Pure has no per-impression collection label; "
            "monthly collection aggregates cannot substitute for one"
        )
    reserved = set(tasks).intersection(SPECIAL_TASKS)
    if reserved:
        raise ValueError(f"Binary task names are reserved: {sorted(reserved)}")
    return tasks


def _parse_negative_tasks(
    config: dict[str, Any],
    binary_tasks: dict[str, str],
) -> set[str]:
    configured = config.get("tasks", {}).get("negative")
    if configured is None:
        negative = {
            name for name, column in binary_tasks.items() if column == "is_hate"
        }
    elif isinstance(configured, list):
        negative = {str(name) for name in configured}
    else:
        raise ValueError("tasks.negative must be a list of binary task names")
    unknown = negative.difference(binary_tasks)
    if unknown:
        raise ValueError(f"Negative tasks are not binary tasks: {sorted(unknown)}")
    return negative


def _strict_binary_target(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        if "collect" in column.lower():
            raise ValueError(
                "KuaiRand-Pure has no per-impression collection label; "
                "monthly collect_cnt must not be used as a target"
            )
        raise ValueError(f"Multi-task ranking data is missing target {column!r}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isin(values, [0.0, 1.0]).all():
        raise ValueError(f"Binary target {column!r} must contain only 0 and 1")
    return values.astype(np.float32)


def _make_targets(
    frame: pd.DataFrame,
    binary_tasks: dict[str, str],
    tau_seconds: float,
) -> MultiTaskTargets:
    missing = {"play_time_ms", "duration_ms"}.difference(frame.columns)
    if missing:
        raise ValueError(f"Multi-task ranking data is missing: {sorted(missing)}")
    binary = {
        name: _strict_binary_target(frame, column)
        for name, column in binary_tasks.items()
    }
    watch_soft, watch_weight, watch_seconds = watch_time_targets(
        frame["play_time_ms"], tau_seconds=tau_seconds
    )
    completion, completion_mask = completion_targets(
        frame["play_time_ms"], frame["duration_ms"]
    )
    return MultiTaskTargets(
        binary=binary,
        watch_soft=watch_soft,
        watch_weight=watch_weight,
        watch_seconds=watch_seconds,
        completion=completion,
        completion_mask=completion_mask,
    )


def _soft_bce_numpy(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, logits) - targets * logits


def _null_model_statistics(
    targets: MultiTaskTargets,
    binary_tasks: dict[str, str],
    tau_seconds: float,
) -> dict[str, dict[str, float]]:
    """Loss and bias of a featureless constant predictor for each task.

    Dividing by these fixed train-only losses changes task gradient scales but
    not any task's statistical optimum.  It prevents the rare follow/forward/
    hate objectives from disappearing merely because their entropy is small.
    """

    output: dict[str, dict[str, float]] = {}
    epsilon = 1e-7
    for name in binary_tasks:
        rate = float(targets.binary[name].mean())
        probability = float(np.clip(rate, epsilon, 1.0 - epsilon))
        logit = math.log(probability / (1.0 - probability))
        loss = float(_soft_bce_numpy(
            np.full(len(targets.binary[name]), logit), targets.binary[name]
        ).mean())
        output[name] = {
            "constant_prediction": rate,
            "constant_logit": logit,
            "loss": max(loss, epsilon),
        }

    mean_watch = float(targets.watch_seconds.mean())
    watch_logit = math.log(max(mean_watch / tau_seconds, epsilon))
    watch_losses = _soft_bce_numpy(
        np.full(len(targets.watch_soft), watch_logit), targets.watch_soft
    )
    watch_loss = float(
        np.sum(targets.watch_weight.astype(np.float64) * watch_losses)
        / np.sum(targets.watch_weight.astype(np.float64))
    )
    output[WATCH_TASK] = {
        "constant_prediction_seconds": mean_watch,
        "constant_logit": watch_logit,
        "mean_example_weight": float(targets.watch_weight.mean()),
        "loss": max(watch_loss, epsilon),
    }

    valid_completion = targets.completion[targets.completion_mask]
    if not len(valid_completion):
        raise ValueError("Training data has no valid positive video durations")
    mean_completion = float(valid_completion.mean())
    probability = float(np.clip(mean_completion, epsilon, 1.0 - epsilon))
    completion_logit = math.log(probability / (1.0 - probability))
    completion_loss = float(_soft_bce_numpy(
        np.full(len(valid_completion), completion_logit), valid_completion
    ).mean())
    output[COMPLETION_TASK] = {
        "constant_prediction_fraction": mean_completion,
        "constant_logit": completion_logit,
        "loss": max(completion_loss, epsilon),
    }
    return output


def _configured_weights(
    configured: Any,
    task_names: list[str],
    *,
    allow_negative: bool,
    defaults: dict[str, float] | None = None,
) -> dict[str, float]:
    base = (
        {name: float(defaults[name]) for name in task_names}
        if defaults is not None
        else {name: 1.0 for name in task_names}
    )
    if configured is None:
        weights = base
    elif isinstance(configured, dict):
        unknown = set(configured).difference(task_names)
        if unknown:
            raise ValueError(f"Weights contain unknown tasks: {sorted(unknown)}")
        weights = {
            name: float(configured.get(name, base[name])) for name in task_names
        }
    else:
        raise ValueError("Task weights must be a mapping")
    if not all(np.isfinite(value) for value in weights.values()):
        raise ValueError("Task weights must be finite")
    if not allow_negative and any(value < 0.0 for value in weights.values()):
        raise ValueError("Training loss weights cannot be negative")
    if not any(value != 0.0 for value in weights.values()):
        raise ValueError("At least one task weight must be non-zero")
    return weights


def _initialize_task_biases(
    model: DeepFMMMoE,
    null_statistics: dict[str, dict[str, float]],
) -> None:
    with torch.no_grad():
        for index, name in enumerate(model.task_names):
            model.task_biases[index].fill_(null_statistics[name]["constant_logit"])


def _batch_loss(
    logits: torch.Tensor,
    batch: np.ndarray,
    targets: MultiTaskTargets,
    task_index: dict[str, int],
    null_statistics: dict[str, dict[str, float]],
    loss_weights: dict[str, float],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    raw: dict[str, torch.Tensor] = {}
    for name in targets.binary:
        label = torch.as_tensor(
            targets.binary[name][batch], dtype=torch.float32, device=device
        )
        raw[name] = F.binary_cross_entropy_with_logits(
            logits[:, task_index[name]], label
        )
    watch_soft = torch.as_tensor(
        targets.watch_soft[batch], dtype=torch.float32, device=device
    )
    watch_weight = torch.as_tensor(
        targets.watch_weight[batch], dtype=torch.float32, device=device
    )
    raw[WATCH_TASK] = weighted_watch_time_bce(
        logits[:, task_index[WATCH_TASK]],
        watch_soft,
        watch_weight,
        mean_weight=null_statistics[WATCH_TASK]["mean_example_weight"],
    )
    completion = torch.as_tensor(
        targets.completion[batch], dtype=torch.float32, device=device
    )
    completion_mask = torch.as_tensor(
        targets.completion_mask[batch], dtype=torch.bool, device=device
    )
    raw[COMPLETION_TASK] = masked_soft_bce_with_logits(
        logits[:, task_index[COMPLETION_TASK]], completion, completion_mask
    )

    active_weight = sum(loss_weights.values())
    total = sum(
        loss_weights[name] * raw[name] / null_statistics[name]["loss"]
        for name in task_index
    ) / active_weight
    return total, raw


def _predict_logits(
    model: DeepFMMMoE,
    categorical: np.ndarray,
    numeric: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(categorical), batch_size):
            rows.append(
                model(
                    torch.as_tensor(
                        categorical[start : start + batch_size],
                        dtype=torch.long,
                        device=device,
                    ),
                    torch.as_tensor(
                        numeric[start : start + batch_size],
                        dtype=torch.float32,
                        device=device,
                    ),
                ).cpu().numpy()
            )
    if not rows:
        return np.empty((0, len(model.task_names)), dtype=np.float32)
    return np.concatenate(rows, axis=0)


def _raw_losses_numpy(
    logits: np.ndarray,
    targets: MultiTaskTargets,
    task_index: dict[str, int],
) -> dict[str, float]:
    output = {
        name: float(_soft_bce_numpy(
            logits[:, task_index[name]], labels
        ).mean())
        for name, labels in targets.binary.items()
    }
    watch_loss = _soft_bce_numpy(
        logits[:, task_index[WATCH_TASK]], targets.watch_soft
    )
    output[WATCH_TASK] = float(
        np.sum(targets.watch_weight.astype(np.float64) * watch_loss)
        / np.sum(targets.watch_weight.astype(np.float64))
    )
    mask = targets.completion_mask
    output[COMPLETION_TASK] = float(_soft_bce_numpy(
        logits[mask, task_index[COMPLETION_TASK]], targets.completion[mask]
    ).mean())
    return output


def _normalized_total_loss(
    raw_losses: dict[str, float],
    null_statistics: dict[str, dict[str, float]],
    loss_weights: dict[str, float],
) -> float:
    return float(
        sum(
            loss_weights[name] * raw_losses[name] / null_statistics[name]["loss"]
            for name in loss_weights
        )
        / sum(loss_weights.values())
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _watch_predictions(
    logits: np.ndarray,
    tau_seconds: float,
    prediction_cap_seconds: float,
) -> np.ndarray:
    upper = math.log(prediction_cap_seconds / tau_seconds)
    return (
        tau_seconds * np.exp(np.clip(logits.astype(np.float64), -30.0, upper))
    ).astype(np.float32)


def _pointwise_metrics(
    logits: np.ndarray,
    targets: MultiTaskTargets,
    task_index: dict[str, int],
    tau_seconds: float,
    prediction_cap_seconds: float,
) -> dict[str, Any]:
    binary = {
        name: _classification_metrics(labels, logits[:, task_index[name]])
        for name, labels in targets.binary.items()
    }
    watch_prediction = _watch_predictions(
        logits[:, task_index[WATCH_TASK]], tau_seconds, prediction_cap_seconds
    )
    watch_error = watch_prediction.astype(np.float64) - targets.watch_seconds
    watch_raw_loss = _raw_losses_numpy(logits, targets, task_index)[WATCH_TASK]
    mask = targets.completion_mask
    completion_prediction = _sigmoid(logits[:, task_index[COMPLETION_TASK]])
    completion_error = completion_prediction[mask] - targets.completion[mask]
    completion_loss = float(_soft_bce_numpy(
        logits[mask, task_index[COMPLETION_TASK]], targets.completion[mask]
    ).mean())
    return {
        "binary": binary,
        WATCH_TASK: {
            "examples": len(targets.watch_seconds),
            "actual_mean_seconds": float(targets.watch_seconds.mean()),
            "predicted_mean_seconds": float(watch_prediction.mean()),
            "mae_seconds": float(np.mean(np.abs(watch_error))),
            "rmse_seconds": float(np.sqrt(np.mean(np.square(watch_error)))),
            "log1p_mae": float(np.mean(np.abs(
                np.log1p(watch_prediction) - np.log1p(targets.watch_seconds)
            ))),
            "weighted_logistic_loss": watch_raw_loss,
        },
        COMPLETION_TASK: {
            "examples": int(mask.sum()),
            "masked_rows": int((~mask).sum()),
            "actual_mean_fraction": float(targets.completion[mask].mean()),
            "predicted_mean_fraction": float(completion_prediction[mask].mean()),
            "mae_fraction": float(np.mean(np.abs(completion_error))),
            "rmse_fraction": float(np.sqrt(np.mean(np.square(completion_error)))),
            "soft_binary_cross_entropy": completion_loss,
        },
    }


def _recommendations_by_score(
    frame: pd.DataFrame,
    score_column: str,
    *,
    descending: bool = True,
) -> dict[Any, list[Any]]:
    ordered = frame[["user_id", "video_id", "retrieval_rank", score_column]].sort_values(
        ["user_id", score_column, "retrieval_rank"],
        ascending=[True, not descending, True],
        kind="stable",
    )
    return {
        user: group["video_id"].tolist()
        for user, group in ordered.groupby("user_id", sort=False)
    }


def _recommendations_by_rank(frame: pd.DataFrame) -> dict[Any, list[Any]]:
    ordered = frame[["user_id", "video_id", "retrieval_rank"]].sort_values(
        ["user_id", "retrieval_rank"], kind="stable"
    )
    return {
        user: group["video_id"].tolist()
        for user, group in ordered.groupby("user_id", sort=False)
    }


def _candidate_metrics(
    candidates: pd.DataFrame,
    splits: Week1Splits,
    binary_tasks: dict[str, str],
    negative_tasks: set[str],
    k_values: list[int],
) -> dict[str, Any]:
    catalog = splits.train["video_id"].drop_duplicates().tolist()
    training_seen = {
        name: {
            user: set(group["video_id"].tolist())
            for user, group in splits.train.loc[
                splits.train[column] > 0, ["user_id", "video_id"]
            ].groupby("user_id", sort=False)
        }
        for name, column in binary_tasks.items()
    }
    output: dict[str, Any] = {}
    for split_name, interactions in (
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        split_candidates = candidates.loc[candidates["split"] == split_name]
        retrieval = _recommendations_by_rank(split_candidates)
        composite = _recommendations_by_score(split_candidates, "mmoe_score")
        split_output: dict[str, Any] = {
            "candidate_users": len(retrieval),
            "candidate_rows": len(split_candidates),
            "targets": {},
        }
        for name, column in binary_tasks.items():
            ground_truth = build_ground_truth(
                interactions,
                label_col=column,
                catalog=catalog,
                exclude=training_seen[name],
            )
            is_negative_feedback = name in negative_tasks
            head = _recommendations_by_score(
                split_candidates,
                f"p_{name}",
                descending=not is_negative_feedback,
            )
            target_output: dict[str, Any] = {
                "ground_truth_users": len(ground_truth),
                "candidate_ground_truth_users": len(
                    set(retrieval).intersection(ground_truth)
                ),
                "direction": "lower_is_better" if is_negative_feedback else "higher_is_better",
                "retrieval_order": {},
                "mmoe_composite_order": {},
                "task_head_order": {},
            }
            if not set(retrieval).intersection(ground_truth):
                target_output["status"] = "no_candidate_user_has_positive_ground_truth"
                split_output["targets"][name] = target_output
                continue
            for k in k_values:
                baseline = evaluate_recommendations(retrieval, ground_truth, k, catalog)
                reranked = evaluate_recommendations(composite, ground_truth, k, catalog)
                task_ranked = evaluate_recommendations(head, ground_truth, k, catalog)
                target_output["retrieval_order"][str(k)] = baseline
                target_output["mmoe_composite_order"][str(k)] = reranked
                target_output["task_head_order"][str(k)] = task_ranked
                sign = -1.0 if is_negative_feedback else 1.0
                target_output.setdefault("composite_ndcg_improvement", {})[str(k)] = (
                    sign * (reranked[f"ndcg@{k}"] - baseline[f"ndcg@{k}"])
                )
                target_output.setdefault("task_head_ndcg_improvement", {})[str(k)] = (
                    sign * (task_ranked[f"ndcg@{k}"] - baseline[f"ndcg@{k}"])
                )
            split_output["targets"][name] = target_output
        output[split_name] = split_output
    return output


def _parameter_group_counts(model: DeepFMMMoE) -> dict[str, int]:
    groups = {
        "shared_feature_embeddings": 0,
        "task_first_order_and_fm": 0,
        "experts": 0,
        "task_gates": 0,
        "task_towers": 0,
    }
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        if name.startswith(("feature_embeddings", "numeric_embeddings")):
            groups["shared_feature_embeddings"] += count
        elif name.startswith("experts"):
            groups["experts"] += count
        elif name.startswith("gates"):
            groups["task_gates"] += count
        elif name.startswith("towers"):
            groups["task_towers"] += count
        else:
            groups["task_first_order_and_fm"] += count
    groups["total"] = sum(parameter.numel() for parameter in model.parameters())
    return groups


def _utility_components(
    candidate_outputs: dict[str, np.ndarray],
    binary_tasks: dict[str, str],
    watch_reference_seconds: float,
    duration_baseline: np.ndarray,
) -> dict[str, np.ndarray]:
    components = {name: candidate_outputs[f"p_{name}"] for name in binary_tasks}
    watch = candidate_outputs["estimated_watch_seconds"]
    components[WATCH_TASK] = np.clip(
        np.log1p(watch) / np.log1p(watch_reference_seconds), 0.0, 1.0
    )
    predicted = candidate_outputs["estimated_completion_fraction"]
    # sigmoid(log(predicted / baseline)) == predicted / (predicted + baseline).
    # This is bounded for utility fusion; the unbounded ratio is still exported.
    components[COMPLETION_TASK] = predicted / (
        predicted + duration_baseline + 1e-8
    )
    return components


def run_mmoe_ranking(
    splits: Week1Splits,
    config: dict[str, Any],
    candidates: pd.DataFrame | None = None,
    user_features: pd.DataFrame | None = None,
    video_features: pd.DataFrame | None = None,
) -> MMoERun:
    """Train all MMoE heads jointly, then score the stored Week 2 top 100."""

    seed = int(config.get("seed", 2026))
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    evaluation_config = config.get("evaluation", {})
    ranking_config = config.get("ranking", {})
    binary_tasks = _parse_binary_tasks(config)
    negative_tasks = _parse_negative_tasks(config, binary_tasks)
    task_names = list(binary_tasks) + list(SPECIAL_TASKS)

    categorical = list(data_config.get("categorical_features", ["user_id", "video_id"]))
    numeric = list(data_config.get("numeric_features", []))
    feature_names = categorical + numeric
    forbidden = (
        POST_EXPOSURE_OR_POLICY_COLUMNS
        | UNSAFE_AGGREGATE_FEATURES
        | set(binary_tasks.values())
    )
    leaked = forbidden.intersection(feature_names)
    if leaked:
        raise ValueError(
            f"Post-exposure, policy, or future-aggregate fields cannot be features: "
            f"{sorted(leaked)}"
        )
    unapproved = set(feature_names).difference(STRICT_FEATURE_ALLOWLIST)
    if unapproved:
        raise ValueError(
            "Features outside the strict request-time allowlist cannot be features: "
            f"{sorted(unapproved)}"
        )

    tau_seconds = float(training_config.get("watch_time_tau_seconds", 1.0))
    if not np.isfinite(tau_seconds) or tau_seconds <= 0.0:
        raise ValueError("training.watch_time_tau_seconds must be positive")
    prediction_cap_seconds = float(
        evaluation_config.get("watch_time_prediction_cap_seconds", 1200.0)
    )
    if not np.isfinite(prediction_cap_seconds) or prediction_cap_seconds <= 0.0:
        raise ValueError("watch_time_prediction_cap_seconds must be positive")

    # Validate all labels before doing expensive feature work.
    targets = {
        name: _make_targets(frame, binary_tasks, tau_seconds)
        for name, frame in (
            ("train", splits.train),
            ("validation", splits.validation),
            ("test", splits.test),
        )
    }
    no_completion = [
        name for name, split_targets in targets.items()
        if not split_targets.completion_mask.any()
    ]
    if no_completion:
        raise ValueError(
            "Completion evaluation requires at least one valid duration in every "
            f"split; none found in {no_completion}"
        )

    if user_features is None or video_features is None:
        raw_dir = data_config.get("raw_dir")
        needs_static = any(name not in {"user_id", "video_id"} for name in feature_names)
        if needs_static and not raw_dir and user_features is None and video_features is None:
            raise ValueError("data.raw_dir is required for configured static features")
        if raw_dir:
            loaded_users, loaded_videos = load_kuairand_features(raw_dir)
            user_features = loaded_users if user_features is None else user_features
            video_features = loaded_videos if video_features is None else video_features

    if candidates is None:
        candidate_path = data_config.get("candidates_path")
        if not candidate_path:
            raise ValueError("data.candidates_path is required")
        candidates = pd.read_csv(candidate_path)
    candidate_k = int(evaluation_config.get("candidate_k", 100))
    expected_users = evaluation_config.get("expected_users_per_split")
    _validate_candidates(
        candidates,
        candidate_k,
        int(expected_users) if expected_users is not None else None,
    )
    duration_available = "video_duration" in candidates.columns or (
        video_features is not None and "video_duration" in video_features.columns
    )
    if not duration_available:
        raise ValueError(
            "Candidate scoring requires basic-video `video_duration`; provide "
            "data.raw_dir, video_features, or a candidate duration column"
        )

    train_features = _attach_static_features(
        splits.train, user_features, video_features, feature_names
    )
    encoder = DeepFMFeatures().fit(train_features, categorical, numeric)
    encoded = {"train": encoder.transform(train_features)}
    del train_features
    for name, frame in (("validation", splits.validation), ("test", splits.test)):
        split_features = _attach_static_features(
            frame, user_features, video_features, feature_names
        )
        encoded[name] = encoder.transform(split_features)
        del split_features

    duration_curve = DurationCompletionCurve(
        n_bins=int(ranking_config.get("duration_curve_bins", 50))
    ).fit(
        splits.train["duration_ms"],
        targets["train"].completion,
        targets["train"].completion_mask,
    )
    watch_reference_quantile = float(
        ranking_config.get("watch_time_reference_quantile", 0.95)
    )
    if not 0.0 < watch_reference_quantile <= 1.0:
        raise ValueError("watch_time_reference_quantile must be in (0, 1]")
    watch_reference_seconds = max(
        float(np.quantile(targets["train"].watch_seconds, watch_reference_quantile)),
        tau_seconds,
    )

    null_statistics = _null_model_statistics(
        targets["train"], binary_tasks, tau_seconds
    )
    loss_weights = _configured_weights(
        training_config.get("loss_weights"), task_names, allow_negative=False
    )
    if sum(loss_weights.values()) <= 0.0:
        raise ValueError("The sum of training loss weights must be positive")
    default_utility = {name: 1.0 for name in task_names}
    for name in negative_tasks:
        default_utility[name] = -1.0
    utility_weights = _configured_weights(
        ranking_config.get("utility_weights", default_utility),
        task_names,
        allow_negative=True,
        defaults=default_utility,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(model_config.get("device", "cpu"))
    model = DeepFMMMoE(
        cardinalities=encoder.cardinalities,
        numeric_dim=len(numeric),
        task_names=task_names,
        embedding_dim=int(model_config.get("embedding_dim", 16)),
        num_experts=int(model_config.get("num_experts", 4)),
        expert_hidden_dims=tuple(model_config.get("expert_hidden_dims", [128, 64])),
        tower_hidden_dim=int(model_config.get("tower_hidden_dim", 32)),
        dropout=float(model_config.get("dropout", 0.1)),
    ).to(device)
    _initialize_task_biases(model, null_statistics)

    learning_rate = float(training_config.get("learning_rate", 1e-3))
    weight_decay = float(training_config.get("weight_decay", 1e-6))
    batch_size = int(training_config.get("batch_size", 2048))
    epochs = int(training_config.get("epochs", 5))
    patience = int(training_config.get("early_stopping_patience", 2))
    if min(batch_size, epochs) <= 0 or patience < 0:
        raise ValueError("batch_size and epochs must be positive and patience non-negative")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    task_index = {name: index for index, name in enumerate(task_names)}
    train_categorical, train_numeric = encoded["train"]
    order = np.arange(len(train_categorical))
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        rng.shuffle(order)
        model.train()
        total_loss_sum = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            logits = model(
                torch.as_tensor(train_categorical[batch], dtype=torch.long, device=device),
                torch.as_tensor(train_numeric[batch], dtype=torch.float32, device=device),
            )
            loss, _ = _batch_loss(
                logits,
                batch,
                targets["train"],
                task_index,
                null_statistics,
                loss_weights,
                device,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss_sum += float(loss.detach()) * len(batch)

        validation_logits = _predict_logits(
            model, *encoded["validation"], batch_size, device
        )
        validation_raw = _raw_losses_numpy(
            validation_logits, targets["validation"], task_index
        )
        validation_loss = _normalized_total_loss(
            validation_raw, null_statistics, loss_weights
        )
        history.append(
            {
                "epoch": epoch,
                "training_normalized_loss": total_loss_sum / len(order),
                "validation_normalized_loss": validation_loss,
                "validation_raw_losses": validation_raw,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if patience and stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - started

    pointwise: dict[str, Any] = {}
    for split_name in ("validation", "test"):
        logits = _predict_logits(model, *encoded[split_name], batch_size, device)
        pointwise[split_name] = _pointwise_metrics(
            logits,
            targets[split_name],
            task_index,
            tau_seconds,
            prediction_cap_seconds,
        )
        raw = _raw_losses_numpy(logits, targets[split_name], task_index)
        pointwise[split_name]["raw_losses"] = raw
        pointwise[split_name]["normalized_total_loss"] = _normalized_total_loss(
            raw, null_statistics, loss_weights
        )

    # Score in chunks so the one-million-row handoff does not need an extra
    # full encoded feature matrix in memory.
    encoded.pop("train")
    candidate_chunk_size = int(evaluation_config.get("candidate_chunk_size", 100_000))
    if candidate_chunk_size <= 0:
        raise ValueError("evaluation.candidate_chunk_size must be positive")
    candidate_outputs = {
        f"p_{name}": np.empty(len(candidates), dtype=np.float32)
        for name in binary_tasks
    }
    candidate_outputs["estimated_watch_seconds"] = np.empty(
        len(candidates), dtype=np.float32
    )
    candidate_outputs["estimated_completion_fraction"] = np.empty(
        len(candidates), dtype=np.float32
    )
    candidate_duration = np.empty(len(candidates), dtype=np.float64)
    scoring_features = list(dict.fromkeys(feature_names + ["video_duration"]))
    for start in range(0, len(candidates), candidate_chunk_size):
        stop = min(start + candidate_chunk_size, len(candidates))
        chunk = _attach_static_features(
            candidates.iloc[start:stop], user_features, video_features, scoring_features
        )
        chunk_logits = _predict_logits(
            model, *encoder.transform(chunk), batch_size, device
        )
        for name in binary_tasks:
            candidate_outputs[f"p_{name}"][start:stop] = _sigmoid(
                chunk_logits[:, task_index[name]]
            )
        candidate_outputs["estimated_watch_seconds"][start:stop] = _watch_predictions(
            chunk_logits[:, task_index[WATCH_TASK]],
            tau_seconds,
            prediction_cap_seconds,
        )
        candidate_outputs["estimated_completion_fraction"][start:stop] = _sigmoid(
            chunk_logits[:, task_index[COMPLETION_TASK]]
        )
        duration_values = pd.to_numeric(
            chunk["video_duration"], errors="coerce"
        ).to_numpy(dtype=float)
        # KuaiRand uses missing basic-video duration for a small item subset.
        # Store the same explicit zero sentinel used by the interaction logs;
        # the frozen duration curve falls back to its train-only global mean.
        candidate_duration[start:stop] = np.where(
            np.isfinite(duration_values) & (duration_values > 0.0),
            duration_values,
            0.0,
        )

    duration_baseline = duration_curve.predict(candidate_duration)
    components = _utility_components(
        candidate_outputs,
        binary_tasks,
        watch_reference_seconds,
        duration_baseline,
    )
    utility_scale = sum(abs(value) for value in utility_weights.values())
    utility_score = sum(
        utility_weights[name] * components[name] for name in task_names
    ) / utility_scale

    reranked = candidates.copy()
    for name, values in candidate_outputs.items():
        reranked[name] = values
    reranked["video_duration_ms"] = candidate_duration
    reranked["duration_baseline_fraction"] = duration_baseline
    reranked["completion_lift"] = (
        candidate_outputs["estimated_completion_fraction"] / duration_baseline
    )
    reranked["completion_log_lift"] = np.log(
        candidate_outputs["estimated_completion_fraction"] + 1e-8
    ) - np.log(duration_baseline + 1e-8)
    reranked["mmoe_score"] = utility_score.astype(np.float32)
    reranked = reranked.sort_values(
        ["split", "user_id", "mmoe_score", "retrieval_rank"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    reranked["mmoe_rank"] = (
        reranked.groupby(["split", "user_id"], sort=False).cumcount() + 1
    )
    reranked = reranked.sort_values(
        ["split", "user_id", "mmoe_rank"], kind="stable"
    ).reset_index(drop=True)

    k_values = sorted({
        int(k) for k in evaluation_config.get("k_values", [10, 20, 50, 100])
    })
    if not k_values or min(k_values) <= 0 or max(k_values) > candidate_k:
        raise ValueError("evaluation.k_values must be between 1 and candidate_k")
    candidate_ranking = _candidate_metrics(
        reranked, splits, binary_tasks, negative_tasks, k_values
    )

    metadata = {
        "binary_tasks": binary_tasks,
        "negative_tasks": sorted(negative_tasks),
        "task_names": task_names,
        "watch_time_tau_seconds": tau_seconds,
        "watch_time_prediction_cap_seconds": prediction_cap_seconds,
        "watch_time_reference_seconds": watch_reference_seconds,
        "utility_weights": utility_weights,
    }
    results = {
        "week": 3,
        "model": "deepfm_mmoe",
        "seed": seed,
        "training_examples": len(splits.train),
        "candidate_source": str(data_config.get("candidates_path", "in-memory")),
        "candidate_k": candidate_k,
        "features": {
            "categorical": categorical,
            "numeric": numeric,
            "outcomes_used_as_features": False,
            "retrieval_rank_used_as_feature": False,
            "monthly_aggregate_rates_used": False,
        },
        "tasks": {
            "binary": binary_tasks,
            "negative": sorted(negative_tasks),
            "watch_time": {
                "source": "play_time_ms",
                "soft_target": "(t / tau) / (1 + t / tau)",
                "example_weight": "1 + t / tau",
                "estimate_seconds": "tau * exp(logit)",
                "tau_seconds": tau_seconds,
            },
            "completion": {
                "source": "clip(play_time_ms / duration_ms, 0, 1)",
                "invalid_duration": "masked",
                "duration_adjustment": "prediction / train_fitted_f(duration)",
            },
            "collection": {
                "used": False,
                "reason": "no per-impression collection label in KuaiRand-Pure",
            },
        },
        "architecture": {
            "embedding_dim": model.embedding_dim,
            "num_experts": model.num_experts,
            "expert_hidden_dims": list(model.expert_hidden_dims),
            "tower_hidden_dim": model.tower_hidden_dim,
            "task_logit": "task_linear + task_fm_scale * shared_fm + task_mmoe_tower",
            "parameter_groups": _parameter_group_counts(model),
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "dropout": float(model_config.get("dropout", 0.1)),
            "loss_weights": loss_weights,
            "loss_normalization": "divide each loss by its train-only constant-predictor loss",
            "null_model_statistics": null_statistics,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "training_seconds": training_seconds,
            "history": history,
        },
        "ranking_policy": {
            "utility_weights": utility_weights,
            "watch_component": "clip(log1p(predicted_seconds) / log1p(train_quantile), 0, 1)",
            "watch_reference_quantile": watch_reference_quantile,
            "watch_reference_seconds": watch_reference_seconds,
            "completion_component": "predicted / (predicted + train_fitted_f(duration))",
            "note": "utility weights are an explicit policy choice, not learned labels",
        },
        "duration_curve": duration_curve.to_dict(),
        "pointwise": pointwise,
        "candidate_ranking": candidate_ranking,
    }
    return MMoERun(
        results, model, encoder, duration_curve, reranked, metadata
    )


def save_mmoe_run(
    run: MMoERun,
    config: dict[str, Any],
    artifacts_dir: str | Path,
    reranked_candidates_path: str | Path,
) -> None:
    """Persist the model, preprocessing, fitted curve, metrics, and top 100."""

    output = Path(artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)
    reranked_output = Path(reranked_candidates_path)
    reranked_output.parent.mkdir(parents=True, exist_ok=True)
    run.reranked_candidates.to_csv(reranked_output, index=False, compression="gzip")

    encoder_path = output / "week3_mmoe_encoder.json"
    curve_path = output / "week3_mmoe_duration_curve.json"
    model_path = output / "week3_mmoe_model.pt"
    with encoder_path.open("w", encoding="utf-8") as handle:
        json.dump(run.encoder.to_dict(), handle, indent=2)
    with curve_path.open("w", encoding="utf-8") as handle:
        json.dump(run.duration_curve.to_dict(), handle, indent=2)
    torch.save(
        {
            "state_dict": run.model.state_dict(),
            "model_config": config.get("model", {}),
            "categorical_features": run.encoder.categorical,
            "numeric_features": run.encoder.numeric,
            "cardinalities": run.encoder.cardinalities,
            "metadata": run.metadata,
        },
        model_path,
    )

    results = copy.deepcopy(run.results)
    results["outputs"] = {
        "reranked_candidates": str(reranked_output),
        "model": str(model_path),
        "encoder": str(encoder_path),
        "duration_curve": str(curve_path),
    }
    with (output / "week3_mmoe_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    rows: list[dict[str, Any]] = []
    for split, split_result in results["candidate_ranking"].items():
        for target, target_result in split_result["targets"].items():
            for ordering in (
                "retrieval_order",
                "mmoe_composite_order",
                "task_head_order",
            ):
                for k, metrics in target_result[ordering].items():
                    if ordering == "retrieval_order":
                        improvement = 0.0
                    elif ordering == "mmoe_composite_order":
                        improvement = target_result[
                            "composite_ndcg_improvement"
                        ][k]
                    else:
                        improvement = target_result[
                            "task_head_ndcg_improvement"
                        ][k]
                    rows.append(
                        {
                            "split": split,
                            "target": target,
                            "direction": target_result["direction"],
                            "ordering": ordering,
                            "k": int(k),
                            "recall": metrics[f"recall@{k}"],
                            "hit_rate": metrics[f"hit_rate@{k}"],
                            "ndcg": metrics[f"ndcg@{k}"],
                            "coverage": metrics[f"coverage@{k}"],
                            "evaluated_users": metrics["evaluated_users"],
                            "signed_ndcg_improvement": improvement,
                        }
                    )
    keys = list(rows[0]) if rows else [
        "split", "target", "direction", "ordering", "k", "recall",
        "hit_rate", "ndcg", "coverage", "evaluated_users",
        "signed_ndcg_improvement",
    ]
    with (output / "week3_mmoe_ranking_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_mmoe_artifacts(
    model_path: str | Path,
    encoder_path: str | Path,
    duration_curve_path: str | Path,
    device: str = "cpu",
) -> tuple[DeepFMMMoE, DeepFMFeatures, DurationCompletionCurve, dict[str, Any]]:
    """Restore the exact trained model and every train-fitted transformation."""

    with Path(encoder_path).open(encoding="utf-8") as handle:
        encoder = DeepFMFeatures.from_dict(json.load(handle))
    with Path(duration_curve_path).open(encoding="utf-8") as handle:
        duration_curve = DurationCompletionCurve.from_dict(json.load(handle))
    checkpoint = torch.load(
        model_path, map_location=torch.device(device), weights_only=True
    )
    model_config = checkpoint["model_config"]
    metadata = checkpoint["metadata"]
    model = DeepFMMMoE(
        cardinalities=checkpoint["cardinalities"],
        numeric_dim=len(checkpoint["numeric_features"]),
        task_names=metadata["task_names"],
        embedding_dim=int(model_config.get("embedding_dim", 16)),
        num_experts=int(model_config.get("num_experts", 4)),
        expert_hidden_dims=tuple(model_config.get("expert_hidden_dims", [128, 64])),
        tower_hidden_dim=int(model_config.get("tower_hidden_dim", 32)),
        dropout=float(model_config.get("dropout", 0.1)),
    ).to(torch.device(device))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, encoder, duration_curve, metadata


__all__ = [
    "MMoERun",
    "MultiTaskTargets",
    "load_mmoe_artifacts",
    "run_mmoe_ranking",
    "save_mmoe_run",
]
