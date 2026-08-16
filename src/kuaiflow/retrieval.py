"""Training and evaluation pipeline for Week 2 retrieval models."""

from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import pandas as pd

from kuaiflow.data import Week1Splits, load_kuairand_features
from kuaiflow.metrics import build_ground_truth, evaluate_recommendations
from kuaiflow.models import TwoTowerRecommender


def _evaluation_users(
    ground_truth: dict[Any, set[Any]], max_users: int | None, seed: int
) -> list[Any]:
    users = list(ground_truth)
    if max_users is not None and len(users) > max_users:
        rng = np.random.default_rng(seed)
        positions = np.sort(rng.choice(len(users), size=max_users, replace=False))
        users = [users[position] for position in positions]
    return users


def run_week2_retrieval(
    splits: Week1Splits, config: dict[str, Any]
) -> dict[str, Any]:
    seed = int(config.get("seed", 2026))
    label_col = config["data"].get("positive_column", "is_click")
    k_values = sorted({int(k) for k in config["evaluation"]["k_values"]})
    if not k_values or min(k_values) <= 0:
        raise ValueError("evaluation.k_values must contain positive integers")
    max_users = config["evaluation"].get("max_users")
    catalog = splits.train["video_id"].drop_duplicates().tolist()
    training_positives = splits.train.loc[
        splits.train[label_col] > 0, ["user_id", "video_id"]
    ].drop_duplicates()
    training_seen = {
        user: set(group["video_id"].tolist())
        for user, group in training_positives.groupby("user_id", sort=False)
    }
    ground_truth = {
        "validation": build_ground_truth(
            splits.validation,
            label_col=label_col,
            catalog=catalog,
            exclude=training_seen,
        ),
        "test": build_ground_truth(
            splits.test,
            label_col=label_col,
            catalog=catalog,
            exclude=training_seen,
        ),
    }
    users = {
        "validation": _evaluation_users(ground_truth["validation"], max_users, seed),
        "test": _evaluation_users(ground_truth["test"], max_users, seed + 1),
    }

    experiment = config.get("experiment", {})
    variant = str(experiment.get("variant", "feature_history"))
    use_user_features = bool(experiment.get("use_user_features", True))
    use_video_features = bool(experiment.get("use_video_features", True))
    use_history = bool(experiment.get("use_history", True))
    user_features = video_features = None
    raw_dir = config["data"].get("raw_dir")
    if raw_dir and (use_user_features or use_video_features):
        loaded_users, loaded_videos = load_kuairand_features(raw_dir)
        user_features = loaded_users if use_user_features else None
        video_features = loaded_videos if use_video_features else None
    model = TwoTowerRecommender(
        seed=seed, use_history=use_history, **config.get("model", {})
    )
    started = time.perf_counter()
    model.fit(
        splits.train,
        label_col=label_col,
        user_features=user_features,
        video_features=video_features,
    )
    fit_seconds = time.perf_counter() - started

    results: dict[str, Any] = {
        "model": "two_tower",
        "variant": variant,
        "features": {
            "user_static": user_features is not None,
            "video_basic": video_features is not None,
            "causal_history": use_history,
            "max_history": model.max_history,
        },
        "label": label_col,
        "fit_seconds": fit_seconds,
        "training_loss": model.training_history,
        "splits": {},
    }
    largest_k = max(k_values)
    for split_name in ("validation", "test"):
        started = time.perf_counter()
        recommendations = model.recommend(users[split_name], k=largest_k)
        retrieval_seconds = time.perf_counter() - started
        split_results: dict[str, Any] = {
            "retrieval_seconds": retrieval_seconds,
            "milliseconds_per_user": (
                retrieval_seconds * 1000 / max(len(users[split_name]), 1)
            ),
            "metrics": {},
        }
        for k in k_values:
            split_results["metrics"][str(k)] = evaluate_recommendations(
                recommendations,
                ground_truth[split_name],
                k=k,
                catalog=catalog,
            )
        results["splits"][split_name] = split_results
    return results


def save_week2_results(results: dict[str, Any], artifacts_dir: str | Path) -> None:
    output = Path(artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)
    variant = re.sub(r"[^a-zA-Z0-9_-]+", "_", results.get("variant", "two_tower"))
    stem = f"week2_{variant}_results"
    with (output / f"{stem}.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    rows: list[dict[str, Any]] = []
    for split, split_result in results["splits"].items():
        for k, metrics in split_result["metrics"].items():
            rows.append(
                {
                    "model": results["model"],
                    "variant": results.get("variant", "two_tower"),
                    "user_static": results.get("features", {}).get("user_static", False),
                    "video_basic": results.get("features", {}).get("video_basic", False),
                    "causal_history": results.get("features", {}).get("causal_history", False),
                    "split": split,
                    "k": int(k),
                    "recall": metrics[f"recall@{k}"],
                    "hit_rate": metrics[f"hit_rate@{k}"],
                    "ndcg": metrics[f"ndcg@{k}"],
                    "coverage": metrics[f"coverage@{k}"],
                    "evaluated_users": metrics["evaluated_users"],
                    "milliseconds_per_user": split_result["milliseconds_per_user"],
                }
            )
    pd.DataFrame(rows).to_csv(output / f"{stem}.csv", index=False)
