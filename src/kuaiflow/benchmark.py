"""Train and compare all week-one baselines through one evaluation path."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from kuaiflow.data import Week1Splits
from kuaiflow.metrics import build_ground_truth, evaluate_recommendations
from kuaiflow.models import BPRMatrixFactorization, ItemCFRecommender, PopularityRecommender


def _evaluation_users(
    ground_truth: dict[Any, set[Any]],
    max_users: int | None,
    seed: int,
) -> list[Any]:
    users = list(ground_truth)
    if max_users is not None and len(users) > max_users:
        rng = np.random.default_rng(seed)
        positions = np.sort(rng.choice(len(users), size=max_users, replace=False))
        users = [users[position] for position in positions]
    return users


def _make_model(name: str, config: dict[str, Any], seed: int) -> Any:
    if name == "popularity":
        return PopularityRecommender()
    if name == "itemcf":
        return ItemCFRecommender(**config.get("itemcf", {}))
    if name == "bpr":
        return BPRMatrixFactorization(seed=seed, **config.get("bpr", {}))
    raise ValueError(f"Unknown model: {name}")


def run_week1_benchmark(
    splits: Week1Splits,
    config: dict[str, Any],
    selected_models: list[str] | None = None,
) -> dict[str, Any]:
    label_col = config["data"].get("positive_column", "is_click")
    k = int(config["evaluation"].get("k", 20))
    max_users = config["evaluation"].get("max_users")
    seed = int(config.get("seed", 2026))
    catalog = splits.train["video_id"].drop_duplicates().tolist()
    selected_models = selected_models or ["popularity", "itemcf", "bpr"]

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

    results: dict[str, Any] = {
        "label": label_col,
        "k": k,
        "models": {},
    }
    for name in selected_models:
        model = _make_model(name, config.get("models", {}), seed)
        started = time.perf_counter()
        model.fit(splits.train, label_col=label_col)
        fit_seconds = time.perf_counter() - started

        model_results: dict[str, Any] = {"fit_seconds": fit_seconds}
        for split_name in ("validation", "test"):
            recommendations = model.recommend(users[split_name], k=k)
            model_results[split_name] = evaluate_recommendations(
                recommendations,
                ground_truth[split_name],
                k=k,
                catalog=catalog,
            )
        results["models"][name] = model_results
    return results


def save_benchmark_results(results: dict[str, Any], artifacts_dir: str | Path) -> None:
    output = Path(artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "week1_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    rows: list[dict[str, Any]] = []
    for model, model_result in results["models"].items():
        for split in ("validation", "test"):
            rows.append({"model": model, "split": split, **model_result[split]})
    pd.DataFrame(rows).to_csv(output / "week1_results.csv", index=False)
