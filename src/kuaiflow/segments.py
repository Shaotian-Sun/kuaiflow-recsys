"""Segmented evaluation by user activity and item popularity."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
import math

import pandas as pd

from kuaiflow.metrics import evaluate_recommendations


def build_user_activity_groups(
    training_positives: pd.DataFrame,
) -> dict[Hashable, str]:
    counts = training_positives.groupby("user_id")["video_id"].nunique()
    low_cut, high_cut = counts.quantile([1 / 3, 2 / 3])

    def assign(count: int) -> str:
        if count <= low_cut:
            return "low"
        if count <= high_cut:
            return "medium"
        return "high"

    return {user: assign(count) for user, count in counts.items()}


def build_item_popularity_groups(
    training_positives: pd.DataFrame,
    catalog: Sequence[Hashable],
) -> dict[Hashable, str]:
    counts = (
        training_positives.groupby("video_id")["user_id"]
        .nunique()
        .reindex(catalog, fill_value=0)
    )
    tail_cut, head_cut = counts.quantile([0.5, 0.8])

    def assign(count: int) -> str:
        if count <= tail_cut:
            return "tail"
        if count <= head_cut:
            return "mid"
        return "head"

    return {item: assign(count) for item, count in counts.items()}


def evaluate_user_groups(
    recommendations,
    ground_truth,
    user_groups,
    k,
    catalog,
) -> list[dict[str, object]]:
    rows = []

    for group in ("zero_positive", "low", "medium", "high"):
        group_users = [
            user
            for user in recommendations
            if user_groups.get(user, "zero_positive") == group
            and ground_truth.get(user)
        ]
        if not group_users:
            continue

        group_recommendations = {
            user: recommendations[user] for user in group_users
        }
        metrics = evaluate_recommendations(
            group_recommendations,
            ground_truth,
            k=k,
            catalog=catalog,
        )
        rows.append({"group": group, **metrics})

    return rows


def evaluate_item_groups(
    recommendations,
    ground_truth,
    item_groups,
    k,
) -> list[dict[str, object]]:
    rows = []
    all_users = [
        user for user in recommendations if ground_truth.get(user)
    ]
    total_slots = sum(
        len(recommendations[user][:k]) for user in all_users
    )

    for group in ("tail", "mid", "head"):
        group_catalog = {
            item for item, label in item_groups.items() if label == group
        }
        group_truth = {
            user: relevant.intersection(group_catalog)
            for user, relevant in ground_truth.items()
        }
        group_truth = {
            user: relevant
            for user, relevant in group_truth.items()
            if relevant and user in recommendations
        }

        if not group_truth:
            continue

        recall = 0.0
        hit_rate = 0.0
        ndcg = 0.0

        for user, relevant in group_truth.items():
            ranked = recommendations[user][:k]
            hits = set(ranked).intersection(relevant)

            recall += len(hits) / len(relevant)
            hit_rate += float(bool(hits))

            dcg = sum(
                1.0 / math.log2(rank + 2.0)
                for rank, item in enumerate(ranked)
                if item in relevant
            )
            ideal_length = min(len(relevant), k)
            idcg = sum(
                1.0 / math.log2(rank + 2.0)
                for rank in range(ideal_length)
            )
            ndcg += dcg / idcg

        recommended_in_group = [
            item
            for user in all_users
            for item in recommendations[user][:k]
            if item in group_catalog
        ]
        n_users = len(group_truth)

        rows.append(
            {
                "group": group,
                f"recall@{k}": recall / n_users,
                f"hit_rate@{k}": hit_rate / n_users,
                f"ndcg@{k}": ndcg / n_users,
                "recommendation_share": (
                    len(recommended_in_group) / max(total_slots, 1)
                ),
                "group_coverage": (
                    len(set(recommended_in_group))
                    / max(len(group_catalog), 1)
                ),
                "evaluated_users": n_users,
                "catalog_items": len(group_catalog),
            }
        )

    return rows