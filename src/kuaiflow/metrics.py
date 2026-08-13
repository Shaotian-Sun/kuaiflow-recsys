"""Top-k metrics shared by every recommender baseline."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
import math

import pandas as pd


UserId = Hashable
ItemId = Hashable


def build_ground_truth(
    interactions: pd.DataFrame,
    label_col: str = "is_click",
    user_col: str = "user_id",
    item_col: str = "video_id",
    catalog: Iterable[ItemId] | None = None,
    exclude: Mapping[UserId, set[ItemId]] | None = None,
) -> dict[UserId, set[ItemId]]:
    positives = interactions.loc[interactions[label_col] > 0, [user_col, item_col]]
    allowed_items = set(catalog) if catalog is not None else None
    ground_truth: dict[UserId, set[ItemId]] = {}
    for user, group in positives.groupby(user_col, sort=False):
        items = set(group[item_col].tolist())
        if allowed_items is not None:
            items.intersection_update(allowed_items)
        if exclude is not None:
            items.difference_update(exclude.get(user, set()))
        if items:
            ground_truth[user] = items
    return ground_truth


def recall_at_k(recommended: Sequence[ItemId], relevant: set[ItemId], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(recommended[:k]).intersection(relevant)) / len(relevant)


def hit_rate_at_k(recommended: Sequence[ItemId], relevant: set[ItemId], k: int) -> float:
    return float(bool(set(recommended[:k]).intersection(relevant)))


def ndcg_at_k(recommended: Sequence[ItemId], relevant: set[ItemId], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 2.0)
        for rank, item in enumerate(recommended[:k])
        if item in relevant
    )
    ideal_length = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2.0) for rank in range(ideal_length))
    return dcg / idcg


def evaluate_recommendations(
    recommendations: Mapping[UserId, Sequence[ItemId]],
    ground_truth: Mapping[UserId, set[ItemId]],
    k: int,
    catalog: Iterable[ItemId],
) -> dict[str, float]:
    users = [user for user in recommendations if ground_truth.get(user)]
    if not users:
        raise ValueError("No evaluated user has a positive ground-truth item")

    recall = sum(recall_at_k(recommendations[u], ground_truth[u], k) for u in users)
    hit_rate = sum(hit_rate_at_k(recommendations[u], ground_truth[u], k) for u in users)
    ndcg = sum(ndcg_at_k(recommendations[u], ground_truth[u], k) for u in users)
    recommended_items = {
        item for user in users for item in recommendations[user][:k]
    }
    catalog_items = set(catalog)

    return {
        f"recall@{k}": recall / len(users),
        f"hit_rate@{k}": hit_rate / len(users),
        f"ndcg@{k}": ndcg / len(users),
        f"coverage@{k}": len(recommended_items) / max(len(catalog_items), 1),
        "evaluated_users": float(len(users)),
    }
