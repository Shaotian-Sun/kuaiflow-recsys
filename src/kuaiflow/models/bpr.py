"""A small NumPy implementation of Bayesian Personalized Ranking matrix factorization."""

from __future__ import annotations

from collections.abc import Hashable, Iterable

import numpy as np
import pandas as pd
from scipy.special import expit

from kuaiflow.models.common import make_id_map, positive_pairs, top_k_indices


class BPRMatrixFactorization:
    def __init__(
        self,
        factors: int = 64,
        learning_rate: float = 0.03,
        regularization: float = 1e-4,
        epochs: int = 20,
        batch_size: int = 2048,
        seed: int = 2026,
    ) -> None:
        if min(factors, epochs, batch_size) <= 0:
            raise ValueError("factors, epochs, and batch_size must be positive")
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def fit(
        self,
        interactions: pd.DataFrame,
        label_col: str = "is_click",
        user_col: str = "user_id",
        item_col: str = "video_id",
    ) -> "BPRMatrixFactorization":
        positives = positive_pairs(interactions, label_col, user_col, item_col)
        if positives.empty:
            raise ValueError("BPR requires at least one positive interaction")

        self.user_ids, self.user_to_index = make_id_map(interactions[user_col])
        self.item_ids, self.item_to_index = make_id_map(interactions[item_col])
        positive_users = positives[user_col].map(self.user_to_index).to_numpy(dtype=np.int64)
        positive_items = positives[item_col].map(self.item_to_index).to_numpy(dtype=np.int64)

        self.user_seen: dict[Hashable, set[int]] = {
            user: set(group[item_col].map(self.item_to_index).tolist())
            for user, group in positives.groupby(user_col, sort=False)
        }
        seen_by_index = {
            self.user_to_index[user]: items for user, items in self.user_seen.items()
        }
        eligible = np.array(
            [len(seen_by_index[user]) < len(self.item_ids) for user in positive_users],
            dtype=bool,
        )
        positive_users = positive_users[eligible]
        positive_items = positive_items[eligible]
        if positive_users.size == 0:
            raise ValueError("Every training user has interacted with every catalog item")

        rng = np.random.default_rng(self.seed)
        scale = 0.05
        self.user_factors = rng.normal(
            0.0, scale, size=(len(self.user_ids), self.factors)
        ).astype(np.float64)
        self.item_factors = rng.normal(
            0.0, scale, size=(len(self.item_ids), self.factors)
        ).astype(np.float64)
        self.popularity = np.bincount(positive_items, minlength=len(self.item_ids)).astype(float)

        pair_indices = np.arange(len(positive_users))
        for _ in range(self.epochs):
            rng.shuffle(pair_indices)
            for start in range(0, len(pair_indices), self.batch_size):
                batch = pair_indices[start : start + self.batch_size]
                users = positive_users[batch]
                positive = positive_items[batch]
                negative = rng.integers(0, len(self.item_ids), size=len(batch))

                collisions = np.array(
                    [neg in seen_by_index[user] for user, neg in zip(users, negative)]
                )
                while collisions.any():
                    negative[collisions] = rng.integers(
                        0, len(self.item_ids), size=int(collisions.sum())
                    )
                    collisions = np.array(
                        [neg in seen_by_index[user] for user, neg in zip(users, negative)]
                    )

                user_vectors = self.user_factors[users].copy()
                positive_vectors = self.item_factors[positive].copy()
                negative_vectors = self.item_factors[negative].copy()
                margins = np.sum(
                    user_vectors * (positive_vectors - negative_vectors), axis=1
                )
                gradient_weight = expit(-margins)[:, None]

                user_update = gradient_weight * (positive_vectors - negative_vectors)
                user_update -= self.regularization * user_vectors
                positive_update = gradient_weight * user_vectors
                positive_update -= self.regularization * positive_vectors
                negative_update = -gradient_weight * user_vectors
                negative_update -= self.regularization * negative_vectors

                np.add.at(self.user_factors, users, self.learning_rate * user_update)
                np.add.at(self.item_factors, positive, self.learning_rate * positive_update)
                np.add.at(self.item_factors, negative, self.learning_rate * negative_update)
        return self

    def recommend(self, user_ids: Iterable[Hashable], k: int) -> dict[Hashable, list[Hashable]]:
        if not hasattr(self, "item_factors"):
            raise RuntimeError("Call fit before recommend")
        output: dict[Hashable, list[Hashable]] = {}
        popularity_scale = max(float(self.popularity.max()), 1.0)
        fallback = self.popularity / popularity_scale

        for user in user_ids:
            if user in self.user_to_index:
                user_index = self.user_to_index[user]
                scores = self.user_factors[user_index] @ self.item_factors.T
                scores = scores + 1e-8 * fallback
            else:
                scores = fallback.copy()
            seen = self.user_seen.get(user, set())
            if seen:
                scores[np.fromiter(seen, dtype=np.int64)] = -np.inf
            selected = top_k_indices(scores, k)
            output[user] = [self.item_ids[index] for index in selected]
        return output

