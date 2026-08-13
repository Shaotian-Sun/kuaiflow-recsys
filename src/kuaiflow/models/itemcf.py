"""Item-based collaborative filtering with cosine nearest neighbors."""

from __future__ import annotations

from collections.abc import Hashable, Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from kuaiflow.models.common import make_id_map, positive_pairs, top_k_indices


class ItemCFRecommender:
    def __init__(self, neighbor_k: int = 100) -> None:
        if neighbor_k <= 0:
            raise ValueError("neighbor_k must be positive")
        self.neighbor_k = neighbor_k

    def fit(
        self,
        interactions: pd.DataFrame,
        label_col: str = "is_click",
        user_col: str = "user_id",
        item_col: str = "video_id",
    ) -> "ItemCFRecommender":
        positives = positive_pairs(interactions, label_col, user_col, item_col)
        if positives.empty:
            raise ValueError("ItemCF requires at least one positive interaction")

        self.user_ids, self.user_to_index = make_id_map(interactions[user_col])
        self.item_ids, self.item_to_index = make_id_map(interactions[item_col])
        rows = positives[user_col].map(self.user_to_index).to_numpy()
        cols = positives[item_col].map(self.item_to_index).to_numpy()
        matrix = csr_matrix(
            (np.ones(len(positives), dtype=np.float32), (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids)),
        )
        self.user_seen = {
            user: set(group[item_col].map(self.item_to_index).tolist())
            for user, group in positives.groupby(user_col, sort=False)
        }
        self.popularity = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)

        item_user = matrix.T.tocsr()
        n_neighbors = min(self.neighbor_k + 1, len(self.item_ids))
        nearest = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
        nearest.fit(item_user)
        distances, indices = nearest.kneighbors(item_user, return_distance=True)

        self.neighbor_indices: list[np.ndarray] = []
        self.neighbor_similarities: list[np.ndarray] = []
        for item_index, (item_distances, item_indices) in enumerate(zip(distances, indices)):
            keep = item_indices != item_index
            self.neighbor_indices.append(item_indices[keep].astype(np.int64))
            self.neighbor_similarities.append(
                np.clip(1.0 - item_distances[keep], 0.0, 1.0).astype(np.float64)
            )
        return self

    def recommend(self, user_ids: Iterable[Hashable], k: int) -> dict[Hashable, list[Hashable]]:
        if not hasattr(self, "item_ids"):
            raise RuntimeError("Call fit before recommend")
        output: dict[Hashable, list[Hashable]] = {}
        popularity_scale = max(float(self.popularity.max()), 1.0)
        popularity_tie_break = 1e-8 * self.popularity / popularity_scale

        for user in user_ids:
            scores = popularity_tie_break.copy()
            seen = self.user_seen.get(user, set())
            for item_index in seen:
                np.add.at(
                    scores,
                    self.neighbor_indices[item_index],
                    self.neighbor_similarities[item_index],
                )
            if seen:
                scores[np.fromiter(seen, dtype=np.int64)] = -np.inf
            selected = top_k_indices(scores, k)
            output[user] = [self.item_ids[index] for index in selected]
        return output

