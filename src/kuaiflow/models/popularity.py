"""Non-personalized popularity baseline."""

from __future__ import annotations

from collections.abc import Hashable, Iterable

import pandas as pd

from kuaiflow.models.common import positive_pairs


class PopularityRecommender:
    def fit(
        self,
        interactions: pd.DataFrame,
        label_col: str = "is_click",
        user_col: str = "user_id",
        item_col: str = "video_id",
    ) -> "PopularityRecommender":
        self.user_col = user_col
        self.item_col = item_col
        positives = positive_pairs(interactions, label_col, user_col, item_col)
        catalog = interactions[item_col].drop_duplicates().tolist()
        counts = positives[item_col].value_counts().to_dict()
        self.ranking = sorted(catalog, key=lambda item: (-counts.get(item, 0), str(item)))
        self.seen = {
            user: set(group[item_col].tolist())
            for user, group in positives.groupby(user_col, sort=False)
        }
        return self

    def recommend(self, user_ids: Iterable[Hashable], k: int) -> dict[Hashable, list[Hashable]]:
        if not hasattr(self, "ranking"):
            raise RuntimeError("Call fit before recommend")
        output: dict[Hashable, list[Hashable]] = {}
        for user in user_ids:
            seen = self.seen.get(user, set())
            output[user] = [item for item in self.ranking if item not in seen][:k]
        return output

