import unittest

import numpy as np
import pandas as pd

from kuaiflow.models import (
    BPRMatrixFactorization,
    ItemCFRecommender,
    PopularityRecommender,
    TwoTowerRecommender,
)
from kuaiflow.toy import make_toy_splits


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = make_toy_splits().train

    def _assert_valid_recommendations(self, model) -> None:
        model.fit(self.train)
        recommendations = model.recommend([0, 1, 2, 3], k=2)
        for user, items in recommendations.items():
            seen = set(
                self.train.loc[
                    (self.train.user_id == user) & (self.train.is_click > 0), "video_id"
                ]
            )
            self.assertTrue(set(items).isdisjoint(seen))
            self.assertLessEqual(len(items), 2)

    def test_popularity(self) -> None:
        self._assert_valid_recommendations(PopularityRecommender())

    def test_itemcf(self) -> None:
        self._assert_valid_recommendations(ItemCFRecommender(neighbor_k=3))

    def test_bpr(self) -> None:
        self._assert_valid_recommendations(
            BPRMatrixFactorization(factors=8, epochs=3, batch_size=4, seed=1)
        )

    def test_two_tower(self) -> None:
        model = TwoTowerRecommender(
            embedding_dim=8,
            hidden_dim=16,
            learning_rate=0.01,
            epochs=20,
            batch_size=4,
            seed=1,
        )
        self._assert_valid_recommendations(model)
        self.assertEqual(len(model.training_history), 20)
        self.assertTrue(all(np.isfinite(loss) for loss in model.training_history))
        self.assertEqual(model.item_vectors.shape, (4, 8))

    def test_two_tower_uses_features_and_causal_history(self) -> None:
        users = pd.DataFrame({
            "user_id": [0, 1, 2, 3],
            "user_active_degree": ["high", "low", "high", "low"],
            "register_days": [100, 200, 300, 400],
        })
        videos = pd.DataFrame({
            "video_id": [10, 11, 12, 13],
            "author_id": [1, 1, 2, 2],
            "video_type": ["NORMAL"] * 4,
            "video_duration": [10, 20, 30, 40],
        })
        model = TwoTowerRecommender(
            embedding_dim=8, hidden_dim=16, feature_dim=4, max_history=2,
            learning_rate=0.01, epochs=2, batch_size=4, seed=1,
        ).fit(self.train, user_features=users, video_features=videos)
        self.assertEqual(model.user_categorical.shape, (4, 1))
        self.assertEqual(model.item_categorical.shape, (4, 2))
        self.assertTrue(all(len(history) <= 2 for history in model.user_history.values()))
        self.assertEqual(model.recommend([0], 2)[0].__len__(), 2)


if __name__ == "__main__":
    unittest.main()
