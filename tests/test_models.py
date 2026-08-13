import unittest

from kuaiflow.models import BPRMatrixFactorization, ItemCFRecommender, PopularityRecommender
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


if __name__ == "__main__":
    unittest.main()

