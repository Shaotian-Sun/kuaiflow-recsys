import unittest

import pandas as pd

from kuaiflow.metrics import (
    build_ground_truth,
    evaluate_recommendations,
    ndcg_at_k,
    recall_at_k,
)


class MetricTests(unittest.TestCase):
    def test_recall(self) -> None:
        self.assertEqual(recall_at_k([1, 2, 3], {2, 4}, 2), 0.5)

    def test_ndcg_perfect_ranking(self) -> None:
        self.assertAlmostEqual(ndcg_at_k([2, 4, 1], {2, 4}, 2), 1.0)

    def test_aggregate_metrics(self) -> None:
        result = evaluate_recommendations(
            {0: [1, 2], 1: [3, 4]},
            {0: {2}, 1: {5}},
            k=2,
            catalog=range(1, 6),
        )
        self.assertEqual(result["hit_rate@2"], 0.5)
        self.assertEqual(result["coverage@2"], 0.8)

    def test_ground_truth_is_warm_start_and_novel(self) -> None:
        frame = pd.DataFrame(
            [(0, 10, 1), (0, 11, 1), (0, 99, 1), (1, 12, 0)],
            columns=["user_id", "video_id", "is_click"],
        )
        truth = build_ground_truth(
            frame,
            catalog={10, 11, 12},
            exclude={0: {10}},
        )
        self.assertEqual(truth, {0: {11}})


if __name__ == "__main__":
    unittest.main()
