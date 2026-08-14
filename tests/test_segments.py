import unittest

from kuaiflow.segments import (
    evaluate_item_groups,
    evaluate_user_groups,
)


class SegmentTests(unittest.TestCase):
    def test_segment_evaluation_is_complete_and_valid(self) -> None:
        recommendations = {
            0: [10, 12],
            1: [11, 13],
            2: [12, 13],
        }
        ground_truth = {
            0: {10},
            1: {13},
            2: {12},
        }

        # User 2 is deliberately absent and should become zero_positive.
        user_groups = {
            0: "low",
            1: "high",
        }
        user_rows = evaluate_user_groups(
            recommendations,
            ground_truth,
            user_groups,
            k=2,
            catalog=[10, 11, 12, 13],
        )
        rows_by_group = {
            row["group"]: row for row in user_rows
        }

        self.assertIn("zero_positive", rows_by_group)
        self.assertEqual(
            rows_by_group["zero_positive"]["evaluated_users"],
            1.0,
        )
        self.assertEqual(
            sum(row["evaluated_users"] for row in user_rows),
            3.0,
        )

        item_groups = {
            10: "head",
            11: "head",
            12: "mid",
            13: "tail",
        }
        item_rows = evaluate_item_groups(
            recommendations,
            ground_truth,
            item_groups,
            k=2,
        )

        self.assertEqual(
            {row["group"] for row in item_rows},
            {"head", "mid", "tail"},
        )
        self.assertAlmostEqual(
            sum(row["recommendation_share"] for row in item_rows),
            1.0,
        )

        for row in item_rows:
            self.assertGreaterEqual(row["group_coverage"], 0.0)
            self.assertLessEqual(row["group_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()