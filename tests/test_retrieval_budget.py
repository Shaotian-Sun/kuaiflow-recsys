import itertools
import unittest
from kuaiflow.retrieval_budget import oracle_ndcg, interleave, validate_candidates
from kuaiflow.metrics import ndcg_at_k


class RetrievalBudgetTests(unittest.TestCase):
    def test_oracle_matches_exhaustive_ordering_and_full_denominator(self):
        candidates = [1, 2, 3, 4]
        truth = {1, 3, 5, 6}
        observed = oracle_ndcg({7: candidates}, {7: truth}, k=3)
        expected = max(ndcg_at_k(list(order), truth, 3) for order in itertools.permutations(candidates))
        self.assertAlmostEqual(observed, expected)
        self.assertLess(observed, 1)
        self.assertEqual(oracle_ndcg({7: [8, 9]}, {7: truth}, 3), 0)
        self.assertEqual(oracle_ndcg({7: [1, 3, 5]}, {7: truth}, 3), 1)

    def test_oracle_monotonic_and_all_users_in_denominator(self):
        truth = {1: {9}, 2: {8}}
        self.assertEqual(oracle_ndcg({1: [9], 2: [7]}, truth), 0.5)
        self.assertLessEqual(oracle_ndcg({1: [7], 2: [7]}, truth), oracle_ndcg({1: [7, 9], 2: [7, 8]}, truth))

    def test_interleave_deduplicates_without_exceeding_budget(self):
        result = interleave({1: [1, 2, 3, 4]}, {1: [1, 5, 2, 6]}, 4)
        self.assertEqual(result, {1: [1, 2, 5, 3]})
        validate_candidates(result, {1: {7}}, set(range(1, 8)), 4)
        with self.assertRaisesRegex(ValueError, 'Invalid candidate'):
            validate_candidates(result, {1: {2}}, set(range(1, 8)), 4)
        with self.assertRaisesRegex(ValueError, 'Invalid candidate'):
            validate_candidates({1: [1, 1, 2, 3]}, {}, set(range(1, 8)), 4)


class StrictRetrievalHistoryTests(unittest.TestCase):
    def test_tied_clicks_never_enter_each_others_history(self):
        import numpy as np
        from kuaiflow.models.two_tower import _strict_histories
        history, final = _strict_histories([1, 1, 1, 1, 2], np.array([4, 5, 6, 7, 8]), [1, 2, 2, 3, 2], 2)
        np.testing.assert_array_equal(history, [[-1, -1], [-1, 4], [-1, 4], [5, 6], [-1, -1]])
        self.assertEqual(final[1], [4, 5, 6, 7])
        # Shuffling input within tied events cannot affect per-event histories.
        permutation = np.array([2, 0, 4, 3, 1])
        shuffled, _ = _strict_histories(np.array([1, 1, 1, 1, 2])[permutation], np.array([4, 5, 6, 7, 8])[permutation], np.array([1, 2, 2, 3, 2])[permutation], 2)
        np.testing.assert_array_equal(shuffled, history[permutation])
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            _strict_histories([1], np.array([2]), [float('nan')], 2)

    def test_fit_requires_timestamps_when_history_enabled(self):
        import pandas as pd
        from kuaiflow.models.two_tower import TwoTowerRecommender
        data = pd.DataFrame({'user_id': [1, 1], 'video_id': [2, 3], 'is_click': [1, 1]})
        with self.assertRaisesRegex(ValueError, 'require timestamps'):
            TwoTowerRecommender(epochs=1).fit(data)
