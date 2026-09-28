import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from kuaiflow.reranking import DiversityMetadata, diversity_metrics, mmr_order, rerank_candidates, select_strength
from kuaiflow.serving import Week5Pipeline


class RerankingTests(unittest.TestCase):
    def setUp(self):
        self.metadata = DiversityMetadata(pd.DataFrame({
            'video_id': [10, 11, 12, 13], 'author_id': [1, 1, 2, -1], 'tag': ['1,2', '1,2', '3', None]}))
        self.frame = pd.DataFrame({'user_id': [7]*4, 'video_id': [10, 11, 12, 13],
                                   'final_rank': [1, 2, 3, 4]})

    def test_zero_strength_exactly_preserves_order(self):
        ranked = rerank_candidates(self.frame.sample(frac=1, random_state=1), self.metadata, 0, 3)
        self.assertEqual(ranked.video_id.tolist(), [10, 11, 12])
        self.assertEqual(ranked.final_rank.tolist(), [1, 2, 3])

    def test_diversity_changes_order_without_adding_items(self):
        ranked = rerank_candidates(self.frame, self.metadata, .8, 3)
        self.assertEqual(ranked.video_id.tolist(), [10, 12, 13])
        self.assertEqual(ranked.base_rank.tolist(), [1, 3, 4])
        self.assertEqual(len(set(ranked.video_id)), 3)
        self.assertTrue(set(ranked.video_id) <= set(self.frame.video_id))

    def test_missing_metadata_is_not_shared_similarity(self):
        matrix, tags, _, _ = self.metadata.similarities([10, 11, 12, 13])
        self.assertEqual(matrix[0, 1], 1)
        self.assertEqual(matrix[0, 2], 0)
        self.assertEqual(matrix[3, 3], 0)
        metrics = diversity_metrics({7: [10, 11, 12, 13]}, self.metadata)
        self.assertAlmostEqual(metrics['tag_ild'], 2/3, places=6)
        self.assertAlmostEqual(metrics['unique_author_fraction'], 2/3)
        self.assertEqual(metrics['tag_metadata_fraction'], .75)

    def test_validation_floor_can_reject_most_diverse_choice(self):
        rows = [{'strength': w, 'metrics': {'ndcg@20': n, 'tag_ild': d}}
                for w, n, d in [(0, .1, .6), (.1, .099, .7), (.2, .09, .9)]]
        self.assertEqual(select_strength(rows, .02)['strength'], .1)

    def test_invalid_candidate_set_or_parameters_fail(self):
        with self.assertRaises(ValueError):
            rerank_candidates(pd.concat([self.frame, self.frame]), self.metadata, .2, 2)
        for strength in [-1, 1, float('nan')]:
            with self.assertRaises(ValueError):
                mmr_order(np.eye(3), strength, 2)
        with self.assertRaises(ValueError):
            mmr_order(np.eye(3), 0, 4)

    def test_greedy_prefix_and_ties_are_deterministic(self):
        similarity = np.ones((4, 4))
        self.assertEqual(mmr_order(similarity, .5, 4)[0], [0, 1, 2, 3])
        self.assertEqual(mmr_order(similarity, .5, 2)[0], mmr_order(similarity, .5, 4)[0][:2])


class ServingInputTests(unittest.TestCase):
    def test_invalid_requests_fail_before_model_execution(self):
        pipeline = Week5Pipeline.__new__(Week5Pipeline)
        pipeline.k = 20
        for users, k in [(None, 20), ([], 20), ([True], 20), ([-1], 20), ([2**64], 20),
                         ([1.5], 20), ([1]*101, 20), ([1], True), ([1], 21)]:
            with self.assertRaises(ValueError):
                pipeline.recommend(users, k)

    def test_manifest_hash_mismatch_fails_before_loading_model(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            artifact = path/'model.pt'
            artifact.write_text('changed')
            manifest = path/'manifest.json'
            manifest.write_text(json.dumps({'profiles': {'itemcf': {}},
                                            'artifact_sha256': {str(artifact): 'wrong'}}))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                Week5Pipeline(manifest)
