import unittest
import torch
from kuaiflow.models.din import DIN


class PoolingTests(unittest.TestCase):
    def test_exact_pooling_mask_and_shared_initialization(self):
        cat = torch.tensor([[2, 3], [2, 4]])
        num = torch.zeros(2, 1)
        hist = torch.tensor([[0, 1, 2, 3], [0, 0, 0, 0]])
        models = {}
        for mode in ('attention', 'mean', 'sum', 'none'):
            torch.manual_seed(2026)
            model = DIN([5, 8], 1, 1, embedding_dim=4, hidden_dims=[8],
                        dropout=0, history_pooling=mode)
            models[mode] = model
            rep, weights = model.representation(cat, num, hist)
            torch.testing.assert_close(rep[1, -4:], torch.zeros(4))
            torch.testing.assert_close(weights[:, :2], torch.zeros(2, 2))
            expected = model.feature_embeddings[1](torch.tensor([2, 3])).sum(0)
            if mode != 'attention':
                expected = expected / 2 if mode == 'mean' else expected
                expected = torch.zeros_like(expected) if mode == 'none' else expected
                torch.testing.assert_close(rep[0, -4:], expected)
            padded = torch.cat([hist, torch.zeros(2, 3, dtype=torch.long)], 1)
            torch.testing.assert_close(model(cat, num, hist), model(cat, num, padded))
            model(cat, num, hist).sum().backward()
        for model in models.values():
            for key, value in models['attention'].state_dict().items():
                torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_invalid_pooling(self):
        with self.assertRaisesRegex(ValueError, 'history_pooling'):
            DIN([5, 8], 0, 1, history_pooling='bad')


class CandidateIntegrityTests(unittest.TestCase):
    def test_reject_changed_candidates_scores_and_ranks(self):
        import pandas as pd
        from kuaiflow.pooling_ablation import verify_candidates
        original = pd.DataFrame({'split': ['test', 'test'], 'user_id': [1, 1],
                                 'video_id': [2, 3], 'retrieval_rank': [1, 2]})
        ranked = original.assign(din_score=[0.2, 0.8], din_rank=[2, 1])
        verify_candidates(original, ranked)
        for column, value in [('video_id', 99), ('retrieval_rank', 9),
                              ('din_rank', 1), ('din_score', float('nan'))]:
            changed = ranked.copy()
            changed.loc[0, column] = value
            with self.assertRaises((AssertionError, ValueError)):
                verify_candidates(original, changed)
