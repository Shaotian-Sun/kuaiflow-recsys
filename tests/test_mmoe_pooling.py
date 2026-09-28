import copy
import unittest
from unittest.mock import patch
import pandas as pd
import torch
from kuaiflow.data import Week1Splits
from kuaiflow.models.din import DINMMoE
from kuaiflow.mmoe_pooling_ablation import scoring_metrics, verify_source


class MMoEPoolingTests(unittest.TestCase):
    def test_shared_initialization_and_target_independent_mean(self):
        models = []
        for mode in ('mean', 'attention'):
            torch.manual_seed(2026)
            models.append(DINMMoE([6, 8], 1, 1, ['click', 'like'], embedding_dim=4,
                                 num_experts=2, expert_hidden_dims=[8], tower_hidden_dim=4,
                                 dropout=0, history_pooling=mode))
        for key, value in models[0].state_dict().items():
            torch.testing.assert_close(value, models[1].state_dict()[key], rtol=0, atol=0)
        cat = torch.tensor([[2, 3], [2, 4]])
        num = torch.zeros(2, 1)
        history = torch.tensor([[0, 1, 2, 5], [0, 1, 2, 5]])
        rep, weights = models[0].representation(cat, num, history)
        torch.testing.assert_close(rep[0, -4:], rep[1, -4:])
        torch.testing.assert_close(weights, torch.tensor([[0., 0., .5, .5]]).expand(2, -1))
        output = models[0].component_logits(cat, num, history)
        self.assertEqual(output['logits'].shape, (2, 2))
        torch.testing.assert_close(output['gate_weights'].sum(-1), torch.ones(2, 2))
        output['logits'].sum().backward()
        self.assertIsNone(models[0].attention.network[0].weight.grad)
        self.assertGreater(models[0].feature_embeddings[1].weight.grad.abs().sum().item(), 0)

    def test_same_ordering_evaluated_against_different_outcomes(self):
        train = pd.DataFrame({'user_id': [1, 1], 'video_id': [2, 3], 'is_click': [0, 0], 'is_hate': [0, 0]})
        future = train.copy()
        future['is_click'] = [1, 0]
        future['is_hate'] = [0, 1]
        splits = Week1Splits(train, future, future, pd.DataFrame())
        candidates = pd.concat([train.assign(split=s, retrieval_rank=[1, 2], click_score=[.9, .1], composite=[.1, .9])
                                for s in ('validation', 'test')], ignore_index=True)
        metrics = scoring_metrics(candidates, splits, {'click': 'is_click', 'hate': 'is_hate'},
                                  {'click_only': 'click_score', 'combined': 'composite'})
        self.assertGreater(metrics['test']['click']['click_only']['ndcg@20'], metrics['test']['click']['combined']['ndcg@20'])
        self.assertLess(metrics['test']['hate']['click_only']['ndcg@20'], metrics['test']['hate']['combined']['ndcg@20'])

    def test_reject_changed_provenance(self):
        base = {'data': {k: 'value' for k in ('categorical_features', 'numeric_features', 'candidates_path', 'processed_dir', 'raw_dir')},
                'model': {k: 1 for k in ('embedding_dim', 'history_max_length', 'dropout', 'attention_hidden_dims')}}
        single = {'specification': {'seeds': [1, 2, 3]}, 'base_config': copy.deepcopy(base), 'sha256': {'file': 'old'}}
        with patch('kuaiflow.mmoe_pooling_ablation.digest', return_value='new'):
            with self.assertRaisesRegex(ValueError, 'provenance'):
                verify_source(single, base, [1, 2, 3])


class ReportCompletenessTests(unittest.TestCase):
    def test_incomplete_run_rejected(self):
        from kuaiflow.mmoe_pooling_report import validate_complete
        data = {'specification': {'seeds': [1, 2, 3]}, 'runs': [], 'strategies': []}
        with self.assertRaisesRegex(ValueError, 'all matched'):
            validate_complete(data)
