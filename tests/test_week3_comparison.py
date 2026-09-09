import copy
import unittest
from kuaiflow.week3_comparison import comparison_rows


class ComparisonTests(unittest.TestCase):
    def test_rejects_incompatible_runs_and_full_list_changes(self):
        metrics = {'recall@100': 0.2, 'hit_rate@100': 0.3, 'ndcg@100': 0.1,
                   'coverage@100': 0.9, 'evaluated_users': 5.0}
        results = {}
        for name in ('deepfm', 'din', 'mmoe', 'din_mmoe'):
            orders = {'retrieval_order': {'100': dict(metrics)}}
            if name in ('mmoe', 'din_mmoe'):
                orders.update(mmoe_composite_order={'100': dict(metrics)}, task_head_order={'100': dict(metrics)})
                ranking = {'targets': {'click': orders}}
            else:
                orders[name+'_order'] = {'100': dict(metrics)}
                ranking = orders
            ranking.update(candidate_users=5, candidate_rows=500)
            results[name] = {'seed':2026, 'training_examples':1000, 'candidate_k':100,
                             'candidate_source':'same.csv',
                             'candidate_ranking':{'validation':copy.deepcopy(ranking), 'test':copy.deepcopy(ranking)}}
        self.assertEqual(len(comparison_rows(results)),14)
        wrong = copy.deepcopy(results)
        wrong['din']['seed'] = 1
        with self.assertRaisesRegex(ValueError,'seed'):
            comparison_rows(wrong)
        wrong = copy.deepcopy(results)
        wrong['din']['candidate_ranking']['test']['din_order']['100']['recall@100']=0.4
        with self.assertRaisesRegex(ValueError,'membership'):
            comparison_rows(wrong)
        wrong = copy.deepcopy(results)
        wrong['din']['candidate_ranking']['test']['retrieval_order']['100']['ndcg@100']=0.3
        with self.assertRaisesRegex(ValueError,'ground truth'):
            comparison_rows(wrong)


if __name__ == '__main__':
    unittest.main()
