import unittest
import numpy as np
from kuaiflow.pipeline import candidate_union,apply_policy,source_order


class PipelinePolicyTests(unittest.TestCase):
    def setUp(self):
        self.frame=candidate_union({1:[10,11,12],2:[10,11,12]}, {1:[12,13,10],2:[12,13,10]})
        self.frame['ranker_logit']=self.frame.video_id.astype(float)
        self.base={'source':'interleave','candidate_budget':3,'neural_weight':0.,'rrf_constant':60,'itemcf_weight':.5}

    def test_zero_neural_preserves_retrieval_and_is_logit_independent(self):
        first=apply_policy(self.frame,self.base)
        changed=self.frame.copy();changed['ranker_logit']=float('nan')
        second=apply_policy(changed,self.base)
        self.assertEqual(first.video_id.tolist(),second.video_id.tolist())
        self.assertEqual(first.loc[first.user_id==1].video_id.tolist(),[10,12,11])

    def test_pure_sources_are_real_source_lists(self):
        for weight,expected in [(0.,[10,11,12]),(1.,[12,13,10])]:
            policy=dict(self.base,source='rrf',itemcf_weight=weight)
            ranked=apply_policy(self.frame,policy)
            self.assertEqual(ranked.loc[ranked.user_id==1].video_id.tolist(),expected)

    def test_full_neural_orders_within_fixed_candidate_set(self):
        ranked=apply_policy(self.frame,dict(self.base,neural_weight=1.))
        self.assertEqual(ranked.loc[ranked.user_id==1].video_id.tolist(),[12,11,10])
        self.assertEqual(set(ranked.video_id),set(source_order(self.frame,self.base).video_id))
        np.testing.assert_array_equal(ranked.final_rank,[1,2,3,1,2,3])

    def test_tied_scores_use_source_order(self):
        self.frame['ranker_logit']=1.
        ranked=apply_policy(self.frame,dict(self.base,neural_weight=.5))
        self.assertEqual(ranked.loc[ranked.user_id==1].video_id.tolist(),[10,12,11])

    def test_bad_weights_rejected(self):
        for policy in [dict(self.base,neural_weight=-1),dict(self.base,source='rrf',itemcf_weight=2)]:
            with self.assertRaises(ValueError):
                apply_policy(self.frame,policy)

    def test_itemcf_serving_path_skips_two_tower(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from kuaiflow.pipeline import RecommendationPipeline
        pipeline=RecommendationPipeline.__new__(RecommendationPipeline)
        pipeline.cf_only=True
        pipeline.tower_only=False
        pipeline.config={'candidate_budget':3}
        pipeline.policy=dict(self.base,source='rrf',itemcf_weight=1.)
        pipeline.itemcf=SimpleNamespace(recommend=lambda users,k:{u:[12,13,10] for u in users})
        with patch('kuaiflow.pipeline._get_user_embeddings_batch',side_effect=AssertionError('Unused model executed')):
            result=pipeline.recommend([1],2)
        self.assertEqual(result.video_id.tolist(),[12,13])
        with self.assertRaises(ValueError):
            pipeline.recommend([],2)
        with self.assertRaises(ValueError):
            pipeline.recommend([1],4)
