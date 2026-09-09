import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch

from kuaiflow.history import ClickHistory, prepare_histories
from kuaiflow.models.din import DIN, DINMMoE
from kuaiflow.ranking import DeepFMFeatures, run_deepfm_ranking, save_deepfm_run, load_deepfm_artifacts
from kuaiflow.multitask_ranking import run_mmoe_ranking, save_mmoe_run, load_mmoe_artifacts
from kuaiflow.toy import make_multitask_toy_splits
import test_mmoe


class DINTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_strict_time_ties_truncation_unknown_and_saved_history(self):
        train = pd.DataFrame({'user_id': [1]*5, 'video_id': [10,11,12,13,14],
                              'time_ms': [1,2,2,3,4], 'is_click': [1,1,1,0,1]})
        encoder = DeepFMFeatures().fit(train, ['user_id','video_id'], [])
        history = ClickHistory(2).fit(train, encoder)
        vocab = encoder.vocabularies['video_id']
        h = history.transform(train, causal=True)
        self.assertEqual(h.tolist(), [[0,0], [0,vocab['10']], [0,vocab['10']],
                                      [vocab['11'],vocab['12']], [vocab['11'],vocab['12']]])
        shuffled = train.sample(frac=1, random_state=3)
        shuffled_history = ClickHistory(2).fit(shuffled, encoder)
        np.testing.assert_array_equal(shuffled_history.transform(train, causal=True), h)
        query = pd.DataFrame({'user_id': [1,99]}, index=[18,45])
        expected = [[vocab['12'],vocab['14']], [0,0]]
        self.assertEqual(history.transform(query).tolist(), expected)
        with tempfile.TemporaryDirectory() as d:
            history.save(Path(d)/'history.npz')
            loaded = ClickHistory.load(Path(d)/'history.npz')
            np.testing.assert_array_equal(loaded.transform(train, causal=True),h)
            self.assertEqual(loaded.transform(query).tolist(), expected)

    def test_future_labels_never_update_frozen_history(self):
        splits = make_multitask_toy_splits()
        encoder = DeepFMFeatures().fit(splits.train,['user_id','video_id'],[])
        _, first = prepare_histories(splits,encoder,{})
        splits.validation['is_click'] = 1 - splits.validation.is_click
        splits.test['is_click'] = 1 - splits.test.is_click
        _, second = prepare_histories(splits,encoder,{})
        for name in first:
            np.testing.assert_array_equal(first[name],second[name])
        splits.test['time_ms'] = 1
        with self.assertRaisesRegex(ValueError,'strictly after'):
            prepare_histories(splits,encoder,{})

    def test_mask_padding_target_attention_and_gradients(self):
        torch.manual_seed(17)
        model = DIN([5,8],1,1,embedding_dim=4,hidden_dims=[8],dropout=0)
        cat = torch.tensor([[2,3],[2,4]])
        num = torch.zeros(2,1)
        history = torch.tensor([[2,5],[2,5]])
        representation, weights = model.representation(cat,num,history)
        self.assertFalse(torch.equal(weights[0],weights[1]))
        self.assertFalse(torch.equal(representation[0,-4:],representation[1,-4:]))
        self.assertTrue(torch.allclose(model(cat,num,history),model(cat,num,torch.cat([history,torch.zeros(2,3,dtype=torch.long)],1))))
        empty, weights = model.representation(cat,num,torch.zeros(2,3,dtype=torch.long))
        self.assertTrue(torch.equal(empty[:,-4:],torch.zeros(2,4)))
        self.assertTrue(torch.equal(weights,torch.zeros(2,3)))
        model(cat,num,history).sum().backward()
        self.assertGreater(float(model.attention.network[0].weight.grad.abs().sum()),0)
        self.assertGreater(float(model.feature_embeddings[1].weight.grad[5].abs().sum()),0)
        multi = DINMMoE([5,8],1,1,['click','forward'],embedding_dim=4,num_experts=2,expert_hidden_dims=[8],tower_hidden_dim=3,dropout=0)
        components = multi.component_logits(cat,num,history)
        self.assertTrue(torch.allclose(components['gate_weights'].sum(-1),torch.ones(2,2)))
        components['logits'].sum().backward()
        self.assertTrue(all(p.grad is not None for p in multi.parameters()))

    def test_both_pipelines_membership_chunking_and_roundtrip(self):
        splits = make_multitask_toy_splits()
        candidates = test_mmoe.MMoEPipelineTests._candidates()
        for architecture in ('din','din_mmoe'):
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as d:
                config = test_mmoe.MMoEPipelineTests._config()
                config['model'].update(architecture=architecture,hidden_dims=[8],attention_hidden_dims=[6],history_max_length=3)
                config['evaluation']['candidate_chunk_size'] = 3
                multi = architecture == 'din_mmoe'
                runner = run_mmoe_ranking if multi else run_deepfm_ranking
                run = runner(splits,config,candidates=candidates,user_features=pd.DataFrame({"user_id": [0,1,2,3]}),video_features=test_mmoe.MMoEPipelineTests._video_features())
                key = ['split','user_id','video_id']
                self.assertEqual(set(map(tuple,candidates[key].to_numpy())),set(map(tuple,run.reranked_candidates[key].to_numpy())))
                rank = 'mmoe_rank' if multi else 'din_rank'
                self.assertTrue(run.reranked_candidates.groupby(['split','user_id'])[rank].apply(lambda x: sorted(x)==[1,2]).all())
                root=Path(d)
                saver=save_mmoe_run if multi else save_deepfm_run
                saver(run,config,root,root/'candidates.csv.gz')
                paths=[root/f'week3_{architecture}_{s}' for s in ['model.pt','encoder.json']]
                if multi:
                    loaded, enc, _, _ = load_mmoe_artifacts(*paths,root/f'week3_{architecture}_duration_curve.json')
                else:
                    loaded, enc = load_deepfm_artifacts(*paths)
                sample=test_mmoe.MMoEPipelineTests._video_features().merge(candidates,on='video_id')
                cat,num=enc.transform(sample)
                seq=loaded.history_index.transform(sample)
                inputs=(torch.tensor(cat),torch.tensor(num),torch.tensor(seq,dtype=torch.long))
                with torch.no_grad():
                    torch.testing.assert_close(loaded(*inputs),run.model(*inputs),rtol=0,atol=0)
                config['evaluation']['candidate_chunk_size']=100
                second=runner(splits,config,candidates=candidates,user_features=pd.DataFrame({"user_id": [0,1,2,3]}),video_features=test_mmoe.MMoEPipelineTests._video_features())
                score='mmoe_score' if multi else 'din_score'
                np.testing.assert_allclose(run.reranked_candidates[score],second.reranked_candidates[score],atol=1e-7)


if __name__ == '__main__':
    unittest.main()
