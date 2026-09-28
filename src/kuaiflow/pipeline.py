"""Reusable retrieval-aware serving policy, without outcome labels at inference."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from kuaiflow.data import load_kuairand_features
from kuaiflow.pooling_ablation import digest
from kuaiflow.ranking import load_deepfm_artifacts, _attach_static_features, _predict_logits
from kuaiflow.retrieval import FAISSRetriever, _get_user_embeddings_batch, _faiss_search_k, _postprocess_faiss_recommendations
from kuaiflow.retrieval_budget import interleave


def candidate_union(tower, itemcf):
    rows = []
    if set(tower) != set(itemcf):
        raise ValueError('Sources must have identical user membership')
    for user in tower:
        a = {v: i+1 for i,v in enumerate(tower[user])}
        b = {v: i+1 for i,v in enumerate(itemcf[user])}
        mixed = {v:i+1 for i,v in enumerate(interleave({user:tower[user]}, {user:itemcf[user]}, len(tower[user]))[user])}
        for video in dict.fromkeys(tower[user]+itemcf[user]):
            rows.append((user,video,a.get(video,0),b.get(video,0),mixed.get(video,0)))
    return pd.DataFrame(rows,columns=['user_id','video_id','tower_rank','itemcf_rank','mixed_rank'])


def source_order(frame, policy):
    budget = int(policy['candidate_budget'])
    if budget <= 0:
        raise ValueError('candidate_budget must be positive')
    output = frame.copy()
    if policy['source']=='interleave':
        output = output.loc[output.mixed_rank>0].copy()
        output['source_score'] = -output.mixed_rank.astype(float)
    elif policy['source']=='rrf':
        weight, constant = float(policy['itemcf_weight']), float(policy['rrf_constant'])
        if not 0<=weight<=1 or constant<=0:
            raise ValueError('Invalid fusion weight or constant')
        output['source_score'] = ((1-weight)*np.where(output.tower_rank>0,1/(constant+output.tower_rank),0)
                                  + weight*np.where(output.itemcf_rank>0,1/(constant+output.itemcf_rank),0))
        output = output.loc[output.source_score>0].copy()
    else:
        raise ValueError('Unknown candidate source')
    output = output.sort_values(['user_id','source_score','video_id'],ascending=[True,False,True],kind='stable')
    output['source_rank'] = output.groupby('user_id',sort=False).cumcount()+1
    return output.loc[output.source_rank<=budget].copy()


def apply_policy(frame, policy):
    output = source_order(frame,policy)
    weight = float(policy['neural_weight'])
    if not 0<=weight<=1:
        raise ValueError('neural_weight must lie in [0,1]')
    count = output.groupby('user_id').video_id.transform('size')
    denominator = (count-1).clip(lower=1)
    retrieval_percentile = 1-(output.source_rank-1)/denominator
    if weight:
        if 'ranker_logit' not in output or not np.isfinite(output.ranker_logit).all():
            raise ValueError('Finite neural logits required')
        # Tie-breaking begins from deterministic source order.
        neural_rank = output.groupby('user_id').ranker_logit.rank(method='first',ascending=False)
        neural_percentile = 1-(neural_rank-1)/denominator
        output['final_score'] = (1-weight)*retrieval_percentile + weight*neural_percentile
    else:
        output['final_score'] = retrieval_percentile
    output = output.sort_values(['user_id','final_score','source_rank'],ascending=[True,False,True],kind='stable')
    output['final_rank'] = output.groupby('user_id',sort=False).cumcount()+1
    return output


class RecommendationPipeline:
    def __init__(self, config, policy):
        self.config, self.policy = config, policy
        root = Path(config['retrieval_dir'])
        hashes = json.loads((root/'selected_reload_verification.json').read_text())
        for name,key in [('baseline_retriever.pt','baseline_sha256'),('selected_itemcf.pt','itemcf_sha256')]:
            if digest(root/name)!=hashes[key]:
                raise ValueError('Retrieval checkpoint hash mismatch')
        # Only project-produced trusted local full-object caches are accepted here.
        self.cf_only = policy['source']=='rrf' and policy['itemcf_weight']==1
        self.tower_only = policy['source']=='rrf' and policy['itemcf_weight']==0
        self.tower = self.index = self.itemcf = None
        if not self.tower_only:
            self.itemcf = torch.load(root/'selected_itemcf.pt',weights_only=False,map_location='cpu')
        if not self.cf_only:
            self.tower = torch.load(root/'baseline_retriever.pt',weights_only=False,map_location='cpu')
            self.index = FAISSRetriever(index_type='ivf',n_lists=100,n_probe=10,seed=config['seed'])
            self.index.build_index(self.tower.item_vectors,self.tower.item_ids)
        self.ranker = self.encoder = None
        self.users = self.videos = None
        if policy['neural_weight']:
            self._load_ranker()

    def _load_ranker(self):
        if self.ranker is None:
            root = Path(self.config['ranker_dir'])
            self.ranker,self.encoder=load_deepfm_artifacts(root/'week3_din_model.pt',root/'week3_din_encoder.json')
            self.users,self.videos=load_kuairand_features(self.config['raw_dir'])

    def candidates(self, users):
        ids=list(dict.fromkeys(users))
        if not ids:
            raise ValueError('At least one user required')
        k=self.config['candidate_budget']
        if self.cf_only:
            cf=self.itemcf.recommend(ids,k)
            return pd.DataFrame([(u,v,0,i+1,0) for u,items in cf.items() for i,v in enumerate(items)],
                                columns=['user_id','video_id','tower_rank','itemcf_rank','mixed_rank'])
        query=_get_user_embeddings_batch(self.tower,ids)
        raw,_,_=self.index.search(query,_faiss_search_k(self.tower,ids,k))
        tower=_postprocess_faiss_recommendations(self.tower,ids,raw,k)
        if self.tower_only:
            return pd.DataFrame([(u,v,i+1,0,0) for u,items in tower.items() for i,v in enumerate(items)],
                                columns=['user_id','video_id','tower_rank','itemcf_rank','mixed_rank'])
        cf=self.itemcf.recommend(ids,k)
        return candidate_union(tower,cf)

    def neural_scores(self, frame):
        self._load_ranker()
        parts=[]
        for start in range(0,len(frame),100000):
            chunk=frame.iloc[start:start+100000].copy()
            features=_attach_static_features(chunk,self.users,self.videos,self.encoder.categorical+self.encoder.numeric)
            cat,num=self.encoder.transform(features)
            chunk['ranker_logit']=_predict_logits(self.ranker,cat,num,2048,torch.device('cpu'),self.ranker.history_index.transform(chunk))
            parts.append(chunk)
        return pd.concat(parts,ignore_index=True)

    def recommend(self, users, k=20):
        if not 0<k<=self.policy['candidate_budget']:
            raise ValueError('Requested k must be within candidate budget')
        frame=self.candidates(users)
        # Score only selected candidates during inference, rather than the tuning union.
        if self.policy['neural_weight']:
            frame=self.neural_scores(source_order(frame,self.policy))
        ranked=apply_policy(frame,self.policy)
        return ranked.loc[ranked.final_rank<=k].reset_index(drop=True)

    @classmethod
    def load(cls, path):
        payload=json.loads(Path(path).read_text())
        return cls(payload['config'],payload['policy'])
