"""Verify saved budget controls and cache the selected ItemCF component."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from kuaiflow.data import load_prepared
from kuaiflow.pooling_ablation import digest
from kuaiflow.models.itemcf import ItemCFRecommender
from kuaiflow.retrieval import FAISSRetriever, _get_user_embeddings_batch, _faiss_search_k, _postprocess_faiss_recommendations
from kuaiflow.retrieval_budget import interleave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='artifacts/retrieval_budget_causal')
    args = parser.parse_args()
    root = Path(args.root)
    results = json.loads((root/'results.json').read_text())
    torch.set_num_threads(1)
    checks = []
    for split in ('validation','test'):
        baseline = pd.read_csv(root/f'ivf_baseline_{split}_100.csv.gz')
        for k in (200,500):
            expanded = pd.read_csv(root/f'ivf_baseline_{split}_{k}.csv.gz')
            paired = baseline.merge(expanded, on=['split','user_id','video_id'], suffixes=('_100','_larger'))
            if len(paired) != len(baseline) or not (paired.retrieval_rank_100 == paired.retrieval_rank_larger).all():
                raise ValueError('Baseline prefix not preserved')
            np.testing.assert_allclose(paired.ranker_logit_100,paired.ranker_logit_larger,rtol=1e-5,atol=1e-5)
            checks.append({'split': split, 'budget':k, 'top100_preserved':True, 'retrieval_ranks_preserved':True,
                           'max_common_candidate_logit_difference':float(abs(paired.ranker_logit_100-paired.ranker_logit_larger).max())})
    (root/'baseline_integrity.json').write_text(json.dumps(checks,indent=2))
    if results['validation_selected_variant']=='ivf_itemcf_interleave':
        cfg=results['retrieval_config']
        splits=load_prepared(cfg['data']['processed_dir'])
        cf=ItemCFRecommender(neighbor_k=100).fit(splits.train,label_col='is_click')
        torch.save(cf,root/'selected_itemcf.pt')
        cf=torch.load(root/'selected_itemcf.pt',weights_only=False)
        model=torch.load(root/'baseline_retriever.pt',weights_only=False)
        index=FAISSRetriever(**cfg['faiss'],seed=cfg['seed'])
        index.build_index(model.item_vectors,model.item_ids)
        for split in ('validation','test'):
            frame=pd.read_csv(root/f'ivf_itemcf_interleave_{split}_100.csv.gz')
            users=frame.user_id.drop_duplicates().tolist()[:64]
            query=_get_user_embeddings_batch(model,users)
            raw,_,_=index.search(query,_faiss_search_k(model,users,100))
            first=_postprocess_faiss_recommendations(model,users,raw,100)
            actual=interleave(first,cf.recommend(users,100),100)
            expected={u:g.sort_values('retrieval_rank').video_id.tolist() for u,g in frame.loc[frame.user_id.isin(users)].groupby('user_id')}
            if actual != expected:
                raise ValueError('Selected hybrid candidate reload mismatch')
        (root/'selected_reload_verification.json').write_text(json.dumps({'users_per_split':64,'exact_candidate_match':True,
            'baseline_sha256':digest(root/'baseline_retriever.pt'),'itemcf_sha256':digest(root/'selected_itemcf.pt')},indent=2))
    print('Budget invariance and selected hybrid reload checks passed')


if __name__ == '__main__':
    main()
