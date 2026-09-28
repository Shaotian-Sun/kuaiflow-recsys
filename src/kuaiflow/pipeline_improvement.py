"""Tune retrieval-aware ranking on validation users, then confirm and test."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from kuaiflow.pipeline import RecommendationPipeline, apply_policy
from kuaiflow.data import load_prepared
from kuaiflow.metrics import build_ground_truth, evaluate_recommendations, ndcg_at_k
from kuaiflow.pooling_ablation import digest


def recommendations(frame):
    return {u:g.sort_values('final_rank').video_id.tolist() for u,g in frame.groupby('user_id',sort=False)}


def evaluate(frame, truth, catalog):
    return evaluate_recommendations(recommendations(frame),truth,20,catalog)


def paired_interval(first,second,truth,repeats,seed):
    a,b=recommendations(first),recommendations(second)
    users=[u for u in a if truth.get(u)]
    differences=np.asarray([ndcg_at_k(a[u],truth[u],20)-ndcg_at_k(b[u],truth[u],20) for u in users])
    rng=np.random.default_rng(seed)
    samples=np.asarray([differences[rng.integers(len(users),size=len(users))].mean() for _ in range(repeats)])
    return {'mean_difference':float(differences.mean()),'percentile95':np.quantile(samples,[.025,.975]).tolist(),
            'users':len(users),'method':'paired user bootstrap, fixed fitted models and selected policy'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/pipeline_improvement.yaml')
    args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())
    root=Path(config['output_dir'])
    if root.exists():
        raise FileExistsError('Choose a fresh output directory')
    root.mkdir(parents=True)
    torch.set_num_threads(1)
    base={'source':'interleave','candidate_budget':config['candidate_budget'],'neural_weight':0.,'rrf_constant':config['rrf_constant'],'itemcf_weight':.5}
    pipeline=RecommendationPipeline(config,base)
    splits=load_prepared(config['processed_dir'])
    cohort=pd.read_csv(config['candidate_path'])
    ids={s:cohort.loc[cohort.split==s].user_id.drop_duplicates().tolist() for s in ('validation','test')}
    rng=np.random.default_rng(config['seed'])
    shuffled=rng.permutation(ids['validation']).tolist()
    n=int(len(shuffled)*config['validation_tune_fraction'])
    tuning=set(shuffled[:n]);confirmation=set(shuffled[n:])
    catalog=splits.train.video_id.drop_duplicates().tolist()
    seen={u:set(g.video_id) for u,g in splits.train.loc[splits.train.is_click>0].groupby('user_id')}
    truths={s:build_ground_truth(getattr(splits,s),'is_click',catalog=catalog,exclude=seen) for s in ids}
    paths=[Path(args.config),Path(config['candidate_path'])]
    paths += list(Path(config['ranker_dir']).glob('week3_din_*'))
    paths += [Path(config['retrieval_dir'])/s for s in ('baseline_retriever.pt','selected_itemcf.pt')]
    paths += [Path(config['processed_dir'])/(s+'.csv.gz') for s in ('train','validation','test')]
    paths += [Path('src/kuaiflow')/s for s in ('pipeline.py','pipeline_improvement.py','models/two_tower.py')]
    result={'config':config,'sha256':{str(p):digest(p) for p in paths},'tuning_users':sorted(tuning),'confirmation_users':sorted(confirmation),'search':[],'evaluations':{}}
    print('Generating and scoring validation candidate union',flush=True)
    validation=pipeline.neural_scores(pipeline.candidates(ids['validation']))
    validation.to_csv(root/'validation_candidates.csv.gz',index=False,compression='gzip')
    policies=[]
    for source,weight in [('interleave',.5)]+[('rrf',w) for w in config['itemcf_weights']]:
        for neural in config['neural_weights']:
            policies.append(dict(base,source=source,itemcf_weight=weight,neural_weight=neural))
    tuning_frame=validation.loc[validation.user_id.isin(tuning)]
    for policy in policies:
        ranked=apply_policy(tuning_frame,policy)
        metrics=evaluate(ranked,truths['validation'],catalog)
        result['search'].append({'policy':policy,'metrics':metrics})
    chosen=max(result['search'],key=lambda r:(r['metrics']['ndcg@20'],-r['policy']['neural_weight']))['policy']
    result['selected_policy']=chosen
    # Persist the choice before accessing confirmation or test outcome metrics.
    (root/'selected_pipeline.json').write_text(json.dumps({'config':config,'policy':chosen},indent=2))
    print('Selected on tuning users: '+json.dumps(chosen),flush=True)
    baselines={'mixed_retrieval':base,'mixed_neural':dict(base,neural_weight=1.),
               'itemcf_only':dict(base,source='rrf',itemcf_weight=1.),
               'tower_only':dict(base,source='rrf',itemcf_weight=0.),'selected':chosen}
    for split in ('validation_confirmation','validation_all','test'):
        truth=truths['test' if split=='test' else 'validation']
        if split=='test':
            print('Generating and scoring test candidate union after selection',flush=True)
            frame=pipeline.neural_scores(pipeline.candidates(ids['test']))
            frame.to_csv(root/'test_candidates.csv.gz',index=False,compression='gzip')
        elif split=='validation_confirmation':
            frame=validation.loc[validation.user_id.isin(confirmation)]
        else:
            frame=validation
        predictions={name:apply_policy(frame,policy) for name,policy in baselines.items()}
        result['evaluations'][split]={name:evaluate(ranked,truth,catalog) for name,ranked in predictions.items()}
        result.setdefault('paired_intervals',{})[split]={name:paired_interval(predictions['selected'],predictions[name],truth,config['bootstrap_repeats'],config['seed']) for name in ('mixed_retrieval','itemcf_only')}
        for name,ranked in predictions.items():
            if name=='selected' and split!='validation_confirmation':
                ranked.to_csv(root/(split+'_selected.csv.gz'),index=False,compression='gzip')
        print(split+': '+json.dumps(result['evaluations'][split]),flush=True)
        (root/'progress.json').write_text(json.dumps(result,indent=2))
        if split=='test':
            expected=predictions['selected']
    # Independently load the serving policy and compare actual end-to-end recommendations.
    deployed=RecommendationPipeline.load(root/'selected_pipeline.json')
    sample=ids['test'][:256]
    timings=[]
    for iteration in range(4):
        start=time.perf_counter();actual=deployed.recommend(sample,20);elapsed=time.perf_counter()-start
        if iteration:
            timings.append(elapsed*1000/len(sample))
    target=expected.loc[expected.user_id.isin(sample)&(expected.final_rank<=20)]
    pd.testing.assert_frame_equal(actual[['user_id','video_id','final_rank']].reset_index(drop=True),target[['user_id','video_id','final_rank']].reset_index(drop=True))
    for u,g in actual.groupby('user_id'):
        if len(g)!=20 or g.video_id.nunique()!=20 or set(g.video_id)&seen.get(u,set()):
            raise ValueError('Invalid serving recommendations')
    result['serving']={'reload_top20_exact':True,'users':len(sample),'median_ms_per_user':float(np.median(timings)),'samples_ms_per_user':timings}
    (root/'results.json').write_text(json.dumps(result,indent=2))
    print('Completed; serving reload matches saved top20 exactly',flush=True)


if __name__=='__main__':
    main()
