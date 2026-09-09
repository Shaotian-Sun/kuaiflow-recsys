"""Rebuild the four-model Week 3 comparison from measured run artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from kuaiflow.week3_figures import SVG, _render_png

MODELS = [('deepfm', 'DeepFM'), ('din', 'DIN'),
          ('mmoe', 'DeepFM + MMoE'), ('din_mmoe', 'DIN + MMoE')]


def comparison_rows(results, source_root=None):
    source_root = Path(source_root or Path.cwd())
    reference = results['deepfm']
    for name, result in results.items():
        if (source_root / result['candidate_source']).resolve() != (source_root / reference['candidate_source']).resolve():
            raise ValueError('Candidate source files differ')
        for field in ('seed', 'training_examples', 'candidate_k'):
            if result[field] != reference[field]:
                raise ValueError(f'Incompatible comparison: {name} differs on {field}')
        for split in ('validation', 'test'):
            a, b = result['candidate_ranking'][split], reference['candidate_ranking'][split]
            if any(a[field] != b[field] for field in ('candidate_users','candidate_rows')):
                raise ValueError('Candidate cohorts differ')
    rows = []
    for split in ('validation', 'test'):
        for key, metrics in reference['candidate_ranking'][split]['retrieval_order'].items():
            rows.append(dict(split=split, model='retrieval', ordering='Week 2 retrieval', k=int(key),
                             **{m:metrics[f'{m}@{key}'] for m in ('recall','hit_rate','ndcg','coverage')}))
        for name,label in MODELS:
            result = results[name]
            multi = name in ('mmoe','din_mmoe')
            ranking = result['candidate_ranking'][split]
            if multi:
                ranking = ranking['targets']['click']
                orders = [('mmoe_composite_order',label+' composite'),('task_head_order',label+' click head')]
            else:
                orders = [(name+'_order',label)]
            for key in reference['candidate_ranking'][split]['retrieval_order']:
                if ranking['retrieval_order'][key] != reference['candidate_ranking'][split]['retrieval_order'][key]:
                    raise ValueError('Retrieval ground truth or ordering differs')
            for order,display in orders:
                for key,metrics in ranking[order].items():
                    if int(key) == result['candidate_k']:
                        baseline = reference['candidate_ranking'][split]['retrieval_order'][key]
                        if any(metrics[f'{m}@{key}'] != baseline[f'{m}@{key}'] for m in ('recall','hit_rate','coverage')):
                            raise ValueError('Full-list membership invariant failed')
                    rows.append(dict(split=split,model=name,ordering=display,k=int(key),
                                     **{m:metrics[f'{m}@{key}'] for m in ('recall','hit_rate','ndcg','coverage')}))
    return rows


def audit_candidates(results, root):
    """Verify actual output pairs, ranks, and finite scores, beyond metric equality."""
    import numpy as np
    import pandas as pd
    from kuaiflow.ranking import _validate_candidates
    keys = ['split', 'user_id', 'video_id']
    source = pd.read_csv(root / results['deepfm']['candidate_source'])
    reference = source[keys + ['retrieval_rank']].sort_values(keys).reset_index(drop=True)
    audit = {}
    for name, result in results.items():
        frame = pd.read_csv(root / result['outputs']['reranked_candidates'])
        _validate_candidates(frame, result['candidate_k'])
        pd.testing.assert_frame_equal(reference, frame[keys + ['retrieval_rank']].sort_values(keys).reset_index(drop=True))
        rank = name + '_rank' if name in ('deepfm', 'din') else 'mmoe_rank'
        groups = frame.groupby(['split', 'user_id'])[rank]
        valid = (groups.min().eq(1) & groups.max().eq(result['candidate_k']) &
                 groups.nunique().eq(result['candidate_k']))
        if not valid.all() or not np.isfinite(frame.select_dtypes(include='number').to_numpy()).all():
            raise ValueError(f'{name}: invalid ranks or non-finite candidate values')
        if not np.equal(frame[rank], np.floor(frame[rank])).all():
            raise ValueError(f'{name}: fractional candidate ranks')
        audit[name] = {'rows': len(frame), 'candidate_membership_and_retrieval_ranks_match': True,
                       'complete_reranking_per_user': True, 'finite_numeric_values': True}
    return audit


def build_comparison_figure(rows, output):
    svg = SVG(1500,1060,title='Week 3: four ranking models',
              description='Validation and test NDCG and test catalog coverage on identical Week 2 top-100 candidates.')
    svg.rect(0,0,1500,1060,fill='white')
    svg.text(48,54,'Week 3 · DIN and multi-task ranking',size=30,weight=700)
    svg.text(48,89,'Four trained models · fixed Week 2 top 100 · 5,000 users per split · seed 2026',size=18,color='#526078')
    panels=[('validation','ndcg','Validation click NDCG@20 (%)',135,3.0),
            ('test','ndcg','Test click NDCG@20 (%)',425,3.0),
            ('test','coverage','Test catalog coverage@20 (%)',715,100.0)]
    for split,metric,title,top,minimum in panels:
        data=[r for r in rows if r['split']==split and r['k']==20]
        maximum=max(minimum,max(r[metric]*100 for r in data)*1.15)
        x0,width=375,1000
        svg.text(48,top,title,size=22,weight=700)
        for j in range(5):
            x=x0+width*j/4
            svg.line(x,top+20,x,top+224,stroke='#e2e8f0',width=1)
            svg.text(x,top+245,f'{maximum*j/4:.2f}' if metric=='ndcg' else f'{maximum*j/4:.0f}',size=13,anchor='middle',color='#526078')
        for i,row in enumerate(data):
            y=top+26+i*28
            color='#087f8c' if row['model'].startswith('din') else '#64748b'
            svg.text(350,y+15,row['ordering'],size=16,anchor='end')
            svg.rect(x0,y,width*row[metric]*100/maximum,20,fill=color)
            svg.text(x0+width*row[metric]*100/maximum+8,y+15,f'{row[metric]*100:.3f}' if metric=='ndcg' else f'{row[metric]*100:.2f}',size=14)
    svg.text(48,1013,'Source: artifacts/week3_*_results.json · chronological validation/test · values shown as percentages.',size=15,color='#526078')
    svg.text(48,1038,'Teal: DIN family. Gray: existing baselines. Click heads are diagnostics; composite weights are fixed.',size=15,color='#526078')
    svg.save(output)


def build_report(results, rows):
    lines=['# KuaiFlow — Week 3: DeepFM, DIN, and MMoE comparison','',
    'Week 3 now compares four trained ranking models over the exact same Week 2 top-100 candidate handoff. '
    'The original retrieval order is included as a reference. All values below come from full runs, not toy examples.','',
    '![Week 3 model comparison](../figures/Week_3_model_comparison.png)','',
    '## Experimental contract','',
    f'- {results["din"]["training_examples"]:,} training impressions; '
    f'{results["din"]["pointwise"]["validation"]["examples"]:,} validation and '
    f'{results["din"]["pointwise"]["test"]["examples"]:,} test impressions.',
    '- Same chronological splits, train-fitted vocabularies, seven categorical and three numeric static fields, '
    '5,000 candidate users per split, and 100 fixed videos per user. Candidates are scored only; unexposed items are never training negatives.',
    '- Seed 2026, CPU, one OpenMP thread, AdamW at 0.001 with weight decay 1e-6, embedding size 16, dropout 0.1, '
    'five-epoch budget and patience two. Single-task batches are 2,048; multi-task batches are 4,096, matching the existing baselines.',
    '- Single-task models select the lowest validation click log loss. Multi-task models select the lowest '
    'validation normalized ten-task loss. Test data never selects epochs or utility weights.',
    '- DIN uses the last 30 training clicks and a target-conditioned local activation MLP [64, 32]. '
    'Training history excludes the current impression, future events, and all events at the same timestamp. '
    'Validation, test, and candidate histories stay frozen at the training cutoff; validation outcomes never enter test history.',
    '- DIN predicts click with an MLP [128, 64]. DIN + MMoE feeds the same attended representation to four '
    'experts [128, 64], ten task-specific softmax gates, and ten 32-unit towers. '
    'The tasks, loss normalization, duration curve, and composite utility match DeepFM + MMoE.',
    '- This is a model-family comparison, not an isolated attention ablation: DIN replaces the DeepFM '
    'linear/FM branches as well as adding history. PReLU attention and AdamW are explicit implementation choices; '
    'this is not an exact reproduction of the DIN paper\'s Dice and mini-batch-aware regularization.','',
    '## Click candidate ranking','',
    'Composite rankings apply the frozen multi-objective utility. Click-head rankings show click specialization '
    'and are reported separately. Coverage is catalog reach, not per-user diversity.','']
    for split in ('validation','test'):
        lines += [f'### {split.title()}','',
                  '| Ordering | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | NDCG@50 |',
                  '|---|---:|---:|---:|---:|---:|']
        for row in rows:
            if row['split']==split and row['k']==20:
                at50=next(r for r in rows if r['split']==split and r['ordering']==row['ordering'] and r['k']==50)
                lines.append('| '+row['ordering']+' | '+' | '.join(f'{row[m]*100:.3f}%' for m in ('recall','hit_rate','ndcg','coverage'))+f' | {at50["ndcg"]*100:.3f}% |')
        lines.append('')
    lines += ['## Pointwise click prediction and training cost','',
              '| Model | Validation ROC-AUC | Test ROC-AUC | Test PR-AUC | Test log loss ↓ | Best epoch / completed | Training seconds | Parameters |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,label in MODELS:
        r=results[name]; multi=name in ('mmoe','din_mmoe')
        val=r['pointwise']['validation']; test=r['pointwise']['test']
        if multi: val,test=val['binary']['click'],test['binary']['click']
        opt=r['optimization']
        lines.append(f'| {label} | {val["roc_auc"]:.4f} | {test["roc_auc"]:.4f} | {test["pr_auc"]:.4f} | {test["log_loss"]:.4f} | {opt["best_epoch"]} / {opt["epochs_completed"]} | {opt["training_seconds"]:.2f} | {r["architecture"]["parameter_count"]:,} |')
    lines += ['', 'Training time includes epoch validation and checkpoint selection, but excludes feature/history preparation, '
              'final evaluation, candidate scoring, and writing files. These are individual local runs, not repeated latency benchmarks.', '',
              '## Multi-task target comparison','',
              'Test candidate NDCG@20 by target. Each target uses its own eligible cohort. Hate is undesirable '
              'exposure: lower values are preferred, and its dedicated ranking uses ascending predicted hate probability.','',
              '| Target | Users | DeepFM+MMoE composite | DIN+MMoE composite | DeepFM+MMoE head | DIN+MMoE head |',
              '|---|---:|---:|---:|---:|---:|']
    old=results['mmoe']['candidate_ranking']['test']['targets']; new=results['din_mmoe']['candidate_ranking']['test']['targets']
    for target in old:
        a,b=old[target],new[target]
        if '20' not in a['mmoe_composite_order']: continue
        values=[a['mmoe_composite_order']['20']['ndcg@20'],b['mmoe_composite_order']['20']['ndcg@20'],a['task_head_order']['20']['ndcg@20'],b['task_head_order']['20']['ndcg@20']]
        lines.append(f'| {target} | {a["candidate_ground_truth_users"]:,} | '+' | '.join(f'{v*100:.3f}%' for v in values)+' |')
    lines += ['', '### Multi-task logged prediction', '',
              '| Target | DeepFM+MMoE test ROC-AUC | DIN+MMoE test ROC-AUC | DeepFM+MMoE test PR-AUC | DIN+MMoE test PR-AUC |',
              '|---|---:|---:|---:|---:|']
    old_point = results['mmoe']['pointwise']['test']
    new_point = results['din_mmoe']['pointwise']['test']
    for target in old_point['binary']:
        a, b = old_point['binary'][target], new_point['binary'][target]
        values = [a['roc_auc'], b['roc_auc'], a['pr_auc'], b['pr_auc']]
        lines.append('| ' + target + ' | ' + ' | '.join(f'{v:.4f}' if v is not None else 'undefined' for v in values) + ' |')
    lines += ['', '| Continuous target metric (lower is better) | DeepFM+MMoE | DIN+MMoE |', '|---|---:|---:|']
    for target, key, label in [('watch_time','mae_seconds','Watch-time MAE, seconds'),
                               ('watch_time','rmse_seconds','Watch-time RMSE, seconds'),
                               ('completion','mae_fraction','Completion MAE, fraction'),
                               ('completion','rmse_fraction','Completion RMSE, fraction')]:
        lines.append(f'| {label} | {old_point[target][key]:.4f} | {new_point[target][key]:.4f} |')
    lines += ['', f'Completion metrics exclude {new_point["completion"]["masked_rows"]:,} test rows with invalid duration. '
              'Sparse binary targets require PR-AUC and cohort sizes alongside ROC-AUC.', '', '## Interpretation','']
    # Predeclared selection on validation, with test shown only afterwards.
    primary=[r for r in rows if r['split']=='validation' and r['k']==20 and 'click head' not in r['ordering'] and r['model']!='retrieval']
    selected=max(primary,key=lambda r:r['ndcg'])
    test_selected=next(r for r in rows if r['split']=='test' and r['k']==20 and r['ordering']==selected['ordering'])
    test_best=max([r for r in rows if r['split']=='test' and r['k']==20 and r['ordering'] in [x['ordering'] for x in primary]],key=lambda r:r['ndcg'])
    lines += [f'- Among the four primary orderings, **{selected["ordering"]}** has the highest validation NDCG@20 '
              f'({selected["ndcg"]*100:.3f}%). Its held-out test NDCG@20 is {test_selected["ndcg"]*100:.3f}%.',
              f'- The highest observed test NDCG@20 among primary orderings is {test_best["ordering"]} '
              f'({test_best["ndcg"]*100:.3f}%). This test observation is not a tuning decision.']
    for name,base,label in [('din','deepfm','DIN versus DeepFM'),('din_mmoe','mmoe','DIN+MMoE composite versus DeepFM+MMoE composite')]:
        deltas=[]
        for split in ('validation','test'):
            a=next(r for r in rows if r['split']==split and r['k']==20 and r['model']==name and 'click head' not in r['ordering'])
            b=next(r for r in rows if r['split']==split and r['k']==20 and r['model']==base and 'click head' not in r['ordering'])
            deltas.append(f'{split}: {(a["ndcg"]/b["ndcg"]-1)*100:+.1f}% relative NDCG@20')
        lines.append(f'- {label}: '+ '; '.join(deltas)+'.')
    lines += ['- At K=100, Recall, HitRate, and Coverage match retrieval for every ordering; '
              'the comparison builder checks this invariant and verifies each saved candidate pair, retrieval rank, '
              'complete reranking, and finite numeric scores across all four million-row output files. '
              'The rankers cannot recover missing candidates.',
              '- One seed, one history length, fixed utility weights, and different parameter counts limit causal architecture claims. '
              'No repeated-seed uncertainty or online lift is claimed. Rare-action cohorts are small. '
              'A useful next experiment is a matched mean-pooling/no-history ablation, followed by utility tuning on validation and a diversity reranker.','',
              '## Reproduce','', '```bash',
              'OMP_NUM_THREADS=1 make deepfm mmoe din din-mmoe',
              'make week3-comparison', 'python -m unittest discover -s tests -v', '```','',
              'The comparison command reads all four `artifacts/week3_*_results.json` files and writes '
              '`artifacts/week3_model_comparison.json`, `artifacts/week3_model_comparison.csv`, '
              'this report, and `figures/Week_3_model_comparison.svg` / `.png`. '
              'DIN checkpoints include their model configuration, feature encoder, and persisted training-history index; '
              'DIN+MMoE also saves the fitted duration curve. See [DIN implementation notes](week3_din.md) for loading and inference.','',
              'Architecture references: [DIN paper](https://arxiv.org/abs/1706.06978) and '
              '[MMoE paper](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/).','']
    return '\n'.join(lines)


def generate_comparison(root, render_png=True):
    root=Path(root)
    results={name:json.loads((root/f'artifacts/week3_{name}_results.json').read_text()) for name,_ in MODELS}
    rows=comparison_rows(results, root)
    integrity=audit_candidates(results, root)
    (root/'artifacts/week3_model_comparison.json').write_text(json.dumps({'rows': rows, 'integrity': integrity},indent=2)+'\n')
    with (root/'artifacts/week3_model_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (root/'docs/week3_results.md').write_text(build_report(results,rows))
    figure=root/'figures/Week_3_model_comparison.svg'
    build_comparison_figure(rows,figure)
    if render_png: _render_png(figure,1500,1060)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--svg-only',action='store_true')
    args=parser.parse_args()
    generate_comparison(args.repo_root,not args.svg_only)


if __name__=='__main__': main()
