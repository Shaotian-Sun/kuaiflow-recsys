"""Regenerate the whole-pipeline comparison and selected serving verification."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from kuaiflow.pipeline import RecommendationPipeline
from kuaiflow.pooling_ablation import digest


def main():
    root=Path('artifacts/pipeline_improvement')
    data=json.loads((root/'results.json').read_text())
    selected=data['selected_policy']
    # Preserve the evaluated source version; record the optimized serving path separately.
    torch.set_num_threads(1)
    pipeline=RecommendationPipeline.load(root/'selected_pipeline.json')
    saved=pd.read_csv(root/'test_selected.csv.gz')
    users=saved.user_id.drop_duplicates().tolist()[:256]
    timings=[]
    for iteration in range(4):
        start=time.perf_counter();actual=pipeline.recommend(users);elapsed=time.perf_counter()-start
        if iteration:timings.append(elapsed*1000/len(users))
    expected=saved.loc[saved.user_id.isin(users)&(saved.final_rank<=20)]
    pd.testing.assert_frame_equal(actual[['user_id','video_id','final_rank']].reset_index(drop=True),expected[['user_id','video_id','final_rank']].reset_index(drop=True))
    serving={'reload_top20_exact':True,'users':len(users),'median_ms_per_user':float(np.median(timings)),
             'samples_ms_per_user':timings,'source_sha256':digest('src/kuaiflow/pipeline.py'),
             'two_tower_loaded':pipeline.tower is not None,'neural_ranker_loaded':pipeline.ranker is not None}
    (root/'serving_verification.json').write_text(json.dumps(serving,indent=2))
    lines=['# Retrieval-aware pipeline improvement', '',
           'The search selected the existing **ItemCF-only ordering**, with no two-tower contribution and no neural reranking. This improves the recently assembled pipeline by removing harmful stages; it is not a new model outperforming ItemCF.', '',
           'The search used 2,500 validation users. The choice was saved before evaluating the other 2,500 validation users or test. Earlier project work had already examined these periods; they are not newly collected untouched data.', '',
           '## Selected policy', '',f"- Candidate budget: {selected['candidate_budget']}.",
           f"- ItemCF weight: {selected['itemcf_weight']}; neural weight: {selected['neural_weight']}.",
           '- ItemCF trained on training positives only, with 100 item neighbors. Train-clicked items are excluded; cold users receive its training-popularity fallback.',
           '- Selected serving path skips loading and running the two-tower and neural models.', '']
    for split in ('validation_confirmation','validation_all','test'):
        lines += [f'## {split}', '', '| Strategy | NDCG@20 (%) | Recall@20 (%) | HitRate@20 (%) | Coverage@20 (%) |', '| --- | ---: | ---: | ---: | ---: |']
        for name,metrics in data['evaluations'][split].items():
            lines.append('| '+name+' | '+' | '.join(f'{100*metrics[k]:.3f}' for k in ('ndcg@20','recall@20','hit_rate@20','coverage@20'))+' |')
        lines += ['']
    lines += ['## Paired uncertainty', '', 'Selected minus baseline, in NDCG@20 percentage points. Paired user bootstrap, 1,000 replicates. Intervals condition on the fitted models and selected policy; they exclude training/selection uncertainty.', '',
              '| Cohort | Baseline | Difference (pp) | 95% percentile interval (pp) |', '| --- | --- | ---: | --- |']
    for split in ('validation_confirmation','test'):
        for baseline,stats in data['paired_intervals'][split].items():
            lo,hi=stats['percentile95']
            lines.append(f"| {split} | {baseline} | {100*stats['mean_difference']:+.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] |")
    lines += ['', '## Search and limitations', '',
              '- Fixed grid: interleaving plus weighted reciprocal-rank fusion (ItemCF weights 0, 0.25, 0.5, 0.75, 1; rank constant 60), crossed with neural rank-percentile weights 0, 0.05, 0.1, 0.25, 0.5, 1. Total 36 policies.',
              '- Source fusion first chooses 100 candidates. A blend of source-rank percentile and neural-rank percentile then determines the final ordering. Both zero-neural and pure-source endpoints are included.',
              '- Single frozen neural model (seed 2026), corrected causal two-tower and training-only ItemCF. No outcomes are model inputs and no unexposed candidates are added as training negatives.',
              '- Aggregate catalog coverage declines versus the mixed retrieval order. This is a measured relevance/coverage tradeoff, not a claim of universal superiority or within-user diversity.',
              '- These are offline logged-feedback metrics. No online lift, unbiased policy value, or superiority of ItemCF on every recommendation task is established.', '',
              '## Serving and reproduction', '',
              f"Reloaded optimized serving output exactly matches saved top 20 for {len(users)} test users. Median amortized time: {serving['median_ms_per_user']:.3f} ms/user (three 256-user batches after warmup; models loaded, no disk I/O; not online p95).", '',
              '```bash', 'OMP_NUM_THREADS=1 .venv/bin/python -m kuaiflow.pipeline_improvement', '.venv/bin/python -m kuaiflow.pipeline_report', '.venv/bin/python -m kuaiflow.recommend --users 1 2 --k 20', '```', '',
              'The experiment refuses existing output directories. Change `output_dir` in a copied configuration to repeat. `RecommendationPipeline.load("artifacts/pipeline_improvement/selected_pipeline.json").recommend(user_ids, k=20)` is the reusable inference entry point. Numeric user IDs above are examples.', '',
              'Artifacts include all tuning metrics, fixed user partitions, source/checkpoint hashes, selected policy, cached candidate logits, ranked outputs and serving verification. The original evaluated pipeline source is preserved alongside the optimized serving source hash. Earlier experiments remain unchanged.']
    Path('docs/pipeline_improvement_results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(serving),flush=True)


if __name__=='__main__':
    main()
