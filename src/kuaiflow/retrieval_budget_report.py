"""Generate the retrieval budget/ceiling and validation tuning report."""
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='artifacts/retrieval_budget_causal/results.json')
    parser.add_argument('--output', default='docs/retrieval_budget_results.md')
    args = parser.parse_args()
    data = json.loads(Path(args.results).read_text())
    if 'validation_selected_variant' not in data:
        raise ValueError('Experiment is incomplete')
    rows = data['rows']
    flat = [{**{k:v for k,v in r.items() if k not in ('ranking','latency_samples')}, **r['ranking']} for r in rows]
    pd.DataFrame(flat).to_csv(Path(args.results).parent / 'summary.csv', index=False)
    lines = ['# Retrieval budget, oracle ceiling and tuning results', '',
             '**Corrected causal retrieval histories:** equal-time training clicks are excluded. These are new results; historical Week 2 scores used the earlier tie-handling implementation.', '',
             'Seed 2026. Same fixed evaluation users, training catalog and frozen mean-pooling ranker throughout. '
             'No calibration or ranking retraining. See [protocol](retrieval_budget.md).', '',
             f"Validation-selected budget: **{data['validation_selected_budget']}**. Validation-selected retrieval: **{data['validation_selected_variant']}**.", '']
    for split in ('validation', 'test'):
        lines += [f'## {split.title()}: fixed-model candidate budget', '',
                  '| Retrieval | K | Candidate recall (%) | Candidate hit rate (%) | Oracle NDCG@20 (%) | Achieved NDCG@20 (%) | Achieved/oracle (%) | Pipeline ms/user |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in rows:
            if row['split'] != split or row['variant'] not in ('ivf_baseline', 'exact_baseline'):
                continue
            oracle = row['oracle_ndcg20']
            achieved = row['ranking']['ndcg@20']
            lines.append(f"| {row['variant']} | {row['budget']} | {row['candidate_recall']*100:.3f} | {row['candidate_hit_rate']*100:.3f} | {oracle*100:.3f} | {achieved*100:.3f} | {achieved/oracle*100:.2f} | {row['latency_ms_per_user']:.3f} |")
        lines += ['']
    lines += ['## Validation retrieval tuning at the selected budget', '',
              '| Retrieval | Candidate recall (%) | Oracle NDCG@20 (%) | Final NDCG@20 (%) | Pipeline ms/user |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        if row['split']=='validation' and row['budget']==data['validation_selected_budget'] and row['variant']!='exact_baseline':
            lines.append(f"| {row['variant']} | {100*row['candidate_recall']:.3f} | {100*row['oracle_ndcg20']:.3f} | {100*row['ranking']['ndcg@20']:.3f} | {row['latency_ms_per_user']:.3f} |")
    initial = next(r for r in rows if r['variant']=='ivf_baseline' and r['split']=='validation' and r['budget']==100)
    expanded = next(r for r in rows if r['variant']=='ivf_baseline' and r['split']=='validation' and r['budget']==500)
    lines += ['', '## Budget interpretation', '',
              f"Validation K=100 → 500 changes candidate recall from {100*initial['candidate_recall']:.3f}% to {100*expanded['candidate_recall']:.3f}%, "
              f"oracle NDCG@20 from {100*initial['oracle_ndcg20']:.3f}% to {100*expanded['oracle_ndcg20']:.3f}%, "
              f"and achieved NDCG@20 from {100*initial['ranking']['ndcg@20']:.3f}% to {100*expanded['ranking']['ndcg@20']:.3f}%.",
              'The fixed ranker does not turn the larger candidate pool into better top-20 ranking here. Improve candidate discrimination or training/evaluation alignment before increasing the serving budget solely on recall.',
              'Separate saved integrity checks verify that both larger IVF budgets preserve the original top-100 entries and ranks, with identical logits for every shared candidate on both splits.']
    chosen = next(r for r in rows if r['variant']==data['validation_selected_variant'] and r['split']=='test' and r['budget']==data['validation_selected_budget'])
    lines += ['', '## Selected configuration: test result', '',
              f"{chosen['variant']}, K={chosen['budget']}: candidate recall {100*chosen['candidate_recall']:.3f}%, final click NDCG@20 {100*chosen['ranking']['ndcg@20']:.3f}%, oracle NDCG@20 {100*chosen['oracle_ndcg20']:.3f}%, pipeline {chosen['latency_ms_per_user']:.3f} ms/user.", '',
              'Only the validation-selected new retrieval option is evaluated on test. The fixed baseline budget curve and exact reference are diagnostic comparisons, not a test-set selection rule.', '',
              '## Interpretation limits', '',
              '- Oracle scores use held-out labels and represent only the candidate-set ceiling. They are not attainable deployment results or training targets.',
              '- A gap between achieved and oracle NDCG indicates ranking headroom; missing candidate positives indicate retrieval headroom. Larger recall alone need not improve the final top 20.',
              '- Training-timestamp tie correction changes the retrieval baseline; do not attribute differences from historical Week 2 solely to candidate budget or temperature.',
              '- The ranker was trained on logged impressions and is frozen. Larger or mixed candidate pools may change its scoring distribution; these results reflect that fixed ranker.',
              '- Latency is median amortized batch time on 256 users with three repeats, not online-request p95. Artifacts include the individual repeats.',
              '- ItemCF-only with the same ranker was not tested: the hybrid improvement does not isolate complementary contributions from its two sources.',
              '- Single-seed validation search and a previously examined test cohort do not establish statistical significance, online lift, or an optimal hyperparameter setting.',
              '', '## Training cost', '', '| Retrieval fit | Seconds | Final training loss |', '| --- | ---: | ---: |']
    for fit in data['fits']:
        lines.append(f"| {fit['variant']} | {fit['seconds']:.2f} | {fit['loss'][-1]:.5f} |")
    lines += ['', 'Full candidate lists, logits, checkpoint caches, frozen embeddings, provenance and metrics are saved under `artifacts/retrieval_budget_causal/`. Regenerate with `python -m kuaiflow.retrieval_budget_report`.']
    Path(args.output).write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
