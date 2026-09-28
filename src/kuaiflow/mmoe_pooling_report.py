"""Generate a five-strategy comparison from the persisted MMoE pooling experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

STRATEGIES = ['single_mean_click', 'mean_mmoe_click', 'mean_mmoe_composite', 'attention_mmoe_click', 'attention_mmoe_composite']


def validate_complete(data):
    seeds = data['specification']['seeds']
    expected = {(seed, mode) for seed in seeds for mode in ('mean', 'attention')}
    actual = [(r['seed'], r['pooling']) for r in data['runs']]
    strategies = [(r['seed'], r['family']) for r in data['strategies']]
    expected_strategies = {(seed, family) for seed in seeds
                           for family in ('single_mean', 'mean_mmoe', 'attention_mmoe')}
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or set(actual) != expected or len(actual) != len(expected):
        raise ValueError('Report requires all matched seed/pooling runs exactly once')
    if set(strategies) != expected_strategies or len(strategies) != len(expected_strategies):
        raise ValueError('Report requires all five strategies for every seed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='artifacts/mmoe_pooling_ablation/results.json')
    parser.add_argument('--output', default='docs/mmoe_pooling_ablation_results.md')
    args = parser.parse_args()
    data = json.loads(Path(args.results).read_text())
    validate_complete(data)
    rows = []
    for run in data['strategies']:
        for split, targets in run['metrics'].items():
            for target, strategies in targets.items():
                for strategy, metrics in strategies.items():
                    rows.append({'seed': run['seed'], 'split': split, 'target': target, 'strategy': strategy, **metrics})
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(args.results).parent / 'strategy_metrics.csv', index=False)
    lines = ['# MMoE pooling and ranking strategies: measured results', '',
             'Seeds 2026, 2027, 2028. Every ranking uses the same Week 2 top-100 candidates. '
             'See [protocol and interpretation limits](mmoe_pooling_ablation.md). Utility weights are fixed, and no calibration is applied.', '',
             'Values are mean ± sample standard deviation across seeds, not confidence intervals. NDCG is shown as a percentage.', '']
    for split in ('validation', 'test'):
        lines += [f'## {split.title()}: click ranking', '', '| Strategy | Click NDCG@20 (%) | Click Recall@20 (%) | Coverage@20 (%) |', '| --- | ---: | ---: | ---: |']
        for strategy in STRATEGIES:
            subset = frame[(frame.split == split) & (frame.target == 'click') & (frame.strategy == strategy)]
            values = [f'{100*subset[col].mean():.4f} ± {100*subset[col].std():.4f}' for col in ('ndcg@20', 'recall@20', 'coverage@20')]
            lines.append('| ' + strategy + ' | ' + ' | '.join(values) + ' |')
        lines += ['', f'## {split.title()}: other outcomes under those same rankings', '',
                  '| Strategy | Long-view NDCG@20 (%) | Like NDCG@20 (%) | Profile-entry NDCG@20 (%) | Hate NDCG@20 (%) ↓ |', '| --- | ---: | ---: | ---: | ---: |']
        for strategy in STRATEGIES:
            values = []
            for target in ('long_view', 'like', 'profile_enter', 'hate'):
                subset = frame[(frame.split == split) & (frame.target == target) & (frame.strategy == strategy)]
                values.append(f"{100*subset['ndcg@20'].mean():.4f} ± {100*subset['ndcg@20'].std():.4f}")
            lines.append('| ' + strategy + ' | ' + ' | '.join(values) + ' |')
        lines += ['']
    lines += ['## Paired click NDCG differences', '', 'Percentage points; first strategy minus second, paired by seed.', '',
              '| Comparison | Split | Mean difference (pp) | Per-seed differences (pp) |', '| --- | --- | ---: | --- |']
    for left, right in [('mean_mmoe_click', 'single_mean_click'), ('mean_mmoe_composite', 'mean_mmoe_click'),
                        ('mean_mmoe_click', 'attention_mmoe_click'), ('mean_mmoe_composite', 'attention_mmoe_composite')]:
        for split in ('validation', 'test'):
            subset = frame[(frame.split == split) & (frame.target == 'click')].pivot(index='seed', columns='strategy', values='ndcg@20')
            diff = 100 * (subset[left] - subset[right])
            lines.append(f'| {left} − {right} | {split} | {diff.mean():+.4f} | ' + ', '.join(f'{v:+.4f}' for v in diff) + ' |')
    lines += ['', '## MMoE pointwise predictions and training cost', '',
              '| Pooling | Split | Click log loss | Click ROC-AUC | Watch-time MAE (seconds) | Completion MAE |', '| --- | --- | ---: | ---: | ---: | ---: |']
    for mode in ('mean', 'attention'):
        runs = [r for r in data['runs'] if r['pooling'] == mode]
        for split in ('validation', 'test'):
            values = [[r['pointwise'][split]['binary']['click'][key] for r in runs] for key in ('log_loss', 'roc_auc')]
            values += [[r['pointwise'][split]['watch_time']['mae_seconds'] for r in runs], [r['pointwise'][split]['completion']['mae_fraction'] for r in runs]]
            lines.append(f'| {mode} | {split} | ' + ' | '.join(f'{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}' for v in values) + ' |')
    for mode in ('mean', 'attention'):
        runs = [r for r in data['runs'] if r['pooling'] == mode]
        times = [r['optimization']['training_seconds'] for r in runs]
        lines += [f"\n{mode.title()} pooling — average training-plus-validation time: {np.mean(times):.2f} seconds. Best epochs: " + ', '.join(str(r['optimization']['best_epoch']) for r in runs) + '.\n']
    lines += ['', '## Outcome cohort sizes', '', '| Target | Validation users | Test users |', '| --- | ---: | ---: |']
    for target in data['base_config']['tasks']['binary']:
        sizes = [int(frame[(frame.split == split) & (frame.target == target)].evaluated_users.iloc[0]) for split in ('validation', 'test')]
        lines.append(f'| {target} | {sizes[0]} | {sizes[1]} |')
    lines += ['', '## Limits and reproducibility', '',
              '- No utility-weight tuning or test-based selection was performed. Rankings for different outcomes are not combined into a post-hoc winner.',
              '- Single-task versus MMoE differs in architecture, loss, batch size and checkpoint criterion. Mean versus attention MMoE changes only pooling.',
              '- Rare outcomes have small eligible cohorts; observed zeros do not establish safety or absence of negative feedback. Outcomes have different eligible users.',
              '- Reported ranking metrics use logged feedback and do not identify online policy value. Candidate reranking cannot recover retrieval misses.',
              '- All six runs passed original-candidate preservation, finite-score, complete-rank, recall@100 invariance, exact reload and independent click metric checks.',
              '- Input/source/configuration hashes and per-run metrics are in `artifacts/mmoe_pooling_ablation/results.json`; all strategy/outcome/seed rows are in `strategy_metrics.csv`.',
              '', 'Regenerate: `python -m kuaiflow.mmoe_pooling_report`.']
    Path(args.output).write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
