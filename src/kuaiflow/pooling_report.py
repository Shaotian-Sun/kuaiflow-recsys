"""Regenerate the pooling results report from the saved experiment."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='artifacts/pooling_ablation/results.json')
    parser.add_argument('--output', default='docs/pooling_ablation_results.md')
    args = parser.parse_args()
    data = json.loads(Path(args.results).read_text())
    lines = ['# Matched pooling ablation: measured results', '',
             'Single-task click prediction with identical features, history rules, candidate sets and head dimensions. '
             'See [protocol](pooling_ablation.md). No calibration is applied.', '',
             'Three seeds: 2026, 2027, 2028. Values are mean ± sample standard deviation across seeds, not confidence intervals.', '',
             f"Validation-selected pooling (mean NDCG@20): **{data['validation_selected_pooling']}**.", '',
             '| Pooling | Split | NDCG@20 (%) | Click log loss | ROC-AUC |',
             '| --- | --- | ---: | ---: | ---: |']
    for row in data['summary']:
        lines.append(f"| {row['pooling']} | {row['split']} | " + ' | '.join(
            f"{row[key + '_mean'] * scale:.4f} ± {row[key + '_sd'] * scale:.4f}"
            for key, scale in [('ndcg20', 100), ('log_loss', 1), ('roc_auc', 1)]) + ' |')
    lines += ['', '## Individual runs', '',
              '| Seed | Pooling | Best epoch | Epochs run | Validation NDCG@20 (%) | Test NDCG@20 (%) | Training seconds |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for run in data['runs']:
        optimization = run['optimization']
        lines.append(f"| {run['seed']} | {run['pooling']} | {optimization['best_epoch']} | {optimization['epochs_completed']} | "
                     f"{100 * run['candidate_ranking']['validation']['din_order']['20']['ndcg@20']:.4f} | "
                     f"{100 * run['candidate_ranking']['test']['din_order']['20']['ndcg@20']:.4f} | {optimization['training_seconds']:.2f} |")
    lines += ['', '## Paired seed differences', '',
              'NDCG@20 differences in percentage points, first method minus second. Each pair uses the same seed.', '',
              '| Comparison | Split | Mean difference (pp) | Individual seed differences (pp) |',
              '| --- | --- | ---: | --- |']
    for left, right in [('mean', 'none'), ('sum', 'none'), ('attention', 'mean'), ('attention', 'sum')]:
        for split in ('validation', 'test'):
            lookup = {(r['seed'], r['pooling']): r['candidate_ranking'][split]['din_order']['20']['ndcg@20'] for r in data['runs']}
            diff = [100 * (lookup[seed, left] - lookup[seed, right]) for seed in data['specification']['seeds']]
            lines.append(f"| {left} − {right} | {split} | {np.mean(diff):+.4f} | " + ', '.join(f'{v:+.4f}' for v in diff) + ' |')
    lookup = {(r['pooling'], r['split']): r for r in data['summary']}
    gap = 100 * (lookup['attention', 'test']['ndcg20_mean'] - lookup['mean', 'test']['ndcg20_mean'])
    times = {mode: np.mean([r['optimization']['training_seconds'] for r in data['runs'] if r['pooling'] == mode])
             for mode in ('mean', 'attention')}
    lines += ['', '## Practical reading', '',
              f"Attention minus mean pooling on test NDCG@20 is {gap:+.4f} percentage points on average; the individual seed differences above show whether that gain is consistent.",
              f"Mean training time is {times['mean']:.2f}s for mean pooling and {times['attention']:.2f}s for attention ({times['attention'] / times['mean']:.2f} times as long). These are local training-plus-validation times, not serving latency benchmarks.",
              'Use the validation-selected method as the experiment choice, and retain mean pooling as the simpler comparison. A small observed test difference does not demonstrate a substantial attention benefit.',
              '', '## Verification and interpretation limits', '',
              '- Every run passed candidate membership/original-column, finite-score, complete-rank, unchanged recall/hit-rate/coverage at 100, and exact checkpoint reload checks.',
              '- Shared parameters have identical initial values within seed. Attention parameters remain allocated but unused in other modes; effective capacity is not identical.',
              '- The history window and prediction head are matched. Early stopping is matched as a rule, not as an identical number of updates; checkpoints are selected by validation log loss.',
              '- Three seeds measure some optimization variability, not population uncertainty. Do not infer significance or online lift.',
              '- The test cohort has been examined in earlier project work. No new holdout was collected, and test scores did not select the pooling method.',
              '- Saved per-run checkpoints, encoders, histories, candidate scores and metrics reside under `artifacts/pooling_ablation/`. Input/source hashes and complete configuration are in `results.json`.',
              '', 'Regenerate with `python -m kuaiflow.pooling_report`.']
    Path(args.output).write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
