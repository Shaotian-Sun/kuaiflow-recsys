"""Matched mean/attention MMoE and fixed scoring-rule comparison."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from kuaiflow.data import load_prepared, load_kuairand_features
from kuaiflow.pooling_ablation import digest, verify_candidates
from kuaiflow.ranking import _attach_static_features
from kuaiflow.multitask_ranking import run_mmoe_ranking, save_mmoe_run, load_mmoe_artifacts, _recommendations_by_score
from kuaiflow.metrics import build_ground_truth, evaluate_recommendations


def scoring_metrics(frame, splits, binary_tasks, score_columns):
    """Evaluate each fixed ordering against every outcome, including hate descending score."""
    catalog = splits.train.video_id.drop_duplicates().tolist()
    result = {}
    for split in ('validation', 'test'):
        predictions = {label: _recommendations_by_score(frame.loc[frame.split == split], column)
                       for label, column in score_columns.items()}
        result[split] = {}
        for task, column in binary_tasks.items():
            seen = {u: set(g.video_id) for u, g in splits.train.loc[splits.train[column] > 0].groupby('user_id')}
            truth = build_ground_truth(getattr(splits, split), label_col=column, catalog=catalog, exclude=seen)
            result[split][task] = {label: evaluate_recommendations(order, truth, 20, catalog)
                                  for label, order in predictions.items()}
    return result


def verify_source(single, base, seeds):
    if single['specification']['seeds'] != seeds:
        raise ValueError('Single-task seeds differ')
    for key in ('categorical_features', 'numeric_features', 'candidates_path', 'processed_dir', 'raw_dir'):
        if single['base_config']['data'][key] != base['data'][key]:
            raise ValueError('Mismatched input setting: ' + key)
    for key in ('embedding_dim', 'history_max_length', 'dropout', 'attention_hidden_dims'):
        if single['base_config']['model'][key] != base['model'][key]:
            raise ValueError('Mismatched model setting: ' + key)
    for path, expected in single['sha256'].items():
        if digest(path) != expected:
            raise ValueError('Single-task provenance changed: ' + path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/mmoe_pooling_ablation.yaml')
    args = parser.parse_args()
    spec = yaml.safe_load(Path(args.config).read_text())
    base = yaml.safe_load(Path(spec['base_config']).read_text())
    single = json.loads(Path(spec['single_task_results']).read_text())
    verify_source(single, base, spec['seeds'])
    root = Path(spec['output_dir'])
    if root.exists():
        raise FileExistsError('Choose a fresh output_dir; existing experiments are preserved')
    root.mkdir(parents=True)
    torch.set_num_threads(1)
    splits = load_prepared(base['data']['processed_dir'])
    candidates = pd.read_csv(base['data']['candidates_path'])
    users, videos = load_kuairand_features(base['data']['raw_dir'])
    paths = [args.config, spec['base_config'], spec['single_task_results']]
    paths += [str(p) for p in Path('src/kuaiflow').glob('*.py')]
    paths += [str(p) for p in Path('src/kuaiflow/models').glob('*.py')]
    output = {'specification': spec, 'base_config': base, 'single_task_sha256': single['sha256'],
              'sha256': {p: digest(p) for p in paths}, 'torch': torch.__version__, 'runs': [], 'strategies': []}
    for seed in spec['seeds']:
        path = Path(spec['single_task_results']).parent / f'seed_{seed}' / 'mean' / 'candidates.csv.gz'
        frame = pd.read_csv(path)
        verify_candidates(candidates, frame)
        output['sha256'][str(path)] = digest(path)
        metrics = scoring_metrics(frame, splits, base['tasks']['binary'], {'single_mean_click': 'din_score'})
        output['strategies'].append({'seed': seed, 'family': 'single_mean', 'metrics': metrics})
        for mode in ('mean', 'attention'):
            print(f'Training seed={seed} pooling={mode}', flush=True)
            config = copy.deepcopy(base)
            config['seed'] = seed
            config['model']['history_pooling'] = mode
            run = run_mmoe_ranking(splits, config, candidates, users, videos)
            verify_candidates(candidates, run.reranked_candidates.rename(columns={'mmoe_score': 'din_score', 'mmoe_rank': 'din_rank'}))
            numeric = run.reranked_candidates.select_dtypes(include='number')
            if not np.isfinite(numeric.to_numpy()).all():
                raise ValueError('Nonfinite candidate values')
            for split in ('validation', 'test'):
                for task in run.results['candidate_ranking'][split]['targets'].values():
                    if not task['retrieval_order']:
                        continue
                    for order in ('mmoe_composite_order', 'task_head_order'):
                        for metric in ('recall@100', 'hit_rate@100', 'coverage@100'):
                            assert task[order]['100'][metric] == task['retrieval_order']['100'][metric]
            directory = root / f'seed_{seed}' / mode
            save_mmoe_run(run, config, directory, directory / 'candidates.csv.gz')
            prefix = directory / 'week3_din_mmoe'
            loaded, encoder, _, _ = load_mmoe_artifacts(str(prefix) + '_model.pt', str(prefix) + '_encoder.json', str(prefix) + '_duration_curve.json')
            sample = candidates.groupby('split', sort=False).head(64)
            features = _attach_static_features(sample, users, videos, encoder.categorical + encoder.numeric)
            cat, num = encoder.transform(features)
            inputs = (torch.tensor(cat), torch.tensor(num), torch.tensor(loaded.history_index.transform(sample), dtype=torch.long))
            with torch.no_grad():
                torch.testing.assert_close(loaded(*inputs), run.model(*inputs), rtol=0, atol=0)
            metrics = scoring_metrics(run.reranked_candidates, splits, base['tasks']['binary'],
                                      {f'{mode}_mmoe_click': 'p_click', f'{mode}_mmoe_composite': 'mmoe_score'})
            # Independent evaluation must reproduce existing click-head/composite metrics.
            for split in ('validation', 'test'):
                original = run.results['candidate_ranking'][split]['targets']['click']
                assert metrics[split]['click'][f'{mode}_mmoe_click'] == original['task_head_order']['20']
                assert metrics[split]['click'][f'{mode}_mmoe_composite'] == original['mmoe_composite_order']['20']
            output['strategies'].append({'seed': seed, 'family': mode + '_mmoe', 'metrics': metrics})
            run.results['pooling'] = mode
            run.results['verification'] = {'candidate_integrity': True, 'reload_exact': True, 'recall100_unchanged': True}
            output['runs'].append(run.results)
            (root / 'progress.json').write_text(json.dumps(output, indent=2))
            print(f'Completed seed={seed} pooling={mode}', flush=True)
            del run, loaded
    (root / 'results.json').write_text(json.dumps(output, indent=2))
    print('All six runs complete', flush=True)


if __name__ == '__main__':
    main()
