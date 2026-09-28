"""Matched single-task DIN pooling experiment; never overwrites Week 3/4 artifacts."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import torch
import yaml
from kuaiflow.data import load_prepared, load_kuairand_features
from kuaiflow.ranking import run_deepfm_ranking, save_deepfm_run, load_deepfm_artifacts, _attach_static_features

MODES = ('none', 'mean', 'sum', 'attention')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_candidates(original, ranked):
    keys = ['split', 'user_id', 'video_id']
    a = original.sort_values(keys).reset_index(drop=True)
    b = ranked.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b[a.columns], check_dtype=False)
    if not np.isfinite(b.din_score).all():
        raise ValueError('Nonfinite candidate scores')
    for _, group in b.groupby(['split', 'user_id']):
        np.testing.assert_array_equal(np.sort(group.din_rank), np.arange(1, len(group) + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pooling_ablation.yaml')
    args = parser.parse_args()
    specification = yaml.safe_load(Path(args.config).read_text())
    base = yaml.safe_load(Path(specification['base_config']).read_text())
    root = Path(specification['output_dir'])
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'results.json').exists():
        raise FileExistsError('Use a new output_dir to preserve the previous experiment')
    torch.set_num_threads(1)
    splits = load_prepared(base['data']['processed_dir'])
    candidates = pd.read_csv(base['data']['candidates_path'])
    users, videos = load_kuairand_features(base['data']['raw_dir'])
    paths = [Path(base['data']['processed_dir']) / f'{name}.csv.gz'
             for name in ('train', 'validation', 'test')]
    paths += [Path(base['data']['candidates_path']), Path(specification['base_config']), Path(args.config)]
    paths += sorted(Path(base['data']['raw_dir']).rglob('user_features_pure.csv'))
    paths += sorted(Path(base['data']['raw_dir']).rglob('video_features_basic_pure.csv'))
    paths += [Path('src/kuaiflow') / name for name in ('models/din.py', 'history.py', 'ranking.py', 'pooling_ablation.py')]
    output = {'specification': specification, 'base_config': base,
              'sha256': {str(p): digest(p) for p in paths},
              'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                          'numpy': np.__version__, 'threads': torch.get_num_threads()}, 'runs': []}
    for seed in specification['seeds']:
        for mode in MODES:
            directory = root / f'seed_{seed}' / mode
            if directory.exists():
                raise FileExistsError(f'Existing run directory: {directory}')
            config = copy.deepcopy(base)
            config['seed'] = seed
            config['model']['history_pooling'] = mode
            print(f'Training seed={seed} pooling={mode}', flush=True)
            run = run_deepfm_ranking(splits, config, candidates, users, videos)
            verify_candidates(candidates, run.reranked_candidates)
            # Candidate membership implies unchanged recall@100; verify metric invariant too.
            for split in ('validation', 'test'):
                scores = run.results['candidate_ranking'][split]
                for metric in ('recall@100', 'hit_rate@100', 'coverage@100'):
                    assert scores['din_order']['100'][metric] == scores['retrieval_order']['100'][metric]
            save_deepfm_run(run, config, directory, directory / 'candidates.csv.gz')
            loaded, encoder = load_deepfm_artifacts(directory / 'week3_din_model.pt', directory / 'week3_din_encoder.json')
            sample = candidates.iloc[:128]
            features = _attach_static_features(sample, users, videos, encoder.categorical + encoder.numeric)
            cat, num = encoder.transform(features)
            hist = loaded.history_index.transform(sample)
            inputs = (torch.tensor(cat), torch.tensor(num), torch.tensor(hist, dtype=torch.long))
            with torch.no_grad():
                torch.testing.assert_close(loaded(*inputs), run.model(*inputs), rtol=0, atol=0)
            result = run.results
            result['pooling'] = mode
            result['active_parameter_count'] = sum(p.numel() for n, p in run.model.named_parameters()
                if mode == 'attention' or not n.startswith('attention.'))
            result['verification'] = {'candidate_integrity': True, 'reload_exact': True, 'recall100_unchanged': True}
            output['runs'].append(result)
            (root / 'progress.json').write_text(json.dumps(output, indent=2))
            print(f"Finished: validation NDCG@20={result['candidate_ranking']['validation']['din_order']['20']['ndcg@20']:.6f}", flush=True)
            del run, loaded
    summary = []
    for mode in MODES:
        for split in ('validation', 'test'):
            rows = [r for r in output['runs'] if r['pooling'] == mode]
            values = {'ndcg20': [r['candidate_ranking'][split]['din_order']['20']['ndcg@20'] for r in rows],
                      'log_loss': [r['pointwise'][split]['log_loss'] for r in rows],
                      'roc_auc': [r['pointwise'][split]['roc_auc'] for r in rows]}
            summary.append({'pooling': mode, 'split': split, **{f'{k}_{stat}': float(fn(v))
                for k, v in values.items() for stat, fn in [('mean', np.mean), ('sd', lambda x: np.std(x, ddof=1))]}})
    output['summary'] = summary
    output['validation_selected_pooling'] = max((r for r in summary if r['split'] == 'validation'), key=lambda r: r['ndcg20_mean'])['pooling']
    (root / 'results.json').write_text(json.dumps(output, indent=2))
    pd.DataFrame(summary).to_csv(root / 'summary.csv', index=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
