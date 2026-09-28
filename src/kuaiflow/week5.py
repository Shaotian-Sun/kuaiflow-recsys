"""Validation-selected diversity reranking on ItemCF and the complete DIN path."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from kuaiflow.data import load_prepared, load_kuairand_features, _find_unique_file, VIDEO_FEATURE_FILE, USER_FEATURE_FILE
from kuaiflow.metrics import build_ground_truth, evaluate_recommendations
from kuaiflow.pipeline import RecommendationPipeline, apply_policy, source_order
from kuaiflow.pipeline_improvement import paired_interval, recommendations
from kuaiflow.pooling_ablation import digest
from kuaiflow.reranking import DiversityMetadata, diversity_metrics, rerank_candidates, select_strength
from kuaiflow.retrieval_budget import validate_candidates


def metrics(frame, truth, catalog, metadata):
    recs = recommendations(frame)
    result = evaluate_recommendations(recs, truth, 20, catalog)
    result.update(diversity_metrics({u: items for u, items in recs.items() if truth.get(u)}, metadata))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/week5.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    root = Path(config['output_dir'])
    if root.exists():
        raise FileExistsError('Choose a fresh Week 5 output directory')
    if config['k'] != 20 or 0 not in config['strengths']:
        raise ValueError('This experiment requires k=20 and a zero-strength control')
    root.mkdir(parents=True)
    torch.set_num_threads(1)
    prior = json.loads(Path(config['prior_results']).read_text())
    tuning = set(prior['tuning_users'])
    confirmation = set(prior['confirmation_users'])
    if tuning & confirmation:
        raise ValueError('Overlapping validation partitions')
    splits = load_prepared(config['processed_dir'])
    catalog = splits.train.video_id.drop_duplicates().tolist()
    seen = {u: set(g.video_id) for u, g in splits.train.loc[splits.train.is_click > 0].groupby('user_id')}
    _, videos = load_kuairand_features(config['raw_dir'])
    videos = videos.loc[videos.video_id.isin(catalog), ['video_id', 'author_id', 'tag']]
    if set(videos.video_id) != set(catalog):
        raise ValueError('Missing catalog metadata')
    metadata_path = root / 'metadata.csv'
    videos.to_csv(metadata_path, index=False)
    metadata = DiversityMetadata(videos)
    base = {'source': 'interleave', 'candidate_budget': config['candidate_budget'],
            'neural_weight': 1., 'itemcf_weight': .5, 'rrf_constant': 60}
    policies = {'itemcf': dict(base, source='rrf', itemcf_weight=1., neural_weight=0.),
                'hybrid_din': base}
    scorer = RecommendationPipeline(config, base)
    if scorer.ranker.history_pooling != 'attention':
        raise ValueError('The industry path requires the frozen attention DIN checkpoint')
    artifact_paths = [metadata_path, Path(config['retrieval_dir']) / 'baseline_retriever.pt',
                      Path(config['retrieval_dir']) / 'selected_itemcf.pt',
                      Path(config['retrieval_dir']) / 'selected_reload_verification.json']
    artifact_paths += [Path(config['ranker_dir']) / f'week3_din_{name}' for name in ('model.pt', 'encoder.json', 'history.npz')]
    artifact_paths += [_find_unique_file(Path(config['raw_dir']), name) for name in (VIDEO_FEATURE_FILE, USER_FEATURE_FILE)]
    evidence_paths = artifact_paths + [Path(args.config), Path(config['prior_results'])]
    evidence_paths += [Path(config['processed_dir']) / (s + '.csv.gz') for s in ('train', 'validation', 'test')]
    evidence_paths += [Path(config['candidate_dir']) / (s + '_candidates.csv.gz') for s in ('validation', 'test')]
    evidence_paths += [Path('src/kuaiflow') / f'{s}.py' for s in ('week5', 'reranking', 'serving', 'pipeline')]
    # Verify unchanged model/data inputs shared with the preceding experiment.
    for path in evidence_paths:
        if path.suffix == '.py':
            continue  # Earlier source hash predates the verified serving optimization.
        expected = prior['sha256'].get(str(path))
        if expected and digest(path) != expected:
            raise ValueError('Changed prior experiment input: ' + str(path))
    serving_check = json.loads((Path(config['candidate_dir']) / 'serving_verification.json').read_text())
    if digest('src/kuaiflow/pipeline.py') != serving_check['source_sha256']:
        raise ValueError('Changed verified serving implementation')
    result = {'config': config, 'sha256': {str(p): digest(p) for p in evidence_paths},
              'tuning_users': sorted(tuning), 'confirmation_users': sorted(confirmation),
              'search': {}, 'evaluations': {}, 'paired_intervals': {}, 'checks': {}}
    profiles = {}

    def rank_split(split):
        cached = pd.read_csv(Path(config['candidate_dir']) / (split + '_candidates.csv.gz'))
        cached = cached.drop(columns='ranker_logit')
        users = cached.user_id.drop_duplicates().tolist()
        # Regenerate a sample from checkpoints before relying on the candidate cache.
        sample = users[:64]
        actual = scorer.candidates(sample)
        columns = ['user_id', 'video_id', 'tower_rank', 'itemcf_rank', 'mixed_rank']
        sort = ['user_id', 'video_id']
        pd.testing.assert_frame_equal(actual[columns].sort_values(sort).reset_index(drop=True),
                                      cached.loc[cached.user_id.isin(sample), columns].sort_values(sort).reset_index(drop=True))
        print(f'{split}: scoring fresh attention DIN on the fixed hybrid top 100', flush=True)
        neural = scorer.neural_scores(source_order(cached, base))
        frames = {'itemcf': apply_policy(cached, policies['itemcf']),
                  'hybrid_din': apply_policy(neural, base)}
        for profile, frame in frames.items():
            validate_candidates(recommendations(frame), seen, set(catalog), config['candidate_budget'])
            frame.to_csv(root / f'{split}_{profile}_candidates.csv.gz', index=False, compression='gzip')
        result['checks'][split] = {'candidate_reload_users': len(sample), 'reload_exact': True,
                                   'unique_unseen_catalog_candidates_per_user': config['candidate_budget']}
        return frames

    validation = rank_split('validation')
    if set(validation['itemcf'].user_id) != tuning | confirmation:
        raise ValueError('Validation cohort changed')
    validation_truth = build_ground_truth(splits.validation, 'is_click', catalog=catalog, exclude=seen)
    for profile, frame in validation.items():
        frame = frame.loc[frame.user_id.isin(tuning)]
        rows = []
        for strength in config['strengths']:
            ranked = rerank_candidates(frame, metadata, strength, config['k'])
            row = {'strength': strength, 'metrics': metrics(ranked, validation_truth, catalog, metadata)}
            rows.append(row)
            print(f'Tune {profile} strength={strength}: NDCG={row["metrics"]["ndcg@20"]:.5f}, tag ILD={row["metrics"]["tag_ild"]:.5f}', flush=True)
        selected = select_strength(rows, config['max_relative_ndcg_loss'])
        result['search'][profile] = rows
        profiles[profile] = {'policy': policies[profile], 'strength': selected['strength']}

    # Freeze both choices before evaluating confirmation/test outcomes.
    manifest = {'config': config, 'profiles': profiles, 'k': config['k'],
                'metadata_path': str(metadata_path),
                'artifact_sha256': {str(p): digest(p) for p in artifact_paths}}
    (root / 'serving_manifest.json').write_text(json.dumps(manifest, indent=2))
    result['profiles'] = profiles
    print('Saved serving manifest and frozen validation choices', flush=True)
    for split in ('validation_confirmation', 'test'):
        if split == 'test':
            frames = rank_split('test')
            truth = build_ground_truth(splits.test, 'is_click', catalog=catalog, exclude=seen)
        else:
            frames = {name: f.loc[f.user_id.isin(confirmation)] for name, f in validation.items()}
            truth = validation_truth
        result['evaluations'][split] = {}
        result['paired_intervals'][split] = {}
        for profile, frame in frames.items():
            baseline = rerank_candidates(frame, metadata, 0, config['k'])
            selected = rerank_candidates(frame, metadata, profiles[profile]['strength'], config['k'])
            for name, ranked in [('baseline', baseline), ('reranked', selected)]:
                result['evaluations'][split][f'{profile}_{name}'] = metrics(ranked, truth, catalog, metadata)
                if name == 'reranked':
                    ranked.to_csv(root / f'{split}_{profile}_top20.csv.gz', index=False, compression='gzip')
            result['paired_intervals'][split][profile] = paired_interval(selected, baseline, truth, config['bootstrap_repeats'], config['seed'])
            print(f'{split} {profile}: ' + json.dumps(result['evaluations'][split][f'{profile}_reranked']), flush=True)
        (root / 'progress.json').write_text(json.dumps(result, indent=2))
    (root / 'results.json').write_text(json.dumps(result, indent=2))
    print('Week 5 offline evaluation complete', flush=True)


if __name__ == '__main__':
    main()
