"""Frozen-ranker retrieval budget, oracle ceiling and validation tuning experiment."""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from kuaiflow.data import load_prepared, load_kuairand_features
from kuaiflow.models.two_tower import TwoTowerRecommender
from kuaiflow.models.itemcf import ItemCFRecommender
from kuaiflow.retrieval import (FAISSRetriever, ExactRetriever, _get_user_embeddings_batch,
                               _faiss_search_k, _postprocess_faiss_recommendations, _candidate_rows)
from kuaiflow.ranking import load_deepfm_artifacts, _attach_static_features, _predict_logits, _recommendation_map
from kuaiflow.metrics import build_ground_truth, evaluate_recommendations
from kuaiflow.pooling_ablation import digest


def oracle_ndcg(candidates, truth, k=20):
    """Ideal order within candidate set; denominator uses ALL eligible positives."""
    values = []
    for user, items in candidates.items():
        relevant = truth.get(user, set())
        if not relevant:
            continue
        count = min(k, len(set(items) & relevant))
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
        values.append(sum(1 / math.log2(i + 2) for i in range(count)) / ideal)
    if not values:
        raise ValueError('No eligible users')
    return float(np.mean(values))


def interleave(first, second, k):
    result = {}
    for user, items in first.items():
        merged = list(dict.fromkeys(item for pair in zip(items, second[user]) for item in pair))
        merged += [item for item in items if item not in merged]
        if len(merged) < k:
            raise ValueError('Insufficient hybrid candidates')
        result[user] = merged[:k]
    return result


def score_candidates(frame, model, encoder, users, videos):
    parts = []
    for start in range(0, len(frame), 100000):
        chunk = frame.iloc[start:start + 100000].copy()
        feature = _attach_static_features(chunk, users, videos, encoder.categorical + encoder.numeric)
        cat, num = encoder.transform(feature)
        hist = model.history_index.transform(chunk)
        chunk['ranker_logit'] = _predict_logits(model, cat, num, 2048, torch.device('cpu'), hist)
        parts.append(chunk)
    scored = pd.concat(parts, ignore_index=True)
    if not np.isfinite(scored.ranker_logit).all():
        raise ValueError('Nonfinite ranker scores')
    ranked = scored.sort_values(['user_id', 'ranker_logit', 'retrieval_rank'], ascending=[True, False, True], kind='stable')
    ranked['ranker_rank'] = ranked.groupby('user_id', sort=False).cumcount() + 1
    return ranked


def validate_candidates(recs, seen, catalog, budget):
    for user, items in recs.items():
        if len(items) != budget or len(set(items)) != budget or set(items) - catalog or set(items) & seen.get(user, set()):
            raise ValueError('Invalid candidate membership, count or seen-item exclusion')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/retrieval_budget.yaml')
    args = parser.parse_args()
    spec = yaml.safe_load(Path(args.config).read_text())
    cfg = yaml.safe_load(Path(spec['retrieval_config']).read_text())
    root = Path(spec['output_dir'])
    if root.exists():
        raise FileExistsError('Choose a fresh output_dir')
    root.mkdir(parents=True)
    torch.set_num_threads(1)
    splits = load_prepared(cfg['data']['processed_dir'])
    users, videos = load_kuairand_features(cfg['data']['raw_dir'])
    old_candidates = pd.read_csv(cfg['data']['candidates_path'])
    user_ids = {s: old_candidates.loc[old_candidates.split == s].user_id.drop_duplicates().tolist() for s in ('validation', 'test')}
    catalog = splits.train.video_id.drop_duplicates().tolist()
    seen = {u: set(g.video_id) for u, g in splits.train.loc[splits.train.is_click > 0].groupby('user_id')}
    truth = {s: build_ground_truth(getattr(splits, s), 'is_click', catalog=catalog, exclude=seen) for s in user_ids}
    directory = Path(spec['ranker_dir'])
    ranker, encoder = load_deepfm_artifacts(directory / 'week3_din_model.pt', directory / 'week3_din_encoder.json')
    if ranker.history_pooling != 'mean':
        raise ValueError('Expected frozen mean-pooling ranker')
    # Refuse reuse if the previous ranker experiment's inputs or source have changed.
    prior = json.loads((directory.parent.parent / 'results.json').read_text())
    for path, expected in prior['sha256'].items():
        if digest(path) != expected:
            raise ValueError('Changed baseline provenance: ' + path)
    paths = [args.config, spec['retrieval_config'], cfg['data']['candidates_path']]
    paths += [str(p) for p in directory.glob('week3_din_*') if p.suffix != '.csv']
    paths += [str(p) for p in Path('src/kuaiflow').rglob('*.py')]
    output = {'specification': spec, 'retrieval_config': cfg, 'sha256': {p: digest(p) for p in paths},
              'prior_input_sha256': prior['sha256'], 'rows': [], 'fits': [],
              'history_correction': 'strictly earlier timestamps; replaces legacy within-timestamp ordering',
              'protocol': 'seed 2026; fixed user cohorts; validation selection; same frozen mean ranker; no test tuning'}
    print('Training corrected causal baseline retrieval', flush=True)
    model = TwoTowerRecommender(seed=cfg['seed'], use_history=True, **cfg['model'])
    started = time.perf_counter()
    model.fit(splits.train, user_features=users, video_features=videos)
    output['fits'].append({'variant': 'baseline', 'seconds': time.perf_counter()-started, 'loss': model.training_history})
    # Trusted local cache of the full fitted object for future controlled experiments.
    torch.save(model, root / 'baseline_retriever.pt')
    restored = torch.load(root / 'baseline_retriever.pt', weights_only=False, map_location='cpu')
    np.testing.assert_array_equal(restored.item_vectors, model.item_vectors)
    q = {s: _get_user_embeddings_batch(model, ids) for s, ids in user_ids.items()}
    np.testing.assert_array_equal(_get_user_embeddings_batch(restored, user_ids['validation'][:8]), q['validation'][:8])
    np.savez_compressed(root / 'baseline_embeddings.npz', item_ids=np.asarray(model.item_ids), item_vectors=model.item_vectors,
                        **{s+'_users': np.asarray(ids) for s, ids in user_ids.items()}, **q)
    del restored
    ivf = FAISSRetriever(**cfg['faiss'], seed=cfg['seed'])
    ivf.build_index(model.item_vectors, model.item_ids)
    exact = ExactRetriever(model.item_vectors, model.item_ids)

    def retrieve(fitted, retriever, ids, embeddings, budget):
        raw, _, _ = retriever.search(embeddings, _faiss_search_k(fitted, ids, budget))
        return _postprocess_faiss_recommendations(fitted, ids, raw, budget)

    def evaluate(variant, fitted, retriever, split, budget, queries=None, second=None):
        ids = user_ids[split]
        embeddings = queries if queries is not None else _get_user_embeddings_batch(fitted, ids)
        recs = retrieve(fitted, retriever, ids, embeddings, budget)
        if second is not None:
            recs = interleave(recs, second.recommend(ids, budget), budget)
        validate_candidates(recs, seen, set(catalog), budget)
        frame = pd.DataFrame(_candidate_rows(split, ids, recs))
        ranked = score_candidates(frame, ranker, encoder, users, videos)
        rankings = _recommendation_map(ranked, 'ranker_rank')
        metrics = evaluate_recommendations(rankings, truth[split], 20, catalog)
        recall = evaluate_recommendations(recs, truth[split], budget, catalog)
        ceiling = oracle_ndcg(recs, truth[split])
        if ceiling + 1e-12 < metrics['ndcg@20']:
            raise ValueError('Oracle below achieved ranking')
        # Measure the complete in-memory pipeline on a fixed subset; labels and disk I/O excluded.
        subset = ids[:spec['latency_users']]
        durations = []
        for repeat in range(spec['latency_repeats'] + 1):
            start = time.perf_counter()
            queries_small = _get_user_embeddings_batch(fitted, subset)
            small = retrieve(fitted, retriever, subset, queries_small, budget)
            if second is not None:
                small = interleave(small, second.recommend(subset, budget), budget)
            score_candidates(pd.DataFrame(_candidate_rows(split, subset, small)), ranker, encoder, users, videos)
            if repeat:
                durations.append((time.perf_counter()-start)*1000/len(subset))
        name = f'{variant}_{split}_{budget}'
        ranked.to_csv(root / (name + '.csv.gz'), index=False, compression='gzip')
        row = {'variant': variant, 'split': split, 'budget': budget,
               'candidate_recall': recall[f'recall@{budget}'], 'candidate_hit_rate': recall[f'hit_rate@{budget}'],
               'oracle_ndcg20': ceiling, 'ranking': metrics,
               'latency_ms_per_user': float(np.median(durations)), 'latency_samples': durations}
        output['rows'].append(row)
        (root / 'progress.json').write_text(json.dumps(output, indent=2))
        print(json.dumps(row), flush=True)
        return row

    for split in user_ids:
        for budget in spec['budgets']:
            evaluate('ivf_baseline', model, ivf, split, budget, q[split])
    # Exact reference separates ANN approximation from learned retrieval quality.
    for split in user_ids:
        evaluate('exact_baseline', model, exact, split, 100, q[split])
    validation = [r for r in output['rows'] if r['split']=='validation' and r['variant']=='ivf_baseline']
    selected_budget = max(validation, key=lambda r: (r['ranking']['ndcg@20'], -r['budget']))['budget']
    output['validation_selected_budget'] = selected_budget
    print(f'Validation-selected budget: {selected_budget}; starting fixed temperature grid', flush=True)
    # Two train-only refits, selected by validation end-to-end click NDCG at the chosen budget.
    fitted_options = {'ivf_baseline': (model, ivf, None)}
    for temperature in spec['temperatures']:
        variant = f'temperature_{temperature}'
        options = dict(cfg['model'], temperature=temperature)
        candidate = TwoTowerRecommender(seed=cfg['seed'], use_history=True, **options)
        start = time.perf_counter()
        candidate.fit(splits.train, user_features=users, video_features=videos)
        output['fits'].append({'variant': variant, 'seconds': time.perf_counter()-start, 'loss': candidate.training_history})
        torch.save(candidate, root / (variant + '_retriever.pt'))
        index = FAISSRetriever(**cfg['faiss'], seed=cfg['seed'])
        index.build_index(candidate.item_vectors, candidate.item_ids)
        evaluate(variant, candidate, index, 'validation', selected_budget)
        fitted_options[variant] = (candidate, index, None)
    print('Testing fixed ItemCF interleaving on validation', flush=True)
    cf = ItemCFRecommender(neighbor_k=100).fit(splits.train, label_col='is_click')
    evaluate('ivf_itemcf_interleave', model, ivf, 'validation', selected_budget, q['validation'], cf)
    fitted_options['ivf_itemcf_interleave'] = (model, ivf, cf)
    eligible = [r for r in output['rows'] if r['split']=='validation' and r['budget']==selected_budget and r['variant'] in fitted_options]
    winner = max(eligible, key=lambda r: r['ranking']['ndcg@20'])['variant']
    output['validation_selected_variant'] = winner
    (root / 'selection.json').write_text(json.dumps({'budget': selected_budget, 'variant': winner, 'criterion': 'validation end-to-end click NDCG@20'}, indent=2))
    if winner != 'ivf_baseline':
        fitted, retriever, second = fitted_options[winner]
        evaluate(winner, fitted, retriever, 'test', selected_budget, second=second)
    (root / 'results.json').write_text(json.dumps(output, indent=2))
    print('Experiment complete: ' + winner + ', budget=' + str(selected_budget), flush=True)


if __name__ == '__main__':
    main()
