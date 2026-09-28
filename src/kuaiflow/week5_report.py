"""Build the Week 5 and five-week completion report from measured artifacts."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='artifacts/week5')
    args = parser.parse_args()
    root = Path(args.directory)
    result = json.loads((root / 'results.json').read_text())
    serving = json.loads((root / 'serving_verification.json').read_text())
    lines = ['# Week 5: diversity reranking and local serving', '',
             'The five-week offline industry-style prototype is implemented: chronological data and baselines, '
             'two-tower/FAISS retrieval, DeepFM/DIN/MMoE ranking, exposure calibration, diversity reranking, '
             'and a working local recommendation API. ItemCF remains the selected relevance baseline. '
             'The complete two-tower + ItemCF → DIN attention → MMR path is retained as a separate runnable profile. '
             '**Diversity reranking is opt-in.** The selected ItemCF setting missed the 98% relevance-retention target on confirmation and test. '
             'After that check, the serving default was conservatively left at zero reranking strength; the experimental choice and results remain unchanged.', '',
             '## Frozen protocol', '',
             '- Same training cutoff, warm catalog, novel-positive ground truth, and 5,000 validation/test users as the preceding experiments.',
             '- Frozen seed-2026 DIN attention checkpoint; corrected strict-history two-tower; training-only ItemCF. No retraining or test-based hyperparameter selection.',
             '- Reuse the preceding 2,500 tuning / 2,500 confirmation validation-user partition. These users and periods have been examined before; confirmation is not a fresh independent holdout.',
             '- Each profile starts with 100 unique unseen candidates. Reranking chooses 20 only from those candidates, so it cannot improve candidate Recall@100.',
             '- The hybrid profile interleaves two-tower/FAISS and ItemCF candidates, then orders them with DIN attention click logits. The ItemCF profile retains its source ordering.',
             '- MMR greedily maximizes `(1-strength) * relevance - strength * max_similarity_to_selected`. Relevance is the base-rank percentile; similarity is half same-author indicator plus half tag-set Jaccard similarity. Ties retain base order.',
             '- Missing tags/authors contribute no matching similarity. Reported tag diversity excludes unknown-tag pairs and separately reports metadata coverage.',
             '- Search strengths: 0, 0.05, 0.1, 0.2, 0.4. Maximize tuning-user mean tag diversity subject to NDCG@20 at least 98% of that profile\'s no-reranking control. The 2% tolerance is an illustrative offline product constraint, not a proven business requirement.',
             '- Both selected strengths are written to the serving manifest before confirmation/test metrics are computed. A zero-strength control is always eligible.', '',
             '## Validation selection', '',
             '| Profile | Strength | NDCG@20 (%) | Tag diversity | Unique-author fraction |',
             '| --- | ---: | ---: | ---: | ---: |']
    for profile, rows in result['search'].items():
        for row in rows:
            m = row['metrics']
            chosen = ' **selected**' if row['strength'] == result['profiles'][profile]['strength'] else ''
            lines.append(f"| {profile}{chosen} | {row['strength']} | {100*m['ndcg@20']:.3f} | {m['tag_ild']:.4f} | {m['unique_author_fraction']:.4f} |")
    for split, rows in result['evaluations'].items():
        lines += ['', f'## {split}', '',
                  '| Profile | NDCG@20 (%) | Recall@20 (%) | Coverage@20 (%) | Tag diversity | Unique-author fraction |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for name, m in rows.items():
            lines.append(f"| {name} | {100*m['ndcg@20']:.3f} | {100*m['recall@20']:.3f} | {100*m['coverage@20']:.3f} | {m['tag_ild']:.4f} | {m['unique_author_fraction']:.4f} |")
        lines += ['', 'Tag diversity is mean within-list `1 - Jaccard(tags)` over known-tag pairs; '
                  'author diversity is distinct known authors divided by known-author slots. These are metadata proxies, not user satisfaction. '
                  'Catalog coverage is a separate population-level metric.', '']
        for profile, interval in result['paired_intervals'][split].items():
            low, high = interval['percentile95']
            lines.append(f"- {profile}: reranked minus baseline NDCG@20 **{100*interval['mean_difference']:+.3f} pp**, paired 95% interval [{100*low:+.3f}, {100*high:+.3f}] pp.")
            baseline = rows[f'{profile}_baseline']['ndcg@20']
            ratio = rows[f'{profile}_reranked']['ndcg@20'] / baseline
            lines.append(f"  Observed NDCG retention: {100*ratio:.2f}%; the 98% tuning constraint {'holds' if ratio >= .98 else 'does not hold'} on this cohort.")
        minimum = min(m['tag_metadata_fraction'] for m in rows.values())
        lines.append(f'- Minimum known-tag slot fraction across these lists: {100*minimum:.2f}%.')
    lines += ['', 'Intervals use 1,000 paired user-bootstrap replicates and condition on fixed models and chosen strengths. They do not include training or policy-selection uncertainty.', '',
              '## Serving verification', '',
              'The service binds only to `127.0.0.1`. Both profiles verify checkpoint/metadata hashes at startup. '
              'The hybrid profile builds its IVF-100/10 index from the corrected saved embeddings; ItemCF skips the two-tower and DIN. '
              'The `--diversity` flag enables the frozen tuning-selected setting; default requests preserve the baseline order. '
              'Inference consumes user IDs and frozen training state, never evaluation outcomes.', '',
              '| Profile | Startup (s) | Single request median (ms) | Single request p95 (ms) | Batch amortized (ms/user) | Exact replay users |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, s in serving['profiles'].items():
        lines.append(f"| {name} | {s['startup_seconds']:.3f} | {s['single_request_median_ms']:.3f} | {s['single_request_p95_ms']:.3f} | {s['batch_amortized_ms_per_user']:.3f} | {s['http_exact_top20_users']} |")
    lines += ['', 'Latency includes loopback HTTP, JSON serialization, and the in-memory recommendation path. '
              'Single-request values use 32 sequential one-user requests after five warmups; batching uses three 64-user requests. '
              'These are local CPU observations, not a production p95 or load-test SLA. Startup is excluded from request timings.', '',
              'Replay checks verify exact saved top-20 order, top-5 prefix consistency, duplicate user handling, unknown-user fallback, '
              'catalog membership, and invalid-input rejection. Temporary verification servers are stopped after testing.', '',
              '## Five-week status', '',
              '1. **Week 1 complete:** chronological preprocessing and Popularity/ItemCF/BPR controls. [Report](week1_results.md).',
              '2. **Week 2 complete, with correction:** learned retrieval, FAISS comparisons, and strict tied-timestamp history exclusion. Earlier saved retrieval metrics remain historical. [Corrected retrieval report](retrieval_budget_results.md).',
              '3. **Week 3 complete:** DeepFM, DIN, DeepFM+MMoE, DIN+MMoE, matched pooling and fixed-utility comparisons. Neural stages have not beaten ItemCF on the current novel-item click benchmark. [Ranking report](week3_results.md), [pooling](pooling_ablation_results.md), [MMoE](mmoe_pooling_ablation_results.md).',
              '4. **Week 4 complete:** frozen standard/random exposure calibration diagnostics. Calibrators remain exposure-specific; they are not silently applied to the serving policy. A positive-slope single-task calibration cannot change item ordering. [Report](week4_results.md).',
              '5. **Week 5 complete:** measured relevance/diversity selection, local API with ItemCF and complete neural profiles, replay checks, latency measurements, and this final report.', '',
              '## Scope and remaining work', '',
              'This completes the five-week **offline prototype**, not a deployed industrial recommender. '
              'No live feature ingestion, impression logging, online learning, authenticated public endpoint, concurrent load test, '
              'A/B test, or causal policy lift is claimed. Serving uses training-cutoff histories and requires local data/checkpoints. '
              'Month-aggregated video engagement statistics are excluded. MMoE remains a measured offline alternative rather than the default API scorer.', '',
              'Future model work should investigate training/evaluation mismatch and retrieval-aware ranking against the ItemCF control. '
              'It is not a prerequisite for finishing the pipeline or a reason to conceal the baseline result. '
              'The original serving source captured during evaluation is preserved as `serving_at_evaluation.py`; '
              'HTTP replay records the updated serving-source hash after adding the explicit opt-in flag.', '',
              '## Reproduction', '',
              'See [Week 5 operations guide](week5.md) for build order, API examples, artifacts, and restart instructions. '
              'The evaluator refuses to overwrite an existing output directory. Change `output_dir` in a copied YAML to repeat; '
              'verification and report generation accept `--directory` for the new result directory.']
    Path('docs/week5_results.md').write_text('\n'.join(lines) + '\n')
    print('Wrote docs/week5_results.md')


if __name__ == '__main__':
    main()
