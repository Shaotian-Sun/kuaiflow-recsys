"""Replay the frozen Week 5 policies through real loopback HTTP requests."""
import argparse
import json
import threading
import time
from http.server import HTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import torch

from kuaiflow.pooling_ablation import digest
from kuaiflow.serving import Week5Pipeline, make_handler


def request_json(url, payload=None):
    request = Request(url, data=None if payload is None else json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='artifacts/week5')
    args = parser.parse_args()
    root = Path(args.directory)
    manifest = root / 'serving_manifest.json'
    torch.set_num_threads(1)
    results = {'profiles': {}, 'manifest_sha256': digest(manifest),
               'source_sha256': {str(p): digest(p) for p in [Path('src/kuaiflow/serving.py'),
                                                            Path('src/kuaiflow/reranking.py')]}}
    for profile, diversity in [('itemcf', False), ('itemcf', True), ('hybrid_din', False), ('hybrid_din', True)]:
        start = time.perf_counter()
        pipeline = Week5Pipeline(manifest, profile, diversity=diversity)
        startup = time.perf_counter() - start
        saved = pd.read_csv(root / (f'test_{profile}_top20.csv.gz' if diversity else f'test_{profile}_candidates.csv.gz'))
        saved = saved.loc[saved.final_rank <= 20]
        users = saved.user_id.drop_duplicates().tolist()[:128]
        expected = {u: g.video_id.tolist() for u, g in saved.loc[saved.user_id.isin(users)].groupby('user_id')}
        server = HTTPServer(('127.0.0.1', 0), make_handler(pipeline))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f'http://127.0.0.1:{server.server_port}'
        try:
            health = request_json(url + '/health')
            assert health['profile'] == profile and health['status'] == 'ready'
            assert health['diversity_enabled'] == diversity
            for start in range(0, len(users), 64):
                batch = users[start:start+64]
                response = request_json(url + '/recommend', {'user_ids': batch, 'k': 20})
                actual = {r['user_id']: r['video_ids'] for r in response['recommendations']}
                assert actual == {u: expected[u] for u in batch}, 'HTTP recommendations differ from offline output'
            # Unknown users exercise training-only popularity fallback plus any ranking/reranking.
            source = pipeline.base.itemcf or pipeline.base.tower
            unknown = max(int(u) for u in source.user_ids) + 1000000
            cold = request_json(url + '/recommend', {'user_ids': [unknown, unknown], 'k': 20})
            assert len(cold['recommendations']) == 1
            cold_items = cold['recommendations'][0]['video_ids']
            assert len(cold_items) == len(set(cold_items)) == 20
            assert set(cold_items) <= set(pipeline.metadata.lookup)
            small = request_json(url + '/recommend', {'user_ids': [users[0]], 'k': 5})
            assert small['recommendations'][0]['video_ids'] == expected[users[0]][:5]
            for payload in [{'user_ids': []}, {'user_ids': [1], 'k': 21}, {'user_ids': [True]},
                            {'user_ids': [1], 'model_path': 'untrusted'}, ['not-an-object']]:
                try:
                    request_json(url + '/recommend', payload)
                    raise AssertionError('Invalid request accepted')
                except HTTPError as exc:
                    assert exc.code == 400
            for _ in range(5):
                request_json(url + '/recommend', {'user_ids': [users[0]], 'k': 20})
            timings = []
            for user in users[:32]:
                start = time.perf_counter()
                request_json(url + '/recommend', {'user_ids': [user], 'k': 20})
                timings.append((time.perf_counter() - start) * 1000)
            batch_times = []
            for _ in range(3):
                start = time.perf_counter()
                request_json(url + '/recommend', {'user_ids': users[:64], 'k': 20})
                batch_times.append((time.perf_counter() - start) * 1000)
            name = profile + ('_diversity' if diversity else '_baseline')
            results['profiles'][name] = {
                'startup_seconds': startup, 'http_exact_top20_users': len(users),
                'cold_user_checked': True, 'request_validation_checked': True,
                'single_request_users': 1, 'single_request_samples': len(timings),
                'single_request_median_ms': float(np.median(timings)),
                'single_request_p95_ms': float(np.quantile(timings, .95)),
                'single_request_samples_ms': timings, 'batch_users': 64,
                'batch_median_ms': float(np.median(batch_times)),
                'batch_amortized_ms_per_user': float(np.median(batch_times) / 64),
                'two_tower_loaded': pipeline.base.tower is not None,
                'faiss_index_loaded': pipeline.base.index is not None,
                'din_loaded': pipeline.base.ranker is not None,
            }
            print(name + ': ' + json.dumps(results['profiles'][name]), flush=True)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    (root / 'serving_verification.json').write_text(json.dumps(results, indent=2))
    print('Both profiles verified over HTTP; all temporary servers stopped', flush=True)


if __name__ == '__main__':
    main()
