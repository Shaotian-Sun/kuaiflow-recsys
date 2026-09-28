"""Local Week 5 HTTP service with fixed, verified offline model profiles."""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pandas as pd
import torch

from kuaiflow.pipeline import RecommendationPipeline
from kuaiflow.pooling_ablation import digest
from kuaiflow.reranking import DiversityMetadata, rerank_candidates


class Week5Pipeline:
    def __init__(self, manifest, profile='itemcf', diversity=False):
        payload = json.loads(Path(manifest).read_text())
        if profile not in payload['profiles']:
            raise ValueError('Unknown serving profile')
        for path, expected in payload['artifact_sha256'].items():
            if digest(path) != expected:
                raise ValueError('Serving artifact hash mismatch: ' + path)
        self.profile = profile
        self.spec = payload['profiles'][profile]
        self.diversity = diversity
        # Opt-in: the tuning-selected setting exceeded the relevance budget on confirmation.
        self.strength = self.spec['strength'] if diversity else 0.
        self.k = payload['k']
        self.base = RecommendationPipeline(payload['config'], self.spec['policy'])
        self.metadata = DiversityMetadata(pd.read_csv(payload['metadata_path'], dtype={'tag': str}))

    def recommend(self, users, k=20):
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= self.k:
            raise ValueError(f'k must be an integer from 1 to {self.k}')
        if not isinstance(users, list) or not 1 <= len(users) <= 100:
            raise ValueError('user_ids must be a list containing 1 to 100 IDs')
        if any(not isinstance(u, int) or isinstance(u, bool) or u < 0 or u > 2**63-1 for u in users):
            raise ValueError('user_ids must contain nonnegative signed 64-bit integers')
        users = list(dict.fromkeys(users))
        frame = self.base.recommend(users, self.spec['policy']['candidate_budget'])
        return rerank_candidates(frame, self.metadata, self.strength, k)


def make_handler(pipeline):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, status, payload):
            body = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/health':
                self.send_json(200, {'status': 'ready', 'profile': pipeline.profile,
                                     'candidate_budget': pipeline.spec['policy']['candidate_budget'],
                                     'diversity_enabled': pipeline.diversity,
                                     'mmr_strength': pipeline.strength})
            else:
                self.send_json(404, {'error': 'Not found'})

        def do_POST(self):
            if self.path != '/recommend':
                self.send_json(404, {'error': 'Not found'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 65536:
                    self.send_json(413, {'error': 'Body must contain 1 to 65536 bytes'})
                    return
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict) or set(request) - {'user_ids', 'k'}:
                    raise ValueError('Expected an object with user_ids and optional k')
                started = time.perf_counter()
                frame = pipeline.recommend(request.get('user_ids'), request.get('k', 20))
                results = [{'user_id': int(u), 'video_ids': g.video_id.astype(int).tolist()}
                           for u, g in frame.groupby('user_id', sort=False)]
                self.send_json(200, {'profile': pipeline.profile, 'diversity_enabled': pipeline.diversity,
                                     'recommendations': results,
                                     'compute_ms': (time.perf_counter() - started) * 1000})
            except (ValueError, TypeError, UnicodeDecodeError) as exc:
                self.send_json(400, {'error': str(exc)})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='artifacts/week5/serving_manifest.json')
    parser.add_argument('--profile', choices=['itemcf', 'hybrid_din'], default='itemcf')
    parser.add_argument('--diversity', action='store_true', help='Opt into the measured diversity/relevance tradeoff')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    pipeline = Week5Pipeline(args.manifest, args.profile, diversity=args.diversity)
    # Serialized local requests avoid concurrent mutable model/index access.
    server = HTTPServer(('127.0.0.1', args.port), make_handler(pipeline))
    print(f'Ready: http://127.0.0.1:{server.server_port} profile={args.profile}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
