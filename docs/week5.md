# Week 5 operations guide

Week 5 completes the offline five-week prototype with diversity-aware reranking,
a local JSON API, reproducible comparisons, and a [final report](week5_results.md).
The highest-relevance baseline and the full neural architecture are both runnable.

## Build and evaluate

Run from the repository root in the installed environment, with KuaiRand-Pure
downloaded/prepared and the CPU FAISS extra installed. The experiment uses local,
project-produced checkpoints. Generated data and checkpoints are not in Git.

Required predecessors, in order:

1. `make prepare` creates the chronological splits from the downloaded data.
2. `OMP_NUM_THREADS=1 kuaiflow retrieval --config configs/week2_faiss_ivf.yaml --mode faiss`
   creates the reference candidate cohorts when starting from scratch.
3. `make pooling-ablation` creates the mean and attention DIN checkpoints.
4. `make retrieval-budget` then `make retrieval-budget-report` create and verify
   the corrected retrieval and ItemCF caches.
5. `make pipeline-improvement` then `make pipeline-report` create the source
   candidate union, tuning/confirmation partitions, and optimized serving verification.

Existing experiments refuse to overwrite their result directories. Do not rerun
predecessors just to start the service if their verified artifacts already exist.
A clean rebuild produces a new experiment lineage; historical figures remain
snapshots, and each runner records its actual inputs.

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m kuaiflow.week5
OMP_NUM_THREADS=1 .venv/bin/python -m kuaiflow.week5_verify
.venv/bin/python -m kuaiflow.week5_report
```

The first command selects diversity strength on the pre-existing validation
tuning users, writes both choices, then evaluates confirmation and test users.
The second sends real HTTP requests to temporary loopback servers, compares
responses to saved offline recommendations, and measures local latency. The
third regenerates the final report. `make week5` runs all three.

## Start a service

```bash
# Selected relevance baseline (diversity is off by default):
OMP_NUM_THREADS=1 .venv/bin/python -m kuaiflow.serving --profile itemcf --port 8000

# Full hybrid retrieval -> DIN attention -> diversity reranking:
OMP_NUM_THREADS=1 .venv/bin/python -m kuaiflow.serving --profile hybrid_din --diversity --port 8001
```

Run each service in its own terminal. Stop it with Ctrl-C. User IDs below are
examples; requests use the frozen training state for the specified IDs.
Add `--diversity` to either profile to use its frozen tuning-selected setting.
It is opt-in because ItemCF's diversity setting exceeded the relevance-loss
budget on confirmation and test; the experiment was not retuned on those users.

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"user_ids": [1, 2], "k": 20}'
```

Response fields are the profile, recommendation lists with user/video IDs, and
server computation time. Requests accept 1–100 nonnegative integer user IDs and
`k` from 1 to 20. Duplicate user IDs are collapsed. Unknown users start from the
training-popularity fallback; the neural profile can then score/rerank that list.
Training-clicked items are excluded for known users. This service is local and
serialized; it is not a public deployment or concurrent-throughput benchmark.

For Python use:

```python
from kuaiflow.serving import Week5Pipeline

pipeline = Week5Pipeline("artifacts/week5/serving_manifest.json", "hybrid_din", diversity=True)
recommendations = pipeline.recommend([1, 2], k=20)
```

## Modules and artifacts

- `reranking.py`: metadata similarity, greedy MMR, diversity metrics, and
  validation relevance constraint. No labels enter reranking.
- `week5.py`: fixed cohorts, attention-DIN scoring, validation choice,
  confirmation/test comparison, hashes, and saved recommendations.
- `serving.py`: verified manifest loader and loopback HTTP endpoints.
- `week5_verify.py`: real HTTP replay, input/fallback checks, and request timing.
- `week5_report.py`: generated completion report.

`artifacts/week5/` contains the serving manifest, catalog author/tag snapshot,
full ranked candidate caches, selected top-20 lists, validation grid, input/code
hashes, bootstrap intervals, and serving verification. The manifest references
trusted local model files and verifies their hashes before loading. It is not a
portable model bundle; paths resolve from the repository root. Historical
calibrators are kept separate because no deployment exposure policy has been
established.

## Interpretation

Completing the architecture and beating a strong baseline are separate outcomes.
The project implements all five planned stages. The measured ItemCF result stays
visible, the neural path remains available for controlled experiments, and
diversity improvements are reported with their relevance cost. The 2% validation
tolerance is an explicit illustrative choice, not a guarantee on new traffic.
