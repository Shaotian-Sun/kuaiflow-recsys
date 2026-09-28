# Retrieval budget, ranking ceiling and controlled tuning

This experiment keeps the trained single-task mean-pooling ranker from seed 2026 fixed. It measures candidate budgets 100, 200 and 500 on the same 5,000 validation and 5,000 test users, evaluates an oracle ranking ceiling, and runs a small validation-only retrieval search.

## History correction discovered during audit

The original two-tower implementation populated training history row by row. Equal-time clicks from the same user could therefore enter one another's histories. Among the retained earliest positive user/video pairs, 45,249 rows belong to duplicated user/timestamp groups. This is a count of rows in tied groups, not a count of leaked labels or affected examples.

The corrected implementation uses strict timestamp search: a training example sees only clicks with a smaller timestamp. A regression test covers tied clicks, padding, separate users and input permutations. Evaluation histories remain frozen to the training period. The previous retrieval training run was interrupted before any results were used. Earlier Week 2 results and downstream experiments are preserved as historical artifacts; they do not claim the strict tie exclusion now implemented. Rerunning Week 2 with current source will train the corrected model.

The budget experiment therefore starts from a newly trained corrected baseline with the original configuration (temperature 0.07, seed 2026), rather than claiming to reuse a nonexistent Week 2 model checkpoint or reproduce the legacy candidates. This distinction matters: both the historical baseline and its tie-handling issue should be disclosed in downstream comparisons.

## Measurements

- Candidate Recall@K: mean fraction of eligible held-out clicked videos retrieved, with train-seen positives excluded and the catalog restricted to training videos.
- Candidate HitRate@K: fraction of evaluated users with at least one eligible positive in their candidate list.
- Achieved NDCG@20: the frozen mean ranker orders candidates, with retrieval rank as the tie breaker.
- Oracle NDCG@20: put all retrieved held-out positives first, but normalize by the ideal ranking of **all** eligible positives, including those missing from candidates. Users with no retrieved positives contribute zero. This uses held-out labels only as an offline diagnostic upper bound and is not a deployable score.
- Pipeline time: query encoding, retrieval, filtering, candidate construction, metadata joins, ranker feature/history preparation, inference and sorting. Report median milliseconds per user across three repeated batched runs, after one warmup, on the first 256 fixed users. Models and static tables are already loaded. Disk I/O and metrics are excluded. These are amortized batch timings, not individual online-request latency percentiles.

Exact retrieval at K=100 provides an approximation-control reference. Budgets use the same frozen retrieval model, same IVF lists/probes, same seen-item policy, same cohorts and same ranker. Approximate search with different requested K need not be assumed perfectly nested; actual memberships and scores are saved for inspection.

## Validation-only selection

Before running, the candidate budgets and training temperatures were fixed in `configs/retrieval_budget.yaml`. Choose budget by validation final NDCG@20 (smaller K breaks ties). At that budget, compare:

- Corrected original-temperature two-tower + IVF.
- Two-tower temperature 0.04.
- Two-tower temperature 0.10.
- Alternating two-tower and ItemCF candidates, deduplicated, at the same total budget; ItemCF uses 100 neighbors.

The temperature refits change only that training hyperparameter. ItemCF fits training clicks only. The highest validation end-to-end click NDCG@20 selects the retrieval option; write `selection.json` before testing a newly selected option. The baseline budget curve is reported on both splits, but test scores never choose a setting. No ranking model, calibration map, or utility weight is updated. This is a small single-seed search, not an exhaustive optimum or independent new holdout: the existing test period was examined in prior work.

## Reproduce

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m kuaiflow.retrieval_budget
OMP_NUM_THREADS=1 .venv/bin/python -m kuaiflow.retrieval_budget_verify
.venv/bin/python -m kuaiflow.retrieval_budget_report
```

The runner refuses to overwrite its output directory. Change `output_dir` for a repeat. It checks prior ranker provenance, candidate uniqueness/counts, catalog and train-seen filtering, finite scores and the oracle upper bound. The baseline retrieval object is saved and reloaded to verify item/query embeddings, and embedding arrays are saved separately. `baseline_retriever.pt` is a trusted local full-object PyTorch cache: only load files produced by this project, never untrusted pickle files. Selected temperature models use the same local cache format.

Results, configurations, input/source hashes, training curves, candidate scores, frozen embeddings and selection records are stored under `artifacts/retrieval_budget_causal`. See [measured results](retrieval_budget_results.md).

The verification command separately checks unchanged top-100 prefixes and identical common-candidate logits across budgets. For a selected hybrid it also saves/reloads the ItemCF component and verifies exact regenerated candidate lists on 64 users per split. ItemCF-only plus this frozen ranker is not included in this search, so a hybrid gain does not by itself prove complementary contributions from both sources.
