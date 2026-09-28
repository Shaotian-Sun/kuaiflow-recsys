# Commit the experiments and Week 5 separately

Unexpected results belong in version control. The experiment commit records a
reproducible comparison and a correctness fix; it should not claim that the
neural architecture beats ItemCF. This guide records the two-commit grouping;
the commits were created at the user's request.
Week 4 and the single-task pooling ablation are already committed in this checkout.

Run the following from the repository root. First review `git status --short`
and `git diff`; preserve any additional changes made outside this task.

## 1. Earlier experiment checkpoint

```bash
git add configs/mmoe_pooling_ablation.yaml configs/retrieval_budget.yaml configs/pipeline_improvement.yaml \
  src/kuaiflow/mmoe_pooling_ablation.py src/kuaiflow/mmoe_pooling_report.py \
  src/kuaiflow/models/two_tower.py \
  src/kuaiflow/retrieval_budget.py src/kuaiflow/retrieval_budget_report.py src/kuaiflow/retrieval_budget_verify.py \
  src/kuaiflow/pipeline.py src/kuaiflow/pipeline_improvement.py src/kuaiflow/pipeline_report.py src/kuaiflow/recommend.py \
  tests/test_mmoe_pooling.py tests/test_retrieval_budget.py tests/test_pipeline.py \
  docs/week2_results.md docs/mmoe_pooling_ablation.md docs/mmoe_pooling_ablation_results.md \
  docs/retrieval_budget.md docs/retrieval_budget_results.md docs/pipeline_improvement_results.md
git add -p README.md Makefile
git diff --cached --stat
git diff --cached
git commit -m "Evaluate pooling and retrieval policies; retain ItemCF baseline"
```

In the interactive README/Makefile step, stage only the MMoE-pooling,
retrieval-budget, and pipeline-comparison additions. Leave the Week 5/service
changes for the second commit; use `s` to split a combined hunk if necessary.
It is also safe to defer both shared files entirely to the second commit.

The first commit records three findings:

- The matched MMoE/pooling comparison and fixed scoring policies.
- Strict exclusion of same-timestamp training history, plus corrected retrieval
  budgets, oracle ceilings, and checkpoint verification.
- Whole-pipeline controls, validation source/score selection, and ItemCF as the
  best tested relevance baseline. Earlier metrics remain identified as historical.

Large model/data files stay ignored. The tracked configs, experiment runners,
tests, and generated Markdown reports retain the implementation and measured
findings. Preserve the local `artifacts/` directory separately if exact checkpoint
replay is needed; a Git commit does not back up ignored weights or datasets.

## 2. Complete Week 5

```bash
git add configs/week5.yaml src/kuaiflow/reranking.py src/kuaiflow/serving.py \
  src/kuaiflow/week5.py src/kuaiflow/week5_verify.py src/kuaiflow/week5_report.py \
  tests/test_week5.py docs/week5.md docs/week5_results.md docs/commit_plan.md README.md Makefile
git diff --cached --check
git diff --cached --stat
git diff --cached
git commit -m "Complete Week 5 reranking and local recommendation serving"
```

The second commit contains the MMR tradeoff, frozen validation choices, local
API, exact HTTP replay, and five-week report. The default keeps baseline ordering;
diversity remains opt-in after its ItemCF setting misses the held-out relevance
constraint. This is completion of an offline industry-style prototype, not a
production rollout or online-lift claim.
