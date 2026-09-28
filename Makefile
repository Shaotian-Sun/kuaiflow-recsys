.PHONY: install test demo download prepare benchmark deepfm mmoe din din-mmoe week3-figures week3-comparison week4 week4-report

install:
	python -m pip install -e .

test:
	python -m unittest discover -s tests -v

demo:
	python -m kuaiflow.cli demo

download:
	python -m kuaiflow.cli download --config configs/week1.yaml

prepare:
	python -m kuaiflow.cli prepare --config configs/week1.yaml

benchmark:
	python -m kuaiflow.cli benchmark --config configs/week1.yaml

deepfm:
	python -m kuaiflow.cli deepfm --config configs/week3_deepfm.yaml

mmoe:
	python -m kuaiflow.cli mmoe --config configs/week3_mmoe.yaml

week3-figures:
	python -m kuaiflow.week3_figures

din:
	python -m kuaiflow.cli din --config configs/week3_din.yaml

din-mmoe:
	python -m kuaiflow.cli din-mmoe --config configs/week3_din_mmoe.yaml

week3-comparison:
	python -m kuaiflow.week3_comparison

week4:
	python -m kuaiflow.cli week4 --config configs/week4.yaml

week4-report:
	python -m kuaiflow.week4_report

.PHONY: pooling-ablation pooling-report
pooling-ablation:
	OMP_NUM_THREADS=1 python -m kuaiflow.pooling_ablation

pooling-report:
	python -m kuaiflow.pooling_report

.PHONY: mmoe-pooling-ablation mmoe-pooling-report
mmoe-pooling-ablation:
	OMP_NUM_THREADS=1 python -m kuaiflow.mmoe_pooling_ablation

mmoe-pooling-report:
	python -m kuaiflow.mmoe_pooling_report

.PHONY: retrieval-budget retrieval-budget-report
retrieval-budget:
	OMP_NUM_THREADS=1 python -m kuaiflow.retrieval_budget

retrieval-budget-report:
	OMP_NUM_THREADS=1 python -m kuaiflow.retrieval_budget_verify
	python -m kuaiflow.retrieval_budget_report

.PHONY: pipeline-improvement pipeline-report
pipeline-improvement:
	OMP_NUM_THREADS=1 python -m kuaiflow.pipeline_improvement

pipeline-report:
	OMP_NUM_THREADS=1 python -m kuaiflow.pipeline_report

.PHONY: week5 week5-evaluate week5-verify week5-report serve
week5: week5-evaluate
	$(MAKE) week5-verify
	$(MAKE) week5-report

week5-evaluate:
	OMP_NUM_THREADS=1 python -m kuaiflow.week5

week5-verify:
	OMP_NUM_THREADS=1 python -m kuaiflow.week5_verify

week5-report:
	python -m kuaiflow.week5_report

serve:
	OMP_NUM_THREADS=1 python -m kuaiflow.serving
