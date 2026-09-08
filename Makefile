.PHONY: install test demo download prepare benchmark deepfm mmoe week3-figures

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
