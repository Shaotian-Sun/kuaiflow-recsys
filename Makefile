.PHONY: install test demo download prepare benchmark

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

