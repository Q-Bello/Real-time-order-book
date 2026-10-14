# Full pipeline: make record MINUTES=120 && make research
MINUTES ?= 60

.PHONY: install record dataset evaluate report research live dashboard test lint demo

install:
	pip install -r requirements.txt

record:
	python -m lobpred.record --minutes $(MINUTES)

dataset:
	python -m lobpred.dataset data/raw/*.jsonl.gz

evaluate:
	python -m lobpred.evaluate

report:
	python -m lobpred.report

research: dataset evaluate report

live:
	python -m lobpred.live

dashboard:
	python dashboard/app.py

test:
	python -m pytest -q

lint:
	python -m ruff check src tests dashboard

# Runs the whole pipeline on simulated data in a scratch folder (no network needed).
demo:
	python -m lobpred.simulate --minutes 60 --out data/demo/raw
	python -m lobpred.dataset data/demo/raw/*.jsonl.gz --out data/demo/dataset.parquet --report-dir data/demo/reports
	python -m lobpred.evaluate --data data/demo/dataset.parquet --report-dir data/demo/reports --oof-dir data/demo/oof --model-dir data/demo/models --fast
	python -m lobpred.report --report-dir data/demo/reports --oof-dir data/demo/oof
