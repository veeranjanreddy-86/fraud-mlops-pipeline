# Card-fraud MLOps pipeline - developer entry points.
VENV   ?= .venv
PYTHON ?= python3.11
BIN    := $(VENV)/bin
CLI    := $(BIN)/fraud-mlops
DATA   := data/transactions.parquet

export MLFLOW_DISABLE_AGENT_HINT ?= 1

.PHONY: help install lint format test generate train train-smote gate drift export serve docker-build pipeline clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install:  ## Create the virtualenv and install pinned dependencies + package (editable)
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -e . --no-deps

lint:  ## Ruff lint + format check
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:  ## Auto-fix lint issues and format
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

test:  ## Run the test suite
	$(BIN)/pytest

$(DATA):
	$(CLI) generate --out $(DATA)

generate: $(DATA)  ## Generate the 50k-row synthetic training set

train: $(DATA)  ## Train + register a candidate (class weights, precision-target threshold)
	$(CLI) train --run-name baseline-xgb

train-smote: $(DATA)  ## Train + register a SMOTE candidate
	$(CLI) train --smote --run-name xgb-smote

gate: $(DATA)  ## Champion/challenger gate on the latest registered version
	$(CLI) gate

drift: $(DATA)  ## Generate a shifted "production" batch and write a PSI/KS drift report
	$(CLI) generate --rows 10000 --seed 202 --start 2025-04-01 --drift 1.0 --out data/current.parquet
	$(CLI) drift --reference $(DATA) --current data/current.parquet

export:  ## Export the @champion model to artifacts/champion (used by the Docker image)
	$(CLI) export --out artifacts/champion

serve:  ## Serve @champion from the local MLflow registry on :8000
	$(CLI) serve --host 127.0.0.1 --port 8000

pipeline: $(DATA)  ## End-to-end demo: two candidates, two gate decisions, drift check
	$(CLI) train --run-name baseline-xgb
	$(CLI) gate
	$(CLI) train --smote --run-name xgb-smote
	$(CLI) gate
	$(MAKE) drift

docker-build:  ## Build the self-contained serving image
	docker build -t fraud-mlops:latest .

clean:  ## Remove generated data, reports, MLflow store and caches
	rm -rf data reports artifacts mlruns .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
