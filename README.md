# GRAFT

**G**ated **R**estricted **A**daptation via **F**ine-Tuning **T**ransfer

A research project exploring delta-reranking methods where a fine-tuned small model
steers a larger frozen model at inference time.

## Setup

```bash
uv sync
```

## Run the oracle sanity check

```bash
uv run python scripts/oracle_sanity_check.py
```

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright src/
uv run pytest -v
```
