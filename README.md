# GRAFT

**G**ated **R**estricted **A**daptation via **F**ine-Tuning **T**ransfer

Make a large base model follow instructions **without fine-tuning it**, by
grafting the behavioral delta from a smaller instruction-tuned model pair.

## Key Result

The 8B instruct-vs-base delta successfully steers a 14B base model:
- **85.8%** top-1 match with 8B instruct across 541 IFEval prompts
- **92.1%** at k=10 (restriction to top-k acts as regularization)
- Full generation produces instruction-following text matching the instruct model

See `summary.html` for the full report with math and examples.

## Setup

```bash
uv sync
```

## Reproducing All Findings

### Step 0: Download and quantize models

The instruct model downloads automatically. The base models must be quantized locally:

```bash
# 8B base (downloads ~16GB bf16, outputs ~4.6GB 4-bit)
uv run python -m mlx_lm.convert \
  --hf-path mlx-community/Qwen3-8B-Base-bf16 \
  --mlx-path models/Qwen3-8B-Base-4bit \
  -q --q-bits 4

# 14B base (downloads ~28GB bf16, outputs ~8.5GB 4-bit)
huggingface-cli download Qwen/Qwen3-14B-Base
uv run python -m mlx_lm.convert \
  --hf-path Qwen/Qwen3-14B-Base \
  --mlx-path models/Qwen3-14B-Base-4bit \
  -q --q-bits 4
```

### Step 1: Oracle math check

Verifies the proxy tuning formula using a single model (large = raw = ft).
Expected: KL = 0, delta = 0.

```bash
uv run python scripts/oracle_sanity_check.py
```

### Step 2: Same-model delta transfer

Uses raw = 8B-base, ft = 8B-instruct, large = 8B-base (same as raw).
Expected: KL = 0 (large=raw identity), delta > 0 (instruct differs from base).

```bash
uv run python scripts/step2_delta_transfer.py
```

### Step 3: Cross-model transfer (5 prompts)

Uses raw = 8B-base, ft = 8B-instruct, large = **14B-base**.
This is the actual hypothesis test.

```bash
uv run python scripts/step3_cross_model.py
```

### Step 4: IFEval benchmark (541 prompts, first-token logits)

Runs the cross-model transfer analysis at scale on IFEval.
Loads each model sequentially (~12 min total on M3 Pro).

```bash
uv run python scripts/eval_ifeval_logits.py
```

Results are saved to `results/ifeval_logits.json`.

### Step 5: Full generation comparison (50 prompts)

Generates multi-token responses with all 3 models loaded simultaneously.
Requires ~18GB unified memory. (~15 min on M3 Pro).

```bash
uv run python scripts/eval_generate.py
```

Results are saved to `results/generation_samples.json`.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright src/
uv run pytest -v
```

## Hardware Requirements

- Apple Silicon Mac with 32GB+ unified memory (for Step 3+)
- Steps 1-2 work with 16GB
- All models are 4-bit quantized via mlx-lm
