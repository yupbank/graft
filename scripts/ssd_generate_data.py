#!/usr/bin/env python3
"""
Generate training data for SSD, CFG-SSD, and Proxy-SSD experiments.

Three data sources from the same prompts:
  1. SSD: sample from 4B-instruct (temp=1.0, top_p=0.95)
  2. CFG-SSD: sample from amplified 4B (s_inst + alpha*(s_inst - s_base))
  3. Proxy-SSD: sample from 4B-instruct + (0.6B-inst - 0.6B-base) delta

Each prompt gets N solutions. Output as JSONL in chat format for mlx_lm.lora.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from datetime import date
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Models
MODEL_4B_INSTRUCT = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
MODEL_4B_BASE = "mlx-community/Qwen3-4B-4bit"
MODEL_06B_INSTRUCT = str(PROJECT_DIR / "models" / "Qwen3-0.6B-Instruct-4bit")
MODEL_06B_BASE = "mlx-community/Qwen3-0.6B-4bit"

# Generation params
TEMP = 1.0
TOP_P = 0.95
MAX_TOKENS = 256
N_SOLUTIONS = 3
CFG_ALPHA = 1.0

mx.random.seed(42)


def _flush() -> None:
    sys.stdout.flush()


def logsumexp(x: mx.array) -> mx.array:
    c = x.max()
    return c + mx.log(mx.exp(x - c).sum())


def load_model(path: str):  # type: ignore[no-untyped-def]
    from mlx_lm import load

    return load(path)


def unload(*models: object) -> None:
    for m in models:
        del m
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------
def sample_from_logits(logits: mx.array, temp: float, top_p: float) -> mx.array:
    """Sample one token from logits with temperature and top-p."""
    if temp <= 0:
        return mx.argmax(logits)

    logits = logits / temp
    # Top-p (nucleus) sampling
    probs = mx.softmax(logits, axis=-1)
    sorted_idx = mx.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumsum = mx.cumsum(sorted_probs)

    # Create mask: keep tokens where cumsum <= top_p (plus the first one that exceeds)
    mask = cumsum - sorted_probs <= top_p
    sorted_probs = sorted_probs * mask
    sorted_probs = sorted_probs / sorted_probs.sum()

    # Sample from filtered distribution
    token_idx = mx.random.categorical(mx.log(sorted_probs + 1e-10))
    mx.eval(token_idx)
    return sorted_idx[token_idx]


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------
def generate_ssd(
    model: nn.Module,
    prompt_tokens: mx.array,
    max_tokens: int,
    eos_id: int,
    temp: float,
    top_p: float,
) -> list[int]:
    """Standard SSD: sample from model's own distribution."""
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    logits = model(prompt_tokens[None], cache=cache)[:, -1, :].squeeze(0)
    token = sample_from_logits(logits, temp, top_p)
    mx.eval(token)
    tokens = [token.item()]

    for _ in range(max_tokens - 1):
        logits = model(token.reshape(1, 1), cache=cache)[:, -1, :].squeeze(0)
        token = sample_from_logits(logits, temp, top_p)
        mx.eval(token)
        tid = token.item()
        tokens.append(tid)
        if tid == eos_id:
            break

    del cache
    return tokens


def generate_cfg_ssd(
    model_inst: nn.Module,
    model_base: nn.Module,
    prompt_tokens: mx.array,
    max_tokens: int,
    eos_id: int,
    temp: float,
    top_p: float,
    alpha: float,
) -> list[int]:
    """CFG-SSD: sample from amplified distribution s_inst + alpha*(s_inst - s_base)."""
    from mlx_lm.models.cache import make_prompt_cache

    ci = make_prompt_cache(model_inst)
    cb = make_prompt_cache(model_base)

    lg_i = model_inst(prompt_tokens[None], cache=ci)[:, -1, :].squeeze(0)
    lg_b = model_base(prompt_tokens[None], cache=cb)[:, -1, :].squeeze(0)
    score = lg_i + alpha * (lg_i - lg_b)
    token = sample_from_logits(score, temp, top_p)
    mx.eval(token)
    tokens = [token.item()]

    for _ in range(max_tokens - 1):
        inp = token.reshape(1, 1)
        lg_i = model_inst(inp, cache=ci)[:, -1, :].squeeze(0)
        lg_b = model_base(inp, cache=cb)[:, -1, :].squeeze(0)
        score = lg_i + alpha * (lg_i - lg_b)
        token = sample_from_logits(score, temp, top_p)
        mx.eval(token)
        tid = token.item()
        tokens.append(tid)
        if tid == eos_id:
            break

    del ci, cb
    return tokens


def generate_proxy_ssd(
    model_4b: nn.Module,
    model_06b_inst: nn.Module,
    model_06b_base: nn.Module,
    prompt_tokens: mx.array,
    max_tokens: int,
    eos_id: int,
    temp: float,
    top_p: float,
) -> list[int]:
    """Proxy-SSD: sample from 4B + (0.6B-inst - 0.6B-base) delta."""
    from mlx_lm.models.cache import make_prompt_cache

    c4 = make_prompt_cache(model_4b)
    ci = make_prompt_cache(model_06b_inst)
    cb = make_prompt_cache(model_06b_base)

    lg4 = model_4b(prompt_tokens[None], cache=c4)[:, -1, :].squeeze(0)
    lgi = model_06b_inst(prompt_tokens[None], cache=ci)[:, -1, :].squeeze(0)
    lgb = model_06b_base(prompt_tokens[None], cache=cb)[:, -1, :].squeeze(0)
    score = lg4 + (lgi - lgb)
    token = sample_from_logits(score, temp, top_p)
    mx.eval(token)
    tokens = [token.item()]

    for _ in range(max_tokens - 1):
        inp = token.reshape(1, 1)
        lg4 = model_4b(inp, cache=c4)[:, -1, :].squeeze(0)
        lgi = model_06b_inst(inp, cache=ci)[:, -1, :].squeeze(0)
        lgb = model_06b_base(inp, cache=cb)[:, -1, :].squeeze(0)
        score = lg4 + (lgi - lgb)
        token = sample_from_logits(score, temp, top_p)
        mx.eval(token)
        tid = token.item()
        tokens.append(tid)
        if tid == eos_id:
            break

    del c4, ci, cb
    return tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def get_prompts() -> list[dict]:  # type: ignore[type-arg]
    """Get coding prompts from LiveCodeBench + IFEval."""
    from datasets import load_dataset

    prompts = []

    # LiveCodeBench code generation
    try:
        lcb = load_dataset("livecodebench/code_generation_lite", split="test")
        for ex in lcb:
            prompts.append(
                {
                    "source": "livecodebench",
                    "prompt": ex.get("question_content") or ex.get("prompt", ""),
                    "id": ex.get("question_id", ""),
                }
            )
        print(f"  LiveCodeBench: {len(prompts)} prompts")
    except Exception as e:
        print(f"  LiveCodeBench failed: {e}")

    # Also include IFEval for instruction-following eval
    ifeval = load_dataset("google/IFEval", split="train")
    n_ifeval = min(100, len(ifeval))  # first 100 for quick eval
    for ex in list(ifeval)[:n_ifeval]:
        prompts.append(
            {
                "source": "ifeval",
                "prompt": ex["prompt"],
                "id": str(ex["key"]),
            }
        )
    print(f"  IFEval: {n_ifeval} prompts")
    print(f"  Total: {len(prompts)} prompts")

    return prompts


def generate_condition(
    condition: str,
    prompts: list[dict],  # type: ignore[type-arg]
    tokenizer,  # type: ignore[no-untyped-def]
    eos_id: int,
    **models: nn.Module,
) -> list[dict]:  # type: ignore[type-arg]
    """Generate N solutions per prompt for one condition."""
    results = []
    t0 = time.time()

    for i, p in enumerate(prompts):
        messages = [{"role": "user", "content": p["prompt"]}]
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = mx.array(token_ids)

        for n in range(N_SOLUTIONS):
            if condition == "ssd":
                toks = generate_ssd(models["inst"], prompt_tokens, MAX_TOKENS, eos_id, TEMP, TOP_P)
            elif condition == "cfg_ssd":
                toks = generate_cfg_ssd(
                    models["inst"],
                    models["base"],
                    prompt_tokens,
                    MAX_TOKENS,
                    eos_id,
                    TEMP,
                    TOP_P,
                    CFG_ALPHA,
                )
            elif condition == "proxy_ssd":
                toks = generate_proxy_ssd(
                    models["inst"],
                    models["small_inst"],
                    models["small_base"],
                    prompt_tokens,
                    MAX_TOKENS,
                    eos_id,
                    TEMP,
                    TOP_P,
                )
            else:
                raise ValueError(condition)

            text = tokenizer.decode(toks)
            results.append(
                {
                    "messages": [
                        {"role": "user", "content": p["prompt"]},
                        {"role": "assistant", "content": text},
                    ],
                    "source": p["source"],
                    "prompt_id": p["id"],
                    "solution_idx": n,
                }
            )

            gc.collect()
            mx.clear_cache()

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed * 60
        print(f"    [{i + 1}/{len(prompts)}] {N_SOLUTIONS} sols, ({rate:.1f} prompts/min)")
        _flush()

    return results


def save_jsonl(data: list[dict], path: str) -> None:  # type: ignore[type-arg]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} examples to {path}")


def main() -> None:
    print(f"Date: {date.today()}")
    print(f"SSD data generation: N={N_SOLUTIONS}, temp={TEMP}, top_p={TOP_P}")
    print(f"Max tokens: {MAX_TOKENS}")
    _flush()

    prompts = get_prompts()

    # === Condition 1: SSD (4B-instruct only) ===
    print(f"\n=== SSD: Loading {MODEL_4B_INSTRUCT} ===")
    _flush()
    inst_model, tokenizer = load_model(MODEL_4B_INSTRUCT)
    eos_id = tokenizer.eos_token_id or 151645
    print(f"  Loaded. {mx.get_peak_memory() / 1e9:.1f} GB")
    _flush()

    print("  Generating SSD data...")
    _flush()
    ssd_data = generate_condition("ssd", prompts, tokenizer, eos_id, inst=inst_model)
    save_jsonl(ssd_data, str(PROJECT_DIR / "data" / "ssd" / "train.jsonl"))

    unload(inst_model)

    # Summary
    print(f"\n{'=' * 60}")
    print("DATA GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  SSD: {len(ssd_data)} examples")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Solutions per prompt: {N_SOLUTIONS}")


if __name__ == "__main__":
    main()
