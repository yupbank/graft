#!/usr/bin/env python3
"""
Full generation comparison: 14B base vs delta-steered vs 8B instruct.

Loads all 3 models simultaneously (~18GB total for 4-bit) and generates
multi-token responses for a subset of IFEval prompts.

Three generation modes:
  1. 14B-base alone (baseline) — what the large model does without help
  2. 14B-base + delta reranking (GRAFT) — our method
  3. 8B-instruct alone (ceiling) — what perfect transfer would look like

Delta reranking at each step:
  - Forward pass through all 3 models (sharing the same token sequence)
  - Compute delta = log_softmax(s_ft)[top-k] - log_softmax(s_raw)[top-k]
  - Apply: score = log_softmax(s_large)[top-k] + lambda * delta
  - Sample from softmax(score) over top-k candidates

Saves all responses to results/generation_samples.json for inspection.
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MODEL_RAW = str(PROJECT_DIR / "models" / "Qwen3-8B-Base-4bit")
MODEL_FT = "mlx-community/Qwen3-8B-4bit"
MODEL_LARGE = str(PROJECT_DIR / "models" / "Qwen3-14B-Base-4bit")
LAMBDA = 1.0
TOP_K = 50
MAX_TOKENS = 256
NUM_PROMPTS = 50
OUTPUT_JSON = str(PROJECT_DIR / "results" / "generation_samples.json")

mx.random.seed(42)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def logsumexp(x: mx.array) -> mx.array:
    c = x.max()
    return c + mx.log(mx.exp(x - c).sum())


def log_softmax(x: mx.array) -> mx.array:
    return x - logsumexp(x)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path: str):  # type: ignore[no-untyped-def]
    from mlx_lm import load

    model, tokenizer = load(model_path)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Simple greedy generation (single model)
# ---------------------------------------------------------------------------
def _flush() -> None:
    sys.stdout.flush()


def generate_greedy(
    model: nn.Module,
    prompt_tokens: mx.array,
    max_tokens: int,
    eos_token_id: int,
) -> list[int]:
    """Generate tokens greedily from a single model."""
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)

    # Prefill: process prompt
    logits = model(prompt_tokens[None], cache=cache)
    logits = logits[:, -1, :]
    token = mx.argmax(logits, axis=-1).squeeze()
    mx.eval(token)

    tokens = [token.item()]
    for _ in range(max_tokens - 1):
        logits = model(token.reshape(1, 1), cache=cache)
        logits = logits[:, -1, :]
        token = mx.argmax(logits, axis=-1).squeeze()
        mx.eval(token)
        tid = token.item()
        tokens.append(tid)
        if tid == eos_token_id:
            break

    # Free KV cache
    del cache
    return tokens


# ---------------------------------------------------------------------------
# Delta-reranked generation
# ---------------------------------------------------------------------------
def generate_delta(
    model_large: nn.Module,
    model_raw: nn.Module,
    model_ft: nn.Module,
    prompt_tokens: mx.array,
    max_tokens: int,
    eos_token_id: int,
    top_k: int = 50,
    lam: float = 1.0,
) -> list[int]:
    """Generate tokens using delta reranking across 3 models."""
    from mlx_lm.models.cache import make_prompt_cache

    cache_large = make_prompt_cache(model_large)
    cache_raw = make_prompt_cache(model_raw)
    cache_ft = make_prompt_cache(model_ft)

    # Prefill all 3 models
    lg_logits = model_large(prompt_tokens[None], cache=cache_large)[:, -1, :]
    raw_logits = model_raw(prompt_tokens[None], cache=cache_raw)[:, -1, :]
    ft_logits = model_ft(prompt_tokens[None], cache=cache_ft)[:, -1, :]

    token = _delta_sample(
        lg_logits.squeeze(0), raw_logits.squeeze(0), ft_logits.squeeze(0), top_k, lam
    )
    mx.eval(token)
    tokens = [token.item()]

    for _ in range(max_tokens - 1):
        inp = token.reshape(1, 1)
        lg_logits = model_large(inp, cache=cache_large)[:, -1, :]
        raw_logits = model_raw(inp, cache=cache_raw)[:, -1, :]
        ft_logits = model_ft(inp, cache=cache_ft)[:, -1, :]

        token = _delta_sample(
            lg_logits.squeeze(0), raw_logits.squeeze(0), ft_logits.squeeze(0), top_k, lam
        )
        mx.eval(token)
        tid = token.item()
        tokens.append(tid)
        if tid == eos_token_id:
            break

    # Free KV caches
    del cache_large, cache_raw, cache_ft
    return tokens


def _delta_sample(
    s_large: mx.array,
    s_raw: mx.array,
    s_ft: mx.array,
    top_k: int,
    lam: float,
) -> mx.array:
    """Apply delta reranking and return greedy argmax token."""
    vocab_size = s_large.shape[0]
    k = min(top_k, vocab_size)

    # Top-k from large model
    s_t = mx.argpartition(s_large, kth=vocab_size - k)[-k:]

    # Delta in restricted space
    s_ft_k = s_ft[s_t]
    s_raw_k = s_raw[s_t]
    log_pf = s_ft_k - logsumexp(s_ft_k)
    log_pr = s_raw_k - logsumexp(s_raw_k)
    delta = log_pf - log_pr

    # Apply to large model
    log_p_large = log_softmax(s_large)
    score = log_p_large[s_t] + lam * delta

    # Greedy from restricted set
    best_idx = mx.argmax(score)
    return s_t[best_idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Date: {date.today()}")
    print(f"Python: {sys.version}")
    print(f"Config: lambda={LAMBDA}, top_k={TOP_K}, max_tokens={MAX_TOKENS}")
    print()

    # Load IFEval prompts
    print("Loading IFEval dataset ...")
    _flush()
    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    prompts = [ex["prompt"] for ex in ds][:NUM_PROMPTS]
    print(f"  {len(prompts)} prompts loaded")
    _flush()

    # Load models one at a time with progress
    print(f"\n  Loading 8B instruct: {MODEL_FT} ...")
    _flush()
    t0 = time.time()
    ft_model, tokenizer = load_model(MODEL_FT)
    print(f"    done ({time.time() - t0:.1f}s, {mx.get_peak_memory() / 1e9:.1f} GB peak)")
    _flush()

    print(f"  Loading 8B base: {MODEL_RAW} ...")
    _flush()
    t0 = time.time()
    raw_model, _ = load_model(MODEL_RAW)
    print(f"    done ({time.time() - t0:.1f}s, {mx.get_peak_memory() / 1e9:.1f} GB peak)")
    _flush()

    print(f"  Loading 14B base: {MODEL_LARGE} ...")
    _flush()
    t0 = time.time()
    large_model, _ = load_model(MODEL_LARGE)
    print(f"    done ({time.time() - t0:.1f}s, {mx.get_peak_memory() / 1e9:.1f} GB peak)")
    _flush()

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = 151645  # Qwen3 default
    print(f"\n  All models loaded. EOS={eos_id}. Starting generation.\n")
    _flush()

    results = []
    total_t0 = time.time()

    for i, prompt_text in enumerate(prompts):
        prompt_t0 = time.time()
        short = prompt_text[:55].replace("\n", " ")
        print(f"[{i + 1}/{len(prompts)}] {short}...")
        _flush()

        messages = [{"role": "user", "content": prompt_text}]
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = mx.array(token_ids)

        # 1. 14B base alone
        t0 = time.time()
        base_tokens = generate_greedy(large_model, prompt_tokens, MAX_TOKENS, eos_id)
        base_time = time.time() - t0
        base_text = tokenizer.decode(base_tokens)
        print(f"  base:  {len(base_tokens)} tok, {base_time:.1f}s")
        _flush()

        # 2. Delta-steered
        t0 = time.time()
        delta_tokens = generate_delta(
            large_model,
            raw_model,
            ft_model,
            prompt_tokens,
            MAX_TOKENS,
            eos_id,
            top_k=TOP_K,
            lam=LAMBDA,
        )
        delta_time = time.time() - t0
        delta_text = tokenizer.decode(delta_tokens)
        print(f"  delta: {len(delta_tokens)} tok, {delta_time:.1f}s")
        _flush()

        # 3. 8B instruct alone
        t0 = time.time()
        instruct_tokens = generate_greedy(ft_model, prompt_tokens, MAX_TOKENS, eos_id)
        instruct_time = time.time() - t0
        instruct_text = tokenizer.decode(instruct_tokens)
        print(f"  inst:  {len(instruct_tokens)} tok, {instruct_time:.1f}s")

        elapsed = time.time() - total_t0
        rate = (i + 1) / elapsed * 60
        print(f"  total: {time.time() - prompt_t0:.1f}s  ({rate:.1f} prompts/min)")
        _flush()

        # Free memory between prompts
        gc.collect()
        mx.clear_cache()

        results.append(
            {
                "prompt_idx": i,
                "prompt": prompt_text,
                "base_14b": base_text,
                "delta_steered": delta_text,
                "instruct_8b": instruct_text,
                "base_tokens": len(base_tokens),
                "delta_tokens": len(delta_tokens),
                "instruct_tokens": len(instruct_tokens),
                "base_time": base_time,
                "delta_time": delta_time,
                "instruct_time": instruct_time,
            }
        )

        # Save incrementally every 10 prompts
        if (i + 1) % 10 == 0:
            Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_JSON, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  ** saved {len(results)} results to {OUTPUT_JSON}")
            _flush()

    # Final cleanup
    del large_model, raw_model, ft_model
    gc.collect()
    mx.synchronize()
    mx.clear_cache()

    # Save results
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {OUTPUT_JSON}")
    print()

    # Quick automated analysis
    print("=" * 72)
    print("GENERATION SUMMARY")
    print("=" * 72)

    n = len(results)

    # Check if delta output looks more like instruct than base
    def starts_conversational(text: str) -> bool:
        openers = [
            "here",
            "sure",
            "certainly",
            "of course",
            "to ",
            "below",
            "i ",
            "the ",
            "you ",
            "let",
        ]
        t = text.strip().lower()
        return any(t.startswith(o) for o in openers)

    def has_code_block(text: str) -> bool:
        return "```" in text

    base_conv = sum(1 for r in results if starts_conversational(r["base_14b"]))
    delta_conv = sum(1 for r in results if starts_conversational(r["delta_steered"]))
    inst_conv = sum(1 for r in results if starts_conversational(r["instruct_8b"]))

    base_code = sum(1 for r in results if has_code_block(r["base_14b"]))
    delta_code = sum(1 for r in results if has_code_block(r["delta_steered"]))
    inst_code = sum(1 for r in results if has_code_block(r["instruct_8b"]))

    base_len = sum(len(r["base_14b"].split()) for r in results) / n
    delta_len = sum(len(r["delta_steered"].split()) for r in results) / n
    inst_len = sum(len(r["instruct_8b"].split()) for r in results) / n

    base_tps = sum(r["base_tokens"] / r["base_time"] for r in results) / n
    delta_tps = sum(r["delta_tokens"] / r["delta_time"] for r in results) / n
    inst_tps = sum(r["instruct_tokens"] / r["instruct_time"] for r in results) / n

    print(f"  {'Metric':<30s} {'14B base':>10s} {'Delta':>10s} {'8B inst':>10s}")
    print(f"  {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 10}")
    print(f"  {'Conversational opener':<30s} {base_conv:>10d} {delta_conv:>10d} {inst_conv:>10d}")
    print(f"  {'Has code block':<30s} {base_code:>10d} {delta_code:>10d} {inst_code:>10d}")
    print(f"  {'Avg response words':<30s} {base_len:>10.0f} {delta_len:>10.0f} {inst_len:>10.0f}")
    print(f"  {'Avg tokens/sec':<30s} {base_tps:>10.1f} {delta_tps:>10.1f} {inst_tps:>10.1f}")
    print()

    print(
        f"  Conclusion: Delta-steered generation produces conversational "
        f"responses in {delta_conv}/{n} cases "
        f"(vs {base_conv}/{n} for base, {inst_conv}/{n} for instruct)."
    )


if __name__ == "__main__":
    main()
