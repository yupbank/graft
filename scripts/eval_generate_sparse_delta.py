#!/usr/bin/env python3
"""
Sparse delta experiment: GRAFT with top-m delta sparsification.

Double restriction:
  1. Restrict to S_t = top-k from large model (standard GRAFT)
  2. Within S_t, only apply delta to top-m tokens by |delta| magnitude

Tests whether denoising the delta signal improves generation.
Runs on first 50 IFEval prompts with multiple (k, m) configs.
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
MODEL_RAW = str(PROJECT_DIR / "models" / "Qwen3-8B-Base-4bit")
MODEL_FT = "mlx-community/Qwen3-8B-4bit"
MODEL_LARGE = str(PROJECT_DIR / "models" / "Qwen3-14B-Base-4bit")
MAX_TOKENS = 256
NUM_PROMPTS = 50

# Configs to test: (k, m, label)
CONFIGS = [
    (50, 50, "GRAFT k=50 (baseline)"),
    (50, 5, "GRAFT k=50 m=5"),
    (50, 10, "GRAFT k=50 m=10"),
    (50, 20, "GRAFT k=50 m=20"),
    (200, 5, "GRAFT k=200 m=5"),
    (200, 10, "GRAFT k=200 m=10"),
    (200, 200, "GRAFT k=200 (no sparsify)"),
    (151936, 10, "full-vocab m=10"),
]

mx.random.seed(42)


def _flush() -> None:
    sys.stdout.flush()


def logsumexp(x: mx.array) -> mx.array:
    c = x.max()
    return c + mx.log(mx.exp(x - c).sum())


def log_softmax(x: mx.array) -> mx.array:
    return x - logsumexp(x)


def load_model(model_path: str):  # type: ignore[no-untyped-def]
    from mlx_lm import load

    return load(model_path)


def _sparse_delta_sample(
    s_large: mx.array,
    s_raw: mx.array,
    s_ft: mx.array,
    top_k: int,
    top_m: int,
) -> mx.array:
    """GRAFT with sparse delta: restrict to top-k, then sparsify delta to top-m."""
    vocab_size = s_large.shape[0]
    k = min(top_k, vocab_size)

    # Step 1: top-k from large model
    if k < vocab_size:
        s_t = mx.argpartition(s_large, kth=vocab_size - k)[-k:]
    else:
        s_t = mx.arange(vocab_size)
        k = vocab_size

    # Step 2: compute delta in restricted log-prob space
    s_ft_k = s_ft[s_t]
    s_raw_k = s_raw[s_t]
    log_pf = s_ft_k - logsumexp(s_ft_k)
    log_pr = s_raw_k - logsumexp(s_raw_k)
    delta = log_pf - log_pr

    # Step 3: sparsify — keep only top-m by |delta|, zero out rest
    m = min(top_m, k)
    if m < k:
        abs_delta = mx.abs(delta)
        threshold_idx = mx.argpartition(abs_delta, kth=k - m)
        mask = mx.zeros_like(delta)
        mask = mask.at[threshold_idx[-m:]].add(mx.ones(m))
        delta = delta * mask

    # Step 4: apply to large model's log-probs
    log_p_large = log_softmax(s_large)
    score = log_p_large[s_t] + delta

    # Step 5: greedy
    best_idx = mx.argmax(score)
    return s_t[best_idx]


def generate_sparse_delta(
    model_large: nn.Module,
    model_raw: nn.Module,
    model_ft: nn.Module,
    prompt_tokens: mx.array,
    max_tokens: int,
    eos_token_id: int,
    top_k: int,
    top_m: int,
) -> list[int]:
    from mlx_lm.models.cache import make_prompt_cache

    cache_large = make_prompt_cache(model_large)
    cache_raw = make_prompt_cache(model_raw)
    cache_ft = make_prompt_cache(model_ft)

    lg = model_large(prompt_tokens[None], cache=cache_large)[:, -1, :].squeeze(0)
    rw = model_raw(prompt_tokens[None], cache=cache_raw)[:, -1, :].squeeze(0)
    ft = model_ft(prompt_tokens[None], cache=cache_ft)[:, -1, :].squeeze(0)

    token = _sparse_delta_sample(lg, rw, ft, top_k, top_m)
    mx.eval(token)
    tokens = [token.item()]

    for _ in range(max_tokens - 1):
        inp = token.reshape(1, 1)
        lg = model_large(inp, cache=cache_large)[:, -1, :].squeeze(0)
        rw = model_raw(inp, cache=cache_raw)[:, -1, :].squeeze(0)
        ft = model_ft(inp, cache=cache_ft)[:, -1, :].squeeze(0)

        token = _sparse_delta_sample(lg, rw, ft, top_k, top_m)
        mx.eval(token)
        tid = token.item()
        tokens.append(tid)
        if tid == eos_token_id:
            break

    del cache_large, cache_raw, cache_ft
    return tokens


def main() -> None:
    print(f"Date: {date.today()}")
    print(f"Sparse delta experiment: {NUM_PROMPTS} prompts, {len(CONFIGS)} configs")
    _flush()

    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    prompts = [ex["prompt"] for ex in ds][:NUM_PROMPTS]

    # Load models
    print("\nLoading models...")
    _flush()
    ft_model, tokenizer = load_model(MODEL_FT)
    raw_model, _ = load_model(MODEL_RAW)
    large_model, _ = load_model(MODEL_LARGE)
    print(f"  Loaded. {mx.get_peak_memory() / 1e9:.1f} GB peak")
    _flush()

    eos_id = tokenizer.eos_token_id or 151645

    # Also generate proxy-tuning baseline for same 50 prompts

    all_configs_results: dict[str, list[dict]] = {}  # type: ignore[type-arg]

    for k, m, label in CONFIGS:
        print(f"\n{'=' * 60}")
        print(f"Config: {label} (k={k}, m={m})")
        print(f"{'=' * 60}")
        _flush()

        results = []
        t0_total = time.time()

        for i, prompt_text in enumerate(prompts):
            messages = [{"role": "user", "content": prompt_text}]
            token_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_tokens = mx.array(token_ids)

            t0 = time.time()
            tokens = generate_sparse_delta(
                large_model,
                raw_model,
                ft_model,
                prompt_tokens,
                MAX_TOKENS,
                eos_id,
                top_k=k,
                top_m=m,
            )
            elapsed = time.time() - t0
            text = tokenizer.decode(tokens)

            gc.collect()
            mx.clear_cache()

            results.append(
                {
                    "prompt_idx": i,
                    "prompt": prompt_text,
                    "response": text,
                    "num_tokens": len(tokens),
                    "time": elapsed,
                }
            )

            if (i + 1) % 10 == 0:
                rate = (i + 1) / (time.time() - t0_total) * 60
                print(f"  [{i + 1}/{NUM_PROMPTS}] ({rate:.1f}/min)")
                _flush()

        all_configs_results[label] = results
        total = time.time() - t0_total
        print(f"  Done in {total / 60:.1f} min")
        _flush()

    # Cleanup models
    del large_model, raw_model, ft_model
    gc.collect()
    mx.synchronize()
    mx.clear_cache()

    # Score all configs
    import importlib.util

    spec = importlib.util.spec_from_file_location("score", str(SCRIPT_DIR / "score_single.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}\n")

    print(f"  {'Config':<30s} {'Prompt':>8s} {'Instr':>8s}")
    print(f"  {'-' * 30} {'-' * 8} {'-' * 8}")

    for label, results in all_configs_results.items():
        pp, ip, it = 0, 0, 0
        for gen in results:
            ex = ds[gen["prompt_idx"]]
            ev = mod.evaluate_response(gen["response"], ex["instruction_id_list"], ex["kwargs"])
            if ev["prompt_pass"]:
                pp += 1
            for r in ev["instructions"]:
                if not r.get("skipped"):
                    it += 1
                    if r["pass"]:
                        ip += 1

        n = len(results)
        pp_pct = 100 * pp / n
        ip_pct = 100 * ip / it if it > 0 else 0
        print(f"  {label:<30s} {pp:>3d}/{n} ({pp_pct:>4.1f}%) {ip:>3d}/{it} ({ip_pct:>4.1f}%)")

    # Save all results
    output_path = str(PROJECT_DIR / "results" / "sparse_delta_experiment.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {label: results for label, results in all_configs_results.items()},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
