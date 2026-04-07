#!/usr/bin/env python3
"""
Delta rank analysis: measure the effective dimensionality of behavioral transfer.

For each IFEval prompt (first-token), compute the delta vector and measure:
  1. How many tokens have |delta| > threshold (effective support size)
  2. The entropy of |delta| distribution (concentration)
  3. Top-1 and top-5 delta tokens
  4. Correlation between delta rank and IFEval category

Uses pre-computed logits from eval_ifeval_logits.py results.
If not available, computes from scratch.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
LOGITS_JSON = str(PROJECT_DIR / "results" / "ifeval_logits.json")
OUTPUT_JSON = str(PROJECT_DIR / "results" / "delta_rank_analysis.json")

# If logits aren't cached, we need to recompute
MODEL_RAW = str(PROJECT_DIR / "models" / "Qwen3-8B-Base-4bit")
MODEL_FT = "mlx-community/Qwen3-8B-4bit"
MODEL_LARGE = str(PROJECT_DIR / "models" / "Qwen3-14B-Base-4bit")


def logsumexp(x: mx.array) -> mx.array:
    c = x.max()
    return c + mx.log(mx.exp(x - c).sum())


def _flush() -> None:
    sys.stdout.flush()


def compute_delta_stats(s_large: mx.array, s_raw: mx.array, s_ft: mx.array) -> dict:  # type: ignore[type-arg]
    """Compute delta and measure its rank/sparsity."""
    vocab_size = s_large.shape[0]

    # Full-vocab delta in log-prob space
    log_p_ft = s_ft - logsumexp(s_ft)
    log_p_raw = s_raw - logsumexp(s_raw)
    delta_full = log_p_ft - log_p_raw
    mx.eval(delta_full)

    abs_delta = mx.abs(delta_full)
    abs_np = np.array(abs_delta.tolist())

    # Effective support at various thresholds
    thresholds = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    support_sizes = {}
    for t in thresholds:
        support_sizes[str(t)] = int(np.sum(abs_np > t))

    # Delta entropy (how concentrated is the delta?)
    abs_sum = abs_np.sum()
    if abs_sum > 0:
        p_delta = abs_np / abs_sum
        delta_entropy = -np.sum(p_delta[p_delta > 0] * np.log(p_delta[p_delta > 0]))
    else:
        delta_entropy = 0.0

    # Top-10 tokens by |delta|
    top10_idx = np.argsort(abs_np)[-10:][::-1]
    top10 = [(int(idx), float(abs_np[idx])) for idx in top10_idx]

    # Fraction of total |delta| mass in top-k
    sorted_abs = np.sort(abs_np)[::-1]
    cumsum = np.cumsum(sorted_abs)
    total_mass = cumsum[-1] if len(cumsum) > 0 else 1.0
    mass_in_top = {}
    for k in [5, 10, 50, 200]:
        if k <= len(cumsum):
            mass_in_top[str(k)] = float(cumsum[k - 1] / total_mass)
        else:
            mass_in_top[str(k)] = 1.0

    # Mean and max
    delta_mean = float(np.mean(abs_np))
    delta_max = float(np.max(abs_np))
    delta_std = float(np.std(abs_np))

    # Also compute for restricted sets
    restricted_stats = {}
    for k in [10, 50, 200]:
        ek = min(k, vocab_size)
        s_t = mx.argpartition(s_large, kth=vocab_size - ek)[-ek:]
        s_ft_k = s_ft[s_t]
        s_raw_k = s_raw[s_t]
        log_pf_k = s_ft_k - logsumexp(s_ft_k)
        log_pr_k = s_raw_k - logsumexp(s_raw_k)
        delta_k = log_pf_k - log_pr_k
        mx.eval(delta_k)
        abs_dk = np.array(mx.abs(delta_k).tolist())
        restricted_stats[str(k)] = {
            "mean": float(np.mean(abs_dk)),
            "max": float(np.max(abs_dk)),
            "std": float(np.std(abs_dk)),
            "nonzero_01": int(np.sum(abs_dk > 0.1)),
        }

    return {
        "support_sizes": support_sizes,
        "delta_entropy": delta_entropy,
        "top10_tokens": top10,
        "mass_in_top": mass_in_top,
        "delta_mean": delta_mean,
        "delta_max": delta_max,
        "delta_std": delta_std,
        "restricted": restricted_stats,
    }


def main() -> None:
    import gc
    import time

    print(f"Date: {date.today()}")
    print("Delta rank analysis on 541 IFEval prompts")
    _flush()

    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    categories = [ex["instruction_id_list"] for ex in ds]
    n = len(ds)

    # Load models sequentially, extract logits
    from mlx_lm import load

    print("\nLoading 8B instruct...")
    _flush()
    ft_model, tokenizer = load(MODEL_FT)
    ft_logits = []
    t0 = time.time()
    for i, ex in enumerate(ds):
        msgs = [{"role": "user", "content": ex["prompt"]}]
        ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        logits = ft_model(mx.array(ids)[None])[:, -1, :].squeeze(0)
        mx.eval(logits)
        ft_logits.append(logits)
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{n}] {(i + 1) / (time.time() - t0):.1f}/s")
            _flush()
    del ft_model
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    print(f"  Done in {time.time() - t0:.0f}s\n")

    print("Loading 8B base...")
    _flush()
    raw_model, _ = load(MODEL_RAW)
    raw_logits = []
    t0 = time.time()
    for i, ex in enumerate(ds):
        msgs = [{"role": "user", "content": ex["prompt"]}]
        ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        logits = raw_model(mx.array(ids)[None])[:, -1, :].squeeze(0)
        mx.eval(logits)
        raw_logits.append(logits)
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{n}] {(i + 1) / (time.time() - t0):.1f}/s")
            _flush()
    del raw_model
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    print(f"  Done in {time.time() - t0:.0f}s\n")

    print("Loading 14B base...")
    _flush()
    large_model, _ = load(MODEL_LARGE)
    large_logits = []
    t0 = time.time()
    for i, ex in enumerate(ds):
        msgs = [{"role": "user", "content": ex["prompt"]}]
        ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        logits = large_model(mx.array(ids)[None])[:, -1, :].squeeze(0)
        mx.eval(logits)
        large_logits.append(logits)
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{n}] {(i + 1) / (time.time() - t0):.1f}/s")
            _flush()
    del large_model
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    print(f"  Done in {time.time() - t0:.0f}s\n")

    # Compute delta stats for each prompt
    print("Computing delta stats...")
    _flush()
    all_stats = []
    for i in range(n):
        stats = compute_delta_stats(large_logits[i], raw_logits[i], ft_logits[i])
        stats["prompt_idx"] = i
        stats["categories"] = categories[i]
        all_stats.append(stats)
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{n}]")
            _flush()

    # Save raw results
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_stats, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print("DELTA RANK ANALYSIS")
    print(f"{'=' * 60}\n")

    # Average support sizes
    print("  Effective support size (tokens with |delta| > threshold):")
    for t in ["0.01", "0.1", "0.5", "1.0", "2.0", "5.0"]:
        sizes = [s["support_sizes"][t] for s in all_stats]
        m, md = np.mean(sizes), np.median(sizes)
        print(f"    |delta| > {t:>4s}: mean={m:>8.1f}  median={md:>8.1f}")

    # Mass concentration
    print("\n  Fraction of total |delta| mass in top-k tokens:")
    for k in ["5", "10", "50", "200"]:
        masses = [s["mass_in_top"][k] for s in all_stats]
        print(f"    top-{k:>3s}: mean={np.mean(masses):.3f}  median={np.median(masses):.3f}")

    # Delta entropy
    entropies = [s["delta_entropy"] for s in all_stats]
    print(f"\n  Delta entropy: mean={np.mean(entropies):.2f}  median={np.median(entropies):.2f}")
    print(f"  (lower = more concentrated, max possible = ln(151936) = {np.log(151936):.2f})")

    # Per-category
    print("\n  Per-category delta statistics:")
    cat_stats: dict[str, list[float]] = {}
    for s in all_stats:
        for cat_full in s["categories"]:
            cat = cat_full.split(":")[0]
            cat_stats.setdefault(cat, []).append(s["delta_mean"])

    print(f"    {'Category':<25s} {'Mean |delta|':>12s} {'Count':>6s}")
    for cat in sorted(cat_stats.keys()):
        vals = cat_stats[cat]
        print(f"    {cat:<25s} {np.mean(vals):>12.4f} {len(vals):>6d}")

    print(f"\n  Saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
