#!/usr/bin/env python3
"""
Oracle sanity check for GRAFT — bf16 models.

Same as oracle_sanity_check.py but using bf16 (full precision) models
instead of 4-bit quantized ones. This tests whether the float precision
artifacts seen with 4-bit models are reduced or eliminated with bf16.
"""

from __future__ import annotations

import gc
import sys
import time
from datetime import date
from pathlib import Path

import mlx.core as mx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MODEL_NAME = str(PROJECT_DIR / "models" / "Qwen3-8B-Base-bf16")
LAMBDA = 1.0
K_VALUES = [10, 50, 200, 1000]  # vocab_size is appended dynamically
PROMPTS = [
    "Write a Python function that returns the nth Fibonacci number iteratively.",
    "Implement binary search in Python. Return the index or -1 if not found.",
    "Write a Python class for a stack with push, pop, and peek methods.",
    "Given a list of integers, return all pairs that sum to a target value.",
    "Write a Python decorator that measures and prints execution time.",
]
KL_PASS_THRESHOLD = 1e-4

# ---------------------------------------------------------------------------
# Seed and header
# ---------------------------------------------------------------------------
mx.random.seed(42)


def print_header() -> None:
    print(f"Date: {date.today()}")
    print(f"Python: {sys.version}")
    try:
        import mlx_lm

        print(f"mlx-lm: {mlx_lm.__version__}")
    except AttributeError:
        print("mlx-lm: (version unavailable)")
    print(f"Model: {MODEL_NAME}")
    print(f"Lambda: {LAMBDA}")
    print(f"Precision: bf16")
    print()


# ---------------------------------------------------------------------------
# Math helpers (all mlx)
# ---------------------------------------------------------------------------
def logsumexp(x: mx.array) -> mx.array:
    c = x.max()
    return c + mx.log(mx.exp(x - c).sum())


def log_softmax(x: mx.array) -> mx.array:
    return x - logsumexp(x)


def kl_divergence(log_p: mx.array, log_q: mx.array, floor: float = -1e9) -> float:
    """KL(p || q) where inputs are already log-probabilities."""
    log_p_clamped = mx.maximum(log_p, mx.array(floor))
    log_q_clamped = mx.maximum(log_q, mx.array(floor))
    p = mx.exp(log_p_clamped)
    kl = mx.sum(p * (log_p_clamped - log_q_clamped))
    return kl.item()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer():  # type: ignore[no-untyped-def]
    try:
        from mlx_lm import load
    except ImportError:
        print("ERROR: mlx-lm is not installed. Run: uv pip install mlx-lm")
        sys.exit(1)

    try:
        model, tokenizer = load(MODEL_NAME)
    except Exception as e:
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            print(f"ERROR: Model {MODEL_NAME} not found locally.")
            sys.exit(1)
        raise
    return model, tokenizer


def unload_model(model: object, tokenizer: object) -> None:
    del model
    del tokenizer
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


# ---------------------------------------------------------------------------
# Logit extraction
# ---------------------------------------------------------------------------
def get_logits_for_prompts(
    model,  # type: ignore[no-untyped-def]
    tokenizer,  # type: ignore[no-untyped-def]
    prompts: list[str],
) -> list[mx.array]:
    """Get first-token logits for each prompt. Returns list of (vocab_size,) arrays."""
    all_logits: list[mx.array] = []
    for prompt_text in prompts:
        messages = [{"role": "user", "content": prompt_text}]
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = mx.array(token_ids)
        logits = model(prompt_tokens[None])  # (1, seq_len, vocab_size)
        last_logits = logits[:, -1, :]  # (1, vocab_size)
        mx.eval(last_logits)
        all_logits.append(last_logits.squeeze(0))  # (vocab_size,)
    return all_logits


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------
def compute_metrics(
    s_large: mx.array,
    s_raw: mx.array,
    s_ft: mx.array,
    k_values: list[int],
) -> dict:  # type: ignore[type-arg]
    """Compute all metrics for a single prompt."""
    vocab_size = s_large.shape[0]

    # --- Full-vocab math check ---
    score_full = s_large + LAMBDA * (s_ft - s_raw)
    log_p_ft_full = log_softmax(s_ft)
    log_p_recovered_full = log_softmax(score_full)
    kl_full = kl_divergence(log_p_ft_full, log_p_recovered_full)

    # First predicted token
    top1_id = mx.argmax(s_ft).item()

    results: dict = {  # type: ignore[type-arg]
        "vocab_size": vocab_size,
        "kl_full": kl_full,
        "full_pass": abs(kl_full) < KL_PASS_THRESHOLD,
        "top1_id": top1_id,
        "per_k": [],
    }

    for k in k_values:
        effective_k = min(k, vocab_size)
        is_full = effective_k == vocab_size

        # Top-k indices from large model
        if is_full:
            s_t = mx.arange(vocab_size)
        else:
            s_t = mx.argpartition(s_large, kth=vocab_size - effective_k)[-effective_k:]

        # Extract logits for S_t
        s_ft_k = s_ft[s_t]
        s_raw_k = s_raw[s_t]

        # Restricted renormalized delta
        log_p_ft_restricted = s_ft_k - logsumexp(s_ft_k)
        log_p_raw_restricted = s_raw_k - logsumexp(s_raw_k)
        delta = log_p_ft_restricted - log_p_raw_restricted

        # Score: log_softmax(s_large) restricted to S_t + lambda * delta
        log_p_large_full = log_softmax(s_large)
        log_p_large_restricted = log_p_large_full[s_t]
        score_restricted = log_p_large_restricted + LAMBDA * delta

        # Normalize score_restricted to get recovered distribution over S_t
        log_p_recovered_restricted = score_restricted - logsumexp(score_restricted)

        # FT distribution renormalized over S_t
        log_p_ft_restricted_norm = s_ft_k - logsumexp(s_ft_k)

        # KL divergence
        kl_restricted = kl_divergence(log_p_ft_restricted_norm, log_p_recovered_restricted)

        # Top-1 agreement
        top1_recovered = s_t[mx.argmax(log_p_recovered_restricted)].item()
        top1_ft_restricted = s_t[mx.argmax(log_p_ft_restricted_norm)].item()
        top1_match = top1_recovered == top1_ft_restricted

        # Top-5 overlap
        n5 = min(5, effective_k)
        top5_recovered_idx = mx.argpartition(log_p_recovered_restricted, kth=effective_k - n5)[-n5:]
        top5_ft_idx = mx.argpartition(log_p_ft_restricted_norm, kth=effective_k - n5)[-n5:]
        top5_recovered_set = set(s_t[top5_recovered_idx].tolist())
        top5_ft_set = set(s_t[top5_ft_idx].tolist())
        top5_overlap = len(top5_recovered_set & top5_ft_set) / 5.0

        # Delta magnitude
        delta_mean = mx.mean(mx.abs(delta)).item()

        k_label = "full" if is_full else str(k)
        results["per_k"].append(
            {
                "k_label": k_label,
                "k": effective_k,
                "kl": kl_restricted,
                "top1_match": top1_match,
                "top5_overlap": top5_overlap,
                "delta_mean": delta_mean,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def print_prompt_results(idx: int, prompt: str, results: dict, tokenizer) -> None:  # type: ignore[type-arg, no-untyped-def]
    top1_id = results["top1_id"]
    token_str = tokenizer.decode([top1_id])
    short_prompt = prompt[:60] + "..." if len(prompt) > 60 else prompt

    print(f'Prompt {idx + 1}: "{short_prompt}"')
    print(f'  First predicted token: "{token_str}" (id: {top1_id})')
    print()

    status = "PASS" if results["full_pass"] else "FAIL"
    print(f"  Full-vocab math check: KL = {results['kl_full']:.6f} [{status}]")
    print()

    for km in results["per_k"]:
        k_str = f"k={km['k_label']:<5s}"
        print(
            f"  {k_str} KL={km['kl']:<10.6f} "
            f"top1_match={str(km['top1_match']):<6s} "
            f"top5_overlap={km['top5_overlap']:.2f}  "
            f"delta_mean={km['delta_mean']:.6f}"
        )
    print()


def print_summary(all_results: list[dict]) -> None:  # type: ignore[type-arg]
    all_pass = all(r["full_pass"] for r in all_results)
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  All full-vocab math checks passed: {all_pass}")

    # Mean KL at k=50
    kl_50_values = []
    top1_50_count = 0
    for r in all_results:
        for km in r["per_k"]:
            if km["k_label"] == "50":
                kl_50_values.append(km["kl"])
                if km["top1_match"]:
                    top1_50_count += 1

    if kl_50_values:
        mean_kl_50 = sum(kl_50_values) / len(kl_50_values)
        print(f"  Mean KL at k=50 across prompts: {mean_kl_50:.6f}")
        print(f"  Top-1 agreement rate at k=50: {top1_50_count}/{len(all_results)}")

    if all_pass:
        if kl_50_values:
            print(
                f"\n  Conclusion: Math is correct. Oracle delta is zero as expected. "
                f"Restriction to k=50 has mean KL={mean_kl_50:.6f}, confirming "
                f"the renormalization introduces {'negligible' if mean_kl_50 < 0.01 else 'some'} "
                f"divergence in the oracle case."
            )
        else:
            print("\n  Conclusion: Math is correct. Oracle delta is zero as expected.")
    else:
        print("\n  Conclusion: FAIL — formula implementation has a bug. Check details above.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print_header()

    # Load model once (oracle: large = raw = ft = same checkpoint)
    print(f"Loading {MODEL_NAME} ...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print()

    # Get logits for all prompts
    print("Extracting first-token logits for all prompts ...")
    t0 = time.time()
    all_logits = get_logits_for_prompts(model, tokenizer, PROMPTS)
    elapsed_logits = time.time() - t0
    print(f"  Done in {elapsed_logits:.1f}s")
    print()

    vocab_size = all_logits[0].shape[0]
    print(f"Vocabulary size: {vocab_size}")
    print()

    # Unload model to free memory (logits are retained as mx arrays)
    unload_model(model, tokenizer)

    # Reload tokenizer only (for decode in output) — lightweight
    from mlx_lm import load  # noqa: F811

    _, tokenizer_for_decode = load(MODEL_NAME)

    # Compute metrics
    k_values_full = K_VALUES + [vocab_size]
    all_results: list[dict] = []  # type: ignore[type-arg]

    for i, (prompt, logits) in enumerate(zip(PROMPTS, all_logits)):
        # Oracle: all three models produce identical logits
        s_large = logits
        s_raw = logits
        s_ft = logits

        t0 = time.time()
        results = compute_metrics(s_large, s_raw, s_ft, k_values_full)
        elapsed = time.time() - t0

        print_prompt_results(i, prompt, results, tokenizer_for_decode)
        print(f"  (computed in {elapsed:.2f}s)")
        print()

        all_results.append(results)

    print_summary(all_results)


if __name__ == "__main__":
    main()
