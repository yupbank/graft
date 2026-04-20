#!/usr/bin/env python3
"""
Step 2: Same-model delta transfer for GRAFT — bf16 models.

Setup:
  M_raw   = Qwen3-8B-Base-bf16        — the untuned foundation model
  M_ft    = Qwen3-8B-Instruct-bf16    — the instruction-tuned variant
  M_large = Qwen3-8B-Base-bf16        — same as M_raw
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
MODEL_BASE = str(PROJECT_DIR / "models" / "Qwen3-8B-Base-bf16")
MODEL_INSTRUCT = str(PROJECT_DIR / "models" / "Qwen3-8B-Instruct-bf16")
LAMBDA = 1.0
K_VALUES = [10, 50, 200, 1000]
PROMPTS = [
    "Write a Python function that returns the nth Fibonacci number iteratively.",
    "Implement binary search in Python. Return the index or -1 if not found.",
    "Write a Python class for a stack with push, pop, and peek methods.",
    "Given a list of integers, return all pairs that sum to a target value.",
    "Write a Python decorator that measures and prints execution time.",
]
KL_PASS_THRESHOLD = 1e-4

mx.random.seed(42)


def print_header() -> None:
    print(f"Date: {date.today()}")
    print(f"Python: {sys.version}")
    try:
        import mlx_lm
        print(f"mlx-lm: {mlx_lm.__version__}")
    except AttributeError:
        print("mlx-lm: (version unavailable)")
    print(f"Base model (raw/large): {MODEL_BASE}")
    print(f"Instruct model (ft):    {MODEL_INSTRUCT}")
    print(f"Lambda: {LAMBDA}")
    print(f"Precision: bf16")
    print()


def logsumexp(x: mx.array) -> mx.array:
    c = x.max()
    return c + mx.log(mx.exp(x - c).sum())


def log_softmax(x: mx.array) -> mx.array:
    return x - logsumexp(x)


def kl_divergence(log_p: mx.array, log_q: mx.array, floor: float = -1e9) -> float:
    log_p_clamped = mx.maximum(log_p, mx.array(floor))
    log_q_clamped = mx.maximum(log_q, mx.array(floor))
    p = mx.exp(log_p_clamped)
    kl = mx.sum(p * (log_p_clamped - log_q_clamped))
    return kl.item()


def load_model(model_path: str):  # type: ignore[no-untyped-def]
    from mlx_lm import load
    model, tokenizer = load(model_path)
    return model, tokenizer


def unload_model(model: object) -> None:
    del model
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


def get_logits_for_prompts(model, tokenizer, prompts: list[str]) -> list[mx.array]:
    all_logits: list[mx.array] = []
    for prompt_text in prompts:
        messages = [{"role": "user", "content": prompt_text}]
        token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        prompt_tokens = mx.array(token_ids)
        logits = model(prompt_tokens[None])[:, -1, :]
        mx.eval(logits)
        all_logits.append(logits.squeeze(0))
    return all_logits


def compute_metrics(s_large, s_raw, s_ft, k_values):
    vocab_size = s_large.shape[0]
    score_full = s_large + LAMBDA * (s_ft - s_raw)
    log_p_ft_full = log_softmax(s_ft)
    log_p_recovered_full = log_softmax(score_full)
    kl_full = kl_divergence(log_p_ft_full, log_p_recovered_full)

    top1_ft_id = mx.argmax(s_ft).item()
    top1_base_id = mx.argmax(s_large).item()

    results = {
        "vocab_size": vocab_size,
        "kl_full": kl_full,
        "full_pass": abs(kl_full) < KL_PASS_THRESHOLD,
        "top1_ft_id": top1_ft_id,
        "top1_base_id": top1_base_id,
        "per_k": [],
    }

    for k in k_values:
        effective_k = min(k, vocab_size)
        is_full = effective_k == vocab_size

        if is_full:
            s_t = mx.arange(vocab_size)
        else:
            s_t = mx.argpartition(s_large, kth=vocab_size - effective_k)[-effective_k:]

        s_ft_k = s_ft[s_t]
        s_raw_k = s_raw[s_t]

        log_p_ft_restricted = s_ft_k - logsumexp(s_ft_k)
        log_p_raw_restricted = s_raw_k - logsumexp(s_raw_k)
        delta = log_p_ft_restricted - log_p_raw_restricted

        log_p_large_full = log_softmax(s_large)
        log_p_large_restricted = log_p_large_full[s_t]
        score_restricted = log_p_large_restricted + LAMBDA * delta

        log_p_recovered_restricted = score_restricted - logsumexp(score_restricted)
        log_p_ft_restricted_norm = s_ft_k - logsumexp(s_ft_k)

        kl_restricted = kl_divergence(log_p_ft_restricted_norm, log_p_recovered_restricted)

        top1_recovered = s_t[mx.argmax(log_p_recovered_restricted)].item()
        top1_ft_restricted = s_t[mx.argmax(log_p_ft_restricted_norm)].item()
        top1_match = top1_recovered == top1_ft_restricted

        n5 = min(5, effective_k)
        top5_recovered_idx = mx.argpartition(log_p_recovered_restricted, kth=effective_k - n5)[-n5:]
        top5_ft_idx = mx.argpartition(log_p_ft_restricted_norm, kth=effective_k - n5)[-n5:]
        top5_recovered_set = set(s_t[top5_recovered_idx].tolist())
        top5_ft_set = set(s_t[top5_ft_idx].tolist())
        top5_overlap = len(top5_recovered_set & top5_ft_set) / 5.0

        delta_mean = mx.mean(mx.abs(delta)).item()
        delta_max = mx.max(mx.abs(delta)).item()

        mx.eval(delta)
        if effective_k >= 5:
            top5_promote_idx = mx.argpartition(delta, kth=effective_k - 5)[-5:]
            top5_demote_idx = mx.argpartition(-delta, kth=effective_k - 5)[-5:]
            promoted_ids = s_t[top5_promote_idx].tolist()
            demoted_ids = s_t[top5_demote_idx].tolist()
            promoted_deltas = delta[top5_promote_idx].tolist()
            demoted_deltas = delta[top5_demote_idx].tolist()
        else:
            promoted_ids = s_t.tolist()
            demoted_ids = s_t.tolist()
            promoted_deltas = delta.tolist()
            demoted_deltas = delta.tolist()

        k_label = "full" if is_full else str(k)
        results["per_k"].append({
            "k_label": k_label, "k": effective_k, "kl": kl_restricted,
            "top1_match": top1_match, "top5_overlap": top5_overlap,
            "delta_mean": delta_mean, "delta_max": delta_max,
            "promoted_ids": promoted_ids, "promoted_deltas": promoted_deltas,
            "demoted_ids": demoted_ids, "demoted_deltas": demoted_deltas,
        })

    return results


def print_prompt_results(idx, prompt, results, tokenizer):
    top1_ft_id = results["top1_ft_id"]
    top1_base_id = results["top1_base_id"]
    ft_token = tokenizer.decode([top1_ft_id])
    base_token = tokenizer.decode([top1_base_id])
    short_prompt = prompt[:60] + "..." if len(prompt) > 60 else prompt

    print(f'Prompt {idx + 1}: "{short_prompt}"')
    print(f'  Base model predicts:    "{base_token}" (id: {top1_base_id})')
    print(f'  Instruct model predicts: "{ft_token}" (id: {top1_ft_id})')
    print()

    status = "PASS" if results["full_pass"] else "FAIL"
    print(f"  Full-vocab math check (large=raw): KL = {results['kl_full']:.6f} [{status}]")
    print()

    for km in results["per_k"]:
        k_str = f"k={km['k_label']:<5s}"
        print(
            f"  {k_str} KL={km['kl']:<10.6f} "
            f"top1_match={str(km['top1_match']):<6s} "
            f"top5_overlap={km['top5_overlap']:.2f}  "
            f"delta_mean={km['delta_mean']:.4f}  "
            f"delta_max={km['delta_max']:.4f}"
        )

    for km in results["per_k"]:
        if km["k_label"] == "50":
            print()
            print("  Delta analysis (k=50):")
            print("    Top promoted tokens (instruct wants these MORE than base):")
            pairs = sorted(zip(km["promoted_ids"], km["promoted_deltas"]), key=lambda x: -x[1])
            for tid, d in pairs[:5]:
                tok = tokenizer.decode([tid])
                print(f'      "{tok}" (id:{tid})  delta={d:+.4f}')
            print("    Top demoted tokens (instruct wants these LESS than base):")
            pairs = sorted(zip(km["demoted_ids"], km["demoted_deltas"]), key=lambda x: x[1])
            for tid, d in pairs[:5]:
                tok = tokenizer.decode([tid])
                print(f'      "{tok}" (id:{tid})  delta={d:+.4f}')
            break
    print()


def print_summary(all_results):
    all_pass = all(r["full_pass"] for r in all_results)
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  All full-vocab math checks passed (large=raw identity): {all_pass}")

    k_labels = [km["k_label"] for km in all_results[0]["per_k"]]
    for k_label in k_labels:
        kl_vals, top1_count, top5_vals, delta_means = [], 0, [], []
        for r in all_results:
            for km in r["per_k"]:
                if km["k_label"] == k_label:
                    kl_vals.append(km["kl"])
                    if km["top1_match"]: top1_count += 1
                    top5_vals.append(km["top5_overlap"])
                    delta_means.append(km["delta_mean"])

        mean_kl = sum(kl_vals) / len(kl_vals)
        mean_top5 = sum(top5_vals) / len(top5_vals)
        mean_delta = sum(delta_means) / len(delta_means)
        k_str = f"k={k_label:<5s}"
        print(
            f"  {k_str} mean_KL={mean_kl:<10.6f} "
            f"top1={top1_count}/{len(all_results)}  "
            f"mean_top5={mean_top5:.2f}  "
            f"mean_|delta|={mean_delta:.4f}"
        )

    all_delta_nonzero = all(any(km["delta_mean"] > 0.001 for km in r["per_k"]) for r in all_results)
    print()
    if all_pass and all_delta_nonzero:
        print("  Conclusion: The large=raw identity holds. Delta is non-zero. Ready for Step 3.")
    elif all_pass and not all_delta_nonzero:
        print("  WARNING: Full-vocab KL=0 but delta is near zero.")
    else:
        print("  FAIL: Full-vocab KL should be 0 since large=raw.")


def main() -> None:
    print_header()

    print(f"Loading instruct model: {MODEL_INSTRUCT} ...")
    t0 = time.time()
    instruct_model, tokenizer = load_model(MODEL_INSTRUCT)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("  Extracting instruct logits ...")
    t0 = time.time()
    instruct_logits = get_logits_for_prompts(instruct_model, tokenizer, PROMPTS)
    print(f"  Done in {time.time() - t0:.1f}s")
    unload_model(instruct_model)
    print()

    print(f"Loading base model: {MODEL_BASE} ...")
    t0 = time.time()
    base_model, _ = load_model(MODEL_BASE)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("  Extracting base logits ...")
    t0 = time.time()
    base_logits = get_logits_for_prompts(base_model, tokenizer, PROMPTS)
    print(f"  Done in {time.time() - t0:.1f}s")
    unload_model(base_model)
    print()

    vocab_size = base_logits[0].shape[0]
    print(f"Vocabulary size: {vocab_size}")
    print()

    k_values_full = K_VALUES + [vocab_size]
    all_results = []

    for i, (prompt, base_lg, instruct_lg) in enumerate(zip(PROMPTS, base_logits, instruct_logits)):
        s_large = base_lg
        s_raw = base_lg
        s_ft = instruct_lg

        t0 = time.time()
        results = compute_metrics(s_large, s_raw, s_ft, k_values_full)
        elapsed = time.time() - t0

        print_prompt_results(i, prompt, results, tokenizer)
        print(f"  (computed in {elapsed:.2f}s)")
        print()

        all_results.append(results)

    print_summary(all_results)


if __name__ == "__main__":
    main()
