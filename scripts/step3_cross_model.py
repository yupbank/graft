#!/usr/bin/env python3
"""
Step 3: Cross-model delta transfer for GRAFT.

Setup:
  M_raw   = Qwen3-8B-Base-4bit       — the small base model
  M_ft    = Qwen3-8B-Instruct-4bit   — the instruction-tuned small model
  M_large = Qwen3-14B-Base-4bit      — the LARGE base model (different from M_raw!)

THIS IS THE ACTUAL HYPOTHESIS TEST.

For the first time, M_large != M_raw. The proxy tuning identity no longer holds:
  score = s_large + (s_ft - s_raw) != s_ft   (because s_large != s_raw)

Instead, the delta (s_ft - s_raw) learned from the 8B pair is "grafted" onto the
14B model. The key question: does the 8B instruction-following delta meaningfully
steer the 14B base model toward instruction-following behavior?

WHAT SUCCESS LOOKS LIKE:
  - Full-vocab KL is non-zero but finite — the formula produces a valid distribution.
  - The recovered distribution shifts the 14B model's top-1 prediction toward
    instruction-following tokens (like "Here", "Sure") and away from raw
    completion tokens (like "import", "def").
  - Restricted KL increases as k shrinks — restriction now has a real cost
    because the full-vocab Z no longer cancels.
  - Top-1 agreement at k=200+ remains high — the most important tokens are
    in the candidate set.

WHAT FAILURE LOOKS LIKE:
  - The delta does nothing — 14B predictions unchanged, meaning the 8B delta
    doesn't transfer across model sizes.
  - The delta makes things worse — KL diverges wildly or top-1 becomes garbage.
  - Restriction kills the signal at k=50 — too few candidates to capture the
    relevant tokens.
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
MODEL_RAW = str(PROJECT_DIR / "models" / "Qwen3-8B-Base-4bit")
MODEL_FT = "mlx-community/Qwen3-8B-4bit"
MODEL_LARGE = str(PROJECT_DIR / "models" / "Qwen3-14B-Base-4bit")
LAMBDA = 1.0
K_VALUES = [10, 50, 200, 1000]  # vocab_size is appended dynamically
PROMPTS = [
    "Write a Python function that returns the nth Fibonacci number iteratively.",
    "Implement binary search in Python. Return the index or -1 if not found.",
    "Write a Python class for a stack with push, pop, and peek methods.",
    "Given a list of integers, return all pairs that sum to a target value.",
    "Write a Python decorator that measures and prints execution time.",
]

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
    print(f"Raw model (8B base):      {MODEL_RAW}")
    print(f"Fine-tuned model (8B ft): {MODEL_FT}")
    print(f"Large model (14B base):   {MODEL_LARGE}")
    print(f"Lambda: {LAMBDA}")
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


def js_divergence(log_p: mx.array, log_q: mx.array) -> float:
    """Jensen-Shannon divergence — symmetric, bounded [0, ln2]."""
    p = mx.exp(log_p)
    q = mx.exp(log_q)
    m = 0.5 * (p + q)
    log_m = mx.log(mx.maximum(m, mx.array(1e-30)))
    kl_pm = mx.sum(p * (log_p - log_m)).item()
    kl_qm = mx.sum(q * (log_q - log_m)).item()
    return 0.5 * (kl_pm + kl_qm)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path: str):  # type: ignore[no-untyped-def]
    from mlx_lm import load

    try:
        model, tokenizer = load(model_path)
    except Exception as e:
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            print(f"ERROR: Model {model_path} not found.")
            if model_path.startswith("/") or model_path.startswith("."):
                print("  This is a local path. Did you run the quantization step?")
                print(
                    "  uv run python -m mlx_lm.convert "
                    "--hf-path <HF_MODEL> "
                    f"--mlx-path {model_path} -q --q-bits 4"
                )
            else:
                print(f"  Download it: huggingface-cli download {model_path}")
            sys.exit(1)
        raise
    return model, tokenizer


def unload_model(model: object) -> None:
    del model
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

    # --- Full-vocab proxy tuning ---
    score_full = s_large + LAMBDA * (s_ft - s_raw)
    log_p_ft_full = log_softmax(s_ft)
    log_p_recovered_full = log_softmax(score_full)
    log_p_large_full = log_softmax(s_large)

    kl_ft_vs_recovered = kl_divergence(log_p_ft_full, log_p_recovered_full)
    kl_large_vs_recovered = kl_divergence(log_p_large_full, log_p_recovered_full)
    js_ft_recovered = js_divergence(log_p_ft_full, log_p_recovered_full)
    js_large_recovered = js_divergence(log_p_large_full, log_p_recovered_full)

    # Top-1 predictions from each distribution
    top1_large_id = mx.argmax(s_large).item()
    top1_raw_id = mx.argmax(s_raw).item()
    top1_ft_id = mx.argmax(s_ft).item()
    top1_recovered_full_id = mx.argmax(score_full).item()

    results: dict = {  # type: ignore[type-arg]
        "vocab_size": vocab_size,
        "kl_ft_vs_recovered": kl_ft_vs_recovered,
        "kl_large_vs_recovered": kl_large_vs_recovered,
        "js_ft_recovered": js_ft_recovered,
        "js_large_recovered": js_large_recovered,
        "top1_large_id": top1_large_id,
        "top1_raw_id": top1_raw_id,
        "top1_ft_id": top1_ft_id,
        "top1_recovered_full_id": top1_recovered_full_id,
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
        log_p_large_restricted = log_p_large_full[s_t]
        score_restricted = log_p_large_restricted + LAMBDA * delta

        # Normalize over S_t
        log_p_recovered_restricted = score_restricted - logsumexp(score_restricted)

        # FT distribution renormalized over S_t
        log_p_ft_restricted_norm = s_ft_k - logsumexp(s_ft_k)

        # Large distribution renormalized over S_t
        log_p_large_restricted_norm = log_p_large_restricted - logsumexp(log_p_large_restricted)

        # KL: ft_restricted vs recovered_restricted
        kl_restricted = kl_divergence(log_p_ft_restricted_norm, log_p_recovered_restricted)

        # KL: large_restricted vs recovered_restricted (how far did the delta push?)
        kl_large_vs_rec_restricted = kl_divergence(
            log_p_large_restricted_norm, log_p_recovered_restricted
        )

        # Top-1 agreement (recovered vs ft, restricted to S_t)
        top1_recovered = s_t[mx.argmax(log_p_recovered_restricted)].item()
        top1_ft_restricted = s_t[mx.argmax(log_p_ft_restricted_norm)].item()
        top1_large_restricted = s_t[mx.argmax(log_p_large_restricted_norm)].item()
        top1_match_ft = top1_recovered == top1_ft_restricted
        top1_match_large = top1_recovered == top1_large_restricted

        # Top-5 overlap (recovered vs ft)
        n5 = min(5, effective_k)
        top5_recovered_idx = mx.argpartition(log_p_recovered_restricted, kth=effective_k - n5)[-n5:]
        top5_ft_idx = mx.argpartition(log_p_ft_restricted_norm, kth=effective_k - n5)[-n5:]
        top5_recovered_set = set(s_t[top5_recovered_idx].tolist())
        top5_ft_set = set(s_t[top5_ft_idx].tolist())
        top5_overlap = len(top5_recovered_set & top5_ft_set) / 5.0

        # Delta magnitude
        delta_mean = mx.mean(mx.abs(delta)).item()
        delta_max = mx.max(mx.abs(delta)).item()

        # Top promoted/demoted tokens
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
        results["per_k"].append(
            {
                "k_label": k_label,
                "k": effective_k,
                "kl_ft_restricted": kl_restricted,
                "kl_large_restricted": kl_large_vs_rec_restricted,
                "top1_recovered": top1_recovered,
                "top1_ft_restricted": top1_ft_restricted,
                "top1_large_restricted": top1_large_restricted,
                "top1_match_ft": top1_match_ft,
                "top1_match_large": top1_match_large,
                "top5_overlap": top5_overlap,
                "delta_mean": delta_mean,
                "delta_max": delta_max,
                "promoted_ids": promoted_ids,
                "promoted_deltas": promoted_deltas,
                "demoted_ids": demoted_ids,
                "demoted_deltas": demoted_deltas,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def print_prompt_results(
    idx: int,
    prompt: str,
    results: dict,  # type: ignore[type-arg]
    tokenizer,  # type: ignore[no-untyped-def]
) -> None:
    def tok(tid: int) -> str:
        return tokenizer.decode([tid])

    short_prompt = prompt[:60] + "..." if len(prompt) > 60 else prompt

    print(f'Prompt {idx + 1}: "{short_prompt}"')
    lg_id = results["top1_large_id"]
    ft_id = results["top1_ft_id"]
    rec_id = results["top1_recovered_full_id"]
    print(f'  14B base predicts:         "{tok(lg_id)}" (id: {lg_id})')
    print(f'  8B instruct predicts:      "{tok(ft_id)}" (id: {ft_id})')
    print(f'  Recovered (full) predicts: "{tok(rec_id)}" (id: {rec_id})')
    print()

    print(f"  Full-vocab KL(ft || recovered) = {results['kl_ft_vs_recovered']:.6f}")
    print(f"  Full-vocab KL(large || recovered) = {results['kl_large_vs_recovered']:.6f}")
    print(f"  Full-vocab JS(ft, recovered) = {results['js_ft_recovered']:.6f}")
    print(f"  Full-vocab JS(large, recovered) = {results['js_large_recovered']:.6f}")
    print()

    print("  Restricted metrics:")
    for km in results["per_k"]:
        k_str = f"k={km['k_label']:<5s}"
        rec_tok = tok(km["top1_recovered"])
        print(
            f"    {k_str} "
            f"KL(ft||rec)={km['kl_ft_restricted']:<9.4f} "
            f"KL(lg||rec)={km['kl_large_restricted']:<9.4f} "
            f'top1_rec="{rec_tok}"  '
            f"ft_match={str(km['top1_match_ft']):<6s} "
            f"top5={km['top5_overlap']:.2f}  "
            f"|δ|={km['delta_mean']:.2f}"
        )

    # Delta analysis at k=200
    for km in results["per_k"]:
        if km["k_label"] == "200":
            print()
            print("  Delta analysis (k=200):")
            print("    Promoted (instruct > base):")
            pairs = sorted(zip(km["promoted_ids"], km["promoted_deltas"]), key=lambda x: -x[1])
            for tid, d in pairs[:5]:
                print(f'      "{tok(tid)}" (id:{tid})  δ={d:+.4f}')
            print("    Demoted (instruct < base):")
            pairs = sorted(zip(km["demoted_ids"], km["demoted_deltas"]), key=lambda x: x[1])
            for tid, d in pairs[:5]:
                print(f'      "{tok(tid)}" (id:{tid})  δ={d:+.4f}')
            break
    print()


def print_summary(all_results: list[dict], tokenizer) -> None:  # type: ignore[type-arg, no-untyped-def]
    def tok(tid: int) -> str:
        return tokenizer.decode([tid])

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Full-vocab stats
    mean_kl_ft = sum(r["kl_ft_vs_recovered"] for r in all_results) / len(all_results)
    mean_js_ft = sum(r["js_ft_recovered"] for r in all_results) / len(all_results)
    mean_js_lg = sum(r["js_large_recovered"] for r in all_results) / len(all_results)
    print(f"  Full-vocab mean KL(ft || recovered): {mean_kl_ft:.6f}")
    print(f"  Full-vocab mean JS(ft, recovered):   {mean_js_ft:.6f}")
    print(f"  Full-vocab mean JS(large, recovered): {mean_js_lg:.6f}")
    print()

    # Did the delta shift predictions?
    print("  Top-1 prediction shift (full vocab):")
    for i, r in enumerate(all_results):
        large_tok = tok(r["top1_large_id"])
        rec_tok = tok(r["top1_recovered_full_id"])
        changed = r["top1_recovered_full_id"] != r["top1_large_id"]
        shifted = "SHIFTED" if changed else "(unchanged)"
        is_ft = r["top1_recovered_full_id"] == r["top1_ft_id"]
        match_ft = " matches-ft" if is_ft else ""
        print(f'    P{i + 1}: 14B="{large_tok}" -> recovered="{rec_tok}" {shifted}{match_ft}')
    print()

    # Per-k restricted stats
    k_labels = [km["k_label"] for km in all_results[0]["per_k"]]
    print("  Restricted metrics (averaged across prompts):")
    for k_label in k_labels:
        kl_ft_vals = []
        kl_lg_vals = []
        ft_match_count = 0
        top5_vals = []
        delta_vals = []
        for r in all_results:
            for km in r["per_k"]:
                if km["k_label"] == k_label:
                    kl_ft_vals.append(km["kl_ft_restricted"])
                    kl_lg_vals.append(km["kl_large_restricted"])
                    if km["top1_match_ft"]:
                        ft_match_count += 1
                    top5_vals.append(km["top5_overlap"])
                    delta_vals.append(km["delta_mean"])

        n = len(all_results)
        k_str = f"k={k_label:<5s}"
        print(
            f"    {k_str} "
            f"KL(ft||rec)={sum(kl_ft_vals) / n:<9.4f} "
            f"KL(lg||rec)={sum(kl_lg_vals) / n:<9.4f} "
            f"ft_top1={ft_match_count}/{n}  "
            f"top5={sum(top5_vals) / n:.2f}  "
            f"|δ|={sum(delta_vals) / n:.2f}"
        )

    # Conclusion
    shifts = sum(1 for r in all_results if r["top1_recovered_full_id"] != r["top1_large_id"])
    ft_matches = sum(1 for r in all_results if r["top1_recovered_full_id"] == r["top1_ft_id"])
    print()
    print(
        f"  Conclusion: The 8B instruct delta shifted the 14B base model's top-1 "
        f"prediction in {shifts}/{len(all_results)} prompts. "
        f"Of those, {ft_matches}/{len(all_results)} now match the 8B instruct model's "
        f"prediction. Mean JS(ft, recovered) = {mean_js_ft:.4f}, "
        f"mean JS(large, recovered) = {mean_js_lg:.4f}."
    )
    if shifts >= 3:
        print("  The delta transfers meaningfully across model sizes.")
    elif shifts >= 1:
        print("  The delta shows partial transfer. λ tuning may improve results.")
    else:
        print(
            "  The delta did not shift predictions. "
            "Cross-model transfer may not work at this scale."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print_header()

    # --- Load instruct model (8B ft) first for tokenizer ---
    print(f"Loading 8B instruct model: {MODEL_FT} ...")
    t0 = time.time()
    ft_model, tokenizer = load_model(MODEL_FT)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("  Extracting instruct logits ...")
    t0 = time.time()
    ft_logits = get_logits_for_prompts(ft_model, tokenizer, PROMPTS)
    print(f"  Done in {time.time() - t0:.1f}s")
    unload_model(ft_model)
    print()

    # --- Load raw model (8B base) ---
    print(f"Loading 8B base model: {MODEL_RAW} ...")
    t0 = time.time()
    raw_model, _ = load_model(MODEL_RAW)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("  Extracting base 8B logits ...")
    t0 = time.time()
    raw_logits = get_logits_for_prompts(raw_model, tokenizer, PROMPTS)
    print(f"  Done in {time.time() - t0:.1f}s")
    unload_model(raw_model)
    print()

    # --- Load large model (14B base) ---
    print(f"Loading 14B base model: {MODEL_LARGE} ...")
    t0 = time.time()
    large_model, _ = load_model(MODEL_LARGE)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("  Extracting base 14B logits ...")
    t0 = time.time()
    large_logits = get_logits_for_prompts(large_model, tokenizer, PROMPTS)
    print(f"  Done in {time.time() - t0:.1f}s")
    unload_model(large_model)
    print()

    vocab_size = large_logits[0].shape[0]
    print(f"Vocabulary size: {vocab_size}")
    print()

    # --- Compute metrics ---
    k_values_full = K_VALUES + [vocab_size]
    all_results: list[dict] = []  # type: ignore[type-arg]

    for i, (prompt, large_lg, raw_lg, ft_lg) in enumerate(
        zip(PROMPTS, large_logits, raw_logits, ft_logits)
    ):
        t0 = time.time()
        results = compute_metrics(large_lg, raw_lg, ft_lg, k_values_full)
        elapsed = time.time() - t0

        print_prompt_results(i, prompt, results, tokenizer)
        print(f"  (computed in {elapsed:.2f}s)")
        print()

        all_results.append(results)

    print_summary(all_results, tokenizer)


if __name__ == "__main__":
    main()
