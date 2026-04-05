#!/usr/bin/env python3
"""
IFEval distributional metrics at scale.

Runs the Step 3 cross-model transfer analysis on all 541 IFEval prompts
(first-token logits only) to validate that the 8B instruct delta meaningfully
steers the 14B base model across a diverse instruction-following benchmark.

Models loaded sequentially to manage memory:
  1. 8B-Instruct → logits for 541 prompts → unload
  2. 8B-Base     → logits for 541 prompts → unload
  3. 14B-Base    → logits for 541 prompts → unload

Then compute all metrics offline.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from collections import Counter
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
K_VALUES = [10, 50, 200]
OUTPUT_JSON = str(PROJECT_DIR / "results" / "ifeval_logits.json")

mx.random.seed(42)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def logsumexp(x: mx.array) -> mx.array:
    c = x.max()
    return c + mx.log(mx.exp(x - c).sum())


def log_softmax(x: mx.array) -> mx.array:
    return x - logsumexp(x)


def kl_divergence(log_p: mx.array, log_q: mx.array) -> float:
    log_p_c = mx.maximum(log_p, mx.array(-1e9))
    log_q_c = mx.maximum(log_q, mx.array(-1e9))
    p = mx.exp(log_p_c)
    return mx.sum(p * (log_p_c - log_q_c)).item()


def js_divergence(log_p: mx.array, log_q: mx.array) -> float:
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

    model, tokenizer = load(model_path)
    return model, tokenizer


def unload_model(model: object) -> None:
    del model
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


# ---------------------------------------------------------------------------
# Logit extraction
# ---------------------------------------------------------------------------
def get_all_logits(
    model,  # type: ignore[no-untyped-def]
    tokenizer,  # type: ignore[no-untyped-def]
    prompts: list[str],
    batch_label: str = "",
) -> list[mx.array]:
    """Get first-token logits for all prompts. Prints progress."""
    all_logits: list[mx.array] = []
    n = len(prompts)
    t0 = time.time()
    for i, prompt_text in enumerate(prompts):
        messages = [{"role": "user", "content": prompt_text}]
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = mx.array(token_ids)
        logits = model(prompt_tokens[None])[:, -1, :]
        mx.eval(logits)
        all_logits.append(logits.squeeze(0))
        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"    {batch_label} [{i + 1}/{n}] {rate:.1f} prompts/s, {elapsed:.0f}s elapsed")
    return all_logits


# ---------------------------------------------------------------------------
# Metrics per prompt
# ---------------------------------------------------------------------------
def compute_prompt_metrics(
    s_large: mx.array,
    s_raw: mx.array,
    s_ft: mx.array,
    k_values: list[int],
) -> dict:  # type: ignore[type-arg]
    vocab_size = s_large.shape[0]

    # Full-vocab
    score_full = s_large + LAMBDA * (s_ft - s_raw)
    log_p_ft = log_softmax(s_ft)
    log_p_rec = log_softmax(score_full)
    log_p_large = log_softmax(s_large)

    top1_large = mx.argmax(s_large).item()
    top1_ft = mx.argmax(s_ft).item()
    top1_rec = mx.argmax(score_full).item()

    result: dict = {  # type: ignore[type-arg]
        "top1_large": top1_large,
        "top1_ft": top1_ft,
        "top1_rec_full": top1_rec,
        "shifted": top1_rec != top1_large,
        "matches_ft": top1_rec == top1_ft,
        "kl_ft_rec_full": kl_divergence(log_p_ft, log_p_rec),
        "js_ft_rec_full": js_divergence(log_p_ft, log_p_rec),
        "js_large_rec_full": js_divergence(log_p_large, log_p_rec),
        "per_k": {},
    }

    for k in k_values:
        ek = min(k, vocab_size)
        if ek == vocab_size:
            s_t = mx.arange(vocab_size)
        else:
            s_t = mx.argpartition(s_large, kth=vocab_size - ek)[-ek:]

        s_ft_k = s_ft[s_t]
        s_raw_k = s_raw[s_t]

        log_pf = s_ft_k - logsumexp(s_ft_k)
        log_pr = s_raw_k - logsumexp(s_raw_k)
        delta = log_pf - log_pr

        score_r = log_p_large[s_t] + LAMBDA * delta
        log_p_rec_r = score_r - logsumexp(score_r)
        log_p_ft_r = s_ft_k - logsumexp(s_ft_k)

        kl_r = kl_divergence(log_p_ft_r, log_p_rec_r)
        top1_rec_r = s_t[mx.argmax(log_p_rec_r)].item()
        top1_ft_r = s_t[mx.argmax(log_p_ft_r)].item()

        n5 = min(5, ek)
        t5_rec = set(s_t[mx.argpartition(log_p_rec_r, kth=ek - n5)[-n5:]].tolist())
        t5_ft = set(s_t[mx.argpartition(log_p_ft_r, kth=ek - n5)[-n5:]].tolist())

        result["per_k"][k] = {
            "kl_ft_rec": kl_r,
            "top1_match_ft": top1_rec_r == top1_ft_r,
            "top5_overlap": len(t5_rec & t5_ft) / 5.0,
            "delta_mean": mx.mean(mx.abs(delta)).item(),
        }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Date: {date.today()}")
    print(f"Python: {sys.version}")
    print(f"Lambda: {LAMBDA}")
    print()

    # Load IFEval
    print("Loading IFEval dataset ...")
    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    prompts = [ex["prompt"] for ex in ds]
    categories = [ex["instruction_id_list"] for ex in ds]
    print(f"  {len(prompts)} prompts loaded")
    print()

    # --- Extract logits from each model sequentially ---
    print(f"Loading 8B instruct: {MODEL_FT}")
    ft_model, tokenizer = load_model(MODEL_FT)
    ft_logits = get_all_logits(ft_model, tokenizer, prompts, "8B-instruct")
    unload_model(ft_model)
    print()

    print(f"Loading 8B base: {MODEL_RAW}")
    raw_model, _ = load_model(MODEL_RAW)
    raw_logits = get_all_logits(raw_model, tokenizer, prompts, "8B-base")
    unload_model(raw_model)
    print()

    print(f"Loading 14B base: {MODEL_LARGE}")
    large_model, _ = load_model(MODEL_LARGE)
    large_logits = get_all_logits(large_model, tokenizer, prompts, "14B-base")
    unload_model(large_model)
    print()

    # --- Compute metrics ---
    print("Computing metrics ...")
    all_results = []
    t0 = time.time()
    for i in range(len(prompts)):
        r = compute_prompt_metrics(large_logits[i], raw_logits[i], ft_logits[i], K_VALUES)
        r["prompt_idx"] = i
        r["categories"] = categories[i]
        all_results.append(r)
        if (i + 1) % 100 == 0:
            print(f"    [{i + 1}/{len(prompts)}]")
    print(f"  Done in {time.time() - t0:.1f}s")
    print()

    # --- Save raw results ---
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Raw results saved to {OUTPUT_JSON}")
    print()

    # --- Print summary ---
    n = len(all_results)
    shifted_count = sum(1 for r in all_results if r["shifted"])
    ft_match_count = sum(1 for r in all_results if r["matches_ft"])

    kl_ft_vals = [r["kl_ft_rec_full"] for r in all_results]
    js_ft_vals = [r["js_ft_rec_full"] for r in all_results]
    js_lg_vals = [r["js_large_rec_full"] for r in all_results]

    print("=" * 72)
    print("IFEVAL LOGITS EVALUATION SUMMARY")
    print("=" * 72)
    print(f"  Prompts: {n}")
    print(f"  Top-1 shifted (14B->recovered): {shifted_count}/{n} ({100 * shifted_count / n:.1f}%)")
    print(
        f"  Top-1 matches 8B instruct:      {ft_match_count}/{n} ({100 * ft_match_count / n:.1f}%)"
    )
    print()

    def stats(vals: list[float], name: str) -> None:
        s = sorted(vals)
        mean = sum(s) / len(s)
        median = s[len(s) // 2]
        p5 = s[int(0.05 * len(s))]
        p95 = s[int(0.95 * len(s))]
        print(f"  {name:30s}  mean={mean:.4f}  median={median:.4f}  p5={p5:.4f}  p95={p95:.4f}")

    stats(kl_ft_vals, "KL(ft || recovered) full")
    stats(js_ft_vals, "JS(ft, recovered) full")
    stats(js_lg_vals, "JS(large, recovered) full")
    print()

    # Per-k stats
    print("  Restricted metrics (averaged):")
    for k in K_VALUES:
        kl_vals = [r["per_k"][k]["kl_ft_rec"] for r in all_results]
        t1_count = sum(1 for r in all_results if r["per_k"][k]["top1_match_ft"])
        t5_vals = [r["per_k"][k]["top5_overlap"] for r in all_results]
        delta_vals = [r["per_k"][k]["delta_mean"] for r in all_results]
        print(
            f"    k={k:<5d} "
            f"KL(ft||rec)={sum(kl_vals) / n:<9.4f} "
            f"ft_top1={t1_count}/{n}  "
            f"top5={sum(t5_vals) / n:.2f}  "
            f"|delta|={sum(delta_vals) / n:.2f}"
        )
    print()

    # Category breakdown
    print("  Category breakdown (top-1 shift rate):")
    cat_shift: Counter[str] = Counter()
    cat_total: Counter[str] = Counter()
    for r in all_results:
        for cat_full in r["categories"]:
            cat = cat_full.split(":")[0] if ":" in cat_full else cat_full
            cat_total[cat] += 1
            if r["shifted"]:
                cat_shift[cat] += 1
    for cat in sorted(cat_total.keys()):
        rate = cat_shift[cat] / cat_total[cat]
        print(f"    {cat:30s} {cat_shift[cat]:3d}/{cat_total[cat]:<3d} ({100 * rate:.0f}%)")
    print()

    # Decode top shifted tokens
    print("  Most common recovered top-1 tokens (across all shifted prompts):")
    shifted_tokens: Counter[int] = Counter()
    for r in all_results:
        if r["shifted"]:
            shifted_tokens[r["top1_rec_full"]] += 1
    # Reload tokenizer for decode
    _, tok_for_decode = load_model(MODEL_FT)
    for tid, count in shifted_tokens.most_common(10):
        token_str = tok_for_decode.decode([tid])
        print(f'    "{token_str}" (id:{tid}): {count} times')
    unload_model(tok_for_decode)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
