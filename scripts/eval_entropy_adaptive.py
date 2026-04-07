#!/usr/bin/env python3
"""
Entropy-adaptive switching: GRAFT at locks, proxy-tuning at forks.

At each generation step, measure the entropy of the large model's distribution:
  - Low entropy (confident, "lock"): use GRAFT k=50 (restriction filters noise)
  - High entropy (uncertain, "fork"): use proxy-tuning full-vocab (preserve options)

Threshold tau is swept to find optimal. Compared with greedy GRAFT and proxy-tuning.
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

# Entropy thresholds to test + baselines
CONFIGS = [
    ("proxy", 0.0, "Proxy-Tuning (baseline)"),
    ("graft", 0.0, "GRAFT k=50 (baseline)"),
    ("adaptive", 1.0, "Adaptive tau=1.0"),
    ("adaptive", 2.0, "Adaptive tau=2.0"),
    ("adaptive", 3.0, "Adaptive tau=3.0"),
    ("adaptive", 4.0, "Adaptive tau=4.0"),
    ("adaptive", 5.0, "Adaptive tau=5.0"),
]

mx.random.seed(42)


def _flush() -> None:
    sys.stdout.flush()


def logsumexp(x: mx.array) -> mx.array:
    c = x.max()
    return c + mx.log(mx.exp(x - c).sum())


def log_softmax(x: mx.array) -> mx.array:
    return x - logsumexp(x)


def entropy(logits: mx.array) -> float:
    """Compute entropy of softmax distribution from logits."""
    log_p = log_softmax(logits)
    p = mx.exp(log_p)
    h = -mx.sum(p * log_p)
    mx.eval(h)
    return h.item()


def load_model(path: str):  # type: ignore[no-untyped-def]
    from mlx_lm import load

    return load(path)


def _score_graft(s_large: mx.array, s_raw: mx.array, s_ft: mx.array, k: int) -> mx.array:
    vocab = s_large.shape[0]
    ek = min(k, vocab)
    s_t = mx.argpartition(s_large, kth=vocab - ek)[-ek:]

    s_ft_k = s_ft[s_t]
    s_raw_k = s_raw[s_t]
    log_pf = s_ft_k - logsumexp(s_ft_k)
    log_pr = s_raw_k - logsumexp(s_raw_k)
    delta = log_pf - log_pr

    log_p = log_softmax(s_large)
    scores = mx.full((vocab,), -1e9)
    scores = scores.at[s_t].add(log_p[s_t] + delta + 1e9)
    return scores


def _score_proxy(s_large: mx.array, s_raw: mx.array, s_ft: mx.array) -> mx.array:
    return s_large + (s_ft - s_raw)


def generate_adaptive(
    model_large: nn.Module,
    model_raw: nn.Module,
    model_ft: nn.Module,
    prompt_tokens: mx.array,
    max_tokens: int,
    eos_id: int,
    method: str,
    tau: float,
) -> tuple[list[int], dict]:  # type: ignore[type-arg]
    from mlx_lm.models.cache import make_prompt_cache

    cl = make_prompt_cache(model_large)
    cr = make_prompt_cache(model_raw)
    cf = make_prompt_cache(model_ft)

    lg = model_large(prompt_tokens[None], cache=cl)[:, -1, :].squeeze(0)
    rw = model_raw(prompt_tokens[None], cache=cr)[:, -1, :].squeeze(0)
    ft = model_ft(prompt_tokens[None], cache=cf)[:, -1, :].squeeze(0)

    graft_count = 0
    proxy_count = 0

    if method == "proxy":
        score = _score_proxy(lg, rw, ft)
        proxy_count += 1
    elif method == "graft":
        score = _score_graft(lg, rw, ft, 50)
        graft_count += 1
    else:
        h = entropy(lg)
        if h < tau:
            score = _score_graft(lg, rw, ft, 50)
            graft_count += 1
        else:
            score = _score_proxy(lg, rw, ft)
            proxy_count += 1

    token = mx.argmax(score)
    mx.eval(token)
    tokens = [token.item()]

    for _ in range(max_tokens - 1):
        inp = token.reshape(1, 1)
        lg = model_large(inp, cache=cl)[:, -1, :].squeeze(0)
        rw = model_raw(inp, cache=cr)[:, -1, :].squeeze(0)
        ft = model_ft(inp, cache=cf)[:, -1, :].squeeze(0)

        if method == "proxy":
            score = _score_proxy(lg, rw, ft)
            proxy_count += 1
        elif method == "graft":
            score = _score_graft(lg, rw, ft, 50)
            graft_count += 1
        else:
            h = entropy(lg)
            if h < tau:
                score = _score_graft(lg, rw, ft, 50)
                graft_count += 1
            else:
                score = _score_proxy(lg, rw, ft)
                proxy_count += 1

        token = mx.argmax(score)
        mx.eval(token)
        tid = token.item()
        tokens.append(tid)
        if tid == eos_id:
            break

    del cl, cr, cf
    total = graft_count + proxy_count
    return tokens, {
        "graft_steps": graft_count,
        "proxy_steps": proxy_count,
        "graft_pct": 100 * graft_count / total if total > 0 else 0,
    }


def main() -> None:
    print(f"Date: {date.today()}")
    print(f"Entropy-adaptive experiment: {NUM_PROMPTS} prompts, {len(CONFIGS)} configs")
    _flush()

    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    prompts = [ex["prompt"] for ex in ds][:NUM_PROMPTS]

    print("\nLoading models...")
    _flush()
    ft_model, tokenizer = load_model(MODEL_FT)
    raw_model, _ = load_model(MODEL_RAW)
    large_model, _ = load_model(MODEL_LARGE)
    print(f"  Loaded. {mx.get_peak_memory() / 1e9:.1f} GB peak\n")
    _flush()

    eos_id = tokenizer.eos_token_id or 151645
    all_results: dict[str, list[dict]] = {}  # type: ignore[type-arg]

    for method, tau, label in CONFIGS:
        print(f"--- {label} ---")
        _flush()

        results = []
        t0_total = time.time()
        total_graft = 0
        total_proxy = 0

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
            toks, stats = generate_adaptive(
                large_model,
                raw_model,
                ft_model,
                prompt_tokens,
                MAX_TOKENS,
                eos_id,
                method,
                tau,
            )
            elapsed = time.time() - t0
            text = tokenizer.decode(toks)

            total_graft += stats["graft_steps"]
            total_proxy += stats["proxy_steps"]

            gc.collect()
            mx.clear_cache()

            results.append(
                {
                    "prompt_idx": i,
                    "prompt": prompt_text,
                    "response": text,
                    "num_tokens": len(toks),
                    "time": elapsed,
                    **stats,
                }
            )

            if (i + 1) % 25 == 0:
                rate = (i + 1) / (time.time() - t0_total) * 60
                ts = total_graft + total_proxy
                gp = 100 * total_graft / ts if ts > 0 else 0
                print(f"  [{i + 1}/{NUM_PROMPTS}] ({rate:.1f}/min, {gp:.0f}% graft)")
                _flush()

        all_results[label] = results
        total_time = time.time() - t0_total
        total_steps = total_graft + total_proxy
        gp = 100 * total_graft / total_steps if total_steps > 0 else 0
        print(f"  Done in {total_time / 60:.1f} min ({gp:.0f}% graft steps)\n")
        _flush()

    # Score all
    import importlib.util

    spec = importlib.util.spec_from_file_location("score", str(SCRIPT_DIR / "score_single.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    print(f"{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}\n")
    print(f"  {'Config':<28s} {'Prompt':>12s} {'Instr':>12s} {'%GRAFT':>7s}")
    print(f"  {'-' * 28} {'-' * 12} {'-' * 12} {'-' * 7}")

    for label, results in all_results.items():
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
        tg = sum(r["graft_steps"] for r in results)
        tp = sum(r["proxy_steps"] for r in results)
        gp = 100 * tg / (tg + tp) if (tg + tp) > 0 else 0
        pp_pct = 100 * pp / n
        ip_pct = 100 * ip / it if it > 0 else 0
        print(
            f"  {label:<28s} "
            f"{pp:>3d}/{n} ({pp_pct:>4.1f}%) "
            f"{ip:>3d}/{it} ({ip_pct:>4.1f}%) "
            f"{gp:>5.0f}%"
        )

    output = str(PROJECT_DIR / "results" / "entropy_adaptive_experiment.json")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
