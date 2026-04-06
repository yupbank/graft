#!/usr/bin/env python3
"""
Top-k logit boost experiment.

Hypothesis: SSD (self-distillation) ≈ fine-tuning on sharpened outputs ≈ logit update.
If so, we can approximate SSD's effect by simply boosting top-k logits at inference:

  score(i) = s(i) + alpha   if i in top-k(s)
  score(i) = s(i)           otherwise

This is equivalent to adding a uniform reward bonus to the model's own top-k candidates.
No second model needed — just the 14B base with a logit boost.

Tests multiple (k, alpha) combinations on 50 IFEval prompts.
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
MODEL = str(PROJECT_DIR / "models" / "Qwen3-14B-Base-4bit")
MAX_TOKENS = 256
NUM_PROMPTS = 50

CONFIGS = [
    (0, 0.0, "14B base (no boost)"),
    (10, 1.0, "k=10 alpha=1"),
    (10, 2.0, "k=10 alpha=2"),
    (10, 5.0, "k=10 alpha=5"),
    (50, 1.0, "k=50 alpha=1"),
    (50, 2.0, "k=50 alpha=2"),
    (50, 5.0, "k=50 alpha=5"),
    (200, 2.0, "k=200 alpha=2"),
    (200, 5.0, "k=200 alpha=5"),
]

mx.random.seed(42)


def _flush() -> None:
    sys.stdout.flush()


def generate_boosted(
    model: nn.Module,
    prompt_tokens: mx.array,
    max_tokens: int,
    eos_token_id: int,
    top_k: int,
    alpha: float,
) -> list[int]:
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    logits = model(prompt_tokens[None], cache=cache)[:, -1, :].squeeze(0)

    token = _boost_and_sample(logits, top_k, alpha)
    mx.eval(token)
    tokens = [token.item()]

    for _ in range(max_tokens - 1):
        logits = model(token.reshape(1, 1), cache=cache)[:, -1, :].squeeze(0)
        token = _boost_and_sample(logits, top_k, alpha)
        mx.eval(token)
        tid = token.item()
        tokens.append(tid)
        if tid == eos_token_id:
            break

    del cache
    return tokens


def _boost_and_sample(logits: mx.array, top_k: int, alpha: float) -> mx.array:
    if top_k <= 0 or alpha == 0.0:
        return mx.argmax(logits)

    vocab_size = logits.shape[0]
    k = min(top_k, vocab_size)
    top_idx = mx.argpartition(logits, kth=vocab_size - k)[-k:]

    boost = mx.zeros_like(logits)
    boost = boost.at[top_idx].add(mx.full((k,), alpha))
    boosted = logits + boost

    return mx.argmax(boosted)


def main() -> None:
    print(f"Date: {date.today()}")
    print(f"Top-k logit boost experiment: {NUM_PROMPTS} prompts, {len(CONFIGS)} configs")
    print(f"Model: {MODEL}")
    _flush()

    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    prompts = [ex["prompt"] for ex in ds][:NUM_PROMPTS]

    from mlx_lm import load

    print("\nLoading model...")
    _flush()
    model, tokenizer = load(MODEL)
    print(f"  Loaded. {mx.get_peak_memory() / 1e9:.1f} GB peak")
    _flush()

    eos_id = tokenizer.eos_token_id or 151645

    all_results: dict[str, list[dict]] = {}  # type: ignore[type-arg]

    for k, alpha, label in CONFIGS:
        print(f"\n--- {label} (k={k}, alpha={alpha}) ---")
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
            tokens = generate_boosted(
                model,
                prompt_tokens,
                MAX_TOKENS,
                eos_id,
                top_k=k,
                alpha=alpha,
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

            if (i + 1) % 25 == 0:
                rate = (i + 1) / (time.time() - t0_total) * 60
                print(f"  [{i + 1}/{NUM_PROMPTS}] ({rate:.1f}/min)")
                _flush()

        all_results[label] = results
        total = time.time() - t0_total
        print(f"  Done in {total / 60:.1f} min")
        _flush()

    # Score all configs
    import importlib.util

    spec = importlib.util.spec_from_file_location("score", str(SCRIPT_DIR / "score_single.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}\n")

    print(f"  {'Config':<25s} {'Prompt':>12s} {'Instruction':>12s}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 12}")

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
        pp_pct = 100 * pp / n
        ip_pct = 100 * ip / it if it > 0 else 0
        print(f"  {label:<25s} {pp:>3d}/{n} ({pp_pct:>4.1f}%) {ip:>3d}/{it} ({ip_pct:>4.1f}%)")

    # Save
    output_path = str(PROJECT_DIR / "results" / "topk_boost_experiment.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
