#!/usr/bin/env python3
"""
Proxy-Tuning baseline (Liu et al. 2024) on IFEval.

Implements the exact method from the paper: full-vocabulary logit arithmetic.
  score(i) = s_large(i) + alpha * (s_ft(i) - s_raw(i))    for all i in V

This differs from GRAFT which uses log-probability space with top-k restriction.
Comparison shows the effect of our restriction + renormalization.

Loads all 3 models simultaneously, generates token-by-token.
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
ALPHA = 1.0
MAX_TOKENS = 256
NUM_PROMPTS = 541
OUTPUT_JSON = str(PROJECT_DIR / "results" / "generation_proxy_tuning.json")

mx.random.seed(42)


def _flush() -> None:
    sys.stdout.flush()


def load_model(model_path: str):  # type: ignore[no-untyped-def]
    from mlx_lm import load

    return load(model_path)


def generate_proxy_tuning(
    model_large: nn.Module,
    model_raw: nn.Module,
    model_ft: nn.Module,
    prompt_tokens: mx.array,
    max_tokens: int,
    eos_token_id: int,
    alpha: float = 1.0,
) -> list[int]:
    """Proxy-tuning: full-vocab logit arithmetic (Liu et al. 2024)."""
    from mlx_lm.models.cache import make_prompt_cache

    cache_large = make_prompt_cache(model_large)
    cache_raw = make_prompt_cache(model_raw)
    cache_ft = make_prompt_cache(model_ft)

    # Prefill
    lg = model_large(prompt_tokens[None], cache=cache_large)[:, -1, :].squeeze(0)
    rw = model_raw(prompt_tokens[None], cache=cache_raw)[:, -1, :].squeeze(0)
    ft = model_ft(prompt_tokens[None], cache=cache_ft)[:, -1, :].squeeze(0)

    # Proxy-tuning: score = s_large + alpha * (s_ft - s_raw) over full vocab
    score = lg + alpha * (ft - rw)
    token = mx.argmax(score)
    mx.eval(token)
    tokens = [token.item()]

    for _ in range(max_tokens - 1):
        inp = token.reshape(1, 1)
        lg = model_large(inp, cache=cache_large)[:, -1, :].squeeze(0)
        rw = model_raw(inp, cache=cache_raw)[:, -1, :].squeeze(0)
        ft = model_ft(inp, cache=cache_ft)[:, -1, :].squeeze(0)

        score = lg + alpha * (ft - rw)
        token = mx.argmax(score)
        mx.eval(token)
        tid = token.item()
        tokens.append(tid)
        if tid == eos_token_id:
            break

    del cache_large, cache_raw, cache_ft
    return tokens


def main() -> None:
    print(f"Date: {date.today()}")
    print("Method: Proxy-Tuning (Liu et al. 2024)")
    print(f"Config: alpha={ALPHA}, max_tokens={MAX_TOKENS}")
    _flush()

    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    prompts = [ex["prompt"] for ex in ds][:NUM_PROMPTS]
    print(f"Prompts: {len(prompts)}")
    _flush()

    # Load all 3 models
    print(f"\n  Loading 8B instruct: {MODEL_FT} ...")
    _flush()
    t0 = time.time()
    ft_model, tokenizer = load_model(MODEL_FT)
    print(f"    done ({time.time() - t0:.1f}s, {mx.get_peak_memory() / 1e9:.1f} GB)")
    _flush()

    print(f"  Loading 8B base: {MODEL_RAW} ...")
    _flush()
    t0 = time.time()
    raw_model, _ = load_model(MODEL_RAW)
    print(f"    done ({time.time() - t0:.1f}s, {mx.get_peak_memory() / 1e9:.1f} GB)")
    _flush()

    print(f"  Loading 14B base: {MODEL_LARGE} ...")
    _flush()
    t0 = time.time()
    large_model, _ = load_model(MODEL_LARGE)
    print(f"    done ({time.time() - t0:.1f}s, {mx.get_peak_memory() / 1e9:.1f} GB)")
    _flush()

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = 151645
    print(f"\n  All loaded. EOS={eos_id}. Starting generation.\n")
    _flush()

    results = []
    total_t0 = time.time()

    for i, prompt_text in enumerate(prompts):
        prompt_t0 = time.time()
        short = prompt_text[:55].replace("\n", " ")

        messages = [{"role": "user", "content": prompt_text}]
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = mx.array(token_ids)

        gen_tokens = generate_proxy_tuning(
            large_model,
            raw_model,
            ft_model,
            prompt_tokens,
            MAX_TOKENS,
            eos_id,
            alpha=ALPHA,
        )
        elapsed = time.time() - prompt_t0
        text = tokenizer.decode(gen_tokens)

        total_elapsed = time.time() - total_t0
        rate = (i + 1) / total_elapsed * 60
        print(
            f"[{i + 1}/{len(prompts)}] {len(gen_tokens)} tok, "
            f"{elapsed:.1f}s ({rate:.1f}/min)  {short}..."
        )
        _flush()

        gc.collect()
        mx.clear_cache()

        results.append(
            {
                "prompt_idx": i,
                "prompt": prompt_text,
                "response": text,
                "num_tokens": len(gen_tokens),
                "time": elapsed,
            }
        )

        if (i + 1) % 50 == 0:
            Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_JSON, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  ** saved {len(results)} results")
            _flush()

    # Final save
    del large_model, raw_model, ft_model
    gc.collect()
    mx.synchronize()
    mx.clear_cache()

    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = time.time() - total_t0
    print(f"\nDone. {len(results)} prompts in {total / 60:.1f} min")
    print(f"Saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
