#!/usr/bin/env python3
"""
Generate responses from a single model for all IFEval prompts.

Usage:
  uv run python scripts/eval_generate_single.py <model_path> <output_json>

Example:
  uv run python scripts/eval_generate_single.py \
    mlx-community/Qwen3-14B-4bit results/generation_14b_instruct.json
"""

from __future__ import annotations

import gc
import json
import sys
import time
from datetime import date
from pathlib import Path

import mlx.core as mx

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MAX_TOKENS = 256

mx.random.seed(42)


def _flush() -> None:
    sys.stdout.flush()


def main() -> None:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_path> <output_json>")
        sys.exit(1)

    model_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Date: {date.today()}")
    print(f"Model: {model_path}")
    print(f"Max tokens: {MAX_TOKENS}")
    _flush()

    # Load dataset
    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    prompts = [ex["prompt"] for ex in ds]
    print(f"Prompts: {len(prompts)}")
    _flush()

    # Load model
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    print(f"\nLoading {model_path} ...")
    _flush()
    t0 = time.time()
    model, tokenizer = load(model_path)
    print(f"  Loaded in {time.time() - t0:.1f}s, {mx.get_peak_memory() / 1e9:.1f} GB peak")
    _flush()

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = 151645

    results = []
    total_t0 = time.time()

    for i, prompt_text in enumerate(prompts):
        short = prompt_text[:55].replace("\n", " ")
        prompt_t0 = time.time()

        messages = [{"role": "user", "content": prompt_text}]
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = mx.array(token_ids)

        # Greedy generation
        cache = make_prompt_cache(model)
        logits = model(prompt_tokens[None], cache=cache)[:, -1, :]
        token = mx.argmax(logits, axis=-1).squeeze()
        mx.eval(token)
        tokens = [token.item()]

        for _ in range(MAX_TOKENS - 1):
            logits = model(token.reshape(1, 1), cache=cache)[:, -1, :]
            token = mx.argmax(logits, axis=-1).squeeze()
            mx.eval(token)
            tid = token.item()
            tokens.append(tid)
            if tid == eos_id:
                break

        del cache
        gc.collect()
        mx.clear_cache()

        text = tokenizer.decode(tokens)
        elapsed = time.time() - prompt_t0
        total_elapsed = time.time() - total_t0
        rate = (i + 1) / total_elapsed * 60

        print(
            f"[{i + 1}/{len(prompts)}] {len(tokens)} tok, "
            f"{elapsed:.1f}s ({rate:.1f}/min)  {short}..."
        )
        _flush()

        results.append(
            {
                "prompt_idx": i,
                "prompt": prompt_text,
                "response": text,
                "num_tokens": len(tokens),
                "time": elapsed,
            }
        )

        if (i + 1) % 50 == 0:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  ** saved {len(results)} results")
            _flush()

    # Final save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = time.time() - total_t0
    print(f"\nDone. {len(results)} prompts in {total / 60:.1f} min")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
