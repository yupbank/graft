#!/usr/bin/env python3
"""
Full 541-prompt logprobs probe across model families.
Collects top-20 logprobs at first generation token for each model.
Saves incrementally every 50 prompts.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT = str(PROJECT_DIR / "results" / "logprobs_full_541.json")

sys.path.insert(0, "/Users/pengyu/src/github.com/yupbank/skillflow/src")
from skillflow.api_config import get_openai_client  # noqa: E402


def get_api_logprobs(client, model: str, prompt: str) -> dict:  # type: ignore[type-arg]
    """Get first-token top-20 logprobs from API."""
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
        )
        lp_data = r.choices[0].logprobs
        if lp_data and lp_data.content:
            d = lp_data.content[0]
            return {
                "top_token": d.token,
                "top_logprob": d.logprob,
                "top_k": {lp.token: lp.logprob for lp in d.top_logprobs},
            }
    except Exception as e:
        return {"error": str(e)[:100]}
    return {"error": "no logprobs"}


def get_qwen_logprobs(model_path: str, tokenizer, prompt: str, k: int = 20) -> dict:  # type: ignore[type-arg, no-untyped-def]
    """Get first-token top-k logprobs from local Qwen model."""
    import gc

    import mlx.core as mx
    from mlx_lm import load

    model, _ = load(model_path)
    msgs = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    pt = mx.array(ids)
    logits = model(pt[None])[:, -1, :].squeeze(0)
    vocab = logits.shape[0]
    top_idx = mx.argpartition(logits, kth=vocab - k)[-k:]
    max_l = mx.max(logits)
    log_sum = max_l + mx.log(mx.sum(mx.exp(logits - max_l)))
    top_lp = logits[top_idx] - log_sum
    mx.eval(top_idx, top_lp)

    top_k = {}
    for j in range(k):
        tok = tokenizer.decode([top_idx[j].item()])
        top_k[tok] = round(top_lp[j].item(), 4)

    top_token_idx = mx.argmax(logits).item()
    top_token = tokenizer.decode([top_token_idx])

    del model
    gc.collect()
    mx.synchronize()
    mx.clear_cache()

    return {"top_token": top_token, "top_k": top_k}


def main() -> None:
    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    N = len(ds)

    print(f"Date: {date.today()}")
    print(f"Probing {N} prompts across models")
    sys.stdout.flush()

    client = get_openai_client()

    # API models (fast, parallel-safe)
    api_models = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano"]

    # Local models (load once for tokenizer, reload per-prompt is too slow)
    # Instead: batch all prompts per local model
    local_models = [
        ("models/Qwen3-14B-Base-4bit", "Qwen3-14B-Base"),
        ("mlx-community/Qwen3-8B-4bit", "Qwen3-8B-Instruct"),
    ]

    # Load tokenizer once
    from mlx_lm import load as mlx_load

    _, tokenizer = mlx_load("mlx-community/Qwen3-8B-4bit")

    # Try to resume
    all_results = []
    start_from = 0
    if Path(OUTPUT).exists():
        with open(OUTPUT) as f:
            all_results = json.load(f)
        start_from = len(all_results)
        print(f"Resuming from prompt {start_from}")
        sys.stdout.flush()

    t0 = time.time()

    for i in range(start_from, N):
        prompt = ds[i]["prompt"]
        constraints = ds[i]["instruction_id_list"]

        result = {
            "idx": i,
            "constraints": constraints,
            "models": {},
        }

        # API models
        for model in api_models:
            result["models"][model] = get_api_logprobs(client, model, prompt)

        # Local models (load/unload each — slow but memory-safe)
        for path, name in local_models:
            result["models"][name] = get_qwen_logprobs(path, tokenizer, prompt)

        all_results.append(result)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1 - start_from) / elapsed * 60
            print(f"  [{i + 1}/{N}] ({rate:.1f}/min)")
            sys.stdout.flush()

        if (i + 1) % 50 == 0:
            Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT, "w") as f:
                json.dump(all_results, f, ensure_ascii=False)
            print(f"  ** saved {len(all_results)} prompts")
            sys.stdout.flush()

    # Final save
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(all_results, f, ensure_ascii=False)

    total = time.time() - t0
    print(f"\nDone. {len(all_results)} prompts in {total / 60:.1f} min")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
