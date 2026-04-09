#!/usr/bin/env python3
"""
Probe instruction-tuning through token-level logprobs.

Compare black-box models (OpenAI) with open models (Qwen) on the same prompts.
For each prompt, collect top-20 logprobs at the first generation token.
Find "instruction-specific tokens" — tokens that instruct models rank high
but base models ignore.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

sys.path.insert(0, "/Users/pengyu/src/github.com/yupbank/skillflow/src")
from skillflow.api_config import get_openai_client  # noqa: E402


def get_openai_logprobs(client, model: str, prompt: str, n_logprobs: int = 20) -> dict:
    """Get first-token logprobs from an OpenAI model."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
        temperature=0,
        logprobs=True,
        top_logprobs=n_logprobs,
    )
    choice = response.choices[0]
    has_lp = choice.logprobs and choice.logprobs.content
    logprobs_data = choice.logprobs.content[0] if has_lp else None

    if logprobs_data:
        top_token = logprobs_data.token
        top_logprob = logprobs_data.logprob
        top_k = {lp.token: lp.logprob for lp in logprobs_data.top_logprobs}
        return {
            "top_token": top_token,
            "top_logprob": top_logprob,
            "top_k_logprobs": top_k,
            "model": model,
        }
    return {"top_token": None, "top_k_logprobs": {}, "model": model}


def get_qwen_logprobs(model_path: str, prompt: str, k: int = 20) -> dict:
    """Get first-token logprobs from a local Qwen model."""
    import mlx.core as mx
    from mlx_lm import load

    model, tokenizer = load(model_path)
    msgs = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    pt = mx.array(ids)
    logits = model(pt[None])[:, -1, :].squeeze(0)

    # Get top-k
    vocab = logits.shape[0]
    top_idx = mx.argpartition(logits, kth=vocab - k)[-k:]
    top_logits = logits[top_idx]

    # Convert to log-probs
    max_l = mx.max(logits)
    log_sum = max_l + mx.log(mx.sum(mx.exp(logits - max_l)))
    top_logprobs = top_logits - log_sum
    mx.eval(top_idx, top_logprobs)

    top_k = {}
    for j in range(k):
        tid = top_idx[j].item()
        lp = top_logprobs[j].item()
        tok = tokenizer.decode([tid])
        top_k[tok] = lp

    top_token_idx = mx.argmax(logits).item()
    top_token = tokenizer.decode([top_token_idx])
    top_lp = (logits[top_token_idx] - log_sum).item()

    import gc

    del model
    gc.collect()
    mx.synchronize()
    mx.clear_cache()

    return {
        "top_token": top_token,
        "top_logprob": top_lp,
        "top_k_logprobs": top_k,
        "model": model_path,
    }


def main() -> None:
    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")

    # Use prompts that had interesting instruction-specific tokens
    interesting_prompts = [
        (8, "letter writing — 'dear' phenomenon"),
        (6, "all capitals critique"),
        (1, "archaic English itinerary"),
        (15, "JSON format"),
        (4, "lowercase question"),
        (18, "Kannada language"),
        (0, "300+ word summary with highlights"),
        (9, "no-comma email"),
        (27, "explain in French"),
        (19, "lowercase tweet"),
    ]

    print(f"Date: {date.today()}")
    print(f"Probing {len(interesting_prompts)} prompts across models\n")
    sys.stdout.flush()

    client = get_openai_client()

    # Models to probe
    openai_models = ["gpt-4.1-mini", "gpt-4.1-nano"]
    qwen_models = [
        ("models/Qwen3-14B-Base-4bit", "Qwen3-14B-Base"),
        ("mlx-community/Qwen3-8B-4bit", "Qwen3-8B-Instruct"),
    ]

    all_results = []

    for idx, desc in interesting_prompts:
        prompt = ds[idx]["prompt"]
        constraints = ds[idx]["instruction_id_list"]
        short = prompt[:60].replace("\n", " ")

        print(f"{'=' * 70}")
        print(f"Prompt {idx}: {short}...")
        print(f"Constraints: {constraints}")
        print()
        sys.stdout.flush()

        prompt_results = {"idx": idx, "desc": desc, "prompt": prompt, "models": {}}

        # OpenAI models
        for oai_model in openai_models:
            try:
                r = get_openai_logprobs(client, oai_model, prompt)
                prompt_results["models"][oai_model] = r
                # Show top-5
                sorted_toks = sorted(r["top_k_logprobs"].items(), key=lambda x: -x[1])[:5]
                top_str = ", ".join(f"'{t}'({lp:.2f})" for t, lp in sorted_toks)
                print(f"  {oai_model:<20s} top='{r['top_token']}' | {top_str}")
                sys.stdout.flush()
            except Exception as e:
                print(f"  {oai_model:<20s} ERROR: {e}")
                sys.stdout.flush()

        # Qwen models (load/unload each)
        for qwen_path, qwen_name in qwen_models:
            try:
                r = get_qwen_logprobs(qwen_path, prompt)
                r["model"] = qwen_name
                prompt_results["models"][qwen_name] = r
                sorted_toks = sorted(r["top_k_logprobs"].items(), key=lambda x: -x[1])[:5]
                top_str = ", ".join(f"'{t}'({lp:.2f})" for t, lp in sorted_toks)
                print(f"  {qwen_name:<20s} top='{r['top_token']}' | {top_str}")
                sys.stdout.flush()
            except Exception as e:
                print(f"  {qwen_name:<20s} ERROR: {e}")
                sys.stdout.flush()

        # Find instruction-specific tokens: high in instruct models, absent in base
        instruct_tokens = set()
        base_tokens = set()
        for mname, mdata in prompt_results["models"].items():
            if "Base" in mname or "base" in mname:
                base_tokens.update(mdata.get("top_k_logprobs", {}).keys())
            else:
                instruct_tokens.update(mdata.get("top_k_logprobs", {}).keys())

        instruct_only = instruct_tokens - base_tokens
        if instruct_only:
            print("\n  Instruction-specific tokens (in instruct top-20, NOT in base top-20):")
            for tok in list(instruct_only)[:10]:
                sources = []
                for mname, mdata in prompt_results["models"].items():
                    if tok in mdata.get("top_k_logprobs", {}):
                        sources.append(f"{mname}({mdata['top_k_logprobs'][tok]:.2f})")
                print(f"    '{tok}': {', '.join(sources)}")

        print()
        all_results.append(prompt_results)

    # Save
    output = str(PROJECT_DIR / "results" / "logprobs_probe.json")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
