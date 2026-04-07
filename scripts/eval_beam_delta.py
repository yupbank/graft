#!/usr/bin/env python3
"""
Beam search with delta reranking.

Instead of greedy top-k at each step (independent), maintain beam_width
sequences ranked by cumulative log-probability. At each step:
  1. Expand each beam by top-k candidates
  2. Apply delta to score each candidate
  3. Keep top beam_width sequences by cumulative score

This addresses the "candidate drift" problem where greedy per-token
restriction loses good sequences at fork positions.

Tests: greedy GRAFT, greedy proxy-tuning, beam GRAFT, beam proxy-tuning.
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
NUM_PROMPTS = 20

CONFIGS = [
    ("greedy_graft_k50", "Greedy GRAFT k=50"),
    ("greedy_proxy", "Greedy Proxy-Tuning"),
    ("beam5_graft_k50", "Beam=5 GRAFT k=50"),
    ("beam5_proxy", "Beam=5 Proxy-Tuning"),
]

mx.random.seed(42)


def _flush() -> None:
    sys.stdout.flush()


def logsumexp(x: mx.array) -> mx.array:
    c = x.max()
    return c + mx.log(mx.exp(x - c).sum())


def log_softmax(x: mx.array) -> mx.array:
    return x - logsumexp(x)


def load_model(path: str):  # type: ignore[no-untyped-def]
    from mlx_lm import load

    return load(path)


def _score_graft(s_large: mx.array, s_raw: mx.array, s_ft: mx.array, k: int) -> mx.array:
    """Return scores over full vocab, with GRAFT delta applied to top-k."""
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
    scores = scores.at[s_t].add(log_p[s_t] + delta - (-1e9))
    return scores


def _score_proxy(s_large: mx.array, s_raw: mx.array, s_ft: mx.array) -> mx.array:
    """Proxy-tuning: full-vocab logit arithmetic."""
    return s_large + (s_ft - s_raw)


def generate_greedy(
    model_large: nn.Module,
    model_raw: nn.Module,
    model_ft: nn.Module,
    prompt_tokens: mx.array,
    max_tokens: int,
    eos_id: int,
    method: str,
) -> list[int]:
    from mlx_lm.models.cache import make_prompt_cache

    cl = make_prompt_cache(model_large)
    cr = make_prompt_cache(model_raw)
    cf = make_prompt_cache(model_ft)

    lg = model_large(prompt_tokens[None], cache=cl)[:, -1, :].squeeze(0)
    rw = model_raw(prompt_tokens[None], cache=cr)[:, -1, :].squeeze(0)
    ft = model_ft(prompt_tokens[None], cache=cf)[:, -1, :].squeeze(0)

    if method == "graft":
        scores = _score_graft(lg, rw, ft, 50)
    else:
        scores = _score_proxy(lg, rw, ft)

    token = mx.argmax(scores)
    mx.eval(token)
    tokens = [token.item()]

    for _ in range(max_tokens - 1):
        inp = token.reshape(1, 1)
        lg = model_large(inp, cache=cl)[:, -1, :].squeeze(0)
        rw = model_raw(inp, cache=cr)[:, -1, :].squeeze(0)
        ft = model_ft(inp, cache=cf)[:, -1, :].squeeze(0)

        if method == "graft":
            scores = _score_graft(lg, rw, ft, 50)
        else:
            scores = _score_proxy(lg, rw, ft)

        token = mx.argmax(scores)
        mx.eval(token)
        tid = token.item()
        tokens.append(tid)
        if tid == eos_id:
            break

    del cl, cr, cf
    return tokens


def generate_beam(
    model_large: nn.Module,
    model_raw: nn.Module,
    model_ft: nn.Module,
    prompt_tokens: mx.array,
    max_tokens: int,
    eos_id: int,
    method: str,
    beam_width: int = 5,
    expand_k: int = 10,
) -> list[int]:
    """Beam search: keep top beam_width sequences by cumulative score."""
    from mlx_lm.models.cache import make_prompt_cache

    cl = make_prompt_cache(model_large)
    cr = make_prompt_cache(model_raw)
    cf = make_prompt_cache(model_ft)

    lg = model_large(prompt_tokens[None], cache=cl)[:, -1, :].squeeze(0)
    rw = model_raw(prompt_tokens[None], cache=cr)[:, -1, :].squeeze(0)
    ft = model_ft(prompt_tokens[None], cache=cf)[:, -1, :].squeeze(0)

    if method == "graft":
        scores = _score_graft(lg, rw, ft, 50)
    else:
        scores = _score_proxy(lg, rw, ft)

    # Initialize beams: top beam_width tokens
    vocab = scores.shape[0]
    top_idx = mx.argpartition(scores, kth=vocab - beam_width)[-beam_width:]
    top_scores = scores[top_idx]
    mx.eval(top_idx, top_scores)

    beams = []
    for i in range(beam_width):
        tid = top_idx[i].item()
        sc = top_scores[i].item()
        beams.append({"tokens": [tid], "score": sc, "done": tid == eos_id})

    del cl, cr, cf

    # Beam search steps
    for step in range(max_tokens - 1):
        if all(b["done"] for b in beams):
            break

        candidates = []
        for b_idx, beam in enumerate(beams):
            if beam["done"]:
                candidates.append((beam["score"], beam["tokens"], True))
                continue

            # Run full prefix through all 3 models (no cache reuse across beams)
            full_seq = mx.array(list(prompt_tokens.tolist()) + beam["tokens"])
            cl2 = make_prompt_cache(model_large)
            cr2 = make_prompt_cache(model_raw)
            cf2 = make_prompt_cache(model_ft)

            lg2 = model_large(full_seq[None], cache=cl2)[:, -1, :].squeeze(0)
            rw2 = model_raw(full_seq[None], cache=cr2)[:, -1, :].squeeze(0)
            ft2 = model_ft(full_seq[None], cache=cf2)[:, -1, :].squeeze(0)

            del cl2, cr2, cf2

            if method == "graft":
                step_scores = _score_graft(lg2, rw2, ft2, 50)
            else:
                step_scores = _score_proxy(lg2, rw2, ft2)

            # Expand by top expand_k
            top_k_idx = mx.argpartition(step_scores, kth=vocab - expand_k)[-expand_k:]
            top_k_sc = step_scores[top_k_idx]
            mx.eval(top_k_idx, top_k_sc)

            for j in range(expand_k):
                tid = top_k_idx[j].item()
                sc = beam["score"] + top_k_sc[j].item()
                new_tokens = beam["tokens"] + [tid]
                candidates.append((sc, new_tokens, tid == eos_id))

        # Keep top beam_width candidates
        candidates.sort(key=lambda x: -x[0])
        beams = []
        for sc, toks, done in candidates[:beam_width]:
            beams.append({"tokens": toks, "score": sc, "done": done})

        gc.collect()
        mx.clear_cache()

    # Return best beam
    beams.sort(key=lambda x: -x["score"])
    return beams[0]["tokens"]


def main() -> None:
    print(f"Date: {date.today()}")
    print(f"Beam delta experiment: {NUM_PROMPTS} prompts, {len(CONFIGS)} configs")
    _flush()

    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    prompts = [ex["prompt"] for ex in ds][:NUM_PROMPTS]

    print("\nLoading models...")
    _flush()
    ft_model, tokenizer = load_model(MODEL_FT)
    raw_model, _ = load_model(MODEL_RAW)
    large_model, _ = load_model(MODEL_LARGE)
    print(f"  Loaded. {mx.get_peak_memory() / 1e9:.1f} GB peak")
    _flush()

    eos_id = tokenizer.eos_token_id or 151645
    all_results: dict[str, list[dict]] = {}  # type: ignore[type-arg]

    for config_key, label in CONFIGS:
        print(f"\n--- {label} ---")
        _flush()

        is_beam = config_key.startswith("beam")
        method = "graft" if "graft" in config_key else "proxy"
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
            if is_beam:
                tokens = generate_beam(
                    large_model,
                    raw_model,
                    ft_model,
                    prompt_tokens,
                    MAX_TOKENS,
                    eos_id,
                    method,
                    beam_width=5,
                    expand_k=10,
                )
            else:
                tokens = generate_greedy(
                    large_model,
                    raw_model,
                    ft_model,
                    prompt_tokens,
                    MAX_TOKENS,
                    eos_id,
                    method,
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

            rate = (i + 1) / (time.time() - t0_total) * 60
            print(f"  [{i + 1}/{NUM_PROMPTS}] {len(tokens)} tok, {elapsed:.1f}s ({rate:.1f}/min)")
            _flush()

        all_results[label] = results
        print(f"  Done in {(time.time() - t0_total) / 60:.1f} min")

    # Score
    import importlib.util

    spec = importlib.util.spec_from_file_location("score", str(SCRIPT_DIR / "score_single.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}\n")
    print(f"  {'Config':<30s} {'Prompt':>12s} {'Instruction':>12s}")
    print(f"  {'-' * 30} {'-' * 12} {'-' * 12}")

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
        print(f"  {label:<30s} {pp:>3d}/{n} ({pp_pct:>4.1f}%) {ip:>3d}/{it} ({ip_pct:>4.1f}%)")

    output = str(PROJECT_DIR / "results" / "beam_delta_experiment.json")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
