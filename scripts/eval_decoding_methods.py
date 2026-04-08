#!/usr/bin/env python3
"""
Compare decoding-time steering methods on 50 IFEval prompts.

All methods use the same 3 models, same prefixes. Only the scoring formula differs.

Methods:
  1. Proxy-Tuning (baseline):     score = s_large + (s_ft - s_raw)
  2. GRAFT k=50 (baseline):       score = log_p_large[St] + delta_restricted
  3. CFG w=0.5:                    score = (1+w)*log_p_steered - w*log_p_base
  4. CFG w=1.0:                    (stronger guidance)
  5. CFG w=2.0:                    (aggressive guidance)
  6. Contrastive Decoding a=0.5:   score = log_p_steered - a*log_p_base
  7. Contrastive Decoding a=1.0:   (standard CD)
  8. Contrastive Decoding a=1.5:   (stronger contrast)
  9. Adaptive CD:                  a_t = base_alpha * KL(p_steered || p_base) / mean_KL

Where p_steered = softmax(s_large + (s_ft - s_raw)) i.e. proxy-tuned distribution.
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


def _score_proxy(lg: mx.array, rw: mx.array, ft: mx.array) -> mx.array:
    """Proxy-tuning: s_large + (s_ft - s_raw)"""
    return lg + (ft - rw)


def _score_graft(lg: mx.array, rw: mx.array, ft: mx.array, k: int) -> mx.array:
    """GRAFT: restricted log-prob delta on top-k"""
    vocab = lg.shape[0]
    ek = min(k, vocab)
    s_t = mx.argpartition(lg, kth=vocab - ek)[-ek:]
    s_ft_k = ft[s_t]
    s_raw_k = rw[s_t]
    log_pf = s_ft_k - logsumexp(s_ft_k)
    log_pr = s_raw_k - logsumexp(s_raw_k)
    delta = log_pf - log_pr
    log_p = log_softmax(lg)
    restricted_scores = log_p[s_t] + delta
    # Build full-vocab scores: copy log_p, override top-k with delta-adjusted scores
    scores = mx.full((vocab,), -1e30)
    for j in range(ek):
        scores[s_t[j]] = restricted_scores[j]
    return scores


def _score_cfg(lg: mx.array, rw: mx.array, ft: mx.array, w: float) -> mx.array:
    """Classifier-Free Guidance: (1+w)*log_p_steered - w*log_p_base"""
    log_p_steered = log_softmax(lg + (ft - rw))
    log_p_base = log_softmax(lg)
    return (1.0 + w) * log_p_steered - w * log_p_base


def _score_cd(lg: mx.array, rw: mx.array, ft: mx.array, alpha: float) -> mx.array:
    """Contrastive Decoding: log_p_steered - alpha*log_p_base"""
    log_p_steered = log_softmax(lg + (ft - rw))
    log_p_base = log_softmax(lg)
    return log_p_steered - alpha * log_p_base


def _score_adaptive_cd(
    lg: mx.array,
    rw: mx.array,
    ft: mx.array,
    base_alpha: float,
    running_kl: list[float],
) -> mx.array:
    """Adaptive CD: alpha scales with KL divergence at this step."""
    log_p_steered = log_softmax(lg + (ft - rw))
    log_p_base = log_softmax(lg)

    # Compute KL(steered || base)
    p_steered = mx.exp(log_p_steered)
    kl = mx.sum(p_steered * (log_p_steered - log_p_base))
    mx.eval(kl)
    kl_val = max(kl.item(), 0.01)
    running_kl.append(kl_val)

    # Adaptive alpha: scale inversely with KL
    # High KL = models disagree a lot = reduce alpha (trust steered more)
    # Low KL = models agree = increase alpha (contrast more)
    mean_kl = sum(running_kl) / len(running_kl)
    alpha = base_alpha * (mean_kl / kl_val)
    alpha = max(0.1, min(alpha, 3.0))  # clamp

    return log_p_steered - alpha * log_p_base


def generate_with_scorer(
    model_large: nn.Module,
    model_raw: nn.Module,
    model_ft: nn.Module,
    prompt_tokens: mx.array,
    max_tokens: int,
    eos_id: int,
    scorer_name: str,
    scorer_kwargs: dict,  # type: ignore[type-arg]
) -> tuple[list[int], dict]:  # type: ignore[type-arg]
    from mlx_lm.models.cache import make_prompt_cache

    cl = make_prompt_cache(model_large)
    cr = make_prompt_cache(model_raw)
    cf = make_prompt_cache(model_ft)

    lg = model_large(prompt_tokens[None], cache=cl)[:, -1, :].squeeze(0)
    rw = model_raw(prompt_tokens[None], cache=cr)[:, -1, :].squeeze(0)
    ft = model_ft(prompt_tokens[None], cache=cf)[:, -1, :].squeeze(0)

    running_kl: list[float] = []
    score = _apply_scorer(lg, rw, ft, scorer_name, scorer_kwargs, running_kl)
    token = mx.argmax(score)
    mx.eval(token)
    tokens = [token.item()]

    for _ in range(max_tokens - 1):
        inp = token.reshape(1, 1)
        lg = model_large(inp, cache=cl)[:, -1, :].squeeze(0)
        rw = model_raw(inp, cache=cr)[:, -1, :].squeeze(0)
        ft = model_ft(inp, cache=cf)[:, -1, :].squeeze(0)

        score = _apply_scorer(lg, rw, ft, scorer_name, scorer_kwargs, running_kl)
        token = mx.argmax(score)
        mx.eval(token)
        tid = token.item()
        tokens.append(tid)
        if tid == eos_id:
            break

    del cl, cr, cf
    return tokens, {"running_kl_mean": sum(running_kl) / len(running_kl) if running_kl else 0}


def _apply_scorer(
    lg: mx.array,
    rw: mx.array,
    ft: mx.array,
    name: str,
    kwargs: dict,
    running_kl: list[float],  # type: ignore[type-arg]
) -> mx.array:
    if name == "proxy":
        return _score_proxy(lg, rw, ft)
    elif name == "graft":
        return _score_graft(lg, rw, ft, kwargs.get("k", 50))
    elif name == "cfg":
        return _score_cfg(lg, rw, ft, kwargs["w"])
    elif name == "cd":
        return _score_cd(lg, rw, ft, kwargs["alpha"])
    elif name == "adaptive_cd":
        return _score_adaptive_cd(lg, rw, ft, kwargs["base_alpha"], running_kl)
    else:
        raise ValueError(f"Unknown scorer: {name}")


CONFIGS = [
    ("cfg", {"w": 0.5}, "CFG w=0.5"),
    ("cfg", {"w": 1.0}, "CFG w=1.0"),
    ("cfg", {"w": 2.0}, "CFG w=2.0"),
    ("cd", {"alpha": 0.5}, "CD alpha=0.5"),
    ("cd", {"alpha": 1.0}, "CD alpha=1.0"),
    ("cd", {"alpha": 1.5}, "CD alpha=1.5"),
    ("adaptive_cd", {"base_alpha": 1.0}, "Adaptive CD"),
]


def main() -> None:
    print(f"Date: {date.today()}")
    print(f"Decoding methods comparison: {NUM_PROMPTS} prompts, {len(CONFIGS)} methods")
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

    for scorer_name, scorer_kwargs, label in CONFIGS:
        print(f"--- {label} ---")
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
            toks, stats = generate_with_scorer(
                large_model,
                raw_model,
                ft_model,
                prompt_tokens,
                MAX_TOKENS,
                eos_id,
                scorer_name,
                scorer_kwargs,
            )
            elapsed = time.time() - t0
            text = tokenizer.decode(toks)

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
                print(f"  [{i + 1}/{NUM_PROMPTS}] ({rate:.1f}/min)")
                _flush()

        all_results[label] = results
        total = time.time() - t0_total
        print(f"  Done in {total / 60:.1f} min")
        _flush()

        # Save + score after EACH config
        output = str(PROJECT_DIR / "results" / "decoding_methods_experiment.json")
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        import importlib.util

        spec_mod = importlib.util.spec_from_file_location(
            "score", str(SCRIPT_DIR / "score_single.py")
        )
        mod = importlib.util.module_from_spec(spec_mod)
        spec_mod.loader.exec_module(mod)  # type: ignore[union-attr]

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
        print(f"  >> {label}: {pp}/{n} ({pp_pct:.1f}%) prompt, {ip}/{it} ({ip_pct:.1f}%) instr")
        _flush()
        print()

    # Final summary
    print(f"{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}\n")
    print(f"  {'Method':<22s} {'Prompt':>12s} {'Instruction':>12s}")
    print(f"  {'-' * 22} {'-' * 12} {'-' * 12}")

    import importlib.util

    spec_mod2 = importlib.util.spec_from_file_location("score", str(SCRIPT_DIR / "score_single.py"))
    mod = importlib.util.module_from_spec(spec_mod2)
    spec_mod2.loader.exec_module(mod)  # type: ignore[union-attr]

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
        print(f"  {label:<22s} {pp:>3d}/{n} ({pp_pct:>4.1f}%) {ip:>3d}/{it} ({ip_pct:>4.1f}%)")

    output = str(PROJECT_DIR / "results" / "decoding_methods_experiment.json")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
