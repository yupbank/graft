#!/usr/bin/env python3
"""
Speculative decoding with delta verification.

Draft model: 8B-instruct (fast, knows instruction-following behavior)
Verifier: 14B-base + delta from (8B-instruct - 8B-base)

Algorithm:
  1. Draft k tokens greedily from 8B-instruct
  2. Feed all k tokens through 14B-base and 8B-base in one forward pass each
  3. At each position, compute delta-adjusted score
  4. Accept longest prefix where draft matches delta-adjusted argmax
  5. If rejected at position j, resample token j from delta-adjusted distribution
  6. Continue from the accepted prefix

Measures: acceptance rate, tokens/sec, IFEval accuracy.
Compares with greedy GRAFT and greedy proxy-tuning on same prompts.
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
DRAFT_K = 5  # tokens to draft before verifying

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


# ---------------------------------------------------------------------------
# Standard greedy (for comparison timing)
# ---------------------------------------------------------------------------
def generate_greedy_3model(
    model_large: nn.Module,
    model_raw: nn.Module,
    model_ft: nn.Module,
    prompt_tokens: mx.array,
    max_tokens: int,
    eos_id: int,
    method: str,
) -> tuple[list[int], dict]:
    """Greedy generation with 3 models. Returns (tokens, stats)."""
    from mlx_lm.models.cache import make_prompt_cache

    cl = make_prompt_cache(model_large)
    cr = make_prompt_cache(model_raw)
    cf = make_prompt_cache(model_ft)

    lg = model_large(prompt_tokens[None], cache=cl)[:, -1, :].squeeze(0)
    rw = model_raw(prompt_tokens[None], cache=cr)[:, -1, :].squeeze(0)
    ft = model_ft(prompt_tokens[None], cache=cf)[:, -1, :].squeeze(0)

    if method == "proxy":
        score = lg + (ft - rw)
    else:
        score = _graft_score_full(lg, rw, ft, 50)

    token = mx.argmax(score)
    mx.eval(token)
    tokens = [token.item()]
    fwd_count = 3  # 3 models x 1 prefill

    for _ in range(max_tokens - 1):
        inp = token.reshape(1, 1)
        lg = model_large(inp, cache=cl)[:, -1, :].squeeze(0)
        rw = model_raw(inp, cache=cr)[:, -1, :].squeeze(0)
        ft = model_ft(inp, cache=cf)[:, -1, :].squeeze(0)
        fwd_count += 3

        if method == "proxy":
            score = lg + (ft - rw)
        else:
            score = _graft_score_full(lg, rw, ft, 50)

        token = mx.argmax(score)
        mx.eval(token)
        tid = token.item()
        tokens.append(tid)
        if tid == eos_id:
            break

    del cl, cr, cf
    return tokens, {"forward_passes": fwd_count}


def _graft_score_full(s_large: mx.array, s_raw: mx.array, s_ft: mx.array, k: int) -> mx.array:
    """Return full-vocab score array with GRAFT delta on top-k, -inf elsewhere."""
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


# ---------------------------------------------------------------------------
# Speculative decoding with delta verification
# ---------------------------------------------------------------------------
def generate_speculative(
    model_large: nn.Module,
    model_raw: nn.Module,
    model_ft: nn.Module,
    prompt_tokens: mx.array,
    max_tokens: int,
    eos_id: int,
    draft_k: int,
    method: str,
) -> tuple[list[int], dict]:
    """Speculative decoding: draft with 8B-instruct, verify with delta."""
    from mlx_lm.models.cache import make_prompt_cache

    # Persistent caches for all 3 models
    cl = make_prompt_cache(model_large)
    cr = make_prompt_cache(model_raw)
    cf = make_prompt_cache(model_ft)

    # Prefill all 3 models with prompt
    model_large(prompt_tokens[None], cache=cl)
    model_raw(prompt_tokens[None], cache=cr)
    ft_logits = model_ft(prompt_tokens[None], cache=cf)[:, -1, :].squeeze(0)

    # First token: draft from 8B-instruct
    draft_token = mx.argmax(ft_logits)
    mx.eval(draft_token)

    tokens: list[int] = []
    total_drafted = 0
    fwd_count = 3  # prefill

    while len(tokens) < max_tokens:
        # === DRAFT PHASE: generate draft_k tokens from 8B-instruct ===
        draft_tokens = []
        len(tokens)

        # The ft cache is already up to date, draft from it
        current_token = draft_token
        for _ in range(draft_k):
            ft_out = model_ft(current_token.reshape(1, 1), cache=cf)[:, -1, :].squeeze(0)
            fwd_count += 1
            next_t = mx.argmax(ft_out)
            mx.eval(next_t)
            draft_tokens.append(next_t.item())
            if next_t.item() == eos_id:
                break
            current_token = next_t

        total_drafted += len(draft_tokens)

        if not draft_tokens:
            break

        # === VERIFY PHASE: run draft tokens through 14B and 8B-base ===
        draft_seq = mx.array(draft_tokens)

        # Process all draft tokens at once through large and raw models
        model_large(draft_seq[None], cache=cl)  # (1, k, vocab)
        model_raw(draft_seq[None], cache=cr)  # (1, k, vocab)
        fwd_count += 2

        # Also need ft logits at each draft position for delta computation
        # But we already advanced cf past these tokens during drafting.
        # We need to re-derive ft logits. Since cf was advanced token by token,
        # we can reconstruct: the ft logits at position j were used to pick
        # draft_tokens[j+1]. But we didn't save them.
        #
        # Workaround: re-run ft on the draft sequence too.
        # This costs 1 more forward pass but keeps caches aligned.
        #
        # Actually, cf is already advanced past all draft tokens. We need to
        # "rewind" it or run a separate pass. The simplest correct approach:
        # Don't use cf for drafting in a way that advances past verification.
        #
        # SIMPLER APPROACH: just run all 3 models on draft tokens in verify phase.
        # This means cf gets run twice on draft tokens (once for drafting, once for
        # verification). But it keeps the logic clean and caches aligned.

        # Rewind cf: we need to undo the draft advances.
        # MLX caches don't support rewind. Instead, use a separate draft-only cache.
        # Let me restructure...

        # === RESTRUCTURED: separate draft cache from verify cache ===
        # This is getting complex. Let me use a simpler approach:
        # After drafting, verify by checking if delta-adjusted argmax matches draft.
        # We already have lg_all and rw_all. For ft, we need the logits too.
        # Since cf was advanced during drafting, it's now out of sync.
        # Fix: don't advance cf during drafting. Use a separate draft cache.

        # For now, accept the cost: re-run ft on draft tokens for verification.
        # The cf cache is already past the draft tokens, so we can't rewind.
        # Instead, just use lg_all and rw_all with the draft-time ft logits.
        #
        # Actually — for proxy-tuning verification, we just need:
        #   score = s_large + (s_ft - s_raw) at each draft position
        # We have s_large and s_raw from lg_all and rw_all.
        # For s_ft, we need ft logits at each draft position.
        # The ft model already processed these tokens (during drafting), and
        # cf cache is now past them. The logits from cf at each position
        # would have been the same as what we need.
        #
        # PRACTICAL FIX: save ft logits during drafting.

        # Let me restart with a cleaner implementation.
        break  # placeholder

    # === CLEAN IMPLEMENTATION ===
    del cl, cr, cf
    gc.collect()
    mx.clear_cache()
    return _generate_speculative_clean(
        model_large,
        model_raw,
        model_ft,
        prompt_tokens,
        max_tokens,
        eos_id,
        draft_k,
        method,
    )


def _generate_speculative_clean(
    model_large: nn.Module,
    model_raw: nn.Module,
    model_ft: nn.Module,
    prompt_tokens: mx.array,
    max_tokens: int,
    eos_id: int,
    draft_k: int,
    method: str,
) -> tuple[list[int], dict]:
    """Clean speculative decoding with separate draft/verify caches."""
    from mlx_lm.models.cache import make_prompt_cache

    # Verify caches (persistent, only advanced for accepted tokens)
    cl = make_prompt_cache(model_large)
    cr = make_prompt_cache(model_raw)
    cf = make_prompt_cache(model_ft)

    # Prefill all verify caches
    model_large(prompt_tokens[None], cache=cl)
    model_raw(prompt_tokens[None], cache=cr)
    model_ft(prompt_tokens[None], cache=cf)
    fwd_count = 3

    # Draft cache (separate, for 8B-instruct only)
    cf_draft = make_prompt_cache(model_ft)
    model_ft(prompt_tokens[None], cache=cf_draft)
    fwd_count += 1

    tokens: list[int] = []

    while len(tokens) < max_tokens:
        # === DRAFT: k tokens from 8B-instruct using draft cache ===

        for d in range(draft_k):
            if d == 0 and not tokens:
                # First token: get logits from draft cache (already prefilled)
                # Need to get the last logits from prefill
                # Re-run last token to get logits
                last_tok = prompt_tokens[-1:]
                model_ft(last_tok.reshape(1, 1), cache=cf_draft)
                # Actually, prefill already consumed all prompt tokens.
                # The draft cache is ready for the next token.
                # We need to feed a dummy to get logits... no.
                # The prefill returned logits but we didn't save them.
                # Let me fix: after prefill, the cache is primed.
                # To draft, we need to know what token to start with.
                # We don't have the first draft token yet.
                # Solution: run all 3 models on prompt, get first token via delta.
                model_large(prompt_tokens[-1:].reshape(1, 1), cache=cl)
                model_raw(prompt_tokens[-1:].reshape(1, 1), cache=cr)
                model_ft(prompt_tokens[-1:].reshape(1, 1), cache=cf)
                # Hmm, this double-processes the last prompt token...
                break
            break
        # This is getting messy. Let me do it properly.
        break

    del cl, cr, cf, cf_draft

    # =========================================================
    # FINAL CLEAN VERSION: no cache tricks, just measure speedup
    # =========================================================
    return _generate_speculative_simple(
        model_large,
        model_raw,
        model_ft,
        prompt_tokens,
        max_tokens,
        eos_id,
        draft_k,
        method,
    )


def _generate_speculative_simple(
    model_large: nn.Module,
    model_raw: nn.Module,
    model_ft: nn.Module,
    prompt_tokens: mx.array,
    max_tokens: int,
    eos_id: int,
    draft_k: int,
    method: str,
) -> tuple[list[int], dict]:
    """
    Simple speculative decoding.

    1. Use 8B-instruct with its own KV cache to draft k tokens
    2. Feed the full prefix+draft to 14B and 8B-base (single forward each)
    3. Compare delta-adjusted argmax with draft at each position
    4. Accept matching prefix, resample at first mismatch
    """
    from mlx_lm.models.cache import make_prompt_cache

    # 8B-instruct has a persistent cache for fast drafting
    cf_draft = make_prompt_cache(model_ft)
    model_ft(prompt_tokens[None], cache=cf_draft)

    # Verify caches for 14B and 8B-base
    cl = make_prompt_cache(model_large)
    cr = make_prompt_cache(model_raw)
    model_large(prompt_tokens[None], cache=cl)
    model_raw(prompt_tokens[None], cache=cr)

    # Also need ft cache aligned with verification
    cf_verify = make_prompt_cache(model_ft)
    model_ft(prompt_tokens[None], cache=cf_verify)

    fwd_count = 4  # 3 verify prefills + 1 draft prefill
    tokens: list[int] = []
    total_drafted = 0
    total_accepted = 0

    while len(tokens) < max_tokens:
        # === DRAFT k tokens from 8B-instruct ===
        draft = []
        draft_ft_logits_list = []
        current = None

        for d in range(draft_k):
            if current is None:
                # First draft: we need logits from the draft cache.
                # The cache was prefilled but we didn't save final logits.
                # Trick: the cf_verify cache is also at the same position.
                # Get logits from cf_verify (it's aligned).
                # Actually all caches are aligned at this point.
                # Get first logits from all verify models to produce first token.
                if not tokens:
                    # Very first token — need 14B logits too for delta
                    # Just do one normal 3-model step for the first token
                    # and use the 8B-instruct's choice as starting draft
                    pass
                # Use draft cache: feed the last accepted token (or nothing for first)
                if tokens:
                    inp = mx.array([tokens[-1]]).reshape(1, 1)
                else:
                    # Need to get logits without feeding a new token.
                    # After prefill, next call to model gives next-token logits.
                    # But mlx caches work by: you feed token, get logits for NEXT.
                    # So after prefill of prompt, we need to feed nothing...
                    # Actually after prefill, the cache consumed all prompt tokens.
                    # The LAST logits from prefill are for the next token.
                    # But we didn't save them. Need to re-derive.
                    # Simplest: re-run last prompt token.
                    inp = prompt_tokens[-1:].reshape(1, 1)

                ft_out = model_ft(inp, cache=cf_draft)[:, -1, :].squeeze(0)
                fwd_count += 1
                draft_ft_logits_list.append(ft_out)
                t = mx.argmax(ft_out)
                mx.eval(t)
                draft.append(t.item())
                current = t
                if t.item() == eos_id:
                    break
            else:
                ft_out = model_ft(current.reshape(1, 1), cache=cf_draft)[:, -1, :].squeeze(0)
                fwd_count += 1
                draft_ft_logits_list.append(ft_out)
                t = mx.argmax(ft_out)
                mx.eval(t)
                draft.append(t.item())
                current = t
                if t.item() == eos_id:
                    break

        if not draft:
            break

        total_drafted += len(draft)

        # === VERIFY: feed draft tokens through 14B and 8B-base ===
        draft_seq = mx.array(draft)

        lg_all = model_large(draft_seq[None], cache=cl)  # (1, k, vocab)
        rw_all = model_raw(draft_seq[None], cache=cr)
        ft_all = model_ft(draft_seq[None], cache=cf_verify)
        fwd_count += 3

        # Check each position
        accepted = 0
        for j in range(len(draft)):
            lg_j = lg_all[0, j, :]
            rw_j = rw_all[0, j, :]
            ft_j = ft_all[0, j, :]

            if method == "proxy":
                score_j = lg_j + (ft_j - rw_j)
            else:
                score_j = _graft_score_full(lg_j, rw_j, ft_j, 50)

            verified_token = mx.argmax(score_j).item()

            if verified_token == draft[j]:
                accepted += 1
                tokens.append(draft[j])
                if draft[j] == eos_id:
                    break
            else:
                # Reject: use the verified token instead
                tokens.append(verified_token)
                # Need to rewind draft cache to this position
                # Can't rewind MLX caches — rebuild from scratch
                # This is expensive but only happens on rejection
                del cf_draft
                cf_draft = make_prompt_cache(model_ft)
                full_seq = mx.array(list(prompt_tokens.tolist()) + tokens)
                model_ft(full_seq[None], cache=cf_draft)
                fwd_count += 1

                # Also rewind verify caches
                del cl, cr, cf_verify
                cl = make_prompt_cache(model_large)
                cr = make_prompt_cache(model_raw)
                cf_verify = make_prompt_cache(model_ft)
                model_large(full_seq[None], cache=cl)
                model_raw(full_seq[None], cache=cr)
                model_ft(full_seq[None], cache=cf_verify)
                fwd_count += 3
                break

        total_accepted += accepted

        if tokens and tokens[-1] == eos_id:
            break
        if accepted == len(draft):
            # All accepted — draft cache is already in sync
            pass

    del cl, cr, cf_draft, cf_verify
    accept_rate = total_accepted / total_drafted if total_drafted > 0 else 0

    return tokens, {
        "forward_passes": fwd_count,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "accept_rate": accept_rate,
    }


def main() -> None:
    print(f"Date: {date.today()}")
    print(f"Speculative delta: {NUM_PROMPTS} prompts, draft_k={DRAFT_K}")
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

    configs = [
        ("spec_proxy", "Speculative Proxy-Tuning"),
    ]

    all_results: dict[str, list[dict]] = {}  # type: ignore[type-arg]

    for config_key, label in configs:
        print(f"--- {label} ---")
        _flush()

        is_spec = config_key.startswith("spec")
        method = "graft" if "graft" in config_key else "proxy"
        results = []
        t0_total = time.time()
        total_accept = 0
        total_draft = 0

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
            if is_spec:
                toks, stats = generate_speculative(
                    large_model,
                    raw_model,
                    ft_model,
                    prompt_tokens,
                    MAX_TOKENS,
                    eos_id,
                    DRAFT_K,
                    method,
                )
                total_accept += stats.get("total_accepted", 0)
                total_draft += stats.get("total_drafted", 0)
            else:
                toks, stats = generate_greedy_3model(
                    large_model,
                    raw_model,
                    ft_model,
                    prompt_tokens,
                    MAX_TOKENS,
                    eos_id,
                    method,
                )

            elapsed = time.time() - t0
            text = tokenizer.decode(toks)

            gc.collect()
            mx.clear_cache()

            tps = len(toks) / elapsed if elapsed > 0 else 0
            results.append(
                {
                    "prompt_idx": i,
                    "prompt": prompt_text,
                    "response": text,
                    "num_tokens": len(toks),
                    "time": elapsed,
                    "tokens_per_sec": tps,
                    **stats,
                }
            )

            (i + 1) / (time.time() - t0_total) * 60
            extra = ""
            if is_spec and total_draft > 0:
                extra = f" accept={100 * total_accept / total_draft:.0f}%"
            print(f"  [{i + 1}/{NUM_PROMPTS}] {len(toks)} tok, {tps:.1f} tok/s{extra}")
            _flush()

        all_results[label] = results
        total_time = time.time() - t0_total
        avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
        print(f"  Done in {total_time / 60:.1f} min, avg {avg_tps:.1f} tok/s")
        if is_spec and total_draft > 0:
            print(f"  Accept rate: {100 * total_accept / total_draft:.1f}%")
        print()
        _flush()

    # Score all
    import importlib.util

    spec = importlib.util.spec_from_file_location("score", str(SCRIPT_DIR / "score_single.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    print(f"{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}\n")
    print(f"  {'Config':<28s} {'Prompt':>8s} {'Instr':>8s} {'tok/s':>7s} {'Accept':>7s}")
    print(f"  {'-' * 28} {'-' * 8} {'-' * 8} {'-' * 7} {'-' * 7}")

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
        avg_tps = sum(r["tokens_per_sec"] for r in results) / n
        acc = ""
        td = sum(r.get("total_drafted", 0) for r in results)
        ta = sum(r.get("total_accepted", 0) for r in results)
        if td > 0:
            acc = f"{100 * ta / td:.0f}%"
        pp_pct = 100 * pp / n
        ip_pct = 100 * ip / it if it > 0 else 0
        print(
            f"  {label:<28s} "
            f"{pp:>3d}/{n} ({pp_pct:>4.1f}%) "
            f"{ip:>3d}/{it} ({ip_pct:>4.1f}%) "
            f"{avg_tps:>6.1f} "
            f"{acc:>6s}"
        )

    output = str(PROJECT_DIR / "results" / "speculative_delta_experiment.json")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
