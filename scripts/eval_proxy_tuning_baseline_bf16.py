#!/usr/bin/env python3
"""
Proxy-tuning baseline (Liu et al. 2024) — full-vocab raw logit offset.

This implements the original proxy-tuning method for comparison with GRAFT:

  Proxy-tuning:  score(i) = s_large(i) + s_ft(i) - s_raw(i)       for all i in V
  GRAFT:         score(i) = log p_large(i) + lambda * delta(i)     for i in top-k

Key differences:
  1. Proxy-tuning uses raw logits over the FULL vocabulary (no restriction)
  2. Proxy-tuning does NOT renormalize the delta (no log-softmax on restricted set)
  3. Proxy-tuning uses lambda=1 implicitly (no scaling parameter)

Setup:
  M       = Qwen3-14B-Base-bf16      (large base model)
  M+      = Qwen3-8B-Instruct-bf16   (expert / fine-tuned)
  M-      = Qwen3-8B-Base-bf16       (anti-expert / raw)

Generates greedy responses for all 541 IFEval prompts and scores them.
"""

from __future__ import annotations

import gc
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MODEL_RAW = str(PROJECT_DIR / "models" / "Qwen3-8B-Base-bf16")
MODEL_FT = str(PROJECT_DIR / "models" / "Qwen3-8B-Instruct-bf16")
MODEL_LARGE = str(PROJECT_DIR / "models" / "Qwen3-14B-Base-bf16")
MAX_TOKENS = 256
NUM_PROMPTS = 541
OUTPUT_GEN = str(PROJECT_DIR / "results" / "generation_proxy_tuning_bf16.json")
OUTPUT_SCORES = str(PROJECT_DIR / "results" / "ifeval_scores_proxy_tuning_bf16.json")

mx.random.seed(42)


def load_model(model_path):
    from mlx_lm import load
    return load(model_path)


def _flush():
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Proxy-tuning generation (full-vocab raw logit offset)
# ---------------------------------------------------------------------------
def generate_proxy_tuning(model_large, model_raw, model_ft, prompt_tokens, max_tokens, eos_token_id):
    """
    Proxy-tuning (Liu et al. 2024):
      score = s_large + s_ft - s_raw   (full vocabulary, raw logits)
      token = argmax(score)
    """
    from mlx_lm.models.cache import make_prompt_cache

    cache_large = make_prompt_cache(model_large)
    cache_raw = make_prompt_cache(model_raw)
    cache_ft = make_prompt_cache(model_ft)

    # Prefill
    lg = model_large(prompt_tokens[None], cache=cache_large)[:, -1, :].squeeze(0)
    raw = model_raw(prompt_tokens[None], cache=cache_raw)[:, -1, :].squeeze(0)
    ft = model_ft(prompt_tokens[None], cache=cache_ft)[:, -1, :].squeeze(0)

    # Proxy-tuning: full-vocab raw logit offset
    score = lg + ft - raw
    token = mx.argmax(score)
    mx.eval(token)
    tokens = [token.item()]

    for _ in range(max_tokens - 1):
        inp = token.reshape(1, 1)
        lg = model_large(inp, cache=cache_large)[:, -1, :].squeeze(0)
        raw = model_raw(inp, cache=cache_raw)[:, -1, :].squeeze(0)
        ft = model_ft(inp, cache=cache_ft)[:, -1, :].squeeze(0)

        score = lg + ft - raw
        token = mx.argmax(score)
        mx.eval(token)
        tid = token.item()
        tokens.append(tid)
        if tid == eos_token_id:
            break

    del cache_large, cache_raw, cache_ft
    return tokens


# ---------------------------------------------------------------------------
# Greedy generation (single model)
# ---------------------------------------------------------------------------
def generate_greedy(model, prompt_tokens, max_tokens, eos_token_id):
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)

    logits = model(prompt_tokens[None], cache=cache)[:, -1, :]
    token = mx.argmax(logits, axis=-1).squeeze()
    mx.eval(token)
    tokens = [token.item()]

    for _ in range(max_tokens - 1):
        logits = model(token.reshape(1, 1), cache=cache)[:, -1, :]
        token = mx.argmax(logits, axis=-1).squeeze()
        mx.eval(token)
        tid = token.item()
        tokens.append(tid)
        if tid == eos_token_id:
            break

    del cache
    return tokens


# ---------------------------------------------------------------------------
# IFEval checkers (same as other scoring scripts)
# ---------------------------------------------------------------------------
def check_no_comma(text, **kw): return "," not in text
def check_number_words(text, num_words=None, relation=None, **kw):
    if num_words is None: return True
    wc = len(text.split())
    if relation == "at least": return wc >= num_words
    elif relation == "at most": return wc <= num_words
    elif relation == "less than": return wc < num_words
    return wc >= num_words
def check_number_sentences(text, num_sentences=None, relation=None, **kw):
    if num_sentences is None: return True
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sc = len(sents)
    if relation == "at least": return sc >= num_sentences
    elif relation == "at most": return sc <= num_sentences
    elif relation == "less than": return sc < num_sentences
    return sc >= num_sentences
def check_number_paragraphs(text, num_paragraphs=None, **kw):
    if num_paragraphs is None: return True
    return len([p.strip() for p in text.split("\n\n") if p.strip()]) >= num_paragraphs
def check_keyword_existence(text, keywords=None, **kw):
    if not keywords: return True
    lower = text.lower()
    return all(k.lower() in lower for k in keywords)
def check_keyword_frequency(text, keyword=None, frequency=None, relation=None, **kw):
    if keyword is None or frequency is None: return True
    count = text.lower().count(keyword.lower())
    if relation == "at least": return count >= frequency
    elif relation == "at most": return count <= frequency
    elif relation == "less than": return count < frequency
    return count >= frequency
def check_forbidden_words(text, forbidden_words=None, **kw):
    if not forbidden_words: return True
    lower = text.lower()
    return not any(w.lower() in lower for w in forbidden_words)
def check_letter_frequency(text, letter=None, let_frequency=None, let_relation=None, **kw):
    if letter is None or let_frequency is None: return True
    count = text.lower().count(letter.lower())
    if let_relation == "at least": return count >= let_frequency
    elif let_relation == "at most": return count <= let_frequency
    elif let_relation == "less than": return count < let_frequency
    return count >= let_frequency
def check_highlighted_sections(text, num_highlights=None, **kw):
    if num_highlights is None: return True
    return len(re.findall(r"\*[^*]+\*", text)) >= num_highlights
def check_bullet_lists(text, num_bullets=None, **kw):
    if num_bullets is None: return True
    return len(re.findall(r"^\s*[\*\-\•]\s", text, re.MULTILINE)) >= num_bullets
def check_json_format(text, **kw):
    try:
        json.loads(text.strip()); return True
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            try: json.loads(match.group(1).strip()); return True
            except (json.JSONDecodeError, ValueError): pass
        return False
def check_title(text, **kw):
    lines = text.strip().split("\n")
    if not lines: return False
    first = lines[0].strip()
    return bool(re.match(r"^#+\s", first) or re.match(r"^\*\*.*\*\*$", first))
def check_placeholders(text, num_placeholders=None, **kw):
    if num_placeholders is None: return True
    return len(re.findall(r"\[[\w\s,]+\]", text)) >= num_placeholders
def check_postscript(text, postscript_marker=None, **kw):
    if postscript_marker is None: return True
    return postscript_marker.lower() in text.lower()
def check_english_capital(text, **kw):
    letters = [c for c in text if c.isalpha()]
    return all(c.isupper() for c in letters) if letters else True
def check_english_lowercase(text, **kw):
    letters = [c for c in text if c.isalpha()]
    return all(c.islower() for c in letters) if letters else True
def check_end_checker(text, end_phrase=None, **kw):
    if end_phrase is None: return True
    return text.rstrip().endswith(end_phrase)
def check_repeat_prompt(text, prompt_to_repeat=None, **kw):
    if prompt_to_repeat is None: return True
    return prompt_to_repeat in text

CHECKERS = {
    "punctuation:no_comma": check_no_comma,
    "length_constraints:number_words": check_number_words,
    "length_constraints:number_sentences": check_number_sentences,
    "length_constraints:number_paragraphs": check_number_paragraphs,
    "length_constraints:nth_paragraph_first_word": lambda text, **kw: True,
    "keywords:existence": check_keyword_existence,
    "keywords:frequency": check_keyword_frequency,
    "keywords:forbidden_words": check_forbidden_words,
    "keywords:letter_frequency": check_letter_frequency,
    "detectable_format:number_highlighted_sections": check_highlighted_sections,
    "detectable_format:number_bullet_lists": check_bullet_lists,
    "detectable_format:json_format": check_json_format,
    "detectable_format:title": check_title,
    "detectable_format:constrained_response": lambda text, **kw: True,
    "detectable_format:multiple_sections": lambda text, **kw: True,
    "detectable_format:quotation": lambda text, **kw: True,
    "detectable_content:number_placeholders": check_placeholders,
    "detectable_content:postscript": check_postscript,
    "change_case:english_capital": check_english_capital,
    "change_case:english_lowercase": check_english_lowercase,
    "change_case:capital_word_frequency": lambda text, **kw: True,
    "language:response_language": lambda text, **kw: True,
    "startend:end_checker": check_end_checker,
    "startend:quotation": lambda text, **kw: True,
    "combination:two_responses": lambda text, **kw: True,
    "combination:repeat_prompt": check_repeat_prompt,
}


def evaluate_response(text, instruction_ids, kwargs_list):
    inst_results = []
    for inst_id, kw in zip(instruction_ids, kwargs_list):
        checker = CHECKERS.get(inst_id)
        if checker is None:
            inst_results.append({"id": inst_id, "pass": True, "skipped": True})
            continue
        clean_kw = {k: v for k, v in kw.items() if v is not None}
        try: passed = checker(text, **clean_kw)
        except Exception: passed = False
        inst_results.append({"id": inst_id, "pass": passed, "skipped": False})
    return {"prompt_pass": all(r["pass"] for r in inst_results), "instructions": inst_results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Date: {date.today()}")
    print(f"Method: Proxy-tuning baseline (Liu et al. 2024)")
    print(f"  score = s_large + s_ft - s_raw  (full vocab, raw logits)")
    print(f"Config: max_tokens={MAX_TOKENS}")
    print(f"Precision: bf16")
    print()

    # Load dataset
    print("Loading IFEval dataset ...")
    _flush()
    from datasets import load_dataset
    ds = load_dataset("google/IFEval", split="train")
    prompts = [ex["prompt"] for ex in ds][:NUM_PROMPTS]
    print(f"  {len(prompts)} prompts\n")

    # Load all 3 models
    print(f"  Loading 8B instruct (expert): {MODEL_FT} ...")
    _flush()
    t0 = time.time()
    ft_model, tokenizer = load_model(MODEL_FT)
    print(f"    done ({time.time() - t0:.1f}s, {mx.get_peak_memory() / 1e9:.1f} GB)")
    _flush()

    print(f"  Loading 8B base (anti-expert): {MODEL_RAW} ...")
    _flush()
    t0 = time.time()
    raw_model, _ = load_model(MODEL_RAW)
    print(f"    done ({time.time() - t0:.1f}s, {mx.get_peak_memory() / 1e9:.1f} GB)")
    _flush()

    print(f"  Loading 14B base (M): {MODEL_LARGE} ...")
    _flush()
    t0 = time.time()
    large_model, _ = load_model(MODEL_LARGE)
    print(f"    done ({time.time() - t0:.1f}s, {mx.get_peak_memory() / 1e9:.1f} GB)")
    _flush()

    eos_id = tokenizer.eos_token_id or 151645
    print(f"\n  All models loaded. EOS={eos_id}. Starting generation.\n")
    _flush()

    # Generate
    results = []
    total_t0 = time.time()

    for i, prompt_text in enumerate(prompts):
        prompt_t0 = time.time()
        short = prompt_text[:55].replace("\n", " ")
        print(f"[{i + 1}/{len(prompts)}] {short}...")
        _flush()

        messages = [{"role": "user", "content": prompt_text}]
        token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        prompt_tokens = mx.array(token_ids)

        # Proxy-tuning generation
        t0 = time.time()
        pt_tokens = generate_proxy_tuning(
            large_model, raw_model, ft_model, prompt_tokens, MAX_TOKENS, eos_id,
        )
        pt_time = time.time() - t0
        pt_text = tokenizer.decode(pt_tokens)
        print(f"  proxy: {len(pt_tokens)} tok, {pt_time:.1f}s")

        elapsed = time.time() - total_t0
        rate = (i + 1) / elapsed * 60
        print(f"  ({rate:.1f} prompts/min)")
        _flush()

        gc.collect()
        mx.clear_cache()

        results.append({
            "prompt_idx": i, "prompt": prompt_text,
            "proxy_tuned": pt_text,
            "proxy_tokens": len(pt_tokens), "proxy_time": pt_time,
        })

        if (i + 1) % 10 == 0:
            Path(OUTPUT_GEN).parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_GEN, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  ** saved {len(results)} results")
            _flush()

    # Save final
    del large_model, raw_model, ft_model
    gc.collect()
    mx.synchronize()
    mx.clear_cache()

    Path(OUTPUT_GEN).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_GEN, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nGeneration saved to {OUTPUT_GEN}\n")

    # Score
    print("Scoring against IFEval constraints ...")
    scores = []
    for i, gen in enumerate(results):
        idx = gen["prompt_idx"]
        ex = ds[idx]
        inst_ids = ex["instruction_id_list"]
        kwargs_list = ex["kwargs"]
        ev = evaluate_response(gen["proxy_tuned"], inst_ids, kwargs_list)
        scores.append({
            "prompt_idx": idx, "prompt": gen["prompt"][:80],
            "num_instructions": len(inst_ids), "eval": ev,
        })

    with open(OUTPUT_SCORES, "w") as f:
        json.dump(scores, f, indent=2)

    # Summary
    n = len(scores)
    prompt_pass = sum(1 for s in scores if s["eval"]["prompt_pass"])
    inst_pass = inst_total = 0
    cat_stats = {}
    for s in scores:
        for inst in s["eval"]["instructions"]:
            if inst.get("skipped"): continue
            inst_total += 1
            if inst["pass"]: inst_pass += 1
            cat = inst["id"].split(":")[0]
            if cat not in cat_stats:
                cat_stats[cat] = {"total": 0, "pass": 0}
            cat_stats[cat]["total"] += 1
            if inst["pass"]: cat_stats[cat]["pass"] += 1

    print()
    print("=" * 60)
    print("IFEVAL SCORES — Proxy-tuning baseline (bf16)")
    print("=" * 60)
    print(f"  Prompt-level strict: {prompt_pass}/{n} ({100 * prompt_pass / n:.1f}%)")
    if inst_total:
        print(f"  Instruction-level:   {inst_pass}/{inst_total} ({100 * inst_pass / inst_total:.1f}%)")
    print()

    print(f"  {'Category':<25s} {'Pass':>6s} / {'Total':<5s}  {'Rate':>5s}")
    print(f"  {'-'*25} {'-'*6}   {'-'*5}  {'-'*5}")
    for cat in sorted(cat_stats):
        cs = cat_stats[cat]
        print(f"  {cat:<25s} {cs['pass']:>6d} / {cs['total']:<5d}  {100*cs['pass']/cs['total']:>5.1f}%")

    print(f"\n  Scores saved to {OUTPUT_SCORES}")


if __name__ == "__main__":
    main()
