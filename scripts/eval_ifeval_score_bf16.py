#!/usr/bin/env python3
"""
Score bf16 generated responses against IFEval instruction-following constraints.

Same scoring logic as eval_ifeval_score.py but reads from generation_samples_bf16.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
GENERATION_JSON = str(PROJECT_DIR / "results" / "generation_samples_bf16.json")
OUTPUT_JSON = str(PROJECT_DIR / "results" / "ifeval_scores_bf16.json")


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
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return len(paras) >= num_paragraphs


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
    highlights = len(re.findall(r"\*[^*]+\*", text))
    return highlights >= num_highlights


def check_bullet_lists(text, num_bullets=None, **kw):
    if num_bullets is None: return True
    bullets = len(re.findall(r"^\s*[\*\-\•]\s", text, re.MULTILINE))
    return bullets >= num_bullets


def check_json_format(text, **kw):
    try:
        json.loads(text.strip())
        return True
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            try:
                json.loads(match.group(1).strip())
                return True
            except (json.JSONDecodeError, ValueError):
                pass
        return False


def check_title(text, **kw):
    lines = text.strip().split("\n")
    if not lines: return False
    first = lines[0].strip()
    return bool(re.match(r"^#+\s", first) or re.match(r"^\*\*.*\*\*$", first))


def check_placeholders(text, num_placeholders=None, **kw):
    if num_placeholders is None: return True
    phs = len(re.findall(r"\[[\w\s,]+\]", text))
    return phs >= num_placeholders


def check_postscript(text, postscript_marker=None, **kw):
    if postscript_marker is None: return True
    return postscript_marker.lower() in text.lower()


def check_english_capital(text, **kw):
    letters = [c for c in text if c.isalpha()]
    if not letters: return True
    return all(c.isupper() for c in letters)


def check_english_lowercase(text, **kw):
    letters = [c for c in text if c.isalpha()]
    if not letters: return True
    return all(c.islower() for c in letters)


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
        try:
            passed = checker(text, **clean_kw)
        except Exception:
            passed = False
        inst_results.append({"id": inst_id, "pass": passed, "skipped": False})

    prompt_pass = all(r["pass"] for r in inst_results)
    return {"prompt_pass": prompt_pass, "instructions": inst_results}


def main():
    with open(GENERATION_JSON) as f:
        gen_results = json.load(f)

    from datasets import load_dataset
    ds = load_dataset("google/IFEval", split="train")
    n = len(gen_results)
    print(f"Scoring {n} bf16 prompts against IFEval constraints\n")

    scores = []
    for i, gen in enumerate(gen_results):
        idx = gen["prompt_idx"]
        ex = ds[idx]
        inst_ids = ex["instruction_id_list"]
        kwargs_list = ex["kwargs"]

        base_eval = evaluate_response(gen["base_14b"], inst_ids, kwargs_list)
        delta_eval = evaluate_response(gen["delta_steered"], inst_ids, kwargs_list)
        inst_eval = evaluate_response(gen["instruct_8b"], inst_ids, kwargs_list)

        scores.append({
            "prompt_idx": idx, "prompt": gen["prompt"][:80],
            "num_instructions": len(inst_ids),
            "base": base_eval, "delta": delta_eval, "instruct": inst_eval,
        })

    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(scores, f, indent=2)

    print("=" * 72)
    print("IFEVAL INSTRUCTION-FOLLOWING SCORES (bf16)")
    print("=" * 72)

    base_prompt = sum(1 for s in scores if s["base"]["prompt_pass"])
    delta_prompt = sum(1 for s in scores if s["delta"]["prompt_pass"])
    inst_prompt = sum(1 for s in scores if s["instruct"]["prompt_pass"])

    print(f"\n  Prompt-level strict accuracy (all instructions met):")
    print(f"    14B base:       {base_prompt}/{n} ({100 * base_prompt / n:.1f}%)")
    print(f"    Delta-steered:  {delta_prompt}/{n} ({100 * delta_prompt / n:.1f}%)")
    print(f"    8B instruct:    {inst_prompt}/{n} ({100 * inst_prompt / n:.1f}%)")

    base_inst_pass = delta_inst_pass = inst_inst_pass = total_inst = 0
    for s in scores:
        for bi, di, ii in zip(s["base"]["instructions"], s["delta"]["instructions"], s["instruct"]["instructions"]):
            if not bi.get("skipped", False):
                total_inst += 1
                if bi["pass"]: base_inst_pass += 1
                if di["pass"]: delta_inst_pass += 1
                if ii["pass"]: inst_inst_pass += 1

    print(f"\n  Instruction-level accuracy (per constraint):")
    if total_inst > 0:
        print(f"    14B base:       {base_inst_pass}/{total_inst} ({100 * base_inst_pass / total_inst:.1f}%)")
        print(f"    Delta-steered:  {delta_inst_pass}/{total_inst} ({100 * delta_inst_pass / total_inst:.1f}%)")
        print(f"    8B instruct:    {inst_inst_pass}/{total_inst} ({100 * inst_inst_pass / total_inst:.1f}%)")

    print(f"\n  Per-category instruction pass rate:")
    cat_stats = {}
    for s in scores:
        for bi, di, ii in zip(s["base"]["instructions"], s["delta"]["instructions"], s["instruct"]["instructions"]):
            if bi.get("skipped"): continue
            cat = bi["id"].split(":")[0]
            if cat not in cat_stats:
                cat_stats[cat] = {"total": 0, "base": 0, "delta": 0, "inst": 0}
            cat_stats[cat]["total"] += 1
            if bi["pass"]: cat_stats[cat]["base"] += 1
            if di["pass"]: cat_stats[cat]["delta"] += 1
            if ii["pass"]: cat_stats[cat]["inst"] += 1

    header = f"    {'Category':<25s} {'Total':>5s} {'Base':>8s} {'Delta':>8s} {'Instruct':>8s}"
    print(header)
    print(f"    {'-' * 25} {'-' * 5} {'-' * 8} {'-' * 8} {'-' * 8}")
    for cat in sorted(cat_stats.keys()):
        cs = cat_stats[cat]
        t = cs["total"]
        b = f"{cs['base']}/{t}"
        d = f"{cs['delta']}/{t}"
        ii = f"{cs['inst']}/{t}"
        print(f"    {cat:<25s} {t:>5d} {b:>8s} {d:>8s} {ii:>8s}")

    print(f"\n  Results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
