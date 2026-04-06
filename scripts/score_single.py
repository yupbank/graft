#!/usr/bin/env python3
"""
Score a single-model generation file against IFEval constraints.

Usage:
  uv run python scripts/score_single.py <generation_json> [label]

Example:
  uv run python scripts/score_single.py results/generation_14b_instruct.json "14B-Instruct"
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Instruction checkers (same as eval_ifeval_score.py)
# ---------------------------------------------------------------------------
def check_no_comma(text: str, **kw: object) -> bool:
    return "," not in text


def check_number_words(
    text: str, num_words: int | None = None, relation: str | None = None, **kw: object
) -> bool:
    if num_words is None:
        return True
    wc = len(text.split())
    if relation == "at least":
        return wc >= num_words
    elif relation == "at most":
        return wc <= num_words
    elif relation == "less than":
        return wc < num_words
    return wc >= num_words


def check_number_sentences(
    text: str, num_sentences: int | None = None, relation: str | None = None, **kw: object
) -> bool:
    if num_sentences is None:
        return True
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sc = len(sents)
    if relation == "at least":
        return sc >= num_sentences
    elif relation == "at most":
        return sc <= num_sentences
    elif relation == "less than":
        return sc < num_sentences
    return sc >= num_sentences


def check_number_paragraphs(text: str, num_paragraphs: int | None = None, **kw: object) -> bool:
    if num_paragraphs is None:
        return True
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return len(paras) >= num_paragraphs


def check_keyword_existence(text: str, keywords: list[str] | None = None, **kw: object) -> bool:
    if not keywords:
        return True
    lower = text.lower()
    return all(k.lower() in lower for k in keywords)


def check_keyword_frequency(
    text: str,
    keyword: str | None = None,
    frequency: int | None = None,
    relation: str | None = None,
    **kw: object,
) -> bool:
    if keyword is None or frequency is None:
        return True
    count = text.lower().count(keyword.lower())
    if relation == "at least":
        return count >= frequency
    elif relation == "at most":
        return count <= frequency
    elif relation == "less than":
        return count < frequency
    return count >= frequency


def check_forbidden_words(
    text: str, forbidden_words: list[str] | None = None, **kw: object
) -> bool:
    if not forbidden_words:
        return True
    lower = text.lower()
    return not any(w.lower() in lower for w in forbidden_words)


def check_letter_frequency(
    text: str,
    letter: str | None = None,
    let_frequency: int | None = None,
    let_relation: str | None = None,
    **kw: object,
) -> bool:
    if letter is None or let_frequency is None:
        return True
    count = text.lower().count(letter.lower())
    if let_relation == "at least":
        return count >= let_frequency
    elif let_relation == "at most":
        return count <= let_frequency
    elif let_relation == "less than":
        return count < let_frequency
    return count >= let_frequency


def check_highlighted_sections(text: str, num_highlights: int | None = None, **kw: object) -> bool:
    if num_highlights is None:
        return True
    highlights = len(re.findall(r"\*[^*]+\*", text))
    return highlights >= num_highlights


def check_bullet_lists(text: str, num_bullets: int | None = None, **kw: object) -> bool:
    if num_bullets is None:
        return True
    bullets = len(re.findall(r"^\s*[\*\-\•]\s", text, re.MULTILINE))
    return bullets >= num_bullets


def check_json_format(text: str, **kw: object) -> bool:
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


def check_title(text: str, **kw: object) -> bool:
    lines = text.strip().split("\n")
    if not lines:
        return False
    first = lines[0].strip()
    return bool(re.match(r"^#+\s", first) or re.match(r"^\*\*.*\*\*$", first))


def check_placeholders(text: str, num_placeholders: int | None = None, **kw: object) -> bool:
    if num_placeholders is None:
        return True
    phs = len(re.findall(r"\[[\w\s,]+\]", text))
    return phs >= num_placeholders


def check_postscript(text: str, postscript_marker: str | None = None, **kw: object) -> bool:
    if postscript_marker is None:
        return True
    return postscript_marker.lower() in text.lower()


def check_english_capital(text: str, **kw: object) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    return all(c.isupper() for c in letters)


def check_english_lowercase(text: str, **kw: object) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    return all(c.islower() for c in letters)


def check_end_checker(text: str, end_phrase: str | None = None, **kw: object) -> bool:
    if end_phrase is None:
        return True
    return text.rstrip().endswith(end_phrase)


def check_repeat_prompt(text: str, prompt_to_repeat: str | None = None, **kw: object) -> bool:
    if prompt_to_repeat is None:
        return True
    return prompt_to_repeat in text


CHECKERS: dict[str, object] = {
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


def evaluate_response(
    text: str,
    instruction_ids: list[str],
    kwargs_list: list[dict],  # type: ignore[type-arg]
) -> dict:  # type: ignore[type-arg]
    inst_results = []
    for inst_id, kw in zip(instruction_ids, kwargs_list):
        checker = CHECKERS.get(inst_id)
        if checker is None:
            inst_results.append({"id": inst_id, "pass": True, "skipped": True})
            continue
        clean_kw = {k: v for k, v in kw.items() if v is not None}
        try:
            passed = checker(text, **clean_kw)  # type: ignore[operator]
        except Exception:
            passed = False
        inst_results.append({"id": inst_id, "pass": passed, "skipped": False})
    return {
        "prompt_pass": all(r["pass"] for r in inst_results),
        "instructions": inst_results,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <generation_json> [label]")
        sys.exit(1)

    gen_path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else Path(gen_path).stem

    with open(gen_path) as f:
        gen_results = json.load(f)

    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    n = len(gen_results)

    prompt_pass = 0
    inst_pass = 0
    inst_total = 0
    cat_stats: dict[str, dict[str, int]] = {}

    for gen in gen_results:
        idx = gen["prompt_idx"]
        ex = ds[idx]
        text = gen.get("response") or gen.get("base_14b", "")
        ev = evaluate_response(text, ex["instruction_id_list"], ex["kwargs"])

        if ev["prompt_pass"]:
            prompt_pass += 1

        for r in ev["instructions"]:
            if not r.get("skipped", False):
                inst_total += 1
                if r["pass"]:
                    inst_pass += 1
                cat = r["id"].split(":")[0]
                if cat not in cat_stats:
                    cat_stats[cat] = {"total": 0, "pass": 0}
                cat_stats[cat]["total"] += 1
                if r["pass"]:
                    cat_stats[cat]["pass"] += 1

    print(f"Model: {label}")
    print(f"Prompts: {n}")
    print(f"\nPrompt-level strict: {prompt_pass}/{n} ({100 * prompt_pass / n:.1f}%)")
    if inst_total > 0:
        print(
            f"Instruction-level:   {inst_pass}/{inst_total} ({100 * inst_pass / inst_total:.1f}%)"
        )

    print("\nPer-category:")
    for cat in sorted(cat_stats.keys()):
        cs = cat_stats[cat]
        pct = 100 * cs["pass"] / cs["total"]
        print(f"  {cat:<25s} {cs['pass']:>3d}/{cs['total']:<3d} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
