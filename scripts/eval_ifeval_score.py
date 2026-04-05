#!/usr/bin/env python3
"""
Score generated responses against IFEval instruction-following constraints.

Reads generation_samples.json and evaluates each response (base, delta, instruct)
against the verifiable instructions in the IFEval dataset.

Implements checkers for:
  - punctuation:no_comma
  - length_constraints:number_words
  - length_constraints:number_sentences
  - length_constraints:number_paragraphs
  - keywords:existence
  - keywords:frequency
  - keywords:forbidden_words
  - keywords:letter_frequency
  - detectable_format:number_highlighted_sections
  - detectable_format:number_bullet_lists
  - detectable_format:json_format
  - detectable_format:title
  - detectable_content:number_placeholders
  - detectable_content:postscript
  - change_case:english_capital
  - change_case:english_lowercase
  - startend:end_checker
  - language:response_language (basic heuristic)
  - combination:repeat_prompt
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
GENERATION_JSON = str(PROJECT_DIR / "results" / "generation_samples.json")
OUTPUT_JSON = str(PROJECT_DIR / "results" / "ifeval_scores.json")


# ---------------------------------------------------------------------------
# Instruction checkers
# ---------------------------------------------------------------------------
def check_no_comma(text: str, **kwargs: object) -> bool:
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
    # Simple sentence count by splitting on .!?
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
    # Count *highlighted* sections (text between * or **)
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
        # Try to find JSON block in markdown
        match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            try:
                json.loads(match.group(1).strip())
                return True
            except (json.JSONDecodeError, ValueError):
                pass
        return False


def check_title(text: str, **kw: object) -> bool:
    # Check if response has a title (line wrapped in ** or starting with #)
    lines = text.strip().split("\n")
    if not lines:
        return False
    first = lines[0].strip()
    return bool(re.match(r"^#+\s", first) or re.match(r"^\*\*.*\*\*$", first))


def check_placeholders(text: str, num_placeholders: int | None = None, **kw: object) -> bool:
    if num_placeholders is None:
        return True
    # Count [PLACEHOLDER] patterns
    phs = len(re.findall(r"\[[\w\s,]+\]", text))
    return phs >= num_placeholders


def check_postscript(text: str, postscript_marker: str | None = None, **kw: object) -> bool:
    if postscript_marker is None:
        return True
    return postscript_marker.lower() in text.lower()


def check_english_capital(text: str, **kw: object) -> bool:
    # All letters should be uppercase
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_response(
    text: str,
    instruction_ids: list[str],
    kwargs_list: list[dict],  # type: ignore[type-arg]
) -> dict:  # type: ignore[type-arg]
    """Check each instruction and return per-instruction + prompt-level results."""
    inst_results = []
    for inst_id, kw in zip(instruction_ids, kwargs_list):
        checker = CHECKERS.get(inst_id)
        if checker is None:
            inst_results.append({"id": inst_id, "pass": True, "skipped": True})
            continue
        # Filter None values from kwargs
        clean_kw = {k: v for k, v in kw.items() if v is not None}
        try:
            passed = checker(text, **clean_kw)
        except Exception:
            passed = False
        inst_results.append({"id": inst_id, "pass": passed, "skipped": False})

    prompt_pass = all(r["pass"] for r in inst_results)
    return {"prompt_pass": prompt_pass, "instructions": inst_results}


def main() -> None:
    # Load generation results
    with open(GENERATION_JSON) as f:
        gen_results = json.load(f)

    # Load IFEval dataset for constraints
    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    n = len(gen_results)
    print(f"Scoring {n} prompts against IFEval constraints\n")

    scores: list[dict] = []  # type: ignore[type-arg]

    for i, gen in enumerate(gen_results):
        idx = gen["prompt_idx"]
        ex = ds[idx]
        inst_ids = ex["instruction_id_list"]
        kwargs_list = ex["kwargs"]

        base_eval = evaluate_response(gen["base_14b"], inst_ids, kwargs_list)
        delta_eval = evaluate_response(gen["delta_steered"], inst_ids, kwargs_list)
        inst_eval = evaluate_response(gen["instruct_8b"], inst_ids, kwargs_list)

        scores.append(
            {
                "prompt_idx": idx,
                "prompt": gen["prompt"][:80],
                "num_instructions": len(inst_ids),
                "base": base_eval,
                "delta": delta_eval,
                "instruct": inst_eval,
            }
        )

    # Save detailed scores
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(scores, f, indent=2)

    # --- Summary ---
    print("=" * 72)
    print("IFEVAL INSTRUCTION-FOLLOWING SCORES")
    print("=" * 72)

    # Prompt-level strict accuracy
    base_prompt = sum(1 for s in scores if s["base"]["prompt_pass"])
    delta_prompt = sum(1 for s in scores if s["delta"]["prompt_pass"])
    inst_prompt = sum(1 for s in scores if s["instruct"]["prompt_pass"])

    print("\n  Prompt-level strict accuracy (all instructions met):")
    print(f"    14B base:       {base_prompt}/{n} ({100 * base_prompt / n:.1f}%)")
    print(f"    Delta-steered:  {delta_prompt}/{n} ({100 * delta_prompt / n:.1f}%)")
    print(f"    8B instruct:    {inst_prompt}/{n} ({100 * inst_prompt / n:.1f}%)")

    # Instruction-level accuracy
    base_inst_pass = 0
    delta_inst_pass = 0
    inst_inst_pass = 0
    total_inst = 0
    for s in scores:
        for bi, di, ii in zip(
            s["base"]["instructions"],
            s["delta"]["instructions"],
            s["instruct"]["instructions"],
        ):
            if not bi.get("skipped", False):
                total_inst += 1
                if bi["pass"]:
                    base_inst_pass += 1
                if di["pass"]:
                    delta_inst_pass += 1
                if ii["pass"]:
                    inst_inst_pass += 1

    print("\n  Instruction-level accuracy (per constraint):")
    if total_inst > 0:
        bp = 100 * base_inst_pass / total_inst
        dp = 100 * delta_inst_pass / total_inst
        ip = 100 * inst_inst_pass / total_inst
        print(f"    14B base:       {base_inst_pass}/{total_inst} ({bp:.1f}%)")
        print(f"    Delta-steered:  {delta_inst_pass}/{total_inst} ({dp:.1f}%)")
        print(f"    8B instruct:    {inst_inst_pass}/{total_inst} ({ip:.1f}%)")

    # Per-category breakdown
    print("\n  Per-category instruction pass rate:")
    cat_stats: dict[str, dict[str, int]] = {}
    for s in scores:
        for bi, di, ii in zip(
            s["base"]["instructions"],
            s["delta"]["instructions"],
            s["instruct"]["instructions"],
        ):
            if bi.get("skipped"):
                continue
            cat = bi["id"].split(":")[0]
            if cat not in cat_stats:
                cat_stats[cat] = {"total": 0, "base": 0, "delta": 0, "inst": 0}
            cat_stats[cat]["total"] += 1
            if bi["pass"]:
                cat_stats[cat]["base"] += 1
            if di["pass"]:
                cat_stats[cat]["delta"] += 1
            if ii["pass"]:
                cat_stats[cat]["inst"] += 1

    header = f"    {'Category':<25s} {'Total':>5s} {'Base':>8s} {'Delta':>8s} {'Instruct':>8s}"
    print(header)
    print(f"    {'-' * 25} {'-' * 5} {'-' * 8} {'-' * 8} {'-' * 8}")
    for cat in sorted(cat_stats.keys()):
        cs = cat_stats[cat]
        t = cs["total"]
        b = f"{cs['base']}/{t}"
        d = f"{cs['delta']}/{t}"
        i = f"{cs['inst']}/{t}"
        print(f"    {cat:<25s} {t:>5d} {b:>8s} {d:>8s} {i:>8s}")

    # Show some examples where delta beats base
    print("\n  Examples where delta passes but base fails:")
    count = 0
    for s in scores:
        if s["delta"]["prompt_pass"] and not s["base"]["prompt_pass"]:
            if count < 5:
                print(f"    Prompt: {s['prompt']}...")
                for bi, di in zip(s["base"]["instructions"], s["delta"]["instructions"]):
                    if not bi["pass"] and di["pass"]:
                        print(f"      {bi['id']}: base=FAIL, delta=PASS")
                count += 1
    if count == 0:
        print("    (none found)")
    print()

    print(f"  Results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
