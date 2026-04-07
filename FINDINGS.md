# GRAFT — Complete Research Findings

## One-Sentence Summary

Proxy-tuning is a practical zero-cost method to make new base models follow instructions
at inference time, closing ~100% of the gap to the instruct model, but it does not improve
models that are already instruction-tuned, and self-distillation does not help on
instruction-following tasks.

---

## Experiment 1: Cross-Model Behavioral Transfer (14B)

**Setup**: Use the delta between 8B-Instruct and 8B-Base to steer 14B-Base at inference time.

**IFEval Results (541 prompts, full generation)**:

| Model | Prompt Strict | Instruction | Gap Closed |
|-------|:---:|:---:|:---:|
| 14B base | 41.0% | 52.8% | 0% |
| GRAFT k=50 | 59.7% | 70.4% | 87% |
| Proxy-tuning (full vocab) | 62.5% | 72.5% | 100%+ |
| 8B instruct | 61.7% | 71.7% | — |
| 14B instruct | 62.1% | 72.2% | 100% |

**Finding**: Both methods dramatically improve the base model (+19-21pp). Proxy-tuning
slightly edges out the actual 14B instruct model. GRAFT closes 87% of the gap.

**Per-category**: GRAFT wins on precision tasks (change_case 91% vs proxy 85%), proxy
wins on planning tasks (length_constraints 52% vs GRAFT 38%).

---

## Experiment 2: Method Comparison

**Proxy-tuning vs GRAFT**: Two differences — vocabulary (full vs top-k) and space
(raw logits vs log-probabilities). Proxy-tuning wins overall by ~3pp.

**CFG, contrastive decoding, adaptive methods**: All worse than proxy-tuning on generation.
No logit-level formula change improves on simple addition.

**Top-k boost**: Adding a constant to top-k logits barely helps (+2pp vs proxy's +21pp).
Behavioral steering requires directional signal, not uniform amplification.

**Entropy-adaptive switching**: Using GRAFT at low-entropy positions and proxy at
high-entropy: every config worse than pure proxy. Restriction damage cascades.

---

## Experiment 3: Delta Rank Analysis

**Finding**: The behavioral delta is NOT low-rank.

- 99.9% of vocabulary tokens have |delta| > 1.0
- Top-10 tokens capture 0.000% of total delta mass
- Delta entropy is 11.91 out of max 11.93 — nearly uniform

**Revised theory**: The delta is full-rank but importance-weighted. The top-k tokens
by the large model's probability are functionally important (they determine what gets
generated), even though the delta mass on them is negligible. This is why GRAFT closes
87% of the gap with 0.03% of the vocabulary.

---

## Experiment 4: Steering Already-Tuned Models (4B)

**Setup**: Apply 0.6B delta to 4B-Instruct (already instruction-tuned).

**IFEval Results (541 prompts)**:

| Model | Prompt Strict | Instruction |
|-------|:---:|:---:|
| 4B-Instruct baseline | 58.0% | 68.5% |
| 4B + 0.6B proxy delta | 58.4% | 68.6% |
| 4B + 0.6B GRAFT k=50 | 35.1% | 46.9% |

**Finding**: Proxy delta barely helps (+0.4pp) — the model already knows instruction-following.
GRAFT **destroys** the model (-23pp) — restricting to top-k removes tokens the instruct model
needs. The instruct model's distribution is already well-shaped; the restriction doesn't
filter noise (there's no noise to filter), it removes signal.

---

## Experiment 5: Self-Distillation (SSD)

**Setup**: Fine-tune 4B-Instruct on its own outputs (SSD), CFG-amplified outputs, or
proxy-tuned outputs via LoRA.

**IFEval Results (50 prompts)**:

| Condition | Prompt Strict | vs Baseline |
|-----------|:---:|:---:|
| Baseline (no SSD) | 56.0% | — |
| SSD (self-distill) | 48.0% | -8pp |
| Proxy-SSD (0.6B delta) | 46.0% | -10pp |
| CFG-SSD (amplified) | 44.0% | -12pp |

**Finding**: All SSD variants degrade the model on IFEval. Self-distillation's
precision-exploration mechanism (Apple's insight) is specific to code generation,
not general instruction-following. The delta-steered variants are worse because
they push the training distribution further from the model's own, causing LoRA
to overfit to an alien distribution.

---

## Experiment 6: Speculative Decoding

**Setup**: Use 8B-Instruct as draft model, verify with 14B + delta.

**Finding**: 74.7% acceptance rate — the draft model agrees with the delta-steered
verifier 3/4 of the time. But naive implementation is slower than greedy (5.7 vs
7.3 tok/s) due to KV cache rebuild on rejection. Needs proper cache snapshotting.

---

## Precision: bf16 vs 4-bit

All main experiments run at both precisions (PR #1). Key differences:
- bf16 delta is 17% stronger (mean |delta| 28.4 vs 24.3)
- bf16 first-token accuracy: 89.5% vs 85.8%
- Method ranking preserved: proxy-tuning > GRAFT > base at both precisions
- The ~3pp gap between proxy and GRAFT persists in bf16 (method-level, not precision)

---

## What We Learned

1. **Proxy-tuning is the best logit-level method for steering base models.** No formula
   variation (CFG, contrastive decoding, GRAFT restriction, entropy-adaptive) beats
   simple logit addition over the full vocabulary.

2. **The method's value is as a bridge.** When a new base model drops and the instruct
   variant isn't ready yet, proxy-tuning fills the gap at zero training cost. Once the
   instruct model exists, use it directly.

3. **The behavioral delta is full-rank, not low-rank.** The entire vocabulary participates
   in the behavioral shift from base to instruct. Restriction to top-k discards 99.97%
   of the delta mass but preserves 87% of the functional effect — the decision-relevant
   tokens carry disproportionate importance despite carrying negligible mass.

4. **Self-distillation does not generalize from code to instruction-following.** The
   precision-exploration conflict that makes SSD work on code does not exist in diverse
   instruction-following tasks.

5. **Steering an already-tuned model has diminishing returns.** The delta from a smaller
   pair adds almost nothing (+0.4pp) to an already instruction-tuned model.
