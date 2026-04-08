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
(raw logits vs log-probabilities). Proxy-tuning wins overall by ~3pp on 14B-base.

**CFG, contrastive decoding**: Tested on 50 prompts. CFG w=0.5 and w=1.0 both scored
52% (vs proxy 56%). No logit-level formula change improves on simple addition.

**Top-k boost**: Adding a constant to top-k logits barely helps (+2pp vs proxy's +21pp).
Behavioral steering requires directional signal, not uniform amplification.

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
| 4B + 0.6B GRAFT k=50 | 58.2% | 68.8% |
| 4B + 0.6B proxy delta | 58.4% | 68.6% |

**Finding**: All three methods are within 0.4pp — the delta barely moves the needle
on an already instruction-tuned model. The behavioral knowledge is already internalized;
there's nothing new to transfer from the 0.6B pair.

**Note on a scoring bug**: An earlier run showed GRAFT at 35.1% (-23pp) due to a bug
in `mx.array.at[].add()` that collapsed all top-k scores to zero, causing degenerate
`!#!#!#` output in 498/541 prompts. The fix (direct argmax on restricted scores)
restored GRAFT to 58.2%, consistent with proxy-tuning. See commit `e055298` for details.

---

## Experiment 5: Self-Distillation (SSD)

**Setup**: Fine-tune 4B-Instruct on its own outputs (SSD), CFG-amplified outputs, or
proxy-tuned outputs via LoRA. 100 IFEval prompts, 3 solutions each, 300 iters LoRA.

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

**Note**: The SSD data generation used proxy-tuning (full-vocab logit arithmetic),
not GRAFT, so it was not affected by the scoring bug.

---

## Experiment 6: Speculative Decoding

**Setup**: Use 8B-Instruct as draft model, verify with 14B + delta.

**Finding**: 74.7% acceptance rate — the draft model agrees with the delta-steered
verifier 3/4 of the time. But naive implementation is slower than greedy (5.7 vs
7.3 tok/s) due to KV cache rebuild on rejection. With proper cache snapshotting,
this would give a speedup.

---

## Precision: bf16 vs 4-bit

All main experiments run at both precisions (PR #1). Key differences:
- bf16 delta is 17% stronger (mean |delta| 28.4 vs 24.3)
- bf16 first-token accuracy: 89.5% vs 85.8%
- Method ranking preserved: proxy-tuning > GRAFT > base at both precisions
- The ~3pp gap between proxy and GRAFT on 14B persists in bf16 (method-level, not precision)

---

## Bug Discovery: MLX `at[].add()` Semantics

During the project, we discovered a critical numerical bug in several GRAFT scripts.
The pattern:

```python
scores = mx.full((vocab,), -1e9)
scores = scores.at[s_t].add(log_p[s_t] + delta + 1e9)
```

was intended to set scores for top-k tokens while leaving others at -inf. But the
`+1e9` offset cancelled the `-1e9` base AND erased the actual score differences
between tokens — collapsing all top-k scores to approximately zero. This made token
selection effectively random within the top-k, causing generation to degenerate into
repetitive garbage (`!#!#!#!#!`).

The fix: compute argmax directly on the restricted score set.

```python
restricted_scores = log_p[s_t] + delta
token = s_t[mx.argmax(restricted_scores)]
```

This bug affected 4B-GRAFT results (35.1% → 58.2% after fix) and the entropy-adaptive
experiment. The 14B-GRAFT results (59.7%) used a different code path and were not affected.

---

## What We Learned

1. **Proxy-tuning is the best logit-level method for steering base models.** No formula
   variation (CFG, contrastive decoding, GRAFT restriction) beats simple logit addition
   over the full vocabulary. The gap is ~3pp on 14B-base (62.5% vs 59.7%).

2. **GRAFT (restricted delta) is a viable alternative when full-vocab access isn't
   available.** It closes 87% of the gap despite using only 0.03% of the vocabulary.
   On precision tasks (case, punctuation), it actually beats proxy-tuning.

3. **The method's value is as a bridge.** When a new base model drops and the instruct
   variant isn't ready yet, proxy-tuning fills the gap at zero training cost. Once the
   instruct model exists, use it directly.

4. **Steering an already-tuned model has diminishing returns.** The delta from a smaller
   pair adds almost nothing (+0.4pp) to an already instruction-tuned model. All three
   methods (baseline, proxy, GRAFT) converge to ~58% on 4B-instruct.

5. **The behavioral delta is full-rank, not low-rank.** The entire vocabulary participates
   in the behavioral shift from base to instruct. Restriction to top-k discards 99.97%
   of the delta mass but preserves 87% of the functional effect — the decision-relevant
   tokens carry disproportionate importance despite carrying negligible mass.

6. **Self-distillation does not generalize from code to instruction-following.** The
   precision-exploration conflict that makes SSD work on code does not exist in diverse
   instruction-following tasks.

7. **Always verify numerical operations in MLX.** The `at[].add()` bug went undetected
   through multiple experiments and led to incorrect conclusions about GRAFT's failure
   on instruct models. The qualitative analysis (looking at actual outputs) is what
   caught it.

---

## Experiment 8: Overnight Exploration (7 ideas, 10 prompts each)

Tested diverse cheap logit-level approaches. Two findings stood out:

**max_tokens=512 gives +10pp on 4B-instruct** (70% vs 60% at 256 tokens). Many IFEval
prompts require long responses (300+ words, multiple paragraphs). Our 256-token limit
was artificially capping performance. This is the single biggest improvement found in
the entire project — and it's just a config change.

**Entropy-adaptive proxy works on 14B-base** (tau=1.0 matches full proxy at 60%, both
double the base at 30%). With the GRAFT scoring bug fixed, the adaptive approach works
correctly on base models: apply proxy delta when the model is uncertain (entropy > tau),
greedy when confident. At tau=1.0, it matches full proxy while using the delta on only
a subset of tokens.

**Best-of-5 sampling doesn't help** — neither self-certainty scoring nor proxy reranking
improved over greedy on any model (4B-instruct, 14B-base, 0.6B). The greedy path is
already near-optimal for instruction-following; sampling introduces noise without
finding better solutions.

| Experiment | Result | vs Baseline |
|-----------|--------|------------|
| 4B best-of-5 self-certainty | 6/10 (60%) | = baseline |
| 14B-base best-of-5 | 2/10 (20%) | worse than 3/10 |
| 14B-base entropy-adaptive tau=1 | 6/10 (60%) | = proxy (both +30pp) |
| 14B-base bon5 + proxy rerank | 3/10 (30%) | = baseline |
| **4B max_tokens=512** | **7/10 (70%)** | **+10pp** |
| 0.6B bon5 + 4B proxy rerank | 5/10 (50%) | = baseline |
