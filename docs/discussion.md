# Discussion

## Main Finding

The final held-out evaluation shows that calibrated confidence provides the strongest global selective ranking among the five evaluated methods.

On the 3,000-example held-out test set:

- **Confidence only**: AURC `0.292378`
- **Confidence + self-verifier**: AURC `0.309103`
- **Confidence + question-aware semantic V2**: AURC `0.339439`
- **Question-aware semantic V2**: AURC `0.394397`
- **Self-verifier only**: AURC `0.433766`

Because lower AURC is better, the confidence-only baseline ranks best overall.

This ordering is not a single-sample artifact. Confidence only also has the lowest AURC at each nested evaluation size: `200`, `500`, `1000`, `2000`, and `3000`.

## Uncertainty of the Finding

Paired bootstrap analysis with 5,000 resamples supports the stability of the observed ranking on this held-out test set.

Relative to confidence only:

- **Confidence + self-verifier**: Delta AURC `+0.016725`, 95% CI `[0.008614, 0.024690]`
- **Confidence + question-aware semantic V2**: Delta AURC `+0.047061`, 95% CI `[0.036347, 0.057295]`
- **Question-aware semantic V2**: Delta AURC `+0.102019`, 95% CI `[0.086097, 0.118102]`
- **Self-verifier only**: Delta AURC `+0.141389`, 95% CI `[0.124188, 0.158985]`

All observed Delta AURC confidence intervals remain above zero. Because lower AURC is better, this supports the conclusion that the confidence-only ranking advantage is stable under paired resampling of this test set.

These intervals should not be interpreted as establishing a universal performance advantage beyond the observed experimental setting.

## Why Verification Did Not Improve Global Ranking

The negative result does not imply that semantic or self-verification is useless.

A verification signal can be informative without being sufficiently well ordered to improve the entire risk-coverage curve. In this setting, the strong calibrated-confidence baseline already captures a useful ordering of correct and incorrect predictions. Adding a second signal through a fixed geometric mean can reorder some predictions in ways that help locally but hurt globally.

The question-aware path also introduces an additional claim-generation stage. Invalid claims are assigned a score of zero, which is a deliberate safety rule but can penalize examples when claim generation fails for reasons other than answer incorrectness.

## Local Versus Global Performance

The global AURC ranking should be distinguished from local operating-point behavior.

For example, at 20% coverage on the 3,000-example evaluation, the confidence + question-aware semantic method achieves selective risk of approximately `0.1450`, compared with `0.1467` for confidence only.

This is small and does not change the overall AURC conclusion. It does however illustrate that a verification signal can have value at a specific coverage even when it does not improve global ranking.

## Implications

The main implication is methodological: a verification method should be compared against a strong calibrated baseline, not only against raw model confidence or a non-selective system.

A second implication is that greater verification complexity does not automatically produce better selective ranking. The utility of a verifier is not just whether it can detect individual errors, but whether its scores improve the ordering of predictions across the relevant coverage range.

## Scope of the Conclusion

The supported conclusion is narrow:

> Adding semantic or self-verification signals does not necessarily improve selective QA ranking over a strong calibrated-confidence baseline.

This result is specific to the SQuAD v2 held-out evaluation, the extractive QA backbone, and the verification paths implemented in this repository. It should not be extended to general LLM reliability, safety, or verification effectiveness without additional evidence.
