# Discussion

## Main Finding

The final held-out evaluation shows that confidence-only ranking provides the
strongest overall selective performance among the five evaluated methods at
the largest evaluation size.

On the 3,000-example held-out test set:

| Method | AURC | Normalized AURC |
|---|---:|---:|
| Confidence only | **0.292379** | **0.216109** |
| Confidence + self-verifier | 0.309120 | 0.262110 |
| Confidence + question-aware semantic V2 | 0.339417 | 0.345358 |
| Question-aware semantic V2 | 0.394644 | 0.497105 |
| Self-verifier only | 0.433892 | 0.604947 |

Because lower AURC is better, confidence only provides the best global ranking
at N=3,000.

The same pattern appears at N=500, 1,000, and 2,000.

The smallest evaluation is an exception: at N=200, confidence + self-verifier
achieves AURC `0.284548`, compared with `0.292532` for confidence only.

The supported conclusion is therefore not that confidence wins at every sample
size, but that its ranking advantage emerges consistently from N=500 onward
and is strongest in the final 3,000-example comparison.

## Uncertainty of the Final Finding

The final N=3,000 comparison uses 5,000 paired bootstrap resamples.

Relative to confidence only:

| Method | Delta AURC | 95% CI |
|---|---:|---:|
| Confidence + self-verifier | +0.016741 | [0.008561, 0.024773] |
| Confidence + question-aware semantic V2 | +0.047038 | [0.036312, 0.057263] |
| Question-aware semantic V2 | +0.102265 | [0.086357, 0.118286] |
| Self-verifier only | +0.141513 | [0.124280, 0.159087] |

For each method:

```text
Delta AURC = method AURC - confidence-only AURC
```

Because lower AURC is better, positive Delta AURC favors confidence only.

All four paired 95% confidence intervals remain above zero in the final
3,000-example evaluation.

This supports the stability of the observed confidence-only advantage under
paired resampling of this held-out test set.

These intervals quantify uncertainty conditional on the observed held-out
sample. They should not be interpreted as establishing a universal advantage
for confidence-only ranking across datasets, model families, or verification
architectures.

## Why Verification Did Not Improve Global Ranking

The negative result does not imply that the verifier signals contain no useful
information.

Both verification paths are diagnostically related to correctness.

For example, the final error analysis shows that:

- `ENTAILMENT` predictions have substantially higher exact-match accuracy than
  `CONTRADICTION` predictions;
- `SUPPORTED` predictions are more accurate than `UNCERTAIN` predictions;
- `REJECTED` predictions are overwhelmingly incorrect.

However, a verifier can correlate with correctness without producing a better
complete ordering of predictions.

Selective ranking depends on the relative ordering of all examples, not only
on whether broad verifier categories differ in average accuracy.

The final error analysis demonstrates substantial overlap:

- 686 `ENTAILMENT` predictions are incorrect;
- 273 correct predictions have entailment probability at or below `0.2`;
- 344 incorrect predictions are labeled `SUPPORTED`;
- 711 correct predictions are labeled `UNCERTAIN`;
- 14 correct predictions are labeled `REJECTED`.

These overlaps create opportunities for verifier scores to move correct
predictions downward or incorrect predictions upward relative to the
confidence-only ordering.

## Question-Aware Verification Adds an Additional Failure Stage

The question-aware semantic path first converts a question-answer pair into a
declarative claim before NLI scoring.

This additional stage introduces a failure mode that confidence-only ranking
does not have.

In the final prediction set:

- 2,667 generated claims are valid;
- 333 generated claims are invalid.

Invalid claims receive semantic score `0.0` by design.

This is a conservative structural-validation rule, but invalidity is not
equivalent to QA incorrectness: 144 of the 333 invalid-claim examples are
correct under the canonical forced-answer Exact Match definition.

The zero-score rule can therefore downrank correct predictions when claim
generation or structural validation fails independently of QA correctness.

## Shared NLI Backbone

The question-aware semantic verifier and the answer-support/self-verifier are
separate verification paths, but both use:

```text
FacebookAI/roberta-large-mnli
```

Their results should therefore not be interpreted as evidence from two
statistically independent verifier models.

Shared strengths and shared failure modes of the NLI backbone can affect both
signals.

This limits the conclusions that can be drawn from agreement or disagreement
between the two verification paths.

## Fixed Fusion and Calibration-Only Tuning

The primary combined methods use the fixed equal-weight geometric mean:

```text
combined_score = sqrt(confidence * verifier_score)
```

These final comparison rules are not tuned on held-out test labels.

A separate calibration-only fusion-weight analysis also evaluates weighted
geometric combinations.

That tuning selects the confidence endpoint for both verifier combinations,
meaning the best calibration-selected weight assigns no additional weight to
the verifier signal.

This auxiliary result is consistent with the final held-out observation that
the verifier scores do not improve global ranking over confidence alone in this
experimental setting.

It does not establish that every possible learned or nonlinear fusion method
would fail.

## Calibration and Ranking

Temperature scaling substantially improves probability calibration.

The fitted temperature is approximately:

```text
T = 4.604539
```

Calibration negative log-likelihood changes from:

```text
0.966913 -> 0.412469
```

Temperature scaling is a monotonic transformation of the answer-vs-null
margin.

As a result, it improves probability calibration but does not change the
ordering of examples ranked by confidence alone.

The confidence-only AURC result should therefore be interpreted as the ranking
quality of the underlying QA confidence margin, with calibrated probabilities
used to provide a better probability scale and to support comparable fusion
inputs.

## Local Versus Global Performance

Global AURC should be distinguished from behavior at individual coverage
levels.

At approximately 20% coverage in the final 3,000-example evaluation:

- confidence + question-aware semantic V2 has selective risk of approximately
  `0.1450`;
- confidence only has selective risk of approximately `0.1467`.

The combined semantic method is therefore slightly better at this particular
operating point.

This local reversal does not change the global AURC ordering.

It illustrates why matched-coverage metrics and complete risk-coverage curves
are both useful: one method can improve a specific operating point while still
producing a worse overall ranking.

## Rank-Ordering Interpretation

The main failure mechanism can be described as a ranking problem.

A verifier helps global selective performance only when the reordering it
introduces places correct predictions ahead of incorrect predictions often
enough to compensate for harmful reorderings.

The final tie-aware rank diagnostic shows that both verifier-based fusions make
some beneficial pairwise changes, but their harmful strict reorderings exceed
their beneficial strict reorderings overall.

This is consistent with their positive Delta AURC relative to confidence only.

The rank analysis is diagnostic rather than causal, but it provides a more
direct explanation of why apparently informative verifier scores can still
worsen global selective ranking.

## Implications

The first implication is methodological:

**verification methods should be compared against a strong confidence baseline,
not merely against an always-answer system or an uncalibrated heuristic.**

A second implication is that additional model complexity does not automatically
improve selective ranking.

A useful verifier must do more than identify some errors. Its continuous score
must improve the relative ordering of correct and incorrect predictions across
the coverage range of interest.

A third implication is that verifier usefulness can depend on the evaluation
objective.

A verifier that does not improve global AURC may still be useful at a specific
coverage level, as a diagnostic signal, or inside a different decision policy.

Those possibilities are distinct from the global ranking question evaluated
here.

## Scope of the Conclusion

The supported conclusion is deliberately narrow:

> Adding semantic or answer-support verification signals does not necessarily
> improve selective QA ranking over a strong confidence baseline.

In this experiment, confidence only achieves the best global AURC at
N=500, 1,000, 2,000, and 3,000, while confidence + self-verifier is slightly
better at N=200.

At the final N=3,000 evaluation, every evaluated verifier-only or fixed-fusion
alternative has higher AURC than confidence only, and every paired bootstrap
Delta AURC confidence interval remains above zero.

The result is specific to:

- SQuAD v2;
- the pretrained extractive QA backbone;
- the implemented confidence signal;
- the QA2D claim-generation model;
- the shared RoBERTa-large-MNLI verifier backbone;
- the implemented verifier scoring rules;
- the fixed primary fusion rules;
- and the held-out protocol used in this repository.

It should not be generalized to verification methods, large language models,
or high-stakes reliability in general without additional experiments.