# Error Analysis

## Scope

This analysis uses the final 3,000-item held-out test prediction file:

`outputs/predictions/test_with_question_aware_v2_and_self_verification.jsonl`

Correctness is defined using the same exact-match logic as the final evaluator.

## Overall Question-Answering Errors

Out of 3,000 held-out examples:

- correct predictions: *1,270*
- incorrect predictions: *1,730*
- full exact-match accuracy: *0.422333*

The verifiers do not change the underlying QA predictions. Their role in this study is to provide additional ranking signals for selective answering.

## Question-Aware Semantic Verifier

The question-aware verifier produces the following NLI labels:

| Label | Count | Correct | Incorrect | Accuracy |
|---|---:|---:|---:|---:|
|`ENTAILMENT` | 1,622 | 936 | 686 | 0.5771 |
| `CONTRADICTION` | 713 | 57 | 656 | 0.0799 |
| `NEUTRAL` | 332 | 133 | 199 | 0.4006 |
| `INVALID_CLAIM` | 333 | 144 | 189 | 0.4324 |

The labels have diagnostic value: contradiction is strongly associated with incorrect QA predictions, while entailment is associated with higher accuracy. However, the signal is far from perfect.

Among the 1,622 entailment-labeled examples, **686** are still incorrect. At an entailment probability of at least `0.8`, **464** incorrect predictions remain.

This shows that a high semantic-support score is Not equivalent to answer correctness. The context can support a claim that reflects the model's prediction even when that prediction does not exactly match the gold answer.

## Low Entailment Does Not Imply Incorrectness

The opposite failure mode also occurs.

Among correct QA predictions, **276** have a question-aware entailment probability of `0.2` or lower.

This indicates that the verifier can penalize correct answers, for example when claim generation or NLI does not preserve the relevant relationship adequately.

## Claim Validity

Question-aware claim generation produced:

- **2,667** valid claims
- **333** invalid claims

Accuracy was:

- valid claims: `0.4222`
- invalid claims: `0.4324`

Invalid claims are therefore not simply a proxy for incorrect QA predictions. Assigning a semantic score of zero to all invalid claims is a conservative design rule, but it can also downrank correct QA predictions.

## Invalid-Claim Reasons

The most common validation failures are:

| Reason | Count |
|---|---:|
| `ANSWER_NOT_PRESERVED` | 309 |
| `NUMBER_NOT_PRESERVED` | 106 |
| `QUESTION_FORM` | 18 |
| `NEGATION_NOT_PRESERVED` | 5 |
| `ANSWER_ONLY_FRAGMENT` | 3 |
| `TOO_SHORT` | 3 |

These counts are not mutually exclusive. A single invalid claim can have multiple validation reasons, so the reason counts can sum to more than the 333 invalid claims.

## Self-Verifier Error Structure

The self-verifier produced:

| Label | Count | Correct | Incorrect | Accuracy |
|---|---:|---:|---:|---:|
| `SUPPORTED` | 886 | 542 | 344 | 0.6117 |
| `UNCERTAIN` | 1,791 | 712 | 1,079 | 0.3975 |
| `REJECTED` | 323 | 16 | 307 | 0.0495 |

This verifier is also diagnostically informative: rejected answers are almost always incorrect under the exact-match definition, while supported answers have substantially higher accuracy.

However, **344** incorrect predictions are still labeled `SUPPORTED`, and **712** correct predictions are labeled `UNCERTAIN`. This overlap limits the verifier's ability to improve global selective ranking.

## Connection to the AURC Result

The error analysis helps explain why the verification signals can be informative at an individual example level without improving the overall ranking.

The signals separate some correct and incorrect predictions, hut they also produce substantial overlap. When these scores are used alone or combined with confidence, the resulting global ordering is worse than the calibrated-confidence baseline on this held-out test set.

## Interpretation Limit

This analysis is diagnostic. It does not prove causality for the observed AURC ranking.

The findings are specific to this SQuAD v2 held-out set, the extractive QA backbone, and the verification pipelines evaluated here.
