# Error Analysis

## Scope

This analysis uses the final 3,000-item held-out test prediction file:

```text
outputs/predictions/test_with_question_aware_v2_and_self_verification.jsonl
```

Correctness is computed with the same canonical forced-answer definition used
by the final selective-QA evaluation:

- answerable examples are correct when the normalized prediction exactly
  matches at least one normalized reference answer;
- unanswerable forced-answer candidates are incorrect.

The summary is generated reproducibly with:

```bash
python -m src.analysis.generate_final_error_analysis
```

The resulting artifact is stored at:

```text
outputs/analysis/final_error_analysis/summary.json
```

This analysis is diagnostic only. It does not modify ranking rules, thresholds,
fusion weights, calibration parameters, verifier prompts, or final held-out
scores.

## Overall Question-Answering Errors

Out of 3,000 held-out examples:

- correct predictions: **1,267**
- incorrect predictions: **1,733**
- full exact-match accuracy: **0.422333**

The verifier signals do not change the underlying forced-answer QA candidates.
Their role in the final study is to provide alternative ranking signals for
selective answering.

## Question-Aware Semantic Verifier

The question-aware verifier produces the following NLI labels:

| Label | Count | Correct | Incorrect | Accuracy |
|---|---:|---:|---:|---:|
| `ENTAILMENT` | 1,622 | 936 | 686 | 0.5771 |
| `CONTRADICTION` | 713 | 54 | 659 | 0.0757 |
| `NEUTRAL` | 332 | 133 | 199 | 0.4006 |
| `INVALID_CLAIM` | 333 | 144 | 189 | 0.4324 |

The labels have diagnostic value. In particular, contradiction is strongly
associated with incorrect QA predictions, while entailment is associated with
higher exact-match accuracy.

However, semantic support is far from equivalent to QA correctness.

Among the 1,622 entailment-labeled examples, **686** are still incorrect.

At an entailment probability of at least `0.8`, **464 incorrect predictions**
remain.

This shows that a high semantic-support score is not equivalent to answer
correctness. The supplied context can support the generated claim while the
underlying answer still fails the benchmark's normalized Exact Match
criterion.

## Low Entailment Does Not Imply Incorrectness

The opposite failure mode also occurs.

Among correct QA predictions, **273** have a question-aware entailment
probability of `0.2` or lower.

The verifier can therefore substantially downrank some correct answers.

Possible contributing mechanisms include claim-generation errors, incomplete
preservation of the question-answer relationship, or limitations of the NLI
scoring model.

These are diagnostic interpretations rather than causal conclusions.

## Claim Validity

Question-aware claim generation produced:

- **2,667 valid claims**
- **333 invalid claims**

Under the canonical correctness definition:

| Claim validity | Count | Correct | Incorrect | Accuracy |
|---|---:|---:|---:|---:|
| Valid | 2,667 | 1,123 | 1,544 | 0.4211 |
| Invalid | 333 | 144 | 189 | 0.4324 |

Invalid claims are therefore not simply a proxy for incorrect QA predictions.

The final ranking rule assigns semantic score `0.0` to invalid claims. This is
a conservative structural-validation rule, but it can also downrank correct QA
predictions when claim generation fails independently of answer correctness.

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

These counts are not mutually exclusive.

A single invalid claim can fail multiple validation checks, so the reason
counts can sum to more than the 333 invalid claims.

## Self-Verifier Error Structure

The answer-support / self-verifier produces:

| Label | Count | Correct | Incorrect | Accuracy |
|---|---:|---:|---:|---:|
| `SUPPORTED` | 886 | 542 | 344 | 0.6117 |
| `UNCERTAIN` | 1,791 | 711 | 1,080 | 0.3970 |
| `REJECTED` | 323 | 14 | 309 | 0.0433 |

This verifier is also diagnostically informative.

Rejected candidates are overwhelmingly incorrect under the canonical
forced-answer Exact Match definition, while supported candidates have
substantially higher accuracy.

However:

- **344 incorrect predictions** are still labeled `SUPPORTED`;
- **711 correct predictions** are labeled `UNCERTAIN`;
- **14 correct predictions** are labeled `REJECTED`.

The resulting overlap limits the verifier's ability to improve the complete
global selective ranking.

## Connection to the AURC Result

The error analysis helps explain how a verifier can be informative at the
individual-example level while still worsening global AURC.

Both verifier signals separate some correct and incorrect predictions.
However, they also introduce substantial overlap and ranking mistakes.

When used alone or combined with confidence through the fixed geometric-mean
rules, their resulting global orderings are worse than confidence-only ranking
on the final 3,000-example held-out set.

This diagnostic result is consistent with, but does not by itself prove the
cause of, the observed AURC differences.

## Interpretation Limit

This analysis is descriptive rather than causal.

It does not establish that semantic verification or answer-support verification
is ineffective in general.

The findings are specific to:

- the SQuAD v2 held-out evaluation,
- the pretrained extractive QA backbone,
- the QA2D claim-generation model,
- the shared RoBERTa-large-MNLI verification backbone,
- the implemented validation and scoring rules,
- and the canonical forced-answer correctness definition used in this project.

The supported conclusion remains narrow: the evaluated verifier signals are
diagnostically informative, but they do not improve global selective ranking
over the confidence-only baseline in the final experimental setting.