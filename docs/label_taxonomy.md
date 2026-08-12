# Label Taxonomy

## Purpose

This document defines the labels and correctness rules used in the final selective-QA evaluation.

The final analysis distinguishes between:

- dataset answerability
- exact-match QA correctness
- question-aware claim validity
- question-aware NLI labels
- self-verification labels
- ranking scores

These concepts are diagnostic signals and are not interchangeable.

## 1. Dataset Answerability

### `is_answerable = true`

The SQuAD v2 example has at least one reference answer in the supplied context.

### `is_answerable = false`

The SQuAD v2 example is unanswerable under the benchmark annotation.

Answerability is inherited from the dataset and is not manually re-annotated in the final experiment.

## 2. QA Correctness

Final QA correctness uses the same normalized exact-match logic as the evaluator.

For answerable examples, a prediction is correct when its normalized text exactly matches at least one normalized reference answer.

For unanswerable examples, a prediction is correct only when the normalized prediction is empty.

The final evaluation does not use a `PARTIALLY_CORRECT` category and does not replace exact match with manual semantic-equivalence judgment.

## 3. Question-Aware Claim Validity

The question-aware semantic verifier first converts the question-answer pair into a declarative claim.

### `qa_claim_valid = true`

The generated claim passes the structural validation rules required before NLI scoring.

### `qa_claim_valid = false`

The claim fails one or more validation rules.

Observed validation reasons include:

- `ANSWER_NOT_PRESERVED`
- `NUMBER_NOT_PRESERVED`
- `QUESTION_FORM`
- `NEGATION_NOT_PRESERVED`
- `ANSWER_ONLY_FRAGMENT`
- `TOO_SHORT`

A single invalid claim can have multiple validation reasons.

Invalid claims are assigned semantic score `0` and are not passed to the NLI model.

## 4. Question-Aware NLI Labels

For valid claims, the question-aware verifier uses the context as evidence and produces one of:

### `ENTAILMENT`

The NLI model predicts that the context supports the generated claim.

### `CONTRADICTION`

The NLI model predicts that the context contradicts the generated claim.

### `NEUTRAL`

The NLI model predicts that the context neither clearly entails nor contradicts the claim.

### `INVALID_CLAIM`

The claim failed structural validation before NLI evaluation.

These labels are diagnostic evidence signals, not direct correctness labels.

In the final 3,000-example held-out test set:

| Label | Count |
| --- | ---: |
| `ENTAILMENT` | 1,622 |
| `CONTRADICTION` | 713 |
| `NEUTRAL` | 332 |
| `INVALID_CLAIM` | 333 |

## 5. Self-Verification Labels

The self-verification path produces:

### `SUPPORTED`

The self-verification score is above the configured support threshold.

### `UNCERTAIN`

The score lies between the rejection and support thresholds.

### `REJECTED`

The score is below the configured rejection threshold.

In the final 3,000-example held-out test set:

| Label | Count |
| --- | ---: |
| `SUPPORTED` | 886 |
| `UNCERTAIN` | 1,791 |
| `REJECTED` | 323 |

The raw self-verification score is defined on `[-1, 1]` and is mapped to `[0, 1]` for ranking:

```text
normalized_self = (raw_self + 1) / 2
```

## 6. Ranking Signals

The final experiment compares five ranking signals:

1. calibrated confidence only
2. question-aware semantic V2
3. calibrated confidence + question-aware semantic V2
4. self-verifier only
5. calibrated confidence + self-verifier

The two combined methods use a fixed equal-weight geometric mean:

```text
combined_score = sqrt(score_a * score_b)
```

No label threshold or combination weight is tuned using held-out test labels.

## 7. Interpretation Rules

The final analysis keeps the following concepts separate:

- **answerability** describes the benchmark example
- **correctness** describes exact-match agreement with the reference answer
- **claim validity** describes whether the generated declarative claim is structurally usable
- **NLI labels** describe semantic support or contradiction relative to context
- **self-verification labels** describe the verifier's support judgment
- **ranking scores** determine selective ordering

For example, an `ENTAILMENT` label does not guarantee QA correctness, and a `REJECTED` label does not itself change the underlying QA prediction.

## 8. Scope

This taxonomy documents the labels actually used in the final experiment.

It does not define a general annotation standard for open-domain QA, conversational systems, or high-stakes applications.
