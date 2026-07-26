# Label Taxonomy

## Purpose

This document defines the labels used to evaluate answerability, evidence support, correctness, and system decisions.

Answerability and evidence support must be evaluated separately.

## 1. Answerability Labels

### `ANSWERABLE`

The provided context contains enough information to answer the question.

### `UNANSWERABLE`

The provided context does not contain enough information to answer the question reliably.

A question may be unanswerable even when the model knows the answer from external knowledge.

## 2. Evidence-Support Labels

### `SUPPORTED`

The answer is directly supported by the provided evidence.

Requirements:

* the evidence contains the necessary information
* the answer does not add unsupported details
* the answer does not contradict the context

### `UNSUPPORTED`

The answer is not justified by the provided evidence.

This includes:

* fabricated information
* external knowledge not found in the context
* contradictions with the evidence
* unsupported additional details
* answers to unanswerable questions

### `UNCERTAIN`

The available evidence is incomplete, ambiguous, or insufficient for a confident support judgment.

This label should not be used merely because annotation is difficult.

## 3. Correctness Labels

### `CORRECT`

The answer matches the reference answer or expresses an equivalent meaning.

### `INCORRECT`

The answer:

* contradicts the reference
* answers a different question
* contains a significant factual error
* omits essential information

### `PARTIALLY_CORRECT`

The answer contains some correct information but is incomplete or includes a minor unsupported element.

Partially correct answers will be analyzed separately and will not automatically be treated as fully correct.

## 4. Reliability Labels

### `RELIABLE`

An answer is reliable when it is:

* correct
* supported by the provided evidence
* appropriately confident

### `UNRELIABLE`

An answer is unreliable when it is:

* incorrect
* unsupported
* misleadingly incomplete
* produced with unjustifiably high confidence

## 5. Decision Labels

### `ANSWER`

The system returns the answer because evidence support and confidence are sufficient.

### `VERIFY`

The system performs an additional verification step because reliability is unclear.

### `ABSTAIN`

The system does not provide a factual answer because available support is insufficient.

## Decision Mapping

```text
SUPPORTED + HIGH CONFIDENCE
→ ANSWER

UNCERTAIN SUPPORT OR MEDIUM CONFIDENCE
→ VERIFY

UNSUPPORTED OR LOW CONFIDENCE
→ ABSTAIN
```

This mapping is provisional. Final thresholds will be selected using development data.

## Annotation Rules

Annotators must:

1. evaluate only the supplied question, context, and answer
2. avoid using external knowledge
3. judge correctness and evidence support separately
4. record ambiguous cases instead of forcing certainty
5. provide a short reason for `UNSUPPORTED` and `UNCERTAIN` labels

## Example

**Question:** Who developed the theory of relativity?

**Context:** The passage discusses Isaac Newton's laws of motion but does not mention relativity.

**Generated answer:** Albert Einstein.

Labels:

```text
Answerability: UNANSWERABLE
Correctness: CORRECT
Evidence Support: UNSUPPORTED
Recommended Decision: ABSTAIN
```

The answer is factually correct but unsupported by the supplied evidence.

## Quality Control

Before large-scale annotation, the project will conduct a small pilot study.

If multiple annotators are used, the project will report:

* annotation agreement
* disagreement categories
* final adjudication procedure

## Current Status

These definitions are provisional and may be refined after dataset inspection and pilot annotation.
