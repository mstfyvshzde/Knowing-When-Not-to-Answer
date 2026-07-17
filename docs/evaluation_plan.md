# Evaluation Plan

## Evaluation Goal

The evaluation will determine whether self-verification improves reliability without reducing answer coverage excessively.

## Primary Metrics

### Accuracy

The proportion of correct answers among all evaluated examples.

### Coverage

The proportion of examples for which the system provides an answer.

```text
Coverage = Answered Examples / Total Examples
```

### Abstention Rate

The proportion of examples for which the system refuses to answer.

```text
Abstention Rate = Abstained Examples / Total Examples
```

### Selective Risk

The error rate among examples the system chooses to answer.

```text
Selective Risk = Incorrect Answered Examples / Answered Examples
```

### Unsupported Answer Rate

The proportion of returned answers that are not supported by the available evidence.

### False Confidence Rate

The proportion of incorrect answers produced with confidence above a predefined high-confidence threshold.

### Calibration Error

Expected Calibration Error will measure whether predicted confidence corresponds to observed accuracy.

## Classification Metrics

For unreliable-answer detection, the project will report:

* precision
* recall
* F1 score
* confusion matrix

The positive class will represent an answer that should be verified or rejected.

## Risk–Coverage Analysis

Results will be evaluated across multiple thresholds.

The primary visualization will be a risk–coverage curve showing how error changes as the system answers fewer questions.

A useful verification system should reduce risk while preserving meaningful coverage.

## System Comparisons

The following systems will be compared:

1. raw answer baseline
2. confidence-only abstention
3. evidence-only verification
4. combined self-verification framework

Comparisons will be performed at:

* equal coverage
* equal abstention rate
* selected operating thresholds

## Statistical Analysis

Where appropriate, the study will report:

* confidence intervals
* bootstrap estimates
* paired system comparisons

Statistical tests will be selected after the final dataset and prediction format are fixed.

## Error Analysis

Errors will be grouped into categories such as:

* incorrect answer with high confidence
* unsupported answer
* unnecessary abstention
* missed answerable question
* incorrect evidence judgment
* calibration failure

## Reporting Rules

All reported results must include:

* dataset split
* sample size
* model version
* experiment configuration
* selected threshold
* random seed

No single metric will be used as proof that the system is reliable.
