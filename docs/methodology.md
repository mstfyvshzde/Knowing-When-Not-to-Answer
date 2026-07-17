# Methodology

## Research Design

This study will use a controlled experimental design to compare multiple answer-generation and abstention systems under the same dataset, model, and evaluation conditions.

The main independent variable is the verification strategy used before returning an answer.

The main dependent variables are:

* answer accuracy
* coverage
* selective risk
* unsupported answer rate
* false confidence rate
* calibration error

## Experimental Systems

The study will compare four systems.

### 1. Raw Answer Baseline

The model answers every question without verification or abstention.

### 2. Confidence-Based Baseline

The model abstains when its confidence score falls below a selected threshold.

### 3. Evidence-Based Verifier

The generated answer is checked against the provided evidence before it is returned.

### 4. Combined Self-Verification Framework

The final system combines:

* answer confidence
* calibrated confidence
* evidence support
* answerability prediction

The system then selects one of three actions:

* `ANSWER`
* `VERIFY`
* `ABSTAIN`

## Initial Dataset

The initial benchmark will contain both answerable and unanswerable question-answering examples.

SQuAD 2.0 is the primary candidate because it provides questions that can and cannot be answered from the supplied context.

Additional benchmarks may be added only after the initial pipeline is validated.

## Data Splits

The official dataset splits will be preserved whenever possible.

The development set will be used for:

* threshold selection
* calibration
* hyperparameter decisions

The test set will be used only for final evaluation.

No test-set labels will be used during model or threshold development.

## Confidence Estimation

Confidence may be estimated using:

* model probability scores
* answerability probability
* response consistency
* verifier output

Raw and calibrated confidence will be evaluated separately.

## Evidence Verification

The evidence checker will estimate whether the generated answer is supported by the supplied context.

Each answer will be categorized as:

* `SUPPORTED`
* `UNSUPPORTED`
* `UNCERTAIN`

The exact labeling rules will be defined before experiments begin.

## Abstention Policy

The decision policy will use confidence and evidence signals.

A simplified initial rule is:

```text
IF evidence is supported AND confidence is high:
    ANSWER

ELSE IF evidence is uncertain:
    VERIFY

ELSE:
    ABSTAIN
```

Decision thresholds will be selected using development data only.

## Evaluation Procedure

Every system will be evaluated on the same examples.

Results will be reported across multiple confidence thresholds to measure the trade-off between coverage and risk.

The combined framework will be compared against all baselines using identical evaluation metrics.

## Leakage Controls

To reduce evaluation leakage:

* test labels will not be used for threshold selection
* prompts and rules will not be tuned on test failures
* dataset splits will remain fixed
* all preprocessing decisions will be documented
* manual error analysis will be performed after final predictions are saved

## Reproducibility

Experiments will use:

* fixed random seeds
* versioned configuration files
* recorded package versions
* saved predictions
* saved evaluation outputs
* documented execution commands

## Methodological Limitation

This design evaluates reliability under controlled benchmark conditions.

It does not establish that the framework is safe or reliable across all domains, models, or real-world applications.
