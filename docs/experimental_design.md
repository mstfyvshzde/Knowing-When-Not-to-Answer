# Experimental Design

## Objective

The experiments will test whether confidence estimation and evidence verification improve the reliability of question-answering systems.

## Compared Systems

The study will compare four systems:

1. **Raw Baseline**
   Answers every question without verification.

2. **Confidence Baseline**
   Abstains when confidence falls below a threshold.

3. **Evidence Verifier**
   Checks whether the generated answer is supported by the provided context.

4. **Combined Framework**
   Uses confidence, calibration, answerability, and evidence support before deciding to answer or abstain.

## Controlled Variables

All systems will use the same:

* dataset splits
* answer-generation model
* preprocessing pipeline
* evaluation examples
* random seeds

Only the verification and abstention strategy will change.

## Main Experiments

### Experiment 1 — Raw Answer Performance

Measure the accuracy and unsupported answer rate when the model answers every question.

### Experiment 2 — Confidence-Based Abstention

Evaluate performance across multiple confidence thresholds.

### Experiment 3 — Evidence Verification

Measure whether evidence checking reduces unsupported answers.

### Experiment 4 — Combined Framework

Test whether combining confidence and evidence produces a better risk–coverage trade-off.

### Experiment 5 — Ablation Study

Remove one component at a time:

* without calibration
* without evidence verification
* without answerability prediction
* without consistency checking

## Threshold Selection

Thresholds will be selected using development data only.

The test set will not be used for:

* threshold tuning
* prompt modification
* calibration fitting
* system design decisions

## Evaluation Outputs

Each experiment will save:

* predictions
* confidence scores
* evidence scores
* final decisions
* evaluation metrics
* error categories

## Comparison Strategy

Systems will be compared at:

* equal coverage levels
* equal abstention rates
* multiple confidence thresholds

This prevents a system from appearing better only because it abstains more often.

## Expected Result

The combined framework is expected to reduce unsupported and high-confidence incorrect answers.

This is a hypothesis and will not be presented as a conclusion before the experiments are completed.
