# Research Gap

## Established Context

Selective prediction, confidence calibration, selective question answering, answerability detection, and evidence verification are already established research areas.

Accordingly, this project does not claim that abstention, calibration, or self-verification is itself new.

## Specific Research Gap

The specific question addressed here is whether adding semantic or self-verification signals actually improves selective QA ranking when the comparison is made against a strong calibrated-confidence baseline under the same held-out evaluation conditions.

This comparison is important because a verifier can appear diagnostically informative on individual examples without necessarily improving the global risk-coverage ordering.

## Contribution

The project provides a controlled and reproducible comparison of five ranking methods:

1. calibrated confidence only
2. question-aware semantic verification
3. confidence + question-aware semantic verification
4. self-verifier only
5. confidence + self-verifier

The design includes separate calibration and held-out test partitions, fixed score-combination rules, deterministic nested evaluation subsets, AURC and normalized AURC, matched-coverage analysis, and paired bootstrap uncertainty analysis.

## Observed Outcome

In the final 3,000-example held-out evaluation, confidence only achieves the lowest global AURC among the five evaluated methods. At N=200, confidence + self-verifier is slightly better; confidence only is best at N=500, 1000, 2000, and 3000.

The finding therefore does not support the hypothesis that adding the evaluated verification signals improves global selective ranking in this experimental setting.

## Scope

The supported conclusion is deliberately narrow:

> Adding semantic or self-verification signals does not necessarily improve selective QA ranking over a strong calibrated-confidence baseline.

This claim is limited to the SQuAD v2 held-out evaluation, the extractive QA backbone, and the verification signals implemented in this repository. It does not establish a general result about LLM reliability, safety, or the effectiveness of verification in other settings.
