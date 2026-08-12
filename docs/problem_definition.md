# Problem Definition

## Research Problem

Question-answering systems can produce incorrect predictions even when their confidence scores are relatively high. In selective question answering, the goal is therefore not only to predict an answer, but also to rank predictions so that the system can answer higher-reliability cases first and abstain as coverage decreases.

This project studies whether additional verification signals improve that selective ranking over a strong calibrated-confidence baseline.

## Core Research Question

Does adding question-aware semantic verification or self-verification improve selective QA ranking over calibrated confidence alone?

## Operational Setting

The final evaluation treats each method as a scoring function over the same held-out QA predictions.

Higher-scored predictions are answered first. Lower-scored predictions are progressively abstained from as coverage decreases.

The five evaluated ranking methods are:

1. calibrated confidence only
2. question-aware semantic verification
3. calibrated confidence + question-aware semantic verification
4. self-verifier only
5. calibrated confidence + self-verifier

The two combined methods use a fixed equal-weight geometric mean and are not tuned on held-out test labels.

## Evaluation Objective

The primary objective is to compare how effectively the five scoring methods order correct and incorrect predictions across the risk-coverage curve.

The primary metric is AURC, where lower values indicate better selective ranking. Normalized AURC and matched-coverage risk and accuracy are reported as complementary measures.

## Scope

The study is restricted to SQuAD v2, a pretrained extractive QA backbone, and the verification signals implemented in this repository.

It does not claim to solve hallucination, establish general language-model reliability, or demonstrate safety in real-world or high-stakes applications.

## Final Framing

The supported conclusion is intentionally narrow:

> Adding semantic or self-verification signals does not necessarily improve selective QA ranking over a strong calibrated-confidence baseline.

This statement describes the observed experimental setting and is not a universal claim about verification methods.
