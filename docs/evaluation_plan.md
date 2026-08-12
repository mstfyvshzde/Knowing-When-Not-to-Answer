# Evaluation Plan

## Evaluation Goal

The final evaluation tests whether question-aware semantic verification or self-verification improves selective QA ranking over a calibrated-confidence baseline.

## Evaluation Set

The final evaluation uses **3,000 held-out SQuAD v2 test examples**. Temperature scaling is fitted on the separate calibration split only and frozen before test evaluation.

A deterministic order with seed **17** defines nested subsets:

```text
200 ⊂ 500 ⊂ 1000 ⊂ 2000 ⊂ 3000
```

## Compared Methods

Five ranking methods are evaluated:

1. **Confidence only**
2. **Question-aware semantic V2**
3. **Confidence + question-aware semantic V2**
4. **Self-verifier only**
5. **Confidence + self-verifier**

Invalid question-aware claims receive semantic score `0`. The raw self-verifier score is mapped from `[-1, 1]` to `[0, 1]` as `(score + 1) / 2`.

Combined methods use a fixed equal-weight geometric mean:

```text
combined_score = sqrt(score_a * score_b)
```

These rules are not tuned on held-out test labels.

## Primary Metrics

The primary metric is **Area Under the Risk-Coverage Curve (AURC)**; lower is better.

```text
Selective Risk = Incorrect Answered Examples / Answered Examples
Coverage = Answered Examples / Total Examples
```

The evaluation also reports normalized AURC and matched-coverage selective accuracy/risk.

## Deterministic Ranking

Scores are sorted descending. Ties are broken by the smaller original deterministic index.

## Statistical Uncertainty

The final 3,000-example comparison uses **5,000 paired bootstrap resamples**, seed **17**, with **95% percentile confidence intervals**.

For each non-baseline method:

```text
Delta AURC = method AURC - confidence-only AURC
```

Because lower AURC is better, positive Delta AURC favors confidence only.

Bootstrap analysis is performed only after final predictions are fixed and is used to quantify uncertainty, not to tune the system.

## Error Analysis

Diagnostic analysis examines high-entailment errors, claim validity, NLI labels, entailment-score bins, invalid-claim reasons, and confidence-entailment relationships. These analyses do not alter final scoring rules.

## Leakage Controls

- calibration and test partitions are separate
- temperature scaling is fitted only on calibration data
- test labels are not used to tune score-combination weights
- verifier rules are not revised using final test failures
- sample ordering and tie-breaking are deterministic
- all primary methods use identical held-out examples

## Interpretation Rule

The conclusions are restricted to this evaluated setting: **SQuAD v2, the pretrained extractive QA backbone, and the verification signals implemented in this repository.**
