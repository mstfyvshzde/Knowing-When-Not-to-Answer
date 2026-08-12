# Methodology

## Research Design

This study uses a controlled selective question-answering evaluation to compare confidence-based and verification-based ranking signals under the same dataset, QA backbone, and held-out test examples.

The primary research question is whether adding semantic or self-verification signals improves selective QA ranking over a strong calibrated-confidence baseline.

The main evaluation quantities are:

- answer correctness
- coverage
- selective risk
- Area Under the Risk-Coverage Curve (AURC)
- normalized AURC
- matched-coverage selective accuracy and risk

Lower AURC indicates better selective ranking.

## Dataset and Splits

The experiments use SQuAD v2.

The original SQuAD v2 validation split is divided into separate calibration and held-out test partitions using a deterministic 50/50 stratified split with seed 17.

The calibration split is used for calibration fitting only.

The held-out test split is used for final evaluation only.

Test labels are not used to fit the temperature, tune ranking rules, select score-combination weights, or revise verification prompts after observing final test outcomes.

## QA Baseline

The extractive QA backbone is:

`deepset/roberta-base-squad2`

The raw baseline always returns an answer.

For robustness, the QA pipeline requests the top five answer candidates and selects the first non-empty candidate with a valid character span. This prevents an empty top-ranked pipeline result from silently removing an example from evaluation.

## Confidence Estimation

Confidence is derived from the QA model's answer-vs-null margin.

For each prediction:

```text
answer score = start logit + end logit
null score   = CLS start logit + CLS end logit
margin       = answer score - null score
```

The uncalibrated confidence is the sigmoid of this margin.

## Temperature Scaling

Temperature scaling is fitted on the calibration split only.

The learned temperature is:

`4.754804205196784`

Calibration negative log-likelihood changed from:

- before scaling: `1.011796`
- after scaling: `0.422054`

After fitting, the temperature is frozen and applied to held-out test predictions without refitting on test labels.

## Question-Aware Semantic Verification

The question-aware semantic V2 path converts each question-answer pair into a declarative claim and verifies that claim against the supplied context.

Claim generation uses:

`domenicrosati/QA2D-t5-base`

Natural-language inference uses:

`FacebookAI/roberta-large-mnli`

Generated claims are checked with structural validity rules before NLI.Invalid claims are assigned a semantic score of 0 and are not passed to the NLI model.

Gold answers are not used by the verifier.

## Self-Verification

The self-verification path separately evaluates whether the predicted answer is supported by the context.

The verifier uses:

`FacebookAI/roberta-large-mnli`

Its raw score is defined on `[-1, 1]` and is mapped to `[0, 1]` before selective ranking:

```text
normalized self score = (raw self score + 1) / 2
```

The semantic and self-verification paths should be interpreted as separate verification signals, not statistically independent models, because they share the same NLI backbone.

## Final Compared Methods

Five primary ranking methods are evaluated:

1. **Confidence only**
2. **Question-aware semantic V2**
3. **Confidence + question-aware semantic V2**
4. **Self-verifier only**
5. **Confidence + self-verifier**

The two combined methods use a fixed equal-weight geometric mean:

```text
combined score = sqrt(score_a * score_b)
```

This combination rule is fixed in advance and is not tuned on held-out test labels.

## Deterministic Ranking and Sample Sizes

The final held-out evaluation uses 3,000 test examples.

A single deterministic shuffled order with seed 17 is created and reused for every sample size.

The evaluated nested subsets are:

```text
200 < 500 < 1000 < 2000 < 3000
```

The subsets are therefore nested prefixes of the same ordering rather than independently sampled sets. Predictions are ranked by score in descending order. Score ties are resolved deterministically by original index.

## Evaluation Metrics

The primary metric is area under the risk-coverage curve (AURC), where lower values indicate better selective ranking.

The analysis also reports:

- normalized AURC
- matched-coverage risk
- matched-coverage accuracy
- full exact-match accuracy

These metrics separate overall ranking quality from performance at individual operating points.

## Statistical Uncertainty

The final 3,000-example comparison uses 5,000 paired bootstrap resamples.

Both the bootstrap seed and evaluation-order seed are 17. The analysis reports 95% percentile confidence intervals for each method and for paired AURC differences relative to confidence only.

Bootstrap analysis is performed after the final predictions and scoring rules are fixed. It quantifies uncertainty on the observed held-out test sample and is not used for tuning.

## Leakage Controls

The experimental design separates calibration from held-out evaluation.

- Temperature scaling is fitted on the calibration split only.
- The fitted temperature is frozen before test evaluation.
- Held-out test labels are not used to tune combination weights.
- The two combined ranking methods use a fixed equal-weight geometric mean.
- Question-aware validation rules are not revised using held-out test performance.
- Final sample ordering and bootstrap seeds are fixed deterministically.

These controls are intended to prevent test-set result chasing.

## Reproducibility

The complete final workflow can be executed with:

```bash
DEVICE=cpu LIMIT=3000 bash scripts/reproduce_results.sh
```

The reference software environment is recorded in `requirements-lock.txt`. Final lightweight evaluation artifacts are retained under `outputs/evaluation/final_sample_size_comparison/`, while large reproducible intermediate predictions and nested subset JSONL files are excluded from version control.

## Scope and Limitations

The methodology applies to the experimental setting implemented in this repository: SQuAD v2, a pretrained extractive QA backbone, and the evaluated semantic/self-verification signals.

The study does not establish a universal result about language-model reliability, hallucination, safety, or verification. Both verification paths also use the same MNLI backbone, so they are treated as separate verification signals rather than independent verifier models.
