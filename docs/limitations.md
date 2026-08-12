# Limitations

## Benchmark Scope

The final experiments use **SQuAD v2** only.

Results on this benchmark do not establish reliability in open-domain, conversational, retrieval-augmented, multilingual, or real-world question-answering settings.

## Model Scope

The answer-generation backbone is the pretrained extractive QA model:

`deepset/roberta-base-squad2`

The conclusions therefore apply to this experimental setting and should not be generalized to other QA architectures, generative language models, model scales, or training regimes without additional evaluation.

## Verification-Model Dependence

The question-aware semantic verifier and the self-verifier are separate verification paths, but both use:

`FacebookAI/roberta-large-mnli`

They should therefore **not** be interpreted as statistically independent verifier models.

Any shared strengths, biases, or failure modes of this NLI backbone can affect both verification signals.

## Claim-Generation Dependence

Question-aware semantic verification depends on converting each question-answer pair into a declarative claim using:

`domenicrosati/QA2D-t5-base`

Claim-generation failures can reduce verification quality even when the downstream NLI model behaves correctly.

Structurally invalid claims are assigned a semantic score of `0` and are not passed to the NLI model.

## Confidence Limitations

Model confidence is not a direct measurement of truth.

Temperature scaling improves calibration on the calibration split but does not guarantee that confidence corresponds perfectly to correctness on every example or future dataset.

## Fixed Score Combination

The combined methods use a fixed equal-weight geometric mean.

This rule is intentionally not tuned on held-out test labels, which protects against leakage but also means other combination functions or calibration procedures may perform differently.

The study does not establish that equal-weight geometric combination is optimal.

## Selective Evaluation Limitations

AURC summarizes global ranking quality across coverage levels, but local operating points can differ from the overall ranking.

A method can be slightly better at a specific coverage level while still having worse overall AURC.

For this reason, matched-coverage results are interpreted together with, rather than instead of, the global risk-coverage analysis.

## Bootstrap Scope

The paired bootstrap analysis quantifies uncertainty **conditional on the observed 3,000-example held-out test sample**.

Its confidence intervals do not represent uncertainty across entirely different datasets, model families, annotation processes, or deployment environments.

The bootstrap is used only after final predictions are fixed and is not used for system tuning.

## Dataset Bias

SQuAD v2 can contain annotation artifacts, benchmark-specific lexical patterns, and domain limitations.

A model may exploit such regularities without learning a generally reliable abstention strategy.

## Computational Cost

Semantic and self-verification require additional model inference beyond the calibrated-confidence baseline.

The project evaluates ranking quality but does not provide a full latency, energy, memory, or cost analysis.

## Safety and Deployment Claims

This work does **not** establish that the evaluated system is safe or reliable for medical, legal, financial, or other high-stakes applications.

It also does not claim that semantic verification or self-verification is ineffective in general.

The supported conclusion is narrower: in this SQuAD v2 extractive-QA setting, the evaluated verification signals did not outperform calibrated confidence for overall selective ranking.

## Generalization

The final findings are limited to:

- SQuAD v2
- the evaluated extractive QA backbone
- the QA2D claim-generation model
- the shared RoBERTa-large MNLI verification backbone
- the implemented scoring rules
- the fixed held-out evaluation protocol

Broader claims require evaluation across additional datasets, models, verifier architectures, and deployment conditions.
