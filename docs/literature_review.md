# Literature Review

## Scope

This review situates the project within research on:

- selective prediction and abstention
- confidence calibration
- selective question answering
- answerability detection
- evidence and claim verification
- recent abstention methods for language models

The project does not claim that abstention, calibration, or verification is novel. Its contribution is a controlled comparison of calibrated confidence and two verification signals for selective ranking in a fixed extractive-QA setting.

## 1. Selective Prediction and Abstention

Selective prediction studies models that may reject or abstain from some predictions rather than answering every input.

Geifman and El-Yaniv (2017) formalized selective classification for deep neural networks around the trade-off between coverage and risk. SelectiveNet (Geifman and El-Yaniv, 2019) later integrated prediction and rejection into a single learned architecture.

This literature motivates the central evaluation principle used here: a useful abstention signal should improve the ordering of predictions so that low-risk examples are retained before high-risk examples as coverage decreases.

The present study differs from methods that train a dedicated selective architecture. It evaluates post-hoc ranking signals on predictions from a fixed pretrained QA model.

## 2. Confidence Calibration

Confidence scores need not correspond to empirical correctness probabilities.

Guo et al. (2017) showed that modern neural networks can be poorly calibrated and found temperature scaling to be a simple and effective post-hoc calibration method.

More recent work on post-hoc selective classification has also emphasized that the quality of the confidence estimator can strongly determine selective performance (Cattelan and Silva, 2024).

These findings motivate the use of a calibrated-confidence baseline rather than treating raw model confidence as a sufficient comparison.

In this project, temperature scaling is fitted on a separate calibration split and frozen before held-out test evaluation.

## 3. Selective Question Answering

Selective QA has direct prior work.

Kamath, Jia, and Liang (2020) studied selective question answering under domain shift. They showed that QA systems need abstention policies that preserve accuracy while maximizing coverage, and that raw softmax probabilities can be unreliable under distribution shift. Their work used learned calibration signals to improve selective QA in mixtures of in-domain and out-of-domain data.

This prior work is especially important for positioning the present project. The contribution here is not the idea of selective QA itself. Instead, this study asks a narrower question within a controlled in-domain SQuAD v2 setting:

> Do question-aware semantic verification or self-verification signals improve selective ranking over a strong calibrated-confidence baseline?

The present work also differs by holding the QA backbone fixed, using separate calibration and held-out evaluation partitions, comparing five post-hoc ranking methods, and reporting global AURC with paired bootstrap uncertainty.

## 4. Answerability and SQuAD v2

Rajpurkar, Jia, and Liang (2018) introduced SQuAD v2 by adding adversarially constructed unanswerable questions to the original SQuAD task.

The benchmark requires a system not only to extract an answer when evidence exists, but also to avoid unsupported guesses when the context does not contain an answer.

This makes SQuAD v2 suitable for controlled study of confidence, answerability, and abstention, although conclusions from a single benchmark cannot establish cross-domain reliability.

## 5. Evidence and Claim Verification

FEVER (Thorne et al., 2018) established a large-scale textual claim-verification setting with `SUPPORTED`, `REFUTED`, and `NOT ENOUGH INFO` labels and associated textual evidence.

Although the present project is not a FEVER-style fact-checking system, FEVER motivates the separation between answer generation and evidence-based verification.

The question-aware verifier in this repository converts a question-answer pair into a declarative claim and evaluates that claim against the provided context using natural-language inference. This creates an additional failure mode absent from direct confidence scoring: claim generation itself can be invalid or can fail to preserve important answer information.

## 6. Truthfulness and Language-Model Abstention

TruthfulQA (Lin, Hilton, and Evans, 2021) demonstrated that language-model answers can reproduce common false beliefs and that benchmark accuracy alone does not imply truthfulness.

More directly, Wen et al. (2024) survey abstention in large language models across methods, benchmarks, and evaluation perspectives. Madhusudhan et al. (2024) introduce an abstention-focused evaluation framework and the Abstain-QA dataset, showing that even strong language models can struggle to withhold answers reliably.

These works concern broader generative-LLM settings and therefore should not be treated as direct evidence about the extractive QA model evaluated in this repository. They instead provide wider motivation for studying when a model should refrain from answering.

## 7. Recent Directions

Recent work continues to move beyond heuristic abstention.

COIN (Wang et al., 2025) studies selective question answering with statistically controlled risk guarantees rather than relying only on heuristic uncertainty thresholds.

Abstain-R1 (Zhai, Liang, and Kang, 2026) studies learned abstention together with post-refusal clarification using reinforcement learning with verifiable rewards.

These directions highlight an important distinction from the present work. This repository evaluates fixed post-hoc ranking signals and does not train an abstention policy, optimize clarification behavior, or provide formal risk guarantees.

## 8. Synthesis and Research Position

Prior work establishes that:

1. selective prediction is a well-developed framework for trading coverage against risk;
2. confidence calibration can materially affect selective decisions;
3. selective QA predates this project and has been studied under domain shift;
4. textual verification can provide evidence-support signals distinct from model confidence;
5. recent abstention research includes learned policies and statistical risk guarantees.

The research contribution of this project is therefore deliberately narrow.

It provides a reproducible controlled comparison of five ranking methods on a fixed extractive QA system:

1. calibrated confidence only;
2. question-aware semantic verification only;
3. calibrated confidence plus question-aware semantic verification;
4. self-verification only;
5. calibrated confidence plus self-verification.

The study evaluates whether the additional verification signals improve the complete selective ranking, rather than assuming that a verifier is beneficial merely because its labels correlate with correctness.

## 9. Research Gap Addressed

The project addresses the following focused empirical question:

> In a fixed SQuAD v2 extractive-QA setting with calibration separated from held-out evaluation, do semantic or self-verification signals improve global selective ranking beyond a strong calibrated-confidence baseline?

This is an empirical comparison rather than a claim that the underlying problem has not previously been studied.

The final result is negative but informative: under the implemented protocol, calibrated confidence achieves lower global AURC than each evaluated verification-only or combined ranking method. This finding is limited to the benchmark, QA backbone, verification implementations, and scoring rules tested here.

## Key References

1. Geifman, Y., & El-Yaniv, R. (2017). *Selective Classification for Deep Neural Networks*.
2. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML.
3. Rajpurkar, P., Jia, R., & Liang, P. (2018). *Know What You Don't Know: Unanswerable Questions for SQuAD*. ACL.
4. Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). *FEVER: a Large-scale Dataset for Fact Extraction and VERification*. NAACL.
5. Geifman, Y., & El-Yaniv, R. (2019). *SelectiveNet: A Deep Neural Network with an Integrated Reject Option*. ICML.
6. Kamath, A., Jia, R., & Liang, P. (2020). *Selective Question Answering under Domain Shift*. ACL.
7. Lin, S., Hilton, J., & Evans, O. (2021). *TruthfulQA: Measuring How Models Mimic Human Falsehoods*.
8. Cattelan, L. F. P., & Silva, D. (2024). *How to Fix a Broken Confidence Estimator: Evaluating Post-hoc Methods for Selective Classification with Deep Neural Networks*. UAI.
9. Wen, B., Yao, J., Feng, S., Xu, C., Tsvetkov, Y., Howe, B., & Wang, L. L. (2024). *Know Your Limits: A Survey of Abstention in Large Language Models*.
10. Madhusudhan, N., Madhusudhan, S. T., Yadav, V., & Hashemi, M. (2024). *Do LLMs Know When to NOT Answer? Investigating Abstention Abilities of Large Language Models*.
11. Wang, Z., Duan, J., Wang, Q., Zhu, X., Chen, T., Shi, X., & Xu, K. (2025). *COIN: Uncertainty-Guarding Selective Question Answering for Foundation Models with Provable Risk Guarantees*.
12. Zhai, H., Liang, J., & Kang, D. (2026). *Abstain-R1: Calibrated Abstention and Post-Refusal Clarification via Verifiable RL*. Findings of ACL.
