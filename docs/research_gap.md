# Research Gap

## Provisional Gap

Existing research separately studies:

* confidence estimation
* hallucination detection
* evidence verification
* answerability prediction
* selective prediction and abstention

However, this project will investigate whether these components can be combined into a single, reproducible decision framework for question answering.

The proposed system will not only predict an answer. It will decide whether to:

* answer directly
* request additional verification
* abstain from answering

## Research Need

A useful reliability framework should evaluate more than final-answer accuracy.

It should also measure:

* whether the answer is supported by evidence
* whether confidence reflects actual correctness
* whether abstention reduces high-confidence errors
* how reliability changes as answer coverage decreases
* which verification component contributes most to performance

## Intended Contribution

The project aims to provide an experimentally controlled comparison between:

1. a raw answer-generation baseline
2. a confidence-based abstention baseline
3. an evidence-based verification system
4. a combined self-verification framework

## Important Limitation

This research gap is provisional.

It will be revised after the literature review confirms:

* which parts have already been studied
* which benchmark settings remain underexplored
* whether the proposed combination provides a meaningful contribution
