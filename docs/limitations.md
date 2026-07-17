# Limitations

## Benchmark Scope

The initial experiments will use controlled question-answering benchmarks.

Performance on these datasets may not represent reliability in open-domain, conversational, or real-world settings.

## Model Dependence

The results may depend on the selected answer-generation model and verifier.

A method that performs well with one model may not generalize to models with different architectures, sizes, or training data.

## Confidence Limitations

Model confidence is not a direct measurement of truth.

A system may assign high confidence to an incorrect answer or low confidence to a correct answer.

Calibration can reduce this mismatch but cannot eliminate it completely.

## Evidence Limitations

The framework can only verify answers against the evidence provided to it.

If the context is incomplete, misleading, outdated, or incorrect, the verifier may still produce an unreliable decision.

## Abstention Trade-Off

Increasing abstention may reduce incorrect answers but also reduce usefulness.

A system can appear safer simply by refusing to answer too often.

For this reason, results must be compared at similar coverage levels.

## Labeling Limitations

The categories `SUPPORTED`, `UNSUPPORTED`, and `UNCERTAIN` may involve ambiguous cases.

Clear annotation rules and inter-annotator agreement will be required if manual labels are introduced.

## Dataset Bias

Benchmark datasets may contain annotation artifacts, repeated patterns, or domain-specific biases.

Models may exploit these patterns without learning general reliability.

## Verification Cost

Additional verification steps increase computational cost and response latency.

The project will measure reliability but may not fully optimize efficiency.

## Safety Claims

This project does not demonstrate that the system is safe for medical, legal, financial, or other high-stakes applications.

It evaluates reliability only under the tested experimental conditions.

## Generalization

The project will not claim to solve hallucination or uncertainty estimation universally.

Any conclusions will be limited to the selected models, datasets, metrics, and experimental setup.
