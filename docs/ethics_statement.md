# Ethics Statement

## Research Purpose

This project studies selective question answering: whether a QA system can reduce selective risk by ranking higher-reliability predictions ahead of lower-reliability predictions and abstaining as coverage decreases.

The work is a controlled benchmark study. It does not establish that the evaluated system is safe, trustworthy, or suitable for deployment in high-stakes settings.

## Potential Benefits

Selective prediction can be useful when an incorrect answer is more costly than abstention. This study contributes a reproducible comparison between calibrated confidence and additional semantic/self-verification signals.

A key benefit of the study is methodological transparency: negative results, failure modes, uncertainty estimates, and limitations are reported rather than hidden.

## Potential Risks

Verification and abstention can create their own failure modes.

A verifier can:

- reject correct answers
- support incorrect answers
- introduce additional model errors
- create a false impression that verification guarantees correctness
- reduce useful coverage without improving global selective ranking

The final error analysis demonstrates that these failure modes occur in the evaluated system.

## Interpretation of Verification

Verifier labels such as `ENTAILMENT`, `SUPPORTED`, `UNCERTAIN`, or `REJECTED` are diagnostic model outputs. They are not guarantees of factual correctness or safety.

For example, the held-out analysis contains incorrect QA predictions labeled as semantically supported, as well as correct QA predictions receiving low semantic-support scores.

The repository therefore avoids presenting verifier outputs as authoritative judgments.

## Fairness

The final experiments do not include a dedicated demographic fairness evaluation.

SQuAD v2 is not used here to support claims about demographic parity, disparate impact, or fairness across protected groups. No broad fairness claim is made from aggregate benchmark performance.

A separate dataset and evaluation design would be required for such conclusions.

## Data and Human Subjects

The study uses an existing public question-answering benchmark and generated model outputs. No new human-subject experiment, user study, or collection of sensitive participant data was conducted for the final evaluation.

Dataset preparation, splits, preprocessing, and evaluation procedures are documented in the repository for reproducibility.

## High-Stakes Applications

The evaluated framework is not presented as suitable for medical, legal, financial, safety-critical, or other high-stakes decision-making.

Performance on SQuAD v2 does not establish real-world safety, factual reliability, or robustness under distribution shift.

## Transparency and Negative Results

The final report records:

- the pretrained QA and verification models
- the calibration/test separation
- deterministic seeds
- fixed score-combination rules
- held-out AURC and normalized AURC results
- paired bootstrap uncertainty intervals
- verifier failure modes
- methodological limitations

The main hypothesis is not supported by the final experiment: the evaluated semantic and self-verification signals do not improve global selective ranking over calibrated confidence in this setting.

This negative result is reported directly rather than reframed as a positive performance claim.

## Responsible Claims

The project does not claim to:

- solve hallucination
- guarantee factual correctness
- establish general LLM reliability
- guarantee safe abstention
- replace human verification
- generalize beyond the tested benchmark, QA backbone, and verifier implementations

All conclusions are restricted to the experimental setting documented in this repository.
