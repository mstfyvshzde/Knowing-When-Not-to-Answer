# Changelog

All notable changes to this project are documented in this file.

The format follows the principles of **Keep a Changelog**, and release
versions use **Semantic Versioning** where applicable.

---

## [Unreleased]

### Added

- Final held-out selective question answering evaluation pipeline.
- Five canonical ranking methods:
  - Confidence only.
  - Question-aware semantic V2.
  - Confidence + question-aware semantic V2.
  - Answer-support/self verifier only.
  - Confidence + answer-support/self verifier.
- Question-aware QA-to-declarative verification using:
  - `domenicrosati/QA2D-t5-base`.
  - `FacebookAI/roberta-large-mnli`.
- Template-based answer-support NLI verification using
  `FacebookAI/roberta-large-mnli`.
- Native SQuAD v2 no-answer baseline using the same QA backbone as the
  forced-answer experiments.
- Calibration-only fusion-weight tuning.
- Nested held-out sample-size evaluation for:
  - N=200.
  - N=500.
  - N=1000.
  - N=2000.
  - N=3000.
- Paired non-parametric bootstrap uncertainty analysis for final AURC results.
- Tie-aware correct-vs-incorrect pair ranking analysis.
- High-entailment and verifier-error diagnostic analyses.
- Publication-facing calibration and ablation figures.

### Changed

- Standardized the project-wide reproducibility seed to `17`.
- Aligned dataset configuration with the implemented SQuAD v2 preparation
  protocol.
- Documented the deterministic 50/50 calibration and held-out partition of the
  official SQuAD v2 validation split.
- Clarified that the pretrained QA backbone is
  `deepset/roberta-base-squad2`.
- Clarified the distinction between:
  - forced-answer candidate correctness,
  - routing decisions,
  - native no-answer behavior.
- Made AURC the primary final selective-ranking metric.
- Added normalized AURC and matched-coverage reporting.
- Defined risk-coverage curves from ranked prefixes rather than an arbitrary
  threshold grid.
- Standardized final fixed fusion rules as equal-weight geometric means.
- Clarified that the question-aware verifier and answer-support/self verifier
  use the same RoBERTa-large-MNLI backbone and should not be described as
  independent verifier models.
- Updated configuration documentation to distinguish final ranking experiments
  from earlier prototype threshold-routing components.
- Improved repository-wide validation and reproducibility documentation.
- Moved publication-facing figures to `assets/figures/`.

### Fixed

- Corrected answer normalization for English articles in evaluation utilities.
- Corrected an unanswerable-example correctness edge case where
  punctuation-only predictions could normalize to an empty string and be
  incorrectly counted as correct.
- Strengthened explicit ANSWER/ABSTAIN decision handling for held-out
  evaluation.
- Replaced arbitrary rank-position analysis with tie-aware pairwise ranking
  diagnostics.
- Ensured deterministic ordering is used consistently for score-tie resolution
  in final evaluation.
- Corrected stale field names, typographical errors, and outdated prototype
  assumptions in tests and documentation.
- Removed obsolete tests that targeted an earlier decision-engine API.
- Removed stale evaluation artifacts that no longer represented the canonical
  final protocol.

### Reproducibility

- Calibration parameters are fitted only on the calibration split.
- Held-out test labels are not used for temperature, threshold, or fusion
  parameter selection.
- Final nested sample ordering uses seed `17`.
- Final paired bootstrap analysis uses:
  - 5,000 bootstrap replicates.
  - bootstrap seed `17`.
  - deterministic ordering seed `17`.
- Added safeguards and metadata describing score definitions, ordering rules,
  and evaluation semantics.

---

## [1.0.1] - 2026-07-27

### Added

- Expanded project documentation.
- System architecture diagram.
- Decision-engine diagram.
- Research pipeline visualization.
- Contribution guidelines.
- Code of Conduct.
- Security policy.
- Changelog.
- GitHub Actions workflow.
- Repository badges.
- Release and citation metadata.

### Changed

- Improved README structure and navigation.
- Improved installation instructions.
- Expanded reproducibility documentation.
- Improved citation information.
- Improved repository organization and archival metadata.

### Notes

- This release primarily focused on documentation, repository quality, and
  archival preparation.
- Experimental results and finalized benchmark evaluations were not yet part of
  this release.

---

## [1.0.0] - 2026-07-27

### Added

- Initial research repository structure.
- Modular selective question answering framework.
- Confidence estimation module.
- Confidence calibration pipeline.
- Evidence verification module.
- Prototype decision engine supporting answer, verify, and abstain actions.
- Evaluation metrics and benchmarking framework.
- Experiment configuration system.
- Reproducibility scripts.
- Documentation and repository architecture diagrams.
- GitHub Actions continuous integration.
- Unit test framework.
- Contribution guidelines.
- Code of Conduct.
- Security policy.
- MIT License.

### Documentation

- Initial README.
- Installation guide.
- Project pipeline documentation.
- System architecture documentation.
- Decision-engine documentation.
- Experimental-results placeholders.

### Maintenance

- Initial public release.
- Repository prepared for subsequent research development.