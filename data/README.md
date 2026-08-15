# Dataset Documentation

## Purpose

This directory contains the dataset files and preprocessing artifacts used by the project.

The project uses **SQuAD v2** as its benchmark. Large dataset files are not committed to the repository; they can be downloaded and prepared reproducibly using the scripts in `src/data/`.

## Directory Structure

```text
data/
├── raw/          Original downloaded SQuAD v2 dataset
├── processed/    Experiment-ready dataset and split metadata
├── external/     Reserved for additional external resources
└── README.md
```

## Benchmark

The final benchmark used in this project is **SQuAD v2** (`rajpurkar/squad_v2`).

SQuAD v2 contains both:

- answerable questions with reference answers
- unanswerable questions for which the provided context does not contain a valid answer

This makes it suitable for studying selective question answering and abstention.

## Experimental Splits

The original SQuAD v2 training split is preserved.

Because gold labels for the official SQuAD v2 test split are not publicly available, the original validation split is deterministically divided into two disjoint subsets:

- **Calibration split:** used for confidence calibration and parameter selection
- **Held-out test split:** reserved for final evaluation

The split is stratified by answerability so both subsets preserve approximately the same proportion of answerable and unanswerable examples.

The split uses:

```text
Seed: 17
Calibration fraction: 0.50
```

Held-out test labels are not used for calibration or parameter tuning.

## Training Data

The original SQuAD v2 training split is retained in the processed dataset for completeness.

The final experiments in this repository use a pretrained extractive QA backbone and do not train or fine-tune that QA model as part of the dataset preparation pipeline.

## Processing Pipeline

Download the raw dataset with:

```bash
python -m src.data.download_data
```

Prepare the experiment-ready dataset with:

```bash
python -m src.data.prepare_data
```

Dataset preparation:

1. loads the raw SQuAD v2 dataset
2. adds an `is_answerable` field to every example
3. stratifies the original validation split by answerability
4. creates separate calibration and held-out test subsets
5. preserves the original training split
6. saves dataset statistics and split metadata

The raw downloaded dataset is not modified.

## Answerability Label

Each processed example contains:

```text
is_answerable = 1
```

when at least one gold answer is available, and:

```text
is_answerable = 0
```

when the example is unanswerable.

Verifier labels such as `ENTAILMENT`, `CONTRADICTION`, `NEUTRAL`, `SUPPORTED`, `UNCERTAIN`, and `REJECTED` are generated later by the verification pipeline and are not dataset annotations created during preprocessing.

## Generated Metadata

The processed dataset directory contains:

```text
statistics.json
split_manifest.json
```

`statistics.json` records the number of answerable and unanswerable examples in each split.

`split_manifest.json` records the source dataset, random seed, calibration fraction, and final split sizes.

## Leakage Prevention

The experimental protocol separates calibration from final evaluation.

The project does not:

- fit calibration parameters using held-out test labels
- select fusion parameters using held-out test labels
- manually modify final test predictions
- move examples between calibration and test after inspecting final results

Calibration parameters are selected on the calibration split and then frozen before held-out evaluation.

## Reproducibility

The calibration/test split is deterministic because it uses a fixed random seed.

Running the same preprocessing code with the same dataset version and seed reproduces the same split assignment.

The project uses:

```text
Seed: 17
Source: rajpurkar/squad_v2
```

## Data Storage

Large raw and processed dataset files are kept outside version control.

Only code, documentation, metadata conventions, and selected experiment artifacts required for reproducibility are stored in the repository.