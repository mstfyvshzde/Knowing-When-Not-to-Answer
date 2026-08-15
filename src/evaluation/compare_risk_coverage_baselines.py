"""
Compare prototype confidence, hybrid, and random risk-coverage diagnostics.

This module combines outputs produced by two earlier evaluation pipelines:

1. confidence and hybrid risk-coverage curves;
2. multi-seed random-abstention results.

The script creates comparison tables at target coverage levels and reports
descriptive differences between the three prototype baselines.

Important
---------
This file is a legacy prototype comparison, not the source of the project's
canonical final AURC results.

The confidence and hybrid curves are produced from ranked score prefixes,
whereas the random baseline is available only at a small set of predefined
coverage levels and across multiple random seeds.

The `random_aurc` value calculated here therefore uses trapezoidal integration
over those sparse coverage points. It should be treated as a descriptive
prototype diagnostic rather than directly compared with the canonical discrete
prefix-based AURC used by the final selective-QA experiments.

Final paper conclusions should rely on the dedicated held-out ranking evaluator
rather than the `best_aurc_method` result produced by this legacy script.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

RISK_COVERAGE_DIR = Path(
    "outputs/evaluation/risk_coverage"
)

CONFIDENCE_CURVE_PATH = (
    RISK_COVERAGE_DIR
    / "confidence_risk_coverage_curve.csv"
)

HYBRID_CURVE_PATH = (
    RISK_COVERAGE_DIR
    / "hybrid_risk_coverage_curve.csv"
)

RISK_COVERAGE_SUMMARY_PATH = (
    RISK_COVERAGE_DIR
    / "risk_coverage_summary.json"
)

RANDOM_RESULTS_PATH = Path(
    "outputs/tables/random_abstention_multi_seed_metrics.json"
)

OUTPUT_CSV_PATH = (
    RISK_COVERAGE_DIR
    / "baseline_risk_coverage_comparison.csv"
)

OUTPUT_JSON_PATH = (
    RISK_COVERAGE_DIR
    / "baseline_risk_coverage_comparison.json"
)


TARGET_COVERAGES = tuple(
    index / 10
    for index in range(
        1,
        11,
    )
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    """
    Load a JSON object required by the comparison script.

    The function fails early when an expected upstream artifact is missing or
    does not contain a top-level dictionary.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: "
            f"{path}"
        )

    with path.open(
        encoding="utf-8",
    ) as file:
        data = json.load(
            file
        )

    if not isinstance(
        data,
        dict,
    ):
        raise TypeError(
            f"Expected a JSON object in "
            f"{path}"
        )

    return data


def load_risk_coverage_curve(
    path: Path,
) -> list[dict[str, float]]:
    """
    Load one confidence or hybrid risk-coverage curve.

    Only the fields needed for cross-baseline comparison are retained:

    - answered count
    - coverage
    - selective accuracy
    - selective risk

    The source curve may also contain `minimum_score`. The zero-coverage
    plotting origin has no score and is therefore written as an empty CSV
    field. Because that value is not needed here, it is intentionally ignored.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: "
            f"{path}"
        )

    rows: list[
        dict[str, float]
    ] = []

    with path.open(
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        required_columns = {
            "answered",
            "coverage",
            "selective_accuracy",
            "selective_risk",
        }

        if reader.fieldnames is None:
            raise ValueError(
                f"CSV file has no header: "
                f"{path}"
            )

        missing_columns = (
            required_columns
            - set(
                reader.fieldnames
            )
        )

        if missing_columns:
            missing = ", ".join(
                sorted(
                    missing_columns
                )
            )

            raise ValueError(
                f"Missing required columns "
                f"in {path}: {missing}"
            )

        for row in reader:
            parsed_row = {
                "answered": float(
                    row[
                        "answered"
                    ]
                ),
                "coverage": float(
                    row[
                        "coverage"
                    ]
                ),
                "selective_accuracy": float(
                    row[
                        "selective_accuracy"
                    ]
                ),
                "selective_risk": float(
                    row[
                        "selective_risk"
                    ]
                ),
            }

            if any(
                not math.isfinite(
                    value
                )
                for value
                in parsed_row.values()
            ):
                raise ValueError(
                    "Risk-coverage curve contains "
                    f"a non-finite value in {path}."
                )

            rows.append(
                parsed_row
            )

    if not rows:
        raise ValueError(
            f"No curve rows found in "
            f"{path}"
        )

    return sorted(
        rows,
        key=lambda row: (
            row[
                "coverage"
            ]
        ),
    )


def nearest_curve_point(
    curve: list[dict[str, float]],
    target_coverage: float,
) -> dict[str, float]:
    """
    Return the curve point nearest to a requested coverage.

    This is an approximate coverage match. Ranked confidence and hybrid curves
    contain one point per answered prefix, so their actual selected coverage can
    differ slightly from the requested target.

    Equal-distance ties prefer the lower coverage point.
    """

    if not curve:
        raise ValueError(
            "Risk-coverage curve "
            "cannot be empty."
        )

    if not (
        0.0
        <= target_coverage
        <= 1.0
    ):
        raise ValueError(
            "target_coverage must lie "
            "between 0 and 1."
        )

    return min(
        curve,
        key=lambda row: (
            abs(
                row[
                    "coverage"
                ]
                - target_coverage
            ),
            row[
                "coverage"
            ],
        ),
    )


def load_random_results(
    path: Path,
) -> dict[
    float,
    dict[str, float],
]:
    """
    Load multi-seed random-abstention summaries indexed by target coverage.

    Each coverage level contains the mean selective risk/accuracy across random
    seeds together with the observed variability.

    These values describe a stochastic control baseline and are not a ranked
    score curve.
    """

    data = load_json(
        path
    )

    raw_results = data.get(
        "results"
    )

    if not isinstance(
        raw_results,
        list,
    ):
        raise TypeError(
            f"Expected 'results' to be "
            f"a list in {path}"
        )

    results: dict[
        float,
        dict[str, float],
    ] = {}

    for item in raw_results:
        if not isinstance(
            item,
            dict,
        ):
            raise TypeError(
                "Every random result must "
                f"be an object in {path}"
            )

        coverage = item.get(
            "coverage"
        )

        summary = item.get(
            "summary"
        )

        if not isinstance(
            coverage,
            (int, float),
        ):
            raise TypeError(
                "Random result has invalid "
                f"coverage in {path}"
            )

        if not isinstance(
            summary,
            dict,
        ):
            raise TypeError(
                "Random result has invalid "
                f"summary in {path}"
            )

        result = {
            "coverage": float(
                summary.get(
                    "actual_coverage",
                    coverage,
                )
            ),
            "selective_accuracy": float(
                summary[
                    "mean_answer_accuracy"
                ]
            ),
            "selective_risk": float(
                summary[
                    "mean_selective_risk"
                ]
            ),
            "risk_std": float(
                summary[
                    "selective_risk_std"
                ]
            ),
            "minimum_risk": float(
                summary[
                    "min_selective_risk"
                ]
            ),
            "maximum_risk": float(
                summary[
                    "max_selective_risk"
                ]
            ),
            "number_of_seeds": float(
                summary[
                    "number_of_seeds"
                ]
            ),
        }

        if any(
            not math.isfinite(
                value
            )
            for value
            in result.values()
        ):
            raise ValueError(
                "Random baseline contains "
                f"a non-finite metric in {path}."
            )

        results[
            round(
                float(
                    coverage
                ),
                10,
            )
        ] = result

    if not results:
        raise ValueError(
            "Random baseline results "
            "cannot be empty."
        )

    return results


def trapezoidal_area(
    points: list[
        tuple[
            float,
            float,
        ]
    ],
) -> float:
    """
    Approximate area under sparse coverage-risk points using trapezoids.

    This helper is used only for the legacy random-baseline diagnostic because
    random abstention is evaluated at predefined coverage levels rather than at
    every ranked prefix.

    It is not the canonical discrete AURC implementation used by the project's
    final ranking experiments.
    """

    if len(
        points
    ) < 2:
        raise ValueError(
            "At least two points are required "
            "for trapezoidal integration."
        )

    ordered_points = sorted(
        points
    )

    area = 0.0

    for index in range(
        1,
        len(
            ordered_points
        ),
    ):
        (
            previous_x,
            previous_y,
        ) = ordered_points[
            index - 1
        ]

        (
            current_x,
            current_y,
        ) = ordered_points[
            index
        ]

        width = (
            current_x
            - previous_x
        )

        if width < 0.0:
            raise RuntimeError(
                "Coverage points are not "
                "monotonically ordered."
            )

        average_height = (
            previous_y
            + current_y
        ) / 2.0

        area += (
            width
            * average_height
        )

    return area


def build_comparison_rows(
    confidence_curve: list[
        dict[str, float]
    ],
    hybrid_curve: list[
        dict[str, float]
    ],
    random_results: dict[
        float,
        dict[str, float],
    ],
) -> list[
    dict[str, float]
]:
    """
    Build descriptive comparisons at predefined target coverage levels.

    Confidence and hybrid values are selected from their nearest ranked curve
    points. Random values come from the corresponding multi-seed coverage run.

    Because these are not guaranteed to have exactly identical realized
    coverage, risk differences should be interpreted as approximate prototype
    comparisons rather than strict matched-coverage effects.
    """

    comparison_rows: list[
        dict[str, float]
    ] = []

    for target_coverage in (
        TARGET_COVERAGES
    ):
        confidence = (
            nearest_curve_point(
                confidence_curve,
                target_coverage,
            )
        )

        hybrid = (
            nearest_curve_point(
                hybrid_curve,
                target_coverage,
            )
        )

        random_key = round(
            target_coverage,
            10,
        )

        if (
            random_key
            not in random_results
        ):
            raise ValueError(
                "Random baseline does not "
                "contain coverage "
                f"{target_coverage:.1f}"
            )

        random_result = (
            random_results[
                random_key
            ]
        )

        comparison_rows.append(
            {
                "target_coverage": (
                    target_coverage
                ),
                "confidence_coverage": (
                    confidence[
                        "coverage"
                    ]
                ),
                "confidence_accuracy": (
                    confidence[
                        "selective_accuracy"
                    ]
                ),
                "confidence_risk": (
                    confidence[
                        "selective_risk"
                    ]
                ),
                "hybrid_coverage": (
                    hybrid[
                        "coverage"
                    ]
                ),
                "hybrid_accuracy": (
                    hybrid[
                        "selective_accuracy"
                    ]
                ),
                "hybrid_risk": (
                    hybrid[
                        "selective_risk"
                    ]
                ),
                "random_coverage": (
                    random_result[
                        "coverage"
                    ]
                ),
                "random_mean_accuracy": (
                    random_result[
                        "selective_accuracy"
                    ]
                ),
                "random_mean_risk": (
                    random_result[
                        "selective_risk"
                    ]
                ),
                "random_risk_std": (
                    random_result[
                        "risk_std"
                    ]
                ),
                "random_minimum_risk": (
                    random_result[
                        "minimum_risk"
                    ]
                ),
                "random_maximum_risk": (
                    random_result[
                        "maximum_risk"
                    ]
                ),
                "confidence_vs_random_risk_improvement": (
                    random_result[
                        "selective_risk"
                    ]
                    - confidence[
                        "selective_risk"
                    ]
                ),
                "hybrid_vs_random_risk_improvement": (
                    random_result[
                        "selective_risk"
                    ]
                    - hybrid[
                        "selective_risk"
                    ]
                ),
                "hybrid_minus_confidence_risk": (
                    hybrid[
                        "selective_risk"
                    ]
                    - confidence[
                        "selective_risk"
                    ]
                ),
            }
        )

    return comparison_rows


def write_csv(
    path: Path,
    rows: list[
        dict[str, float]
    ],
) -> None:
    """Write comparison rows as deterministic UTF-8 CSV."""

    if not rows:
        raise ValueError(
            "Cannot write an empty "
            "comparison CSV."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
            lineterminator="\n",
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def build_output_summary(
    rows: list[
        dict[str, float]
    ],
    risk_coverage_summary: dict[
        str,
        Any,
    ],
    random_results: dict[
        float,
        dict[str, float],
    ],
) -> dict[str, Any]:
    """
    Build the legacy three-baseline comparison summary.

    Confidence and hybrid AURCs are read from the upstream risk-coverage
    evaluator.

    Random "AURC" is only a trapezoidal approximation over the sparse random
    coverage grid. It is retained for historical compatibility and descriptive
    inspection, not as a canonical final AURC comparison.
    """

    random_points = [
        (
            0.0,
            0.0,
        )
    ]

    random_points.extend(
        (
            result[
                "coverage"
            ],
            result[
                "selective_risk"
            ],
        )
        for _, result
        in sorted(
            random_results.items()
        )
    )

    random_aurc = (
        trapezoidal_area(
            random_points
        )
    )

    confidence_aurc = float(
        risk_coverage_summary[
            "confidence_aurc"
        ]
    )

    hybrid_aurc = float(
        risk_coverage_summary[
            "hybrid_aurc"
        ]
    )

    best_method_by_coverage: list[
        dict[str, Any]
    ] = []

    for row in rows:
        risks = {
            "confidence": (
                row[
                    "confidence_risk"
                ]
            ),
            "hybrid": (
                row[
                    "hybrid_risk"
                ]
            ),
            "random": (
                row[
                    "random_mean_risk"
                ]
            ),
        }

        best_method = min(
            risks,
            key=risks.get,
        )

        best_method_by_coverage.append(
            {
                "coverage": (
                    row[
                        "target_coverage"
                    ]
                ),
                "best_method": (
                    best_method
                ),
                "lowest_risk": (
                    risks[
                        best_method
                    ]
                ),
            }
        )

    aurc_values = {
        "confidence": (
            confidence_aurc
        ),
        "hybrid": (
            hybrid_aurc
        ),
        "random": (
            random_aurc
        ),
    }

    best_aurc_method = min(
        aurc_values,
        key=aurc_values.get,
    )

    return {
        "analysis_type": (
            "legacy_prototype_baseline_comparison"
        ),
        "interpretation_note": (
            "Random AURC is a trapezoidal "
            "approximation over sparse random "
            "coverage points and is not the "
            "canonical prefix-based AURC used "
            "for final project conclusions."
        ),
        "inputs": {
            "confidence_curve": str(
                CONFIDENCE_CURVE_PATH
            ),
            "hybrid_curve": str(
                HYBRID_CURVE_PATH
            ),
            "risk_coverage_summary": str(
                RISK_COVERAGE_SUMMARY_PATH
            ),
            "random_results": str(
                RANDOM_RESULTS_PATH
            ),
        },
        "total_predictions": (
            risk_coverage_summary.get(
                "total_predictions"
            )
        ),
        "target_coverages": list(
            TARGET_COVERAGES
        ),
        "random_seed_count": int(
            next(
                iter(
                    random_results.values()
                )
            )[
                "number_of_seeds"
            ]
        ),
        "aurc": (
            aurc_values
        ),
        "best_aurc_method": (
            best_aurc_method
        ),
        "best_aurc": (
            aurc_values[
                best_aurc_method
            ]
        ),
        "aurc_improvement_over_random": {
            "confidence": (
                random_aurc
                - confidence_aurc
            ),
            "hybrid": (
                random_aurc
                - hybrid_aurc
            ),
        },
        "best_method_by_coverage": (
            best_method_by_coverage
        ),
        "comparison_rows": (
            rows
        ),
    }


def main() -> None:
    """
    Load upstream prototype outputs and create comparison CSV/JSON artifacts.
    """

    confidence_curve = (
        load_risk_coverage_curve(
            CONFIDENCE_CURVE_PATH
        )
    )

    hybrid_curve = (
        load_risk_coverage_curve(
            HYBRID_CURVE_PATH
        )
    )

    random_results = (
        load_random_results(
            RANDOM_RESULTS_PATH
        )
    )

    risk_coverage_summary = (
        load_json(
            RISK_COVERAGE_SUMMARY_PATH
        )
    )

    comparison_rows = (
        build_comparison_rows(
            confidence_curve=(
                confidence_curve
            ),
            hybrid_curve=(
                hybrid_curve
            ),
            random_results=(
                random_results
            ),
        )
    )

    output_summary = (
        build_output_summary(
            rows=(
                comparison_rows
            ),
            risk_coverage_summary=(
                risk_coverage_summary
            ),
            random_results=(
                random_results
            ),
        )
    )

    write_csv(
        OUTPUT_CSV_PATH,
        comparison_rows,
    )

    OUTPUT_JSON_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_JSON_PATH.write_text(
        json.dumps(
            output_summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "Legacy baseline risk-coverage "
        "comparison completed."
    )

    print(
        f"CSV:  "
        f"{OUTPUT_CSV_PATH}"
    )

    print(
        f"JSON: "
        f"{OUTPUT_JSON_PATH}"
    )

    print()

    print(
        "Prototype AURC diagnostics:"
    )

    for method, aurc in (
        output_summary[
            "aurc"
        ].items()
    ):
        print(
            f"  {method}: "
            f"{aurc:.6f}"
        )

    print()

    print(
        "Legacy best AURC diagnostic: "
        f"{output_summary['best_aurc_method']}"
    )

    print(
        "Note: do not use the random "
        "trapezoidal AURC as a canonical "
        "final-project AURC comparison."
    )


if __name__ == "__main__":
    main()