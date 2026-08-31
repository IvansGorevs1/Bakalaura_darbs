from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "judge_validation"
    / "consistency_crossjudge_72"
)

SCORE_METRICS = [
    "faithfulness",
    "numerical_accuracy",
    "completeness",
    "coherence",
]

COUNT_METRICS = [
    "factual_total",
    "numerical_total",
]

ALL_METRICS = SCORE_METRICS + COUNT_METRICS

RATIO_METRICS = {
    "faithfulness",
    "numerical_accuracy",
}

ORDINAL_METRICS = {
    "completeness",
    "coherence",
}

MAX_SCORE = {
    "faithfulness": 1.0,
    "numerical_accuracy": 1.0,
    "completeness": 4.0,
    "coherence": 4.0,
}

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "numerical_accuracy": "Numerical Accuracy",
    "completeness": "Completeness",
    "coherence": "Coherence",
    "factual_total": "Factual Claim Count",
    "numerical_total": "Numerical Claim Count",
}

RUN_PAIRS = [
    (1, 2),
    (1, 3),
    (2, 3),
]

FLOAT_TOLERANCE = 1e-9

RATIO_TOLERANCE = 0.05

ORDINAL_TOLERANCE = 1.0

COUNT_TOLERANCE = 1.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(

        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(
            csv.DictReader(
                f
            )
        )


def normalize_run_rows(
    rows: list[dict],
    run_number: int,
) -> dict[str, dict]:
    if len(rows) != 72:
        raise RuntimeError(
            f"Run {run_number}: expected 72 rows, found {len(rows)}."
        )

    normalized = {}

    for row in rows:
        selection_id = str(
            row.get(
                "selection_id",
                "",
            )
        ).strip()

        if not selection_id:
            raise RuntimeError(
                f"Run {run_number}: row without selection_id."
            )

        if selection_id in normalized:
            raise RuntimeError(
                f"Run {run_number}: duplicate selection_id {selection_id}."
            )

        model_key = (
            row.get(
                "generator_key"
            )
            or row.get(
                "model_key"
            )
            or ""
        ).strip()

        norm = {
            "selection_id": selection_id,
            "report_type": str(
                row.get(
                    "report_type",
                    "",
                )
            ).strip(),
            "generation_mode": str(
                row.get(
                    "generation_mode",
                    "",
                )
            ).strip(),
            "model_key": model_key,
            "prompt_type": str(
                row.get(
                    "prompt_type",
                    "",
                )
            ).strip(),
        }

        report_number = row.get(
            "report_number"
        )

        if report_number not in {
            None,
            "",
        }:
            norm[
                "report_number"
            ] = int(
                float(
                    report_number
                )
            )
        else:
            report_name = str(
                row.get(
                    "report",
                    "",
                )
            )

            digits = ""
            for char in reversed(
                report_name.split(
                    "."
                )[0]
            ):
                if char.isdigit():
                    digits = (
                        char
                        + digits
                    )
                elif digits:
                    break

            norm[
                "report_number"
            ] = (
                int(
                    digits
                )
                if digits
                else None
            )

        for metric in ALL_METRICS:
            raw_value = row.get(
                metric
            )

            if raw_value in {
                None,
                "",
            }:
                raise RuntimeError(
                    f"Run {run_number}, {selection_id}: "
                    f"missing metric {metric}."
                )

            norm[
                metric
            ] = float(
                raw_value
            )

        normalized[
            selection_id
        ] = norm

    return normalized


def validate_alignment(
    run1: dict[str, dict],
    run2: dict[str, dict],
    run3: dict[str, dict],
):
    ids1 = set(
        run1
    )
    ids2 = set(
        run2
    )
    ids3 = set(
        run3
    )

    if not (
        ids1
        == ids2
        == ids3
    ):
        raise RuntimeError(
        )

    metadata_fields = [
        "report_type",
        "generation_mode",
        "model_key",
        "prompt_type",
    ]

    for selection_id in sorted(
        ids1
    ):
        reference = run1[
            selection_id
        ]

        for run_number, current in [
            (2, run2[selection_id]),
            (3, run3[selection_id]),
        ]:
            for field in metadata_fields:
                if (
                    reference[
                        field
                    ]
                    != current[
                        field
                    ]
                ):
                    raise RuntimeError(
                        f"Metadata mismatch for {selection_id}: "
                        f"Run 1 {field}={reference[field]!r}, "
                        f"Run {run_number} {field}={current[field]!r}."
                    )


def is_close(
    a: float,
    b: float,
    tolerance: float = FLOAT_TOLERANCE,
) -> bool:
    return abs(
        a - b
    ) <= tolerance


def sample_sd(
    values: list[float],
) -> float:
    if len(
        values
    ) < 2:
        return 0.0

    return statistics.stdev(
        values
    )


def pearson_r(
    xs: list[float],
    ys: list[float],
):
    if len(
        xs
    ) != len(
        ys
    ):
        raise ValueError(
        )

    if len(
        xs
    ) < 2:
        return None

    mean_x = statistics.mean(
        xs
    )
    mean_y = statistics.mean(
        ys
    )

    dx = [
        x - mean_x
        for x in xs
    ]
    dy = [
        y - mean_y
        for y in ys
    ]

    ss_x = sum(
        value * value
        for value in dx
    )
    ss_y = sum(
        value * value
        for value in dy
    )

    if (
        ss_x <= 0
        or ss_y <= 0
    ):
        return None

    numerator = sum(
        a * b
        for a, b in zip(
            dx,
            dy,
        )
    )

    return numerator / math.sqrt(
        ss_x
        * ss_y
    )


def icc_absolute_agreement_single(
    matrix: list[list[float]],
):
  
    n = len(
        matrix
    )

    if n < 2:
        return None

    k = len(
        matrix[
            0
        ]
    )

    if k < 2:
        return None

    grand_mean = statistics.mean(
        value
        for row in matrix
        for value in row
    )

    row_means = [
        statistics.mean(
            row
        )
        for row in matrix
    ]

    column_means = [
        statistics.mean(
            row[
                column
            ]
            for row in matrix
        )
        for column in range(
            k
        )
    ]

    ss_rows = k * sum(
        (
            row_mean
            - grand_mean
        ) ** 2
        for row_mean in row_means
    )

    ss_columns = n * sum(
        (
            column_mean
            - grand_mean
        ) ** 2
        for column_mean in column_means
    )

    ss_error = 0.0

    for i in range(
        n
    ):
        for j in range(
            k
        ):
            residual = (
                matrix[
                    i
                ][
                    j
                ]
                - row_means[
                    i
                ]
                - column_means[
                    j
                ]
                + grand_mean
            )

            ss_error += (
                residual
                ** 2
            )

    ms_rows = (
        ss_rows
        / (
            n - 1
        )
    )

    ms_columns = (
        ss_columns
        / (
            k - 1
        )
    )

    ms_error = (
        ss_error
        / (
            (
                n - 1
            )
            * (
                k - 1
            )
        )
    )

    denominator = (
        ms_rows
        + (
            k - 1
        )
        * ms_error
        + k
        * (
            ms_columns
            - ms_error
        )
        / n
    )

    if abs(
        denominator
    ) <= 1e-15:
        return None

    return (
        ms_rows
        - ms_error
    ) / denominator


def quadratic_weighted_kappa(
    xs: list[float],
    ys: list[float],
):

    categories = [
        0,
        1,
        2,
        3,
        4,
    ]

    index = {
        category: position
        for position, category in enumerate(
            categories
        )
    }

    observed = [
        [
            0.0
            for _ in categories
        ]
        for _ in categories
    ]

    n = len(
        xs
    )

    if n == 0:
        return None

    for x, y in zip(
        xs,
        ys,
    ):
        x_int = int(
            round(
                x
            )
        )
        y_int = int(
            round(
                y
            )
        )

        if (
            x_int
            not in index
            or y_int
            not in index
        ):
            raise RuntimeError(
                "Quadratic kappa expects ordinal values on the 0-4 scale."
            )

        observed[
            index[
                x_int
            ]
        ][
            index[
                y_int
            ]
        ] += 1.0

    for i in range(
        len(
            categories
        )
    ):
        for j in range(
            len(
                categories
            )
        ):
            observed[
                i
            ][
                j
            ] /= n

    row_marginal = [
        sum(
            row
        )
        for row in observed
    ]

    column_marginal = [
        sum(
            observed[
                i
            ][
                j
            ]
            for i in range(
                len(
                    categories
                )
            )
        )
        for j in range(
            len(
                categories
            )
        )
    ]

    denominator_scale = (
        len(
            categories
        )
        - 1
    ) ** 2

    weighted_observed = 0.0
    weighted_expected = 0.0

    for i in range(
        len(
            categories
        )
    ):
        for j in range(
            len(
                categories
            )
        ):
            weight = (
                (
                    i - j
                )
                ** 2
            ) / denominator_scale

            expected = (
                row_marginal[
                    i
                ]
                * column_marginal[
                    j
                ]
            )

            weighted_observed += (
                weight
                * observed[
                    i
                ][
                    j
                ]
            )

            weighted_expected += (
                weight
                * expected
            )

    if abs(
        weighted_expected
    ) <= 1e-15:
        if abs(
            weighted_observed
        ) <= 1e-15:
            return 1.0

        return None

    return (
        1.0
        - weighted_observed
        / weighted_expected
    )


def metric_tolerance(
    metric: str,
):
    if metric in RATIO_METRICS:
        return RATIO_TOLERANCE

    if metric in ORDINAL_METRICS:
        return ORDINAL_TOLERANCE

    return COUNT_TOLERANCE


def format_optional(
    value,
    digits=6,
):
    if value is None:
        return ""

    if isinstance(
        value,
        float,
    ):
        if math.isnan(
            value
        ):
            return ""

        return round(
            value,
            digits,
        )

    return value


def build_per_item_rows(
    run1: dict[str, dict],
    run2: dict[str, dict],
    run3: dict[str, dict],
):
    rows = []

    for selection_id in sorted(
        run1
    ):
        meta = run1[
            selection_id
        ]

        output = {
            "selection_id": selection_id,
            "report_type": meta[
                "report_type"
            ],
            "generation_mode": meta[
                "generation_mode"
            ],
            "report_number": meta[
                "report_number"
            ],
            "model_key": meta[
                "model_key"
            ],
            "prompt_type": meta[
                "prompt_type"
            ],
        }

        for metric in ALL_METRICS:
            values = [
                run1[
                    selection_id
                ][
                    metric
                ],
                run2[
                    selection_id
                ][
                    metric
                ],
                run3[
                    selection_id
                ][
                    metric
                ],
            ]

            value_range = (
                max(
                    values
                )
                - min(
                    values
                )
            )

            exact = (
                value_range
                <= FLOAT_TOLERANCE
            )

            tolerance = metric_tolerance(
                metric
            )

            output[
                f"{metric}_run1"
            ] = values[
                0
            ]

            output[
                f"{metric}_run2"
            ] = values[
                1
            ]

            output[
                f"{metric}_run3"
            ] = values[
                2
            ]

            output[
                f"{metric}_mean"
            ] = statistics.mean(
                values
            )

            output[
                f"{metric}_sd"
            ] = sample_sd(
                values
            )

            output[
                f"{metric}_range"
            ] = value_range

            output[
                f"{metric}_all3_exact"
            ] = int(
                exact
            )

            output[
                f"{metric}_all3_within_tolerance"
            ] = int(
                value_range
                <= tolerance
                + FLOAT_TOLERANCE
            )

        rows.append(
            output
        )

    return rows


def pairwise_stats_for_metric(
    metric: str,
    ids: list[str],
    runs: dict[int, dict[str, dict]],
):
    output = []

    for run_a, run_b in RUN_PAIRS:
        xs = [
            runs[
                run_a
            ][
                selection_id
            ][
                metric
            ]
            for selection_id in ids
        ]

        ys = [
            runs[
                run_b
            ][
                selection_id
            ][
                metric
            ]
            for selection_id in ids
        ]

        abs_differences = [
            abs(
                x - y
            )
            for x, y in zip(
                xs,
                ys,
            )
        ]

        squared_differences = [
            (
                x - y
            )
            ** 2
            for x, y in zip(
                xs,
                ys,
            )
        ]

        exact = [
            is_close(
                x,
                y,
            )
            for x, y in zip(
                xs,
                ys,
            )
        ]

        tolerance = metric_tolerance(
            metric
        )

        within_tolerance = [
            abs(
                x - y
            )
            <= tolerance
            + FLOAT_TOLERANCE
            for x, y in zip(
                xs,
                ys,
            )
        ]

        kappa = None

        if metric in ORDINAL_METRICS:
            kappa = (
                quadratic_weighted_kappa(
                    xs,
                    ys,
                )
            )

        output.append(
            {
                "metric": METRIC_LABELS[
                    metric
                ],
                "metric_key": metric,
                "run_a": run_a,
                "run_b": run_b,
                "n": len(
                    ids
                ),
                "run_a_mean": statistics.mean(
                    xs
                ),
                "run_b_mean": statistics.mean(
                    ys
                ),
                "exact_agreement_pct": 100.0
                * statistics.mean(
                    int(
                        value
                    )
                    for value in exact
                ),
                "within_tolerance_pct": 100.0
                * statistics.mean(
                    int(
                        value
                    )
                    for value in within_tolerance
                ),
                "mae": statistics.mean(
                    abs_differences
                ),
                "rmse": math.sqrt(
                    statistics.mean(
                        squared_differences
                    )
                ),
                "pearson_r": pearson_r(
                    xs,
                    ys,
                ),
                "quadratic_weighted_kappa": (
                    kappa
                ),
                "tolerance_used": tolerance,
            }
        )

    return output


def summary_for_metric(
    metric: str,
    ids: list[str],
    runs: dict[int, dict[str, dict]],
):
    matrix = [
        [
            runs[
                run_number
            ][
                selection_id
            ][
                metric
            ]
            for run_number in [
                1,
                2,
                3,
            ]
        ]
        for selection_id in ids
    ]

    run_means = [
        statistics.mean(
            row[
                column
            ]
            for row in matrix
        )
        for column in range(
            3
        )
    ]

    ranges = [
        max(
            row
        )
        - min(
            row
        )
        for row in matrix
    ]

    all_three_exact = [
        value
        <= FLOAT_TOLERANCE
        for value in ranges
    ]

    tolerance = metric_tolerance(
        metric
    )

    all_three_within = [
        value
        <= tolerance
        + FLOAT_TOLERANCE
        for value in ranges
    ]

    pairwise = (
        pairwise_stats_for_metric(
            metric,
            ids,
            runs,
        )
    )

    average_pairwise_exact = statistics.mean(
        row[
            "exact_agreement_pct"
        ]
        for row in pairwise
    )

    average_pairwise_within = statistics.mean(
        row[
            "within_tolerance_pct"
        ]
        for row in pairwise
    )

    average_pairwise_mae = statistics.mean(
        row[
            "mae"
        ]
        for row in pairwise
    )

    average_kappa = None

    if metric in ORDINAL_METRICS:
        kappas = [
            row[
                "quadratic_weighted_kappa"
            ]
            for row in pairwise
            if row[
                "quadratic_weighted_kappa"
            ]
            is not None
        ]

        if kappas:
            average_kappa = (
                statistics.mean(
                    kappas
                )
            )

    icc = (
        icc_absolute_agreement_single(
            matrix
        )
    )

    if (
        icc is not None
        and abs(
            icc
        )
        < 1e-12
    ):
        icc = 0.0

    note = ""

    if metric in MAX_SCORE:
        maximum = MAX_SCORE[
            metric
        ]

        all_values = [
            value
            for row in matrix
            for value in row
        ]

        ceiling_share = 100.0 * statistics.mean(
            int(
                is_close(
                    value,
                    maximum,
                )
            )
            for value in all_values
        )

    return {
        "metric": METRIC_LABELS[
            metric
        ],
        "metric_key": metric,
        "n": len(
            ids
        ),
        "run1_mean": run_means[
            0
        ],
        "run2_mean": run_means[
            1
        ],
        "run3_mean": run_means[
            2
        ],
        "grand_mean": statistics.mean(
            run_means
        ),
        "all_three_exact_agreement_pct": 100.0
        * statistics.mean(
            int(
                value
            )
            for value in all_three_exact
        ),
        "all_three_within_tolerance_pct": 100.0
        * statistics.mean(
            int(
                value
            )
            for value in all_three_within
        ),
        "tolerance_used": tolerance,
        "average_pairwise_exact_agreement_pct": average_pairwise_exact,
        "average_pairwise_within_tolerance_pct": average_pairwise_within,
        "average_pairwise_mae": average_pairwise_mae,
        "mean_item_range": statistics.mean(
            ranges
        ),
        "max_item_range": max(
            ranges
        ),
        "icc_a1_absolute_agreement": icc,
        "average_pairwise_quadratic_weighted_kappa": average_kappa,
        "ceiling_share_pct": ceiling_share,
        "note": note,
    }


def build_breakdown_rows(
    runs: dict[int, dict[str, dict]],
):
    ids = sorted(
        runs[
            1
        ]
    )

    dimensions = [
        "report_type",
        "generation_mode",
        "model_key",
        "prompt_type",
    ]

    rows = []

    # Overall first.
    for metric in ALL_METRICS:
        summary = (
            summary_for_metric(
                metric,
                ids,
                runs,
            )
        )

        rows.append(
            {
                "group_dimension": "overall",
                "group_value": "all",
                **summary,
            }
        )

    for dimension in dimensions:
        group_values = sorted(
            {
                runs[
                    1
                ][
                    selection_id
                ][
                    dimension
                ]
                for selection_id in ids
            }
        )

        for group_value in group_values:
            group_ids = [
                selection_id
                for selection_id in ids
                if runs[
                    1
                ][
                    selection_id
                ][
                    dimension
                ]
                == group_value
            ]

            for metric in ALL_METRICS:
                summary = (
                    summary_for_metric(
                        metric,
                        group_ids,
                        runs,
                    )
                )

                rows.append(
                    {
                        "group_dimension": dimension,
                        "group_value": group_value,
                        **summary,
                    }
                )

    return rows


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str] | None = None,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise RuntimeError(
            f"No rows to write: {path}"
        )

    if fieldnames is None:
        fieldnames = list(
            rows[
                0
            ].keys()
        )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            cleaned = {
                key: format_optional(
                    row.get(
                        key
                    )
                )
                for key in fieldnames
            }

            writer.writerow(
                cleaned
            )


def build_text_report(
    summary_rows: list[dict],
    pairwise_rows: list[dict],
):
    by_metric = {
        row[
            "metric_key"
        ]: row
        for row in summary_rows
    }

    lines = []

    lines.append(
        "NOVA 2 LITE SELF-CONSISTENCY — THREE RUNS"
    )
    lines.append(
        "=" * 72
    )
    lines.append(
        "Sample: 72 summaries evaluated independently three times."
    )
    lines.append(
        ""
    )

    lines.append(
        "PRIMARY SCORE STABILITY"
    )
    lines.append(
        "-" * 72
    )

    for metric in SCORE_METRICS:
        row = by_metric[
            metric
        ]

        lines.append(
            f"{METRIC_LABELS[metric]}:"
        )

        lines.append(
            "  Run means: "
            f"{row['run1_mean']:.4f} | "
            f"{row['run2_mean']:.4f} | "
            f"{row['run3_mean']:.4f}"
        )

        lines.append(
            "  All three exactly equal: "
            f"{row['all_three_exact_agreement_pct']:.1f}%"
        )

        lines.append(
            "  Average pairwise exact agreement: "
            f"{row['average_pairwise_exact_agreement_pct']:.1f}%"
        )

        lines.append(
            "  Average pairwise MAE: "
            f"{row['average_pairwise_mae']:.4f}"
        )

        lines.append(
            "  ICC(A,1): "
            + (
                f"{row['icc_a1_absolute_agreement']:.4f}"
                if row[
                    "icc_a1_absolute_agreement"
                ]
                is not None
                else "N/A"
            )
        )

        if metric in ORDINAL_METRICS:
            lines.append(
                "  Average pairwise quadratic weighted kappa: "
                + (
                    f"{row['average_pairwise_quadratic_weighted_kappa']:.4f}"
                    if row[
                        "average_pairwise_quadratic_weighted_kappa"
                    ]
                    is not None
                    else "N/A"
                )
            )

        if row[
            "note"
        ]:
            lines.append(
                "  Note: "
                + row[
                    "note"
                ]
            )

        lines.append(
            ""
        )

    lines.append(
        "CLAIM-DECOMPOSITION STABILITY"
    )
    lines.append(
        "-" * 72
    )

    for metric in COUNT_METRICS:
        row = by_metric[
            metric
        ]

        lines.append(
            f"{METRIC_LABELS[metric]}:"
        )

        lines.append(
            "  Run means: "
            f"{row['run1_mean']:.2f} | "
            f"{row['run2_mean']:.2f} | "
            f"{row['run3_mean']:.2f}"
        )

        lines.append(
            "  All three counts exactly equal: "
            f"{row['all_three_exact_agreement_pct']:.1f}%"
        )

        lines.append(
            "  Average pairwise MAE: "
            f"{row['average_pairwise_mae']:.2f} claims"
        )

        lines.append(
            "  Mean item range: "
            f"{row['mean_item_range']:.2f} claims"
        )

        lines.append(
            ""
        )

    lines.append(
        "INTERPRETATION NOTE"
    )
    lines.append(
        "-" * 72
    )

    return "\n".join(
        lines
    )


def main():
    args = parse_args()

    input_dir = args.input_dir.resolve()

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (
            input_dir
            / "self_consistency_analysis"
        )
    )

    run_paths = {
        1: (
            input_dir
            / "nova_run1_scores_72.csv"
        ),
        2: (
            input_dir
            / "nova_run2_scores_72.csv"
        ),
        3: (
            input_dir
            / "nova_run3_scores_72.csv"
        ),
    }

    runs = {}

    for run_number in [
        1,
        2,
        3,
    ]:
        rows = (
            read_csv_rows(
                run_paths[
                    run_number
                ]
            )
        )

        runs[
            run_number
        ] = (
            normalize_run_rows(
                rows,
                run_number,
            )
        )

    validate_alignment(
        runs[
            1
        ],
        runs[
            2
        ],
        runs[
            3
        ],
    )

    ids = sorted(
        runs[
            1
        ]
    )

    per_item_rows = (
        build_per_item_rows(
            runs[
                1
            ],
            runs[
                2
            ],
            runs[
                3
            ],
        )
    )

    summary_rows = [
        summary_for_metric(
            metric,
            ids,
            runs,
        )
        for metric in ALL_METRICS
    ]

    pairwise_rows = []

    for metric in ALL_METRICS:
        pairwise_rows.extend(
            pairwise_stats_for_metric(
                metric,
                ids,
                runs,
            )
        )

    breakdown_rows = (
        build_breakdown_rows(
            runs
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        output_dir
        / "nova_self_consistency_per_item.csv",
        per_item_rows,
    )

    write_csv(
        output_dir
        / "nova_self_consistency_summary.csv",
        summary_rows,
    )

    write_csv(
        output_dir
        / "nova_self_consistency_pairwise.csv",
        pairwise_rows,
    )

    write_csv(
        output_dir
        / "nova_self_consistency_breakdown.csv",
        breakdown_rows,
    )

    report_text = (
        build_text_report(
            summary_rows,
            pairwise_rows,
        )
    )

    report_path = (
        output_dir
        / "nova_self_consistency_report.txt"
    )

    report_path.write_text(
        report_text,
        encoding="utf-8",
    )

    print(
        "=" * 100
    )
    print(
        "NOVA SELF-CONSISTENCY ANALYSIS COMPLETE"
    )
    print(
        "=" * 100
    )
    print(
        "Validated: 72 identical selection IDs across Run 1 / Run 2 / Run 3"
    )
    print()

    for row in summary_rows:
        if row[
            "metric_key"
        ] not in SCORE_METRICS:
            continue

        print(
            f"{row['metric']:<20} | "
            f"all-3 exact={row['all_three_exact_agreement_pct']:6.1f}% | "
            f"pairwise exact={row['average_pairwise_exact_agreement_pct']:6.1f}% | "
            f"MAE={row['average_pairwise_mae']:.4f} | "
            f"ICC(A,1)="
            + (
                f"{row['icc_a1_absolute_agreement']:.4f}"
                if row[
                    "icc_a1_absolute_agreement"
                ]
                is not None
                else "N/A"
            )
        )

    print()
    print(
        f"Output folder: {output_dir}"
    )
    print(
        f"Summary:       {output_dir / 'nova_self_consistency_summary.csv'}"
    )
    print(
        f"Pairwise:      {output_dir / 'nova_self_consistency_pairwise.csv'}"
    )
    print(
        f"Per item:      {output_dir / 'nova_self_consistency_per_item.csv'}"
    )
    print(
        f"Breakdown:     {output_dir / 'nova_self_consistency_breakdown.csv'}"
    )
    print(
        f"Text report:   {report_path}"
    )


if __name__ == "__main__":
    main()
