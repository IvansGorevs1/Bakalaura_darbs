from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "judge_validation"
    / "consistency_crossjudge_72"
)

PRIMARY_METRICS = [
    "faithfulness",
    "numerical_accuracy",
    "completeness",
    "coherence",
]

COUNT_METRICS = [
    "factual_total",
    "numerical_total",
]

ALL_METRICS = (
    PRIMARY_METRICS
    + COUNT_METRICS
)

RATIO_METRICS = {
    "faithfulness",
    "numerical_accuracy",
}

ORDINAL_METRICS = {
    "completeness",
    "coherence",
}

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "numerical_accuracy": "Numerical Accuracy",
    "completeness": "Completeness",
    "coherence": "Coherence",
    "factual_total": "Factual Claim Count",
    "numerical_total": "Numerical Claim Count",
}

MAX_SCORE = {
    "faithfulness": 1.0,
    "numerical_accuracy": 1.0,
    "completeness": 4.0,
    "coherence": 4.0,
}

FLOAT_TOLERANCE = 1e-9


RATIO_TOLERANCE = 0.05
ORDINAL_TOLERANCE = 1.0
COUNT_TOLERANCE = 1.0


def parse_args():
    parser = argparse.ArgumentParser(
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


def read_csv_rows(
    path: Path,
) -> list[dict]:
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


def normalize_rows(
    rows: list[dict],
    judge_name: str,
) -> dict[str, dict]:
    """
    Normalize schema differences:
      Nova Run 1 -> model_key + report_number
      Grok 4.3   -> generator_key + report
    """
    if len(
        rows
    ) != 72:
        raise RuntimeError(
            f"{judge_name}: expected 72 rows, found {len(rows)}."
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
                f"{judge_name}: row without selection_id."
            )

        if selection_id in normalized:
            raise RuntimeError(
                f"{judge_name}: duplicate selection_id {selection_id}."
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

        report_number = row.get(
            "report_number"
        )

        if report_number not in {
            None,
            "",
        }:
            parsed_report_number = int(
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

            stem = report_name.rsplit(
                ".",
                1,
            )[0]

            digits = ""

            for char in reversed(
                stem
            ):
                if char.isdigit():
                    digits = (
                        char
                        + digits
                    )
                elif digits:
                    break

            parsed_report_number = (
                int(
                    digits
                )
                if digits
                else None
            )

        current = {
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
            "report_number": parsed_report_number,
            "model_key": model_key,
            "prompt_type": str(
                row.get(
                    "prompt_type",
                    "",
                )
            ).strip(),
        }

        for metric in ALL_METRICS:
            raw = row.get(
                metric
            )

            if raw in {
                None,
                "",
            }:
                raise RuntimeError(
                    f"{judge_name}, {selection_id}: missing {metric}."
                )

            current[
                metric
            ] = float(
                raw
            )

        normalized[
            selection_id
        ] = current

    return normalized


def validate_alignment(
    nova: dict[str, dict],
    grok: dict[str, dict],
):
    if set(
        nova
    ) != set(
        grok
    ):
        missing_in_grok = sorted(
            set(
                nova
            )
            - set(
                grok
            )
        )

        missing_in_nova = sorted(
            set(
                grok
            )
            - set(
                nova
            )
        )

        raise RuntimeError(
            "Nova and Grok do not contain the same 72 selection IDs.\n"
            f"Missing in Grok: {missing_in_grok}\n"
            f"Missing in Nova: {missing_in_nova}"
        )

    metadata_fields = [
        "report_type",
        "generation_mode",
        "model_key",
        "prompt_type",
    ]

    for selection_id in sorted(
        nova
    ):
        for field in metadata_fields:
            if (
                nova[
                    selection_id
                ][
                    field
                ]
                != grok[
                    selection_id
                ][
                    field
                ]
            ):
                raise RuntimeError(
                    f"Metadata mismatch for {selection_id}, {field}: "
                    f"Nova={nova[selection_id][field]!r}, "
                    f"Grok={grok[selection_id][field]!r}"
                )


def metric_tolerance(
    metric: str,
) -> float:
    if metric in RATIO_METRICS:
        return RATIO_TOLERANCE

    if metric in ORDINAL_METRICS:
        return ORDINAL_TOLERANCE

    return COUNT_TOLERANCE


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
            "Pearson inputs have different lengths."
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
    """
    ICC(A,1): two-way random-effects, absolute-agreement, single-measure.

    Rows    = summaries
    Columns = Nova Run 1, Grok 4.3
    """
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
        )
        ** 2
        for row_mean in row_means
    )

    ss_columns = n * sum(
        (
            column_mean
            - grand_mean
        )
        ** 2
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

    value = (
        ms_rows
        - ms_error
    ) / denominator

    if abs(
        value
    ) < 1e-12:
        return 0.0

    return value


def quadratic_weighted_kappa(
    xs: list[float],
    ys: list[float],
):
    """
    Quadratic weighted Cohen's kappa for the fixed 0-4 ordinal scale.
    """
    categories = [
        0,
        1,
        2,
        3,
        4,
    ]

    index = {
        value: idx
        for idx, value in enumerate(
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
        xi = int(
            round(
                x
            )
        )

        yi = int(
            round(
                y
            )
        )

        if (
            xi not in index
            or yi not in index
        ):
            raise RuntimeError(
                "Quadratic weighted kappa expects 0-4 ratings."
            )

        observed[
            index[
                xi
            ]
        ][
            index[
                yi
            ]
        ] += 1.0

    for i in range(
        5
    ):
        for j in range(
            5
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
                5
            )
        )
        for j in range(
            5
        )
    ]

    weighted_observed = 0.0
    weighted_expected = 0.0

    scale = 16.0

    for i in range(
        5
    ):
        for j in range(
            5
        ):
            weight = (
                (
                    i - j
                )
                ** 2
            ) / scale

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


def metric_summary(
    metric: str,
    ids: list[str],
    nova: dict[str, dict],
    grok: dict[str, dict],
):
    nova_values = [
        nova[
            selection_id
        ][
            metric
        ]
        for selection_id in ids
    ]

    grok_values = [
        grok[
            selection_id
        ][
            metric
        ]
        for selection_id in ids
    ]

    differences = [
        g - n
        for n, g in zip(
            nova_values,
            grok_values,
        )
    ]

    absolute_differences = [
        abs(
            value
        )
        for value in differences
    ]

    squared_differences = [
        value
        ** 2
        for value in differences
    ]

    exact = [
        abs(
            value
        )
        <= FLOAT_TOLERANCE
        for value in differences
    ]

    tolerance = metric_tolerance(
        metric
    )

    within_tolerance = [
        abs(
            value
        )
        <= tolerance
        + FLOAT_TOLERANCE
        for value in differences
    ]

    matrix = [
        [
            nova[
                selection_id
            ][
                metric
            ],
            grok[
                selection_id
            ][
                metric
            ],
        ]
        for selection_id in ids
    ]

    icc = (
        icc_absolute_agreement_single(
            matrix
        )
    )

    kappa = None

    if metric in ORDINAL_METRICS:
        kappa = (
            quadratic_weighted_kappa(
                nova_values,
                grok_values,
            )
        )

    note = ""
    nova_ceiling = None
    grok_ceiling = None

    if metric in MAX_SCORE:
        maximum = MAX_SCORE[
            metric
        ]

        nova_ceiling = 100.0 * statistics.mean(
            int(
                abs(
                    value
                    - maximum
                )
                <= FLOAT_TOLERANCE
            )
            for value in nova_values
        )

        grok_ceiling = 100.0 * statistics.mean(
            int(
                abs(
                    value
                    - maximum
                )
                <= FLOAT_TOLERANCE
            )
            for value in grok_values
        )

        if (
            nova_ceiling >= 80.0
            or grok_ceiling >= 80.0
        ):
            note = (
                "MAE."
            )

    return {
        "metric": METRIC_LABELS[
            metric
        ],
        "metric_key": metric,
        "n": len(
            ids
        ),
        "nova_run1_mean": statistics.mean(
            nova_values
        ),
        "grok43_mean": statistics.mean(
            grok_values
        ),
        "mean_difference_grok_minus_nova": statistics.mean(
            differences
        ),
        "median_difference_grok_minus_nova": statistics.median(
            differences
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
        "tolerance_used": tolerance,
        "mae": statistics.mean(
            absolute_differences
        ),
        "rmse": math.sqrt(
            statistics.mean(
                squared_differences
            )
        ),
        "max_absolute_difference": max(
            absolute_differences
        ),
        "pearson_r": pearson_r(
            nova_values,
            grok_values,
        ),
        "icc_a1_absolute_agreement": icc,
        "quadratic_weighted_kappa": kappa,
        "nova_ceiling_share_pct": nova_ceiling,
        "grok_ceiling_share_pct": grok_ceiling,
        "note": note,
    }


def per_item_rows(
    nova: dict[str, dict],
    grok: dict[str, dict],
):
    rows = []

    for selection_id in sorted(
        nova
    ):
        meta = nova[
            selection_id
        ]

        row = {
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
            nova_value = nova[
                selection_id
            ][
                metric
            ]

            grok_value = grok[
                selection_id
            ][
                metric
            ]

            difference = (
                grok_value
                - nova_value
            )

            row[
                f"{metric}_nova_run1"
            ] = nova_value

            row[
                f"{metric}_grok43"
            ] = grok_value

            row[
                f"{metric}_difference_grok_minus_nova"
            ] = difference

            row[
                f"{metric}_absolute_difference"
            ] = abs(
                difference
            )

            row[
                f"{metric}_exact_agreement"
            ] = int(
                abs(
                    difference
                )
                <= FLOAT_TOLERANCE
            )

            row[
                f"{metric}_within_tolerance"
            ] = int(
                abs(
                    difference
                )
                <= metric_tolerance(
                    metric
                )
                + FLOAT_TOLERANCE
            )

        rows.append(
            row
        )

    return rows


def breakdown_rows(
    nova: dict[str, dict],
    grok: dict[str, dict],
):
    ids = sorted(
        nova
    )

    rows = []

    for metric in ALL_METRICS:
        rows.append(
            {
                "group_dimension": "overall",
                "group_value": "all",
                **metric_summary(
                    metric,
                    ids,
                    nova,
                    grok,
                ),
            }
        )

    dimensions = [
        "report_type",
        "generation_mode",
        "model_key",
        "prompt_type",
    ]

    for dimension in dimensions:
        group_values = sorted(
            {
                nova[
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
                if nova[
                    selection_id
                ][
                    dimension
                ]
                == group_value
            ]

            for metric in ALL_METRICS:
                rows.append(
                    {
                        "group_dimension": dimension,
                        "group_value": group_value,
                        **metric_summary(
                            metric,
                            group_ids,
                            nova,
                            grok,
                        ),
                    }
                )

    return rows


def optional_value(
    value,
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
            6,
        )

    return value


def write_csv(
    path: Path,
    rows: list[dict],
):
    if not rows:
        raise RuntimeError(
            f"No rows to write: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
            writer.writerow(
                {
                    field: optional_value(
                        row.get(
                            field
                        )
                    )
                    for field in fieldnames
                }
            )


def build_text_report(
    summaries: list[dict],
):
    by_metric = {
        row[
            "metric_key"
        ]: row
        for row in summaries
    }

    lines = []

    lines.append(
        "NOVA RUN 1 vs GROK 4.3 — CROSS-JUDGE AGREEMENT"
    )
    lines.append(
        "=" * 78
    )
    lines.append(
        "Sample: the same 72 summaries from the fixed validation subset."
    )
    lines.append(
        "Nova Run 1 is the original Nova evaluation used in the main experiment."
    )
    lines.append(
        ""
    )

    lines.append(
        "PRIMARY METRICS"
    )
    lines.append(
        "-" * 78
    )

    for metric in PRIMARY_METRICS:
        row = by_metric[
            metric
        ]

        lines.append(
            f"{row['metric']}:"
        )

        lines.append(
            "  Mean score — Nova / Grok: "
            f"{row['nova_run1_mean']:.4f} / "
            f"{row['grok43_mean']:.4f}"
        )

        lines.append(
            "  Mean difference (Grok - Nova): "
            f"{row['mean_difference_grok_minus_nova']:+.4f}"
        )

        lines.append(
            "  Exact agreement: "
            f"{row['exact_agreement_pct']:.1f}%"
        )

        lines.append(
            "  Within tolerance: "
            f"{row['within_tolerance_pct']:.1f}% "
            f"(tolerance={row['tolerance_used']})"
        )

        lines.append(
            "  MAE / RMSE: "
            f"{row['mae']:.4f} / "
            f"{row['rmse']:.4f}"
        )

        lines.append(
            "  Pearson r: "
            + (
                f"{row['pearson_r']:.4f}"
                if row[
                    "pearson_r"
                ]
                is not None
                else "N/A"
            )
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
                "  Quadratic weighted kappa: "
                + (
                    f"{row['quadratic_weighted_kappa']:.4f}"
                    if row[
                        "quadratic_weighted_kappa"
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
        "CLAIM-DECOMPOSITION COMPARISON"
    )
    lines.append(
        "-" * 78
    )

    for metric in COUNT_METRICS:
        row = by_metric[
            metric
        ]

        lines.append(
            f"{row['metric']}:"
        )

        lines.append(
            "  Mean count — Nova / Grok: "
            f"{row['nova_run1_mean']:.2f} / "
            f"{row['grok43_mean']:.2f}"
        )

        lines.append(
            "  Exact count agreement: "
            f"{row['exact_agreement_pct']:.1f}%"
        )

        lines.append(
            "  MAE: "
            f"{row['mae']:.2f} claims"
        )

        lines.append(
            ""
        )

    lines.append(
        "INTERPRETATION"
    )
    lines.append(
        "-" * 78
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
            / "cross_judge_analysis"
        )
    )

    nova_path = (
        input_dir
        / "nova_run1_scores_72.csv"
    )

    grok_path = (
        input_dir
        / "grok43_run1_scores_72.csv"
    )

    nova = normalize_rows(
        read_csv_rows(
            nova_path
        ),
        "Nova Run 1",
    )

    grok = normalize_rows(
        read_csv_rows(
            grok_path
        ),
        "Grok 4.3",
    )

    validate_alignment(
        nova,
        grok,
    )

    ids = sorted(
        nova
    )

    summaries = [
        metric_summary(
            metric,
            ids,
            nova,
            grok,
        )
        for metric in ALL_METRICS
    ]

    item_rows = (
        per_item_rows(
            nova,
            grok,
        )
    )

    grouped_rows = (
        breakdown_rows(
            nova,
            grok,
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_dir
        / "nova1_grok43_summary.csv"
    )

    per_item_path = (
        output_dir
        / "nova1_grok43_per_item.csv"
    )

    breakdown_path = (
        output_dir
        / "nova1_grok43_breakdown.csv"
    )

    report_path = (
        output_dir
        / "nova1_grok43_report.txt"
    )

    write_csv(
        summary_path,
        summaries,
    )

    write_csv(
        per_item_path,
        item_rows,
    )

    write_csv(
        breakdown_path,
        grouped_rows,
    )

    report_text = build_text_report(
        summaries
    )

    report_path.write_text(
        report_text,
        encoding="utf-8",
    )

    print(
        "=" * 110
    )

    print(
        "NOVA RUN 1 vs GROK 4.3 CROSS-JUDGE ANALYSIS COMPLETE"
    )

    print(
        "=" * 110
    )

    print(
        "Validated: 72 identical selection IDs in Nova Run 1 and Grok 4.3"
    )

    print()

    for row in summaries:
        if row[
            "metric_key"
        ] not in PRIMARY_METRICS:
            continue

        extra = ""

        if row[
            "metric_key"
        ] in ORDINAL_METRICS:
            extra = (
                " | weighted_kappa="
                + (
                    f"{row['quadratic_weighted_kappa']:.4f}"
                    if row[
                        "quadratic_weighted_kappa"
                    ]
                    is not None
                    else "N/A"
                )
            )

        print(
            f"{row['metric']:<20} | "
            f"Nova={row['nova_run1_mean']:.4f} | "
            f"Grok={row['grok43_mean']:.4f} | "
            f"exact={row['exact_agreement_pct']:6.1f}% | "
            f"MAE={row['mae']:.4f} | "
            f"ICC={row['icc_a1_absolute_agreement'] if row['icc_a1_absolute_agreement'] is not None else 'N/A'}"
            f"{extra}"
        )

    print()
    print(
        f"Output folder: {output_dir}"
    )
    print(
        f"Summary:       {summary_path}"
    )
    print(
        f"Per item:      {per_item_path}"
    )
    print(
        f"Breakdown:     {breakdown_path}"
    )
    print(
        f"Text report:   {report_path}"
    )


if __name__ == "__main__":
    main()
