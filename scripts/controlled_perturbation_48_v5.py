from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path


# =============================================================================
# CONTROLLED PERTURBATION VALIDATION — FINAL EXPERIMENT V5
# =============================================================================
#
# Dataset:
#   data/judge_validation/perturbation_48/
#       modified_answers/          <- 48 manually corrupted summaries
#       original/                  <- 12 untouched source summaries
#       perturbation_manifest_48.csv
#
# Design:
#   12 original summaries × 4 isolated corruption types = 48 modified summaries
#
# Corruption types:
#   factual_corruption
#   numerical_corruption
#   completeness_corruption
#   coherence_corruption
#
# Default mode is TARGET-ONLY to save time/cost:
#
#   factual corruption
#       -> run Factual/Numerical request
#       -> target = Faithfulness
#       -> Numerical Accuracy is retained as a secondary cross-check
#
#   numerical corruption
#       -> run Factual/Numerical request
#       -> target = Numerical Accuracy
#       -> Faithfulness is retained as a secondary cross-check
#
#   completeness corruption
#       -> run Completeness only
#
#   coherence corruption
#       -> run Coherence only
#
# Therefore the default experiment requires only:
#   12 + 12 + 12 + 12 = 48 logical Judge requests
#
# instead of 48 × 3 = 144 requests.
#
# Optional:
#   --all-metrics
# evaluates all three Judge requests for every corrupted summary.
#
# Baseline:
#   The untouched score is NOT regenerated. The script loads the exact
#   Nova merged result from the original main experiment:
#
#       normal          -> data/judge_results/
#       reasoning_high  -> data/judge_results_reasoning_high/
#
# Completeness:
#   Reuses the exact original verified report-level key-information rubric from:
#       data/judge_shared/nova_2_lite/...
#
# This keeps the validity test controlled:
#   original score vs manually corrupted version,
#   same Judge methodology / same Completeness rubric.
#
# =============================================================================


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the final controlled-corruption validity test with Nova 2 Lite "
            "on the 48 manually modified summaries."
        )
    )

    parser.add_argument(
        "--worker",
        type=int,
        default=1,
        help="Worker number, starting from 1.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Total number of parallel workers.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "No API calls. Validate the 48 filenames, 12 originals, baseline "
            "Judge results and 12×4 perturbation structure."
        ),
    )

    parser.add_argument(
        "--collect",
        action="store_true",
        help=(
            "No API calls. Collect all successful saved perturbation results "
            "and calculate the final sensitivity tables."
        ),
    )

    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help=(
            "Evaluate Factual/Numerical + Completeness + Coherence for every "
            "modified answer. Default is target-only (48 logical requests)."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rerun already successful paid results. Normally DO NOT use this."
        ),
    )

    return parser.parse_args()


ARGS = parse_args()

if ARGS.worker < 1:
    raise SystemExit("--worker must be >= 1")

if ARGS.workers < 1:
    raise SystemExit("--workers must be >= 1")

if ARGS.worker > ARGS.workers:
    raise SystemExit("--worker cannot exceed --workers")


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

ROOT = (
    DATA_DIR
    / "judge_validation"
    / "perturbation_48"
)

MODIFIED_DIR = (
    ROOT
    / "modified_answers"
)

ORIGINAL_DIR = (
    ROOT
    / "original"
)

MANIFEST = (
    ROOT
    / "perturbation_manifest_48.csv"
)

OUTPUT_ROOT = (
    ROOT
    / "nova_controlled_validation"
)

RAW_ROOT = (
    OUTPUT_ROOT
    / "raw"
)

RESULTS_ROOT = (
    OUTPUT_ROOT
    / "results"
)

WORKER_ROOT = (
    OUTPUT_ROOT
    / "worker_results"
)

WORKER_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

WORKER_CSV = (
    WORKER_ROOT
    / f"worker_{ARGS.worker}_of_{ARGS.workers}.csv"
)

WORKER_LOG = (
    WORKER_ROOT
    / f"worker_{ARGS.worker}_of_{ARGS.workers}.log.txt"
)

FINAL_RESULTS_CSV = (
    OUTPUT_ROOT
    / "perturbation_results_48.csv"
)

SUMMARY_CSV = (
    OUTPUT_ROOT
    / "perturbation_summary.csv"
)

PER_ORIGINAL_CSV = (
    OUTPUT_ROOT
    / "perturbation_per_original.csv"
)

TEXT_REPORT = (
    OUTPUT_ROOT
    / "perturbation_report.txt"
)


# -----------------------------------------------------------------------------
# Naming / metadata
# -----------------------------------------------------------------------------

GENERATOR_NAMES = {
    "openai": "GPT-5.6 Luna",
    "gemini": "Gemini 3.5 Flash-Lite",
    "qwen36": "Qwen3.6-35B-A3B",
}

TARGET_METRIC = {
    "factual": "faithfulness",
    "numerical": "numerical_accuracy",
    "completeness": "completeness",
    "coherence": "coherence",
}

SECONDARY_METRIC = {
    "factual": "numerical_accuracy",
    "numerical": "faithfulness",
    "completeness": None,
    "coherence": None,
}

MIN_SCORE = {
    "faithfulness": 0.0,
    "numerical_accuracy": 0.0,
    "completeness": 0.0,
    "coherence": 0.0,
}

MAX_SCORE = {
    "faithfulness": 1.0,
    "numerical_accuracy": 1.0,
    "completeness": 4.0,
    "coherence": 4.0,
}

TARGET_EVALUATIONS = {
    "factual": [
        "factual_numerical",
    ],
    "numerical": [
        "factual_numerical",
    ],
    "completeness": [
        "completeness",
    ],
    "coherence": [
        "coherence",
    ],
}

ALL_EVALUATIONS = [
    "factual_numerical",
    "completeness",
    "coherence",
]


MODIFIED_RE = re.compile(
    r"^(?P<mode>normal|reasoning_high)_+"
    r"report_(?P<rtype>annual|quarterly)_(?P<number>\d+)_"
    r"(?P<prompt>zero_shot|few_shot|chain_of_event)_"
    r"(?P<model>openai|gemini|qwen36)_+"
    r"(?P<corruption>factual|numerical|completeness|coherence)"
    r"_corruption\.txt$",
    flags=re.IGNORECASE,
)

# `original/` contains untouched paired copies that intentionally keep the
# corruption suffix in the filename, e.g.
#   ...qwen36__coherence_corruption.txt
# These are not corrupted files: they are the pre-edit originals paired with
# each modified answer.
ORIGINAL_RE = re.compile(
    r"^(?P<mode>normal|reasoning_high)_+"
    r"report_(?P<rtype>annual|quarterly)_(?P<number>\d+)_"
    r"(?P<prompt>zero_shot|few_shot|chain_of_event)_"
    r"(?P<model>openai|gemini|qwen36)_+"
    r"(?P<corruption>factual|numerical|completeness|coherence)"
    r"_corruption\.txt$",
    flags=re.IGNORECASE,
)

# Also support a true 12-file original layout if such files are later used.
ORIGINAL_BASE_RE = re.compile(
    r"^(?P<mode>normal|reasoning_high)_+"
    r"report_(?P<rtype>annual|quarterly)_(?P<number>\d+)_"
    r"(?P<prompt>zero_shot|few_shot|chain_of_event)_"
    r"(?P<model>openai|gemini|qwen36)"
    r"(?:_+(?:original|baseline|source))?"
    r"\.txt$",
    flags=re.IGNORECASE,
)


def metadata_key(
    meta: dict,
):
    return (
        meta[
            "generation_mode"
        ],
        meta[
            "report_type"
        ],
        int(
            meta[
                "report_number"
            ]
        ),
        meta[
            "prompt_type"
        ],
        meta[
            "model_key"
        ],
    )


def parse_modified(
    path: Path,
) -> dict:
    match = MODIFIED_RE.match(
        path.name
    )

    if not match:
        raise RuntimeError(
            "Modified answer filename does not match the expected pattern:\n"
            f"  {path.name}\n\n"
            "Expected example:\n"
            "  normal__report_annual_1_chain_of_event_qwen36__factual_corruption.txt"
        )

    groups = match.groupdict()

    return {
        "modified_file": path,
        "generation_mode": groups[
            "mode"
        ].lower(),
        "report_type": (
            "annual_report"
            if groups[
                "rtype"
            ].lower()
            == "annual"
            else "quarterly_report"
        ),
        "report_number": int(
            groups[
                "number"
            ]
        ),
        "prompt_type": groups[
            "prompt"
        ].lower(),
        "model_key": groups[
            "model"
        ].lower(),
        "corruption_type": groups[
            "corruption"
        ].lower(),
    }


def parse_original(
    path: Path,
):
    match = ORIGINAL_RE.match(
        path.name
    )

    paired = True

    if not match:
        match = ORIGINAL_BASE_RE.match(
            path.name
        )
        paired = False

    if not match:
        return None

    groups = match.groupdict()

    return {
        "original_file": path,
        "generation_mode": groups[
            "mode"
        ].lower(),
        "report_type": (
            "annual_report"
            if groups[
                "rtype"
            ].lower()
            == "annual"
            else "quarterly_report"
        ),
        "report_number": int(
            groups[
                "number"
            ]
        ),
        "prompt_type": groups[
            "prompt"
        ].lower(),
        "model_key": groups[
            "model"
        ].lower(),
        "corruption_type": (
            groups.get(
                "corruption"
            ).lower()
            if (
                paired
                and groups.get(
                    "corruption"
                )
            )
            else None
        ),
    }


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def log(
    message="",
):
    print(
        message,
        flush=True,
    )

    with WORKER_LOG.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            str(
                message
            )
            + "\n"
        )


def read_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(
            f
        )


def write_json(
    path: Path,
    data: dict,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def read_csv(
    path: Path,
):
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


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
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
        writer.writerows(
            rows
        )


def report_prefix(
    report_type: str,
):
    return (
        "annual"
        if report_type
        == "annual_report"
        else "quarterly"
    )


def discover_modified() -> list[dict]:
    if not MODIFIED_DIR.exists():
        raise FileNotFoundError(
            f"modified_answers folder not found:\n{MODIFIED_DIR}"
        )

    files = sorted(
        path
        for path in MODIFIED_DIR.rglob(
            "*.txt"
        )
        if path.is_file()
    )

    if len(
        files
    ) != 48:
        raise RuntimeError(
            f"Expected exactly 48 .txt files in modified_answers, found {len(files)}."
        )

    rows = [
        parse_modified(
            path
        )
        for path in files
    ]

    return rows


def build_original_index():
    if not ORIGINAL_DIR.exists():
        raise FileNotFoundError(
            f"original folder not found:\n{ORIGINAL_DIR}"
        )

    parsed_files = []

    for path in ORIGINAL_DIR.rglob(
        "*.txt"
    ):
        parsed = parse_original(
            path
        )

        if parsed is not None:
            parsed_files.append(
                parsed
            )

    if not parsed_files:
        raise RuntimeError(
            f"No parseable originals found in:\n{ORIGINAL_DIR}"
        )

    # Two supported layouts:
    #   A) 48 paired untouched originals, each retaining its corruption suffix.
    #   B) 12 base originals, one per selected model summary.
    paired_count = sum(
        1
        for item in parsed_files
        if item.get(
            "corruption_type"
        )
        is not None
    )

    base_count = (
        len(
            parsed_files
        )
        - paired_count
    )

    if (
        paired_count > 0
        and base_count > 0
    ):
        raise RuntimeError(
            "The original/ folder mixes paired 48-file originals and "
            "12-file base originals. Keep only one layout."
        )

    index = {}

    if paired_count > 0:
        if paired_count != 48:
            raise RuntimeError(
                f"Expected 48 paired untouched originals, found {paired_count}."
            )

        for parsed in parsed_files:
            key = (
                metadata_key(
                    parsed
                )
                + (
                    parsed[
                        "corruption_type"
                    ],
                )
            )

            if key in index:
                raise RuntimeError(
                    "Duplicate paired original metadata:\n"
                    f"  {index[key]['original_file']}\n"
                    f"  {parsed['original_file']}"
                )

            index[
                key
            ] = parsed

        return {
            "layout": "paired_48",
            "items": index,
        }

    if base_count != 12:
        raise RuntimeError(
            f"Expected 12 base originals, found {base_count}."
        )

    for parsed in parsed_files:
        key = metadata_key(
            parsed
        )

        if key in index:
            raise RuntimeError(
                "Duplicate base original metadata:\n"
                f"  {index[key]['original_file']}\n"
                f"  {parsed['original_file']}"
            )

        index[
            key
        ] = parsed

    return {
        "layout": "base_12",
        "items": index,
    }


def validate_structure(
    modified_rows: list[dict],
    original_index: dict,
):
    grouped = defaultdict(
        set
    )

    duplicate_keys = set()
    seen_modified = set()

    for row in modified_rows:
        base_key = metadata_key(
            row
        )

        perturbation_key = (
            base_key
            + (
                row[
                    "corruption_type"
                ],
            )
        )

        if perturbation_key in seen_modified:
            duplicate_keys.add(
                perturbation_key
            )

        seen_modified.add(
            perturbation_key
        )

        grouped[
            base_key
        ].add(
            row[
                "corruption_type"
            ]
        )

        original_items = original_index[
            "items"
        ]

        if original_index[
            "layout"
        ] == "paired_48":
            original_key = (
                base_key
                + (
                    row[
                        "corruption_type"
                    ],
                )
            )
        else:
            original_key = base_key

        if original_key not in original_items:
            raise RuntimeError(
                "No matching untouched original answer for:\n"
                f"  {row['modified_file']}"
            )

    if duplicate_keys:
        raise RuntimeError(
            f"Duplicate perturbation metadata found: {duplicate_keys}"
        )

    if len(
        grouped
    ) != 12:
        raise RuntimeError(
            f"Expected 12 unique original candidates, found {len(grouped)}."
        )

    expected = {
        "factual",
        "numerical",
        "completeness",
        "coherence",
    }

    problems = []

    for key, corruptions in grouped.items():
        if corruptions != expected:
            problems.append(
                (
                    key,
                    corruptions,
                )
            )

    if problems:
        raise RuntimeError(
            "Not every original has exactly the four required corruption types:\n"
            + "\n".join(
                f"  {key}: {sorted(values)}"
                for key, values in problems
            )
        )

    if MANIFEST.exists():
        manifest_rows = read_csv(
            MANIFEST
        )

        if len(
            manifest_rows
        ) != 48:
            raise RuntimeError(
                f"perturbation_manifest_48.csv has {len(manifest_rows)} rows, expected 48."
            )


# -----------------------------------------------------------------------------
# Existing main-experiment baseline
# -----------------------------------------------------------------------------

def baseline_root(
    generation_mode: str,
) -> Path:
    if generation_mode == "normal":
        return (
            DATA_DIR
            / "judge_results"
        )

    if generation_mode == "reasoning_high":
        return (
            DATA_DIR
            / "judge_results_reasoning_high"
        )

    raise ValueError(
        generation_mode
    )


def baseline_merged_path(
    row: dict,
) -> Path:
    prefix = report_prefix(
        row[
            "report_type"
        ]
    )

    filename = (
        f"report_{prefix}_"
        f"{row['report_number']}_"
        f"{row['prompt_type']}_"
        f"{row['model_key']}_judge.json"
    )

    root = baseline_root(
        row[
            "generation_mode"
        ]
    )

    category_candidates = [
        row[
            "report_type"
        ],
    ]

    if (
        row[
            "report_type"
        ]
        == "quarterly_report"
    ):
        category_candidates.append(
            "quartely_report"
        )

    direct_candidates = []

    for category in category_candidates:
        direct_candidates.extend(
            [
                (
                    root
                    / "nova_2_lite"
                    / category
                    / "merged"
                    / filename
                ),
                (
                    root
                    / category
                    / "merged"
                    / filename
                ),
            ]
        )

    for path in direct_candidates:
        if path.exists():
            return path

    matches = [
        path
        for path in root.rglob(
            filename
        )
        if path.is_file()
    ]

    if len(
        matches
    ) == 1:
        return matches[
            0
        ]

    raise FileNotFoundError(
        "Could not uniquely find main-experiment Nova merged baseline:\n"
        f"  generation_mode={row['generation_mode']}\n"
        f"  expected filename={filename}\n"
        f"  root={root}\n"
        f"  matches={matches}"
    )


def load_baseline(
    row: dict,
) -> tuple[dict, Path]:
    path = baseline_merged_path(
        row
    )

    data = read_json(
        path
    )

    if data.get(
        "status"
    ) != "success":
        raise RuntimeError(
            f"Baseline merged result is not success:\n{path}"
        )

    scores = data.get(
        "scores",
        {},
    )

    required = {
        "faithfulness",
        "numerical_accuracy",
        "completeness",
        "coherence",
    }

    if not required.issubset(
        scores
    ):
        raise RuntimeError(
            f"Baseline is missing one or more scores:\n{path}"
        )

    compact = {
        "faithfulness": scores[
            "faithfulness"
        ].get(
            "score"
        ),
        "numerical_accuracy": scores[
            "numerical_accuracy"
        ].get(
            "score"
        ),
        "completeness": scores[
            "completeness"
        ].get(
            "score"
        ),
        "coherence": scores[
            "coherence"
        ].get(
            "score"
        ),
    }

    return (
        compact,
        path,
    )


# -----------------------------------------------------------------------------
# Dry-run checks that do not import / call the Judge core
# -----------------------------------------------------------------------------

def dry_run():
    modified_rows = discover_modified()
    originals = build_original_index()

    validate_structure(
        modified_rows,
        originals,
    )

    baseline_paths = set()

    for row in modified_rows:
        baseline, path = load_baseline(
            row
        )

        baseline_paths.add(
            path.resolve()
        )

    print(
        "=" * 100
    )
    print(
        "PERTURBATION VALIDATION DRY RUN — NO API CALLS"
    )
    print(
        "=" * 100
    )
    print(
        f"Modified summaries:      {len(modified_rows)}"
    )
    print(
        f"Original files:          {len(originals['items'])}"
    )
    print(
        f"Original layout:         {originals['layout']}"
    )
    print(
        "Unique original summaries: 12"
    )
    print(
        f"Unique Nova baselines:   {len(baseline_paths)}"
    )
    print(
        f"Manifest:                {'OK (48 rows)' if MANIFEST.exists() else 'not present'}"
    )
    print()

    counts = defaultdict(
        int
    )

    mode_counts = defaultdict(
        int
    )

    report_counts = defaultdict(
        int
    )

    for row in modified_rows:
        counts[
            row[
                "corruption_type"
            ]
        ] += 1

        mode_counts[
            row[
                "generation_mode"
            ]
        ] += 1

        report_counts[
            row[
                "report_type"
            ]
        ] += 1

    print(
        "Corruption distribution:"
    )

    for corruption in [
        "factual",
        "numerical",
        "completeness",
        "coherence",
    ]:
        print(
            f"  {corruption:<14} {counts[corruption]}"
        )

    print()
    print(
        "Generation mode:"
    )

    for key in sorted(
        mode_counts
    ):
        print(
            f"  {key:<18} {mode_counts[key]}"
        )

    print()
    print(
        "Report type:"
    )

    for key in sorted(
        report_counts
    ):
        print(
            f"  {key:<18} {report_counts[key]}"
        )

    print()
    print(
        "STRUCTURE OK: 12 originals × 4 isolated corruption types = 48."
    )
    print(
        "No Bedrock request was made."
    )


# -----------------------------------------------------------------------------
# Judge core import (only needed for paid run)
# -----------------------------------------------------------------------------

core = None


def ensure_core():
    global core

    if core is not None:
        return core

    # The production core itself requires this key.
    if not os.getenv(
        "AWS_BEARER_TOKEN_BEDROCK"
    ):
        raise RuntimeError(
            "AWS_BEARER_TOKEN_BEDROCK is not set.\n"
            'PowerShell:\n$env:AWS_BEARER_TOKEN_BEDROCK="..."'
        )

    try:
        import nova_judge_core_v5_2 as imported_core
    except ImportError as exc:
        raise RuntimeError(
            "Could not import nova_judge_core_v5_2.py. "
            "Put this runner in the same scripts/ folder."
        ) from exc

    core = imported_core

    # Explicitly preserve the final production policy.
    core.SERVICE_TIER = "flex"
    core.CANDIDATE_TEMPERATURE = 0.0

    core.CANDIDATE_REASONING_MODES = dict(
        core.CANDIDATE_REASONING_MODES
    )

    core.CANDIDATE_REASONING_MODES[
        "factual_numerical"
    ] = "low"

    core.CANDIDATE_REASONING_MODES[
        "completeness"
    ] = "off"

    core.CANDIDATE_REASONING_MODES[
        "coherence"
    ] = "off"

    core.FORCE_RERUN_ALL = ARGS.force

    def perturbation_raw_result_path(
        evaluation_type: str,
        candidate_stem: str,
    ) -> Path:
        return (
            RAW_ROOT
            / evaluation_type
            / f"{candidate_stem}.json"
        )

    core.raw_result_path = (
        perturbation_raw_result_path
    )

    return core


# -----------------------------------------------------------------------------
# Source / fixed completeness rubric
# -----------------------------------------------------------------------------

def find_source_with_core(
    row: dict,
):
    c = ensure_core()

    c.CATEGORY = row[
        "report_type"
    ]

    c.REPORT_NUMBER = int(
        row[
            "report_number"
        ]
    )

    c.SUMMARY_PROMPT = row[
        "prompt_type"
    ]

    c.RUN_MODE = row[
        "generation_mode"
    ]

    return c.find_report_file()


def load_original_shared_rubric(
    source_path: Path,
):
    c = ensure_core()

    verified_path = c.key_verified_path(
        source_path
    )

    if not verified_path.exists():
        raise FileNotFoundError(
            "Original verified Completeness rubric missing:\n"
            f"{verified_path}"
        )

    verified = c.load_json(
        verified_path
    )

    if verified.get(
        "status"
    ) != "success":
        raise RuntimeError(
            f"Original verified rubric is not success:\n{verified_path}"
        )

    units = (
        verified.get(
            "evaluation",
            {},
        ).get(
            "key_information_units",
            [],
        )
        or []
    )

    if not units:
        raise RuntimeError(
            f"No key_information_units in:\n{verified_path}"
        )

    return (
        verified,
        units,
        verified_path,
    )


# -----------------------------------------------------------------------------
# Modified score extraction
# -----------------------------------------------------------------------------

def _claim_trace_ids_are_valid(
    claim: dict,
    sentence_map: dict[str, str],
) -> bool:
    sentence_ids = claim.get(
        "candidate_sentence_ids"
    )

    if not isinstance(
        sentence_ids,
        list,
    ) or not sentence_ids:
        return False

    valid_ids = set(
        sentence_map
    )

    return all(
        isinstance(
            sentence_id,
            str,
        )
        and sentence_id in valid_ids
        for sentence_id in sentence_ids
    )


def _selective_faithfulness_score(
    evaluation: dict,
    sentence_map: dict[str, str],
):
    """
    Compute Faithfulness independently of the numerical layer.

    This is intentionally used only for the target-only perturbation test.
    The combined Factual/Numerical core validator can reject the whole JSON
    because of an invalid NUMERICAL label or a numerical coverage warning.
    Those failures do not invalidate a fully well-formed factual layer.
    """
    claims = evaluation.get(
        "factual_claims"
    )

    if not isinstance(
        claims,
        list,
    ) or not claims:
        return None

    valid_labels = {
        "SUPPORTED",
        "CONTRADICTED",
        "NOT_FOUND",
    }

    supported = 0

    for claim in claims:
        if not isinstance(
            claim,
            dict,
        ):
            return None

        claim_text = str(
            claim.get(
                "claim",
                "",
            )
        ).strip()

        label = str(
            claim.get(
                "label",
                "",
            )
        ).strip().upper()

        if (
            not claim_text
            or label not in valid_labels
            or not _claim_trace_ids_are_valid(
                claim,
                sentence_map,
            )
        ):
            return None

        if label == "SUPPORTED":
            supported += 1

    return supported / len(
        claims
    )


def _selective_numerical_score(
    evaluation: dict,
    sentence_map: dict[str, str],
):
    """
    Compute Numerical Accuracy independently of the factual layer.

    Numerical score is accepted only if:
      - all numerical claims use CORRECT/INCORRECT;
      - trace IDs are valid;
      - the production numerical sentence-coverage guard passes.

    Therefore a factual taxonomy error such as PARTIALLY_SUPPORTED cannot
    invalidate an otherwise complete numerical target result, while a true
    numerical_coverage_failed result is still rejected for Numerical Accuracy.
    """
    c = ensure_core()

    claims = evaluation.get(
        "numerical_claims"
    )

    if not isinstance(
        claims,
        list,
    ) or not claims:
        return None

    valid_labels = {
        "CORRECT",
        "INCORRECT",
    }

    correct = 0

    for claim in claims:
        if not isinstance(
            claim,
            dict,
        ):
            return None

        claim_text = str(
            claim.get(
                "claim",
                "",
            )
        ).strip()

        label = str(
            claim.get(
                "label",
                "",
            )
        ).strip().upper()

        if (
            not claim_text
            or label not in valid_labels
            or not _claim_trace_ids_are_valid(
                claim,
                sentence_map,
            )
        ):
            return None

        if label == "CORRECT":
            correct += 1

    try:
        coverage = c.numerical_coverage_check_v52(
            evaluation,
            sentence_map,
        )
    except Exception:
        return None

    if not coverage.get(
        "ok",
        False,
    ):
        return None

    return correct / len(
        claims
    )


def extract_scores_from_raw(
    raw_results: dict,
    sentence_map: dict[str, str],
) -> tuple[dict, dict]:
    c = ensure_core()

    scores = {
        "faithfulness": None,
        "numerical_accuracy": None,
        "completeness": None,
        "coherence": None,
    }

    recovery = {
        "faithfulness": None,
        "numerical_accuracy": None,
        "completeness": None,
        "coherence": None,
    }

    factual_raw = raw_results.get(
        "factual_numerical"
    )

    if factual_raw:
        evaluation = factual_raw.get(
            "evaluation"
        )

        if isinstance(
            evaluation,
            dict,
        ):
            if factual_raw.get(
                "status"
            ) == "success":
                scores[
                    "faithfulness"
                ] = c.calculate_factual_score(
                    evaluation
                )[
                    "score"
                ]

                scores[
                    "numerical_accuracy"
                ] = c.calculate_numerical_score(
                    evaluation
                )[
                    "score"
                ]

                recovery[
                    "faithfulness"
                ] = "full_core_success"

                recovery[
                    "numerical_accuracy"
                ] = "full_core_success"

            else:
                # Target-specific local salvage. No new Judge request.
                factual_score = (
                    _selective_faithfulness_score(
                        evaluation,
                        sentence_map,
                    )
                )

                numerical_score = (
                    _selective_numerical_score(
                        evaluation,
                        sentence_map,
                    )
                )

                if factual_score is not None:
                    scores[
                        "faithfulness"
                    ] = factual_score

                    recovery[
                        "faithfulness"
                    ] = (
                        "selective_local_salvage_from_"
                        + str(
                            factual_raw.get(
                                "status"
                            )
                        )
                    )

                if numerical_score is not None:
                    scores[
                        "numerical_accuracy"
                    ] = numerical_score

                    recovery[
                        "numerical_accuracy"
                    ] = (
                        "selective_local_salvage_from_"
                        + str(
                            factual_raw.get(
                                "status"
                            )
                        )
                    )

    completeness_raw = raw_results.get(
        "completeness"
    )

    if (
        completeness_raw
        and completeness_raw.get(
            "status"
        )
        == "success"
    ):
        scores[
            "completeness"
        ] = c.parse_rubric_score(
            completeness_raw[
                "evaluation"
            ]
        )

        recovery[
            "completeness"
        ] = "full_core_success"

    coherence_raw = raw_results.get(
        "coherence"
    )

    if (
        coherence_raw
        and coherence_raw.get(
            "status"
        )
        == "success"
    ):
        scores[
            "coherence"
        ] = c.parse_rubric_score(
            coherence_raw[
                "evaluation"
            ]
        )

        recovery[
            "coherence"
        ] = "full_core_success"

    return (
        scores,
        recovery,
    )


# -----------------------------------------------------------------------------
# One perturbation
# -----------------------------------------------------------------------------

def result_json_path(
    row: dict,
) -> Path:
    return (
        RESULTS_ROOT
        / (
            row[
                "modified_file"
            ].stem
            + ".json"
        )
    )


def required_evaluations(
    corruption_type: str,
):
    if ARGS.all_metrics:
        return list(
            ALL_EVALUATIONS
        )

    return list(
        TARGET_EVALUATIONS[
            corruption_type
        ]
    )


def process_one(
    row: dict,
    original_index: dict,
):
    c = ensure_core()

    out_path = result_json_path(
        row
    )

    if (
        out_path.exists()
        and not ARGS.force
    ):
        existing = read_json(
            out_path
        )

        expected_mode = (
            "all_metrics"
            if ARGS.all_metrics
            else "target_only"
        )

        if (
            existing.get(
                "status"
            )
            == "success"
            and existing.get(
                "evaluation_scope"
            )
            == expected_mode
        ):
            log(
                f"SKIP existing success: {row['modified_file'].name}"
            )

            return existing

    base_key = metadata_key(
        row
    )

    if original_index[
        "layout"
    ] == "paired_48":
        original_key = (
            base_key
            + (
                row[
                    "corruption_type"
                ],
            )
        )
    else:
        original_key = base_key

    original = original_index[
        "items"
    ][
        original_key
    ]

    baseline_scores, baseline_file = load_baseline(
        row
    )

    c.CATEGORY = row[
        "report_type"
    ]

    c.REPORT_NUMBER = int(
        row[
            "report_number"
        ]
    )

    c.SUMMARY_PROMPT = row[
        "prompt_type"
    ]

    c.RUN_MODE = row[
        "generation_mode"
    ]

    source_path = c.find_report_file()

    source_report = c.read_text(
        source_path
    )

    candidate_path = row[
        "modified_file"
    ]

    candidate_summary = c.read_text(
        candidate_path
    )

    sentence_map = c.build_candidate_sentence_map(
        candidate_summary
    )

    prompts = c.load_judge_prompts()

    verified, final_units, verified_file = (
        load_original_shared_rubric(
            source_path
        )
    )

    key_hash = verified.get(
        "key_information_hash"
    )

    evaluations = required_evaluations(
        row[
            "corruption_type"
        ]
    )

    log()
    log(
        "=" * 105
    )
    log(
        (
            f"{row['modified_file'].name}\n"
            f"corruption={row['corruption_type']} | "
            f"target={TARGET_METRIC[row['corruption_type']]} | "
            f"scope={'all_metrics' if ARGS.all_metrics else 'target_only'}"
        )
    )
    log(
        "=" * 105
    )

    raw_results = {}
    all_success = True

    for evaluation_type in evaluations:
        log(
            (
                f"{evaluation_type} | "
                f"reasoning={c.CANDIDATE_REASONING_MODES[evaluation_type]} | "
                f"tier={c.SERVICE_TIER}"
            )
        )

        raw = c.run_candidate_evaluation(
            evaluation_type=evaluation_type,
            instructions=prompts[
                evaluation_type
            ],
            source_report=source_report,
            candidate_summary=candidate_summary,
            sentence_map=sentence_map,
            source_path=source_path,
            candidate_path=candidate_path,
            generator_model_name=GENERATOR_NAMES[
                row[
                    "model_key"
                ]
            ],
            final_units=(
                final_units
                if evaluation_type
                == "completeness"
                else None
            ),
            key_information_hash_value=(
                key_hash
                if evaluation_type
                == "completeness"
                else None
            ),
            key_information_file=(
                verified_file
                if evaluation_type
                == "completeness"
                else None
            ),
        )

        raw_results[
            evaluation_type
        ] = raw

        if raw.get(
            "status"
        ) != "success":
            all_success = False

        if c.REQUEST_DELAY_SECONDS > 0:
            time.sleep(
                c.REQUEST_DELAY_SECONDS
            )

    (
        modified_scores,
        score_recovery,
    ) = extract_scores_from_raw(
        raw_results,
        sentence_map,
    )

    target = TARGET_METRIC[
        row[
            "corruption_type"
        ]
    ]

    secondary = SECONDARY_METRIC[
        row[
            "corruption_type"
        ]
    ]

    baseline_target = baseline_scores[
        target
    ]

    modified_target = modified_scores[
        target
    ]

    target_drop = (
        (
            float(
                baseline_target
            )
            - float(
                modified_target
            )
        )
        if (
            baseline_target
            is not None
            and modified_target
            is not None
        )
        else None
    )

    target_delta = (
        -target_drop
        if target_drop
        is not None
        else None
    )

    detectability_possible = (
        baseline_target
        is not None
        and float(
            baseline_target
        )
        > MIN_SCORE[
            target
        ]
    )

    detected = (
        target_drop
        is not None
        and target_drop > 0
    )

    nonincrease = (
        target_drop
        is not None
        and target_drop >= 0
    )

    result = {
        # In target-only mode the experiment is successful when the TARGET
        # metric is valid, even if the shared Factual/Numerical request was
        # rejected because only the irrelevant secondary layer failed.
        #
        # Example:
        #   factual_corruption target = Faithfulness
        #   numerical coverage failure != invalid Faithfulness result
        #
        # In --all-metrics mode we retain the original strict behavior.
        "status": (
            "success"
            if (
                modified_target
                is not None
                and (
                    not ARGS.all_metrics
                    or all_success
                )
            )
            else "failed"
        ),
        "evaluation_scope": (
            "all_metrics"
            if ARGS.all_metrics
            else "target_only"
        ),
        "metadata": {
            "generation_mode": row[
                "generation_mode"
            ],
            "report_type": row[
                "report_type"
            ],
            "report_number": row[
                "report_number"
            ],
            "prompt_type": row[
                "prompt_type"
            ],
            "model_key": row[
                "model_key"
            ],
            "generator_model": GENERATOR_NAMES[
                row[
                    "model_key"
                ]
            ],
            "corruption_type": row[
                "corruption_type"
            ],
            "target_metric": target,
            "secondary_metric": secondary,
        },
        "files": {
            "modified": str(
                candidate_path.resolve()
            ),
            "original": str(
                original[
                    "original_file"
                ].resolve()
            ),
            "source_report": str(
                source_path.resolve()
            ),
            "baseline_merged": str(
                baseline_file.resolve()
            ),
            "fixed_completeness_rubric": str(
                verified_file.resolve()
            ),
        },
        "baseline_scores": baseline_scores,
        "modified_scores": modified_scores,
        "effect": {
            "baseline_target": baseline_target,
            "modified_target": modified_target,
            "target_drop_baseline_minus_modified": target_drop,
            "target_delta_modified_minus_baseline": target_delta,
            "detectability_possible": (
                detectability_possible
            ),
            "detected_strict_decrease": (
                detected
            ),
            "nonincrease": (
                nonincrease
            ),
        },
        "raw_status": {
            key: value.get(
                "status"
            )
            for key, value in raw_results.items()
        },
        "score_recovery": score_recovery,
        "target_score_source": score_recovery.get(
            target
        ),
    }

    if secondary is not None:
        baseline_secondary = baseline_scores[
            secondary
        ]

        modified_secondary = modified_scores[
            secondary
        ]

        result[
            "effect"
        ][
            "baseline_secondary"
        ] = baseline_secondary

        result[
            "effect"
        ][
            "modified_secondary"
        ] = modified_secondary

        result[
            "effect"
        ][
            "secondary_delta_modified_minus_baseline"
        ] = (
            (
                float(
                    modified_secondary
                )
                - float(
                    baseline_secondary
                )
            )
            if (
                baseline_secondary
                is not None
                and modified_secondary
                is not None
            )
            else None
        )

    write_json(
        out_path,
        result,
    )

    log(
        (
            f"RESULT | baseline={baseline_target} | "
            f"modified={modified_target} | "
            f"drop={target_drop} | "
            f"detected={detected} | "
            f"score_source={score_recovery.get(target)}"
        )
    )

    return result


# -----------------------------------------------------------------------------
# Compact result row
# -----------------------------------------------------------------------------

RESULT_FIELDS = [
    "modified_file",
    "original_file",
    "generation_mode",
    "report_type",
    "report_number",
    "prompt_type",
    "model_key",
    "generator_model",
    "corruption_type",
    "target_metric",
    "secondary_metric",

    "baseline_faithfulness",
    "modified_faithfulness",
    "baseline_numerical_accuracy",
    "modified_numerical_accuracy",
    "baseline_completeness",
    "modified_completeness",
    "baseline_coherence",
    "modified_coherence",

    "baseline_target",
    "modified_target",
    "target_drop",
    "target_delta",
    "detectability_possible",
    "detected_strict_decrease",
    "nonincrease",

    "baseline_secondary",
    "modified_secondary",
    "secondary_delta",

    "evaluation_scope",
    "result_json",
]


def result_to_row(
    result: dict,
    result_path: Path,
):
    meta = result[
        "metadata"
    ]

    files = result[
        "files"
    ]

    baseline = result[
        "baseline_scores"
    ]

    modified = result[
        "modified_scores"
    ]

    effect = result[
        "effect"
    ]

    return {
        "modified_file": files[
            "modified"
        ],
        "original_file": files[
            "original"
        ],
        "generation_mode": meta[
            "generation_mode"
        ],
        "report_type": meta[
            "report_type"
        ],
        "report_number": meta[
            "report_number"
        ],
        "prompt_type": meta[
            "prompt_type"
        ],
        "model_key": meta[
            "model_key"
        ],
        "generator_model": meta[
            "generator_model"
        ],
        "corruption_type": meta[
            "corruption_type"
        ],
        "target_metric": meta[
            "target_metric"
        ],
        "secondary_metric": (
            meta.get(
                "secondary_metric"
            )
            or ""
        ),

        "baseline_faithfulness": baseline.get(
            "faithfulness"
        ),
        "modified_faithfulness": modified.get(
            "faithfulness"
        ),
        "baseline_numerical_accuracy": baseline.get(
            "numerical_accuracy"
        ),
        "modified_numerical_accuracy": modified.get(
            "numerical_accuracy"
        ),
        "baseline_completeness": baseline.get(
            "completeness"
        ),
        "modified_completeness": modified.get(
            "completeness"
        ),
        "baseline_coherence": baseline.get(
            "coherence"
        ),
        "modified_coherence": modified.get(
            "coherence"
        ),

        "baseline_target": effect.get(
            "baseline_target"
        ),
        "modified_target": effect.get(
            "modified_target"
        ),
        "target_drop": effect.get(
            "target_drop_baseline_minus_modified"
        ),
        "target_delta": effect.get(
            "target_delta_modified_minus_baseline"
        ),
        "detectability_possible": int(
            bool(
                effect.get(
                    "detectability_possible"
                )
            )
        ),
        "detected_strict_decrease": int(
            bool(
                effect.get(
                    "detected_strict_decrease"
                )
            )
        ),
        "nonincrease": int(
            bool(
                effect.get(
                    "nonincrease"
                )
            )
        ),

        "baseline_secondary": effect.get(
            "baseline_secondary",
            "",
        ),
        "modified_secondary": effect.get(
            "modified_secondary",
            "",
        ),
        "secondary_delta": effect.get(
            "secondary_delta_modified_minus_baseline",
            "",
        ),

        "evaluation_scope": result[
            "evaluation_scope"
        ],
        "result_json": str(
            result_path.resolve()
        ),
    }


# -----------------------------------------------------------------------------
# Collect / summary
# -----------------------------------------------------------------------------

def load_all_saved_results(
    modified_rows: list[dict],
):
    results = []
    missing = []

    expected_scope = (
        "all_metrics"
        if ARGS.all_metrics
        else "target_only"
    )

    for row in modified_rows:
        path = result_json_path(
            row
        )

        if not path.exists():
            missing.append(
                f"{row['modified_file'].name}: result missing"
            )
            continue

        data = read_json(
            path
        )

        if data.get(
            "status"
        ) != "success":
            missing.append(
                (
                    f"{row['modified_file'].name}: "
                    f"status={data.get('status')}"
                )
            )
            continue

        if data.get(
            "evaluation_scope"
        ) != expected_scope:
            missing.append(
                (
                    f"{row['modified_file'].name}: "
                    f"scope={data.get('evaluation_scope')} "
                    f"(expected {expected_scope})"
                )
            )
            continue

        results.append(
            (
                data,
                path,
            )
        )

    if missing:
        print()
        print(
            f"COLLECT INCOMPLETE: {len(results)}/48 successful."
        )

        for item in missing:
            print(
                "  "
                + item
            )

        raise SystemExit(
            1
        )

    return results


def mean_or_none(
    values: list[float],
):
    if not values:
        return None

    return statistics.mean(
        values
    )


def build_summary_rows(
    result_rows: list[dict],
):
    output = []

    for corruption in [
        "factual",
        "numerical",
        "completeness",
        "coherence",
    ]:
        rows = [
            row
            for row in result_rows
            if row[
                "corruption_type"
            ]
            == corruption
        ]

        eligible = [
            row
            for row in rows
            if int(
                row[
                    "detectability_possible"
                ]
            )
            == 1
        ]

        drops = [
            float(
                row[
                    "target_drop"
                ]
            )
            for row in rows
            if row[
                "target_drop"
            ]
            not in {
                "",
                None,
            }
        ]

        eligible_detected = sum(
            int(
                row[
                    "detected_strict_decrease"
                ]
            )
            for row in eligible
        )

        detected_all = sum(
            int(
                row[
                    "detected_strict_decrease"
                ]
            )
            for row in rows
        )

        nonincrease = sum(
            int(
                row[
                    "nonincrease"
                ]
            )
            for row in rows
        )

        reverse = sum(
            1
            for row in rows
            if (
                row[
                    "target_drop"
                ]
                not in {
                    "",
                    None,
                }
                and float(
                    row[
                        "target_drop"
                    ]
                )
                < 0
            )
        )

        secondary_rows = [
            row
            for row in rows
            if row[
                "secondary_delta"
            ]
            not in {
                "",
                None,
            }
        ]

        secondary_exact = sum(
            1
            for row in secondary_rows
            if abs(
                float(
                    row[
                        "secondary_delta"
                    ]
                )
            )
            <= 1e-12
        )

        secondary_abs_changes = [
            abs(
                float(
                    row[
                        "secondary_delta"
                    ]
                )
            )
            for row in secondary_rows
        ]

        output.append(
            {
                "corruption_type": corruption,
                "target_metric": TARGET_METRIC[
                    corruption
                ],
                "n": len(
                    rows
                ),
                "detectability_eligible_n": len(
                    eligible
                ),
                "baseline_target_mean": round(
                    statistics.mean(
                        float(
                            row[
                                "baseline_target"
                            ]
                        )
                        for row in rows
                    ),
                    6,
                ),
                "modified_target_mean": round(
                    statistics.mean(
                        float(
                            row[
                                "modified_target"
                            ]
                        )
                        for row in rows
                    ),
                    6,
                ),
                "mean_target_drop": round(
                    statistics.mean(
                        drops
                    ),
                    6,
                ),
                "median_target_drop": round(
                    statistics.median(
                        drops
                    ),
                    6,
                ),
                "strict_detection_count": detected_all,
                "strict_detection_rate_pct": round(
                    100.0
                    * detected_all
                    / len(
                        rows
                    ),
                    3,
                ),
                "eligible_detection_rate_pct": (
                    round(
                        100.0
                        * eligible_detected
                        / len(
                            eligible
                        ),
                        3,
                    )
                    if eligible
                    else ""
                ),
                "nonincrease_count": nonincrease,
                "nonincrease_rate_pct": round(
                    100.0
                    * nonincrease
                    / len(
                        rows
                    ),
                    3,
                ),
                "reverse_direction_count": reverse,
                "secondary_metric": (
                    SECONDARY_METRIC[
                        corruption
                    ]
                    or ""
                ),
                "secondary_n": len(
                    secondary_rows
                ),
                "secondary_exact_unchanged_pct": (
                    round(
                        100.0
                        * secondary_exact
                        / len(
                            secondary_rows
                        ),
                        3,
                    )
                    if secondary_rows
                    else ""
                ),
                "secondary_mean_absolute_change": (
                    round(
                        statistics.mean(
                            secondary_abs_changes
                        ),
                        6,
                    )
                    if secondary_abs_changes
                    else ""
                ),
            }
        )

    return output


def build_per_original_rows(
    result_rows: list[dict],
):
    grouped = defaultdict(
        list
    )

    for row in result_rows:
        key = (
            row[
                "generation_mode"
            ],
            row[
                "report_type"
            ],
            row[
                "report_number"
            ],
            row[
                "prompt_type"
            ],
            row[
                "model_key"
            ],
        )

        grouped[
            key
        ].append(
            row
        )

    output = []

    for key, rows in sorted(
        grouped.items()
    ):
        by_corruption = {
            row[
                "corruption_type"
            ]: row
            for row in rows
        }

        if len(
            by_corruption
        ) != 4:
            raise RuntimeError(
                f"Original group {key} does not contain 4 corruption results."
            )

        output.append(
            {
                "generation_mode": key[
                    0
                ],
                "report_type": key[
                    1
                ],
                "report_number": key[
                    2
                ],
                "prompt_type": key[
                    3
                ],
                "model_key": key[
                    4
                ],

                "factual_baseline": by_corruption[
                    "factual"
                ][
                    "baseline_target"
                ],
                "factual_modified": by_corruption[
                    "factual"
                ][
                    "modified_target"
                ],
                "factual_drop": by_corruption[
                    "factual"
                ][
                    "target_drop"
                ],
                "factual_detected": by_corruption[
                    "factual"
                ][
                    "detected_strict_decrease"
                ],

                "numerical_baseline": by_corruption[
                    "numerical"
                ][
                    "baseline_target"
                ],
                "numerical_modified": by_corruption[
                    "numerical"
                ][
                    "modified_target"
                ],
                "numerical_drop": by_corruption[
                    "numerical"
                ][
                    "target_drop"
                ],
                "numerical_detected": by_corruption[
                    "numerical"
                ][
                    "detected_strict_decrease"
                ],

                "completeness_baseline": by_corruption[
                    "completeness"
                ][
                    "baseline_target"
                ],
                "completeness_modified": by_corruption[
                    "completeness"
                ][
                    "modified_target"
                ],
                "completeness_drop": by_corruption[
                    "completeness"
                ][
                    "target_drop"
                ],
                "completeness_detected": by_corruption[
                    "completeness"
                ][
                    "detected_strict_decrease"
                ],

                "coherence_baseline": by_corruption[
                    "coherence"
                ][
                    "baseline_target"
                ],
                "coherence_modified": by_corruption[
                    "coherence"
                ][
                    "modified_target"
                ],
                "coherence_drop": by_corruption[
                    "coherence"
                ][
                    "target_drop"
                ],
                "coherence_detected": by_corruption[
                    "coherence"
                ][
                    "detected_strict_decrease"
                ],

                "detected_criteria_count": sum(
                    int(
                        row[
                            "detected_strict_decrease"
                        ]
                    )
                    for row in rows
                ),
            }
        )

    return output


def build_text_report(
    summary_rows: list[dict],
    result_rows: list[dict],
):
    lines = [
        "CONTROLLED PERTURBATION VALIDATION — NOVA 2 LITE",
        "=" * 84,
        "",
        "Design:",
        "  12 untouched model summaries × 4 isolated manual corruption types.",
        "  Each corrupted file changes one intended criterion only.",
        "  Baseline scores are the exact original Nova scores from the main experiment.",
        (
            "  Evaluation scope: "
            + (
                "ALL METRICS"
                if ARGS.all_metrics
                else "TARGET ONLY"
            )
        ),
        "",
        "Sensitivity criterion:",
        "  detected = target score strictly decreases after the manual corruption.",
        "",
        "RESULTS",
        "-" * 84,
    ]

    for row in summary_rows:
        lines.append(
            (
                f"{row['corruption_type']:<14} -> "
                f"{row['target_metric']:<20} | "
                f"baseline={row['baseline_target_mean']:.3f} | "
                f"modified={row['modified_target_mean']:.3f} | "
                f"mean drop={row['mean_target_drop']:.3f} | "
                f"detected={row['strict_detection_count']}/{row['n']} "
                f"({row['strict_detection_rate_pct']:.1f}%) | "
                f"nonincrease={row['nonincrease_rate_pct']:.1f}%"
            )
        )

        if row[
            "secondary_metric"
        ]:
            lines.append(
                (
                    f"  secondary {row['secondary_metric']}: "
                    f"exactly unchanged="
                    f"{row['secondary_exact_unchanged_pct']}% | "
                    f"mean abs change="
                    f"{row['secondary_mean_absolute_change']}"
                )
            )

    total_detected = sum(
        int(
            row[
                "detected_strict_decrease"
            ]
        )
        for row in result_rows
    )

    lines.extend(
        [
            "",
            (
                f"Overall targeted detections: "
                f"{total_detected}/48 "
                f"({100.0 * total_detected / 48:.1f}%)"
            ),
            "",
            "INTERPRETATION",
            "-" * 84,
            (
                "This experiment is a controlled sensitivity/validity check, not another "
                "self-consistency test. A high detection rate means the Judge metric "
                "changes in the expected direction when a known criterion-specific "
                "defect is deliberately introduced."
            ),
            (
                "Failure to decrease in an individual case should be inspected manually: "
                "the corruption may be too small for a 0-4 rubric step, the baseline may "
                "already be low, or the Judge may have failed to detect the introduced defect."
            ),
        ]
    )

    return "\n".join(
        lines
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    if ARGS.dry_run:
        dry_run()
        return

    modified_rows = discover_modified()
    originals = build_original_index()

    validate_structure(
        modified_rows,
        originals,
    )

    modified_rows.sort(
        key=lambda row: (
            row[
                "generation_mode"
            ],
            row[
                "report_type"
            ],
            row[
                "report_number"
            ],
            row[
                "prompt_type"
            ],
            row[
                "model_key"
            ],
            row[
                "corruption_type"
            ],
        )
    )

    if ARGS.collect:
        saved = load_all_saved_results(
            modified_rows
        )

        result_rows = [
            result_to_row(
                data,
                path,
            )
            for data, path in saved
        ]

        write_csv(
            FINAL_RESULTS_CSV,
            result_rows,
            RESULT_FIELDS,
        )

        summary_rows = build_summary_rows(
            result_rows
        )

        write_csv(
            SUMMARY_CSV,
            summary_rows,
            list(
                summary_rows[
                    0
                ].keys()
            ),
        )

        per_original_rows = build_per_original_rows(
            result_rows
        )

        write_csv(
            PER_ORIGINAL_CSV,
            per_original_rows,
            list(
                per_original_rows[
                    0
                ].keys()
            ),
        )

        report_text = build_text_report(
            summary_rows,
            result_rows,
        )

        TEXT_REPORT.write_text(
            report_text,
            encoding="utf-8",
        )

        print(
            "=" * 105
        )
        print(
            "CONTROLLED PERTURBATION COLLECT COMPLETE"
        )
        print(
            "=" * 105
        )
        print(
            "Successful perturbations: 48/48"
        )
        print()

        for row in summary_rows:
            print(
                (
                    f"{row['corruption_type']:<14} -> "
                    f"{row['target_metric']:<20} | "
                    f"drop={row['mean_target_drop']:.4f} | "
                    f"detected={row['strict_detection_rate_pct']:.1f}% | "
                    f"nonincrease={row['nonincrease_rate_pct']:.1f}%"
                )
            )

        print()
        print(
            f"Results:      {FINAL_RESULTS_CSV}"
        )
        print(
            f"Summary:      {SUMMARY_CSV}"
        )
        print(
            f"Per original: {PER_ORIGINAL_CSV}"
        )
        print(
            f"Report:       {TEXT_REPORT}"
        )

        return

    # Paid run starts only here.
    ensure_core()

    selected = []

    for index, row in enumerate(
        modified_rows,
        start=1,
    ):
        assigned = (
            (
                index
                - 1
            )
            % ARGS.workers
            + 1
        )

        if assigned == ARGS.worker:
            selected.append(
                row
            )

    log(
        "=" * 105
    )
    log(
        "CONTROLLED PERTURBATION VALIDATION — FINAL EXPERIMENT"
    )
    log(
        "=" * 105
    )
    log(
        f"Worker:              {ARGS.worker}/{ARGS.workers}"
    )
    log(
        f"Modified answers:    {len(selected)}"
    )
    log(
        (
            "Evaluation scope:    "
            + (
                "ALL METRICS (3 requests / answer)"
                if ARGS.all_metrics
                else "TARGET ONLY (~1 logical request / answer)"
            )
        )
    )
    log(
        "Factual/Numerical:   LOW reasoning"
    )
    log(
        "Completeness:        OFF reasoning"
    )
    log(
        "Coherence:           OFF reasoning"
    )
    log(
        "Temperature:         0"
    )
    log(
        "Tier:                FLEX"
    )
    log(
        "Baseline:            reused from main experiment"
    )
    log(
        "Completeness rubric: reused from main experiment"
    )
    log(
        "=" * 105
    )

    successes = []
    failures = []

    for row in selected:
        try:
            result = process_one(
                row,
                originals,
            )

            path = result_json_path(
                row
            )

            if result.get(
                "status"
            ) == "success":
                successes.append(
                    result_to_row(
                        result,
                        path,
                    )
                )

            else:
                failures.append(
                    (
                        row[
                            "modified_file"
                        ].name,
                        "one or more required Judge evaluations failed",
                    )
                )

        except Exception as exc:
            failures.append(
                (
                    row[
                        "modified_file"
                    ].name,
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                )
            )

            log(
                (
                    f"FAILED | "
                    f"{row['modified_file'].name} | "
                    f"{type(exc).__name__}: {exc}"
                )
            )

    write_csv(
        WORKER_CSV,
        successes,
        RESULT_FIELDS,
    )

    log()
    log(
        "=" * 105
    )
    log(
        "WORKER COMPLETE"
    )
    log(
        "=" * 105
    )
    log(
        f"Successful: {len(successes)}/{len(selected)}"
    )
    log(
        f"Failures:   {len(failures)}"
    )
    log(
        f"Worker CSV: {WORKER_CSV}"
    )

    if failures:
        log()
        log(
            "FAILED ITEMS:"
        )

        for filename, error in failures:
            log(
                f"  {filename} | {error}"
            )


if __name__ == "__main__":
    main()
