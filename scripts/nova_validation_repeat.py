from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path



def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Repeat Amazon Nova 2 Lite Judge evaluation for the fixed "
            "72-answer validation subset."
        )
    )

    parser.add_argument(
        "--run-number",
        type=int,
        choices=[2, 3],
        required=True,
        help="Independent repeat number: 2 or 3.",
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
        "--collect",
        action="store_true",
        help=(
            "Do not call Nova. Collect all 72 successful merged JSON files "
            "for this run into nova_runN_scores_72.csv."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rerun successful repeat-run results. Normally do NOT use this."
        ),
    )

    return parser.parse_args()


ARGS = parse_args()

if ARGS.worker < 1:
    raise SystemExit("--worker must be >= 1")

if ARGS.workers < 1:
    raise SystemExit("--workers must be >= 1")

if ARGS.worker > ARGS.workers:
    raise SystemExit("--worker cannot be greater than --workers")


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

VALIDATION_ROOT = (
    DATA_DIR
    / "judge_validation"
    / "consistency_crossjudge_72"
)

MANIFEST = (
    VALIDATION_ROOT
    / "validation_subset_72.csv"
)

ANSWERS_DIR = (
    VALIDATION_ROOT
    / "answers"
)

RUN_ROOT = (
    VALIDATION_ROOT
    / f"nova_run{ARGS.run_number}"
)

WORKER_RESULTS_DIR = (
    RUN_ROOT
    / "worker_results"
)

WORKER_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

WORKER_CSV = (
    WORKER_RESULTS_DIR
    / (
        f"nova_run{ARGS.run_number}_"
        f"worker_{ARGS.worker}_of_{ARGS.workers}.csv"
    )
)

WORKER_LOG = (
    WORKER_RESULTS_DIR
    / (
        f"nova_run{ARGS.run_number}_"
        f"worker_{ARGS.worker}_of_{ARGS.workers}.log.txt"
    )
)

FINAL_CSV = (
    VALIDATION_ROOT
    / f"nova_run{ARGS.run_number}_scores_72.csv"
)

os.environ.setdefault(
    "NOVA_JUDGE_RUN_MODE",
    "normal",
)

try:
    import scripts.nova_judge_v5_2 as core
except ImportError as exc:
    raise SystemExit(
        "\nCould not import nova_judge_core_v5_2.py.\n"
        "Put this runner in the same scripts/ folder as "
        "nova_judge_core_v5_2.py.\n"
    ) from exc


GENERATOR_NAMES = {
    "openai": "GPT-5.6 Luna",
    "gemini": "Gemini 3.5 Flash-Lite",
    "qwen36": "Qwen3.6-35B-A3B",
}

FLEX_INPUT_PER_M = 0.15
FLEX_OUTPUT_PER_M = 1.25

core.FORCE_RERUN_SHARED = False

core.JUDGE_RESULTS_ROOT = RUN_ROOT

core.FORCE_RERUN_ALL = ARGS.force


def log(message=""):
    print(
        message,
        flush=True,
    )

    with WORKER_LOG.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            str(message)
            + "\n"
        )


def estimated_flex_cost(
    input_tokens: int,
    output_tokens: int,
) -> float:
    return (
        input_tokens
        / 1_000_000
        * FLEX_INPUT_PER_M
        + output_tokens
        / 1_000_000
        * FLEX_OUTPUT_PER_M
    )


def load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Validation manifest not found:\n{MANIFEST}\n\n"
            "Run prepare_judge_validation_samples.py first."
        )

    with MANIFEST.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(
            csv.DictReader(
                f
            )
        )

    if len(rows) != 72:
        raise RuntimeError(
            f"Expected 72 rows in validation manifest, found {len(rows)}."
        )

    ids = [
        row["selection_id"]
        for row in rows
    ]

    if len(set(ids)) != 72:
        raise RuntimeError(
            "validation_subset_72.csv contains duplicate selection_id values."
        )

    return rows


def selection_number(
    selection_id: str,
) -> int:
    try:
        return int(
            selection_id.lstrip(
                "Vv"
            )
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid selection_id: {selection_id}"
        ) from exc


def rows_for_this_worker(
    rows: list[dict],
) -> list[dict]:
    """
    Deterministic modulo sharding.

    With 4 workers:
      worker 1 -> 18 answers
      worker 2 -> 18 answers
      worker 3 -> 18 answers
      worker 4 -> 18 answers
    """
    result = []

    for row in rows:
        number = selection_number(
            row[
                "selection_id"
            ]
        )

        assigned_worker = (
            (number - 1)
            % ARGS.workers
            + 1
        )

        if assigned_worker == ARGS.worker:
            result.append(
                row
            )

    return result


def answer_path_from_manifest(
    row: dict,
) -> Path:
    """
    Use the copied answer from the fixed 72-answer subset.

    If the absolute path stored in the CSV cannot be resolved, reconstruct it
    from the validation folder and selection metadata.
    """
    path_text = row.get(
        "copied_answer",
        "",
    ).strip()

    if path_text:
        manifest_path = Path(
            path_text
        )

        if manifest_path.exists():
            return manifest_path

    generation_mode = row[
        "generation_mode"
    ]

    prefix = (
        "annual"
        if row[
            "report_type"
        ] == "annual_report"
        else "quarterly"
    )

    original_name = (
        f"report_{prefix}_"
        f"{int(row['report_number'])}_"
        f"{row['prompt_type']}_"
        f"{row['model_key']}.txt"
    )

    fallback = (
        ANSWERS_DIR
        / (
            f"{generation_mode}__"
            f"{original_name}"
        )
    )

    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        "Selected validation answer was not found.\n"
        f"Selection: {row['selection_id']}\n"
        f"Expected: {fallback}"
    )


def load_shared_results(
    source_path: Path,
):
    """
    Load the original shared extraction and verified criteria.

    This function performs NO Bedrock request.
    """
    draft_path = (
        core.key_draft_path(
            source_path
        )
    )

    verified_path = (
        core.key_verified_path(
            source_path
        )
    )

    if not draft_path.exists():
        raise FileNotFoundError(
            f"Shared draft result missing:\n{draft_path}"
        )

    if not verified_path.exists():
        raise FileNotFoundError(
            f"Shared verified result missing:\n{verified_path}"
        )

    draft = core.load_json(
        draft_path
    )

    verified = core.load_json(
        verified_path
    )

    if draft.get(
        "status"
    ) != "success":
        raise RuntimeError(
            f"Shared draft result is not success:\n{draft_path}"
        )

    if verified.get(
        "status"
    ) != "success":
        raise RuntimeError(
            f"Shared verified result is not success:\n{verified_path}"
        )

    return (
        draft,
        verified,
        verified_path,
    )


def patch_repeat_metadata(
    merged: dict,
    selection_id: str,
) -> dict:
    merged[
        "judge"
    ][
        "reasoning_effort"
    ] = (
        "mixed_by_metric"
    )

    merged[
        "judge"
    ][
        "service_tier"
    ] = (
        core.SERVICE_TIER
    )

    merged[
        "judge"
    ][
        "reasoning_by_metric"
    ] = {
        "factual_numerical": (
            core.CANDIDATE_REASONING_MODES[
                "factual_numerical"
            ]
        ),
        "completeness": (
            core.CANDIDATE_REASONING_MODES[
                "completeness"
            ]
        ),
        "coherence": (
            core.CANDIDATE_REASONING_MODES[
                "coherence"
            ]
        ),
        "key_information_extract": (
            "reused_from_original_run"
        ),
        "key_information_verify": (
            "reused_from_original_run"
        ),
    }

    merged[
        "validation"
    ] = {
        "purpose": (
            "nova_self_consistency"
        ),
        "selection_id": (
            selection_id
        ),
        "repeat_run_number": (
            ARGS.run_number
        ),
        "shared_criteria_regenerated": (
            False
        ),
    }

    return merged


def merged_to_compact_row(
    merged: dict,
    merged_path: Path,
    selection_id: str,
) -> dict:
    scores = merged[
        "scores"
    ]

    faith = scores[
        "faithfulness"
    ]

    numerical = scores[
        "numerical_accuracy"
    ]

    completeness = scores[
        "completeness"
    ]

    coherence = scores[
        "coherence"
    ]

    usage = merged.get(
        "usage",
        {}
    )

    candidate = merged[
        "candidate"
    ]

    return {
        "selection_id": selection_id,
        "repeat_run_number": ARGS.run_number,
        "report": candidate.get("report"),
        "report_type": candidate.get("report_type"),
        "generation_mode": candidate.get("generation_mode"),
        "generator_key": candidate.get("generator_key"),
        "generator_model": candidate.get("generator_model"),
        "prompt_type": candidate.get("prompt_type"),

        "faithfulness": faith.get("score"),
        "factual_supported": faith.get("supported"),
        "factual_contradicted": faith.get("contradicted"),
        "factual_not_found": faith.get("not_found"),
        "factual_total": faith.get("total"),

        "numerical_accuracy": numerical.get("score"),
        "numerical_correct": numerical.get("correct"),
        "numerical_incorrect": numerical.get("incorrect"),
        "numerical_total": numerical.get("total"),

        "completeness": completeness.get("score"),
        "coherence": coherence.get("score"),

        "judge_input_tokens": usage.get("input_tokens_total"),
        "judge_output_tokens": usage.get(
            "output_tokens_total_including_reasoning"
        ),
        "judge_total_tokens": usage.get("total_tokens"),
        "judge_latency_seconds": usage.get("latency_seconds_total"),

        "merged_file": str(
            merged_path.resolve()
        ),
    }


CSV_FIELDS = [
    "selection_id",
    "repeat_run_number",
    "report",
    "report_type",
    "generation_mode",
    "generator_key",
    "generator_model",
    "prompt_type",
    "faithfulness",
    "factual_supported",
    "factual_contradicted",
    "factual_not_found",
    "factual_total",
    "numerical_accuracy",
    "numerical_correct",
    "numerical_incorrect",
    "numerical_total",
    "completeness",
    "coherence",
    "judge_input_tokens",
    "judge_output_tokens",
    "judge_total_tokens",
    "judge_latency_seconds",
    "merged_file",
]


def write_csv(
    path: Path,
    rows: list[dict],
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
            fieldnames=CSV_FIELDS,
        )
        writer.writeheader()
        writer.writerows(
            rows
        )


# =============================================================================
# One selected answer
# =============================================================================

def process_one(
    row: dict,
):
    selection_id = row[
        "selection_id"
    ]

    report_type = row[
        "report_type"
    ]

    generation_mode = row[
        "generation_mode"
    ]

    report_number = int(
        row[
            "report_number"
        ]
    )

    model_key = row[
        "model_key"
    ]

    prompt_type = row[
        "prompt_type"
    ]

    if model_key not in GENERATOR_NAMES:
        raise RuntimeError(
            f"Unknown model_key: {model_key}"
        )

    # Globals used by the production core.
    core.CATEGORY = report_type
    core.REPORT_NUMBER = report_number
    core.SUMMARY_PROMPT = prompt_type
    core.RUN_MODE = generation_mode

    source_path = (
        core.find_report_file()
    )

    reference_path = (
        core.find_reference_file()
    )

    source_report = (
        core.read_text(
            source_path
        )
    )

    candidate_path = (
        answer_path_from_manifest(
            row
        )
    )

    candidate_summary = (
        core.read_text(
            candidate_path
        )
    )

    sentence_map = (
        core.build_candidate_sentence_map(
            candidate_summary
        )
    )

    (
        draft_result,
        verified_result,
        verified_file,
    ) = load_shared_results(
        source_path
    )

    final_units = (
        verified_result[
            "evaluation"
        ][
            "key_information_units"
        ]
    )

    key_information_hash = (
        verified_result[
            "key_information_hash"
        ]
    )

    prompts = (
        core.load_judge_prompts()
    )

    log()
    log("=" * 100)
    log(
        f"{selection_id} | "
        f"Nova Run {ARGS.run_number} | "
        f"{report_type} | "
        f"report {report_number} | "
        f"{generation_mode} | "
        f"{prompt_type} | "
        f"{model_key}"
    )
    log("=" * 100)
    log(
        f"Candidate: {candidate_path}"
    )
    log(
        "Shared criteria: REUSED "
        "(no extraction/verification API calls)"
    )

    raw_results = {}

    for evaluation_type in [
        "factual_numerical",
        "completeness",
        "coherence",
    ]:
        raw = (
            core.run_candidate_evaluation(
                evaluation_type=(
                    evaluation_type
                ),
                instructions=(
                    prompts[
                        evaluation_type
                    ]
                ),
                source_report=(
                    source_report
                ),
                candidate_summary=(
                    candidate_summary
                ),
                sentence_map=(
                    sentence_map
                ),
                source_path=(
                    source_path
                ),
                candidate_path=(
                    candidate_path
                ),
                generator_model_name=(
                    GENERATOR_NAMES[
                        model_key
                    ]
                ),
                final_units=(
                    final_units
                    if evaluation_type
                    == "completeness"
                    else None
                ),
                key_information_hash_value=(
                    key_information_hash
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
        )

        raw_results[
            evaluation_type
        ] = raw

        if (
            core.REQUEST_DELAY_SECONDS
            > 0
        ):
            time.sleep(
                core.REQUEST_DELAY_SECONDS
            )

    merged = (
        core.merge_candidate_results(
            generator_key=(
                model_key
            ),
            generator_model_name=(
                GENERATOR_NAMES[
                    model_key
                ]
            ),
            candidate_path=(
                candidate_path
            ),
            source_path=(
                source_path
            ),
            reference_path=(
                reference_path
            ),
            draft_result=(
                draft_result
            ),
            verified_result=(
                verified_result
            ),
            sentence_map=(
                sentence_map
            ),
            raw_results=(
                raw_results
            ),
        )
    )

    if merged is None:
        return None

    merged = (
        patch_repeat_metadata(
            merged,
            selection_id,
        )
    )

    merged_path = (
        core.merged_result_path(
            candidate_path.stem
        )
    )

    core.save_json(
        merged_path,
        merged,
    )

    compact = (
        merged_to_compact_row(
            merged,
            merged_path,
            selection_id,
        )
    )

    log(
        "SUCCESS | "
        f"faith={compact['faithfulness']} | "
        f"num={compact['numerical_accuracy']} | "
        f"comp={compact['completeness']}/4 | "
        f"coh={compact['coherence']}/4"
    )

    return compact


# =============================================================================
# Collector
# =============================================================================

def collect_run(
    manifest_rows: list[dict],
):
    """
    Build one final 72-row CSV from saved merged JSON files.

    This mode makes NO Nova requests.
    """
    collected = []
    missing = []

    for row in manifest_rows:
        selection_id = row[
            "selection_id"
        ]

        core.CATEGORY = row[
            "report_type"
        ]

        core.RUN_MODE = row[
            "generation_mode"
        ]

        candidate_path = (
            answer_path_from_manifest(
                row
            )
        )

        merged_path = (
            core.merged_result_path(
                candidate_path.stem
            )
        )

        if not merged_path.exists():
            missing.append(
                (
                    selection_id,
                    str(
                        merged_path
                    ),
                )
            )
            continue

        merged = (
            core.load_json(
                merged_path
            )
        )

        if (
            merged.get(
                "status"
            )
            != "success"
        ):
            missing.append(
                (
                    selection_id,
                    (
                        f"{merged_path} "
                        f"(status={merged.get('status')})"
                    ),
                )
            )
            continue

        validation = merged.get(
            "validation",
            {}
        )

        if (
            validation.get(
                "repeat_run_number"
            )
            != ARGS.run_number
        ):
            missing.append(
                (
                    selection_id,
                    (
                        f"{merged_path} "
                        "(wrong/missing repeat_run_number)"
                    ),
                )
            )
            continue

        collected.append(
            merged_to_compact_row(
                merged,
                merged_path,
                selection_id,
            )
        )

    collected.sort(
        key=lambda item: int(
            item[
                "selection_id"
            ][1:]
        )
    )

    if missing:
        print()
        print(
            f"COLLECT INCOMPLETE: "
            f"{len(collected)}/72 successful results found."
        )
        print()
        print(
            "Missing / unsuccessful:"
        )

        for (
            selection_id,
            detail,
        ) in missing:
            print(
                f"  {selection_id}: {detail}"
            )

        raise SystemExit(
            1
        )

    write_csv(
        FINAL_CSV,
        collected,
    )

    input_tokens = sum(
        int(
            row[
                "judge_input_tokens"
            ]
            or 0
        )
        for row in collected
    )

    output_tokens = sum(
        int(
            row[
                "judge_output_tokens"
            ]
            or 0
        )
        for row in collected
    )

    cost = (
        estimated_flex_cost(
            input_tokens,
            output_tokens,
        )
    )

    print()
    print("=" * 100)
    print(
        f"NOVA RUN {ARGS.run_number} COLLECT COMPLETE"
    )
    print("=" * 100)
    print(
        "Successful evaluations: 72/72"
    )
    print(
        f"Input tokens:           {input_tokens:,}"
    )
    print(
        f"Output tokens:          {output_tokens:,}"
    )
    print(
        f"Estimated Flex cost:    ${cost:.2f}"
    )
    print(
        f"CSV:                    {FINAL_CSV}"
    )


# =============================================================================
# Main
# =============================================================================

def main():
    manifest_rows = (
        load_manifest()
    )

    if ARGS.collect:
        collect_run(
            manifest_rows
        )
        return

    selected_rows = (
        rows_for_this_worker(
            manifest_rows
        )
    )

    log("=" * 100)
    log(
        f"NOVA SELF-CONSISTENCY — RUN {ARGS.run_number}"
    )
    log("=" * 100)
    log(
        f"Worker:           {ARGS.worker}/{ARGS.workers}"
    )
    log(
        f"Selected answers: {len(selected_rows)}"
    )
    log(
        f"Result root:      {RUN_ROOT}"
    )
    log(
        f"Service tier:     {core.SERVICE_TIER}"
    )
    log(
        "Shared criteria:  reused from data/judge_shared"
    )
    log(
        "Factual/Numeric:  LOW"
    )
    log(
        "Completeness:     OFF"
    )
    log(
        "Coherence:        OFF"
    )
    log("=" * 100)

    successful_rows = []
    failures = []

    for row in selected_rows:
        try:
            result = (
                process_one(
                    row
                )
            )

            if result is None:
                failures.append(
                    (
                        row[
                            "selection_id"
                        ],
                        "One or more Judge sub-results were not success.",
                    )
                )
            else:
                successful_rows.append(
                    result
                )

        except Exception as exc:
            failures.append(
                (
                    row[
                        "selection_id"
                    ],
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                )
            )

            log(
                "FAILED | "
                f"{row['selection_id']} | "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    successful_rows.sort(
        key=lambda item: int(
            item[
                "selection_id"
            ][1:]
        )
    )

    write_csv(
        WORKER_CSV,
        successful_rows,
    )

    input_tokens = sum(
        int(
            row[
                "judge_input_tokens"
            ]
            or 0
        )
        for row in successful_rows
    )

    output_tokens = sum(
        int(
            row[
                "judge_output_tokens"
            ]
            or 0
        )
        for row in successful_rows
    )

    cost = (
        estimated_flex_cost(
            input_tokens,
            output_tokens,
        )
    )

    log()
    log("=" * 100)
    log(
        f"WORKER COMPLETE — NOVA RUN {ARGS.run_number}"
    )
    log("=" * 100)
    log(
        f"Successful:          "
        f"{len(successful_rows)}/{len(selected_rows)}"
    )
    log(
        f"Failures:            "
        f"{len(failures)}"
    )
    log(
        f"Input tokens:        "
        f"{input_tokens:,}"
    )
    log(
        f"Output tokens:       "
        f"{output_tokens:,}"
    )
    log(
        f"Estimated Flex cost: "
        f"${cost:.2f}"
    )
    log(
        f"Worker CSV:          "
        f"{WORKER_CSV}"
    )
    log(
        f"Worker log:          "
        f"{WORKER_LOG}"
    )

    if failures:
        log()
        log(
            "FAILED ITEMS:"
        )

        for (
            selection_id,
            error,
        ) in failures:
            log(
                f"  {selection_id} | "
                f"{error}"
            )


if __name__ == "__main__":
    main()
