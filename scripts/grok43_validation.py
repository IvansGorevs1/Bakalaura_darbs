from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run xAI Grok 4.3 once on the fixed 72-answer cross-judge subset."
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
        "--collect",
        action="store_true",
        help=(
            "Do not call Grok. Collect all 72 successful merged JSON files "
            "into grok43_run1_scores_72.csv."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rerun successful Grok results. Normally do NOT use this."
        ),
    )

    return parser.parse_args()



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
    / "grok43_run1"
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
        f"grok43_run1_worker_"
        f"{ARGS.worker}_of_{ARGS.workers}.csv"
    )
)

WORKER_LOG = (
    WORKER_RESULTS_DIR
    / (
        f"grok43_run1_worker_"
        f"{ARGS.worker}_of_{ARGS.workers}.log.txt"
    )
)

FINAL_CSV = (
    VALIDATION_ROOT
    / "grok43_run1_scores_72.csv"
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
        "Put run_grok43_validation.py in the same scripts/ folder as "
        "nova_judge_core_v5_2.py.\n"
    ) from exc



try:
    from openai import OpenAI
except ImportError as exc:
    raise SystemExit(
        "\nThe OpenAI Python SDK is required for Bedrock Mantle.\n"
        "Install it with:\n"
        "  pip install -U openai\n"
    ) from exc


AWS_REGION = "us-east-1"
GROK_MODEL_ID = "xai.grok-4.3"
GROK_MODEL_NAME = "xAI Grok 4.3"

GROK_BASE_URL = (
    f"https://bedrock-mantle.{AWS_REGION}.api.aws/openai/v1"
)

SERVICE_TIER = "flex"
TEMPERATURE = 0.0

FLEX_INPUT_PER_M = 0.625
FLEX_OUTPUT_PER_M = 1.25

MAX_API_ATTEMPTS = 4
RETRY_DELAYS_SECONDS = [
    10,
    30,
    60,
]

api_key = (
    os.getenv(
        "OPENAI_API_KEY"
    )
    or os.getenv(
        "AWS_BEARER_TOKEN_BEDROCK"
    )
)

if not api_key:
    raise SystemExit(
        "\nNo Amazon Bedrock API key found.\n\n"
        "Set one of:\n"
        '  $env:OPENAI_API_KEY="..."\n'
        "or\n"
        '  $env:AWS_BEARER_TOKEN_BEDROCK="..."\n'
    )

client = OpenAI(
    api_key=api_key,
    base_url=GROK_BASE_URL,
    timeout=900.0,
    max_retries=0, 
)


GENERATOR_NAMES = {
    "openai": "GPT-5.6 Luna",
    "gemini": "Gemini 3.5 Flash-Lite",
    "qwen36": "Qwen3.6-35B-A3B",
}


def get_attr(
    obj,
    name: str,
    default=None,
):
    if obj is None:
        return default

    if isinstance(
        obj,
        dict,
    ):
        return obj.get(
            name,
            default,
        )

    return getattr(
        obj,
        name,
        default,
    )


def extract_output_text(
    response,
) -> str:
    """
    Responses API normally exposes response.output_text.
    Keep a fallback for SDK/version differences.
    """
    direct = get_attr(
        response,
        "output_text",
        None,
    )

    if isinstance(
        direct,
        str,
    ) and direct.strip():
        return direct.strip()

    pieces = []

    output_items = get_attr(
        response,
        "output",
        [],
    ) or []

    for item in output_items:
        content_items = get_attr(
            item,
            "content",
            [],
        ) or []

        for content in content_items:
            content_type = get_attr(
                content,
                "type",
                "",
            )

            if content_type in {
                "output_text",
                "text",
            }:
                text_value = get_attr(
                    content,
                    "text",
                    "",
                )

                if isinstance(
                    text_value,
                    str,
                ):
                    pieces.append(
                        text_value
                    )

    return "\n".join(
        pieces
    ).strip()


def transient_status_code(
    exc: Exception,
):
    for attribute in [
        "status_code",
        "status",
    ]:
        value = getattr(
            exc,
            attribute,
            None,
        )

        if isinstance(
            value,
            int,
        ):
            return value

    response = getattr(
        exc,
        "response",
        None,
    )

    if response is not None:
        value = getattr(
            response,
            "status_code",
            None,
        )

        if isinstance(
            value,
            int,
        ):
            return value

    return None


def is_transient_exception(
    exc: Exception,
) -> bool:
    status = transient_status_code(
        exc
    )

    if status in {
        408,
        409,
        429,
        500,
        502,
        503,
        504,
    }:
        return True

    name = type(
        exc
    ).__name__

    return name in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }


def grok_reasoning_effort(
    reasoning_mode: str,
) -> str:
    mapping = {
        "high": "high",
        "low": "low",
        "off": "none",
        "none": "none",
    }

    if reasoning_mode not in mapping:
        raise ValueError(
            f"Unknown reasoning mode: {reasoning_mode}"
        )

    return mapping[
        reasoning_mode
    ]


def call_grok_once(
    full_prompt: str,
    reasoning_mode: str = "low",
    max_tokens: int | None = None,
) -> dict:
    """
    Adapter with the same return shape expected by nova_judge_core_v5_2.py.
    """
    effort = (
        grok_reasoning_effort(
            reasoning_mode
        )
    )

    kwargs = {
        "model": GROK_MODEL_ID,
        "input": full_prompt,
        "reasoning": {
            "effort": effort,
        },
        "temperature": TEMPERATURE,
        "service_tier": SERVICE_TIER,
    }

    if max_tokens is not None:
        kwargs[
            "max_output_tokens"
        ] = max_tokens

    start = time.perf_counter()

    response = (
        client.responses.create(
            **kwargs
        )
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    visible_text = (
        extract_output_text(
            response
        )
    )

    if not visible_text:
        raise RuntimeError(
            "Grok Responses API returned no visible output text."
        )

    usage = get_attr(
        response,
        "usage",
        None,
    )

    input_tokens = int(
        get_attr(
            usage,
            "input_tokens",
            0,
        )
        or 0
    )

    output_tokens = int(
        get_attr(
            usage,
            "output_tokens",
            0,
        )
        or 0
    )

    total_tokens = int(
        get_attr(
            usage,
            "total_tokens",
            (
                input_tokens
                + output_tokens
            ),
        )
        or (
            input_tokens
            + output_tokens
        )
    )

    output_details = get_attr(
        usage,
        "output_tokens_details",
        None,
    )

    reasoning_tokens = int(
        get_attr(
            output_details,
            "reasoning_tokens",
            0,
        )
        or 0
    )

    response_status = get_attr(
        response,
        "status",
        None,
    )

    incomplete_details = get_attr(
        response,
        "incomplete_details",
        None,
    )

    incomplete_reason = get_attr(
        incomplete_details,
        "reason",
        None,
    )

    stop_reason = (
        incomplete_reason
        or response_status
        or "completed"
    )

    return {
        "visible_text": visible_text,

        # Grok reasoning content is intentionally not requested.
        # Billing/usage still includes the model's output/reasoning usage.
        "reasoning_texts": [],

        "usage": {
            "input_tokens": input_tokens,
            "output_tokens_including_reasoning": output_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": reasoning_tokens,
        },

        "stop_reason": stop_reason,

        "latency_seconds": round(
            elapsed,
            2,
        ),

        "resolved_service_tier": get_attr(
            response,
            "service_tier",
            SERVICE_TIER,
        ),

        "response_id": get_attr(
            response,
            "id",
            None,
        ),
    }


def call_grok(
    full_prompt: str,
    reasoning_mode: str = "low",
    max_tokens: int | None = None,
) -> dict:
    """
    Logical Grok request with retries only for transient API/network errors.
    """
    last_exception = None

    for attempt in range(
        1,
        MAX_API_ATTEMPTS + 1,
    ):
        if attempt > 1:
            print(
                f"Grok transient retry "
                f"{attempt}/{MAX_API_ATTEMPTS}...",
                flush=True,
            )

        try:
            return call_grok_once(
                full_prompt=full_prompt,
                reasoning_mode=reasoning_mode,
                max_tokens=max_tokens,
            )

        except Exception as exc:
            last_exception = exc

            status = transient_status_code(
                exc
            )

            print(
                "GROK API ERROR | "
                f"{type(exc).__name__}"
                + (
                    f" | HTTP {status}"
                    if status is not None
                    else ""
                )
                + f" | {exc}",
                flush=True,
            )

            if (
                not is_transient_exception(
                    exc
                )
                or attempt
                >= MAX_API_ATTEMPTS
            ):
                raise

            delay_index = min(
                attempt - 1,
                len(
                    RETRY_DELAYS_SECONDS
                )
                - 1,
            )

            delay = (
                RETRY_DELAYS_SECONDS[
                    delay_index
                ]
            )

            print(
                f"Transient failure. Waiting {delay} s...",
                flush=True,
            )

            time.sleep(
                delay
            )

    raise last_exception


# =============================================================================
# Monkey-patch the production core
# =============================================================================

# All existing prompt construction, validation, numerical guard, schema
# normalization, JSON repair and merge logic is reused unchanged.
core.call_nova_once = call_grok_once
core.call_nova = call_grok

core.JUDGE_MODEL_ID = GROK_MODEL_ID
core.JUDGE_MODEL_NAME = GROK_MODEL_NAME
core.SERVICE_TIER = SERVICE_TIER
core.CANDIDATE_TEMPERATURE = TEMPERATURE

# Keep the final production reasoning policy.
core.CANDIDATE_REASONING_MODES = {
    "factual_numerical": "low",
    "completeness": "off",
    "coherence": "off",
}

# Never regenerate shared Nova-derived verified completeness criteria.
core.FORCE_RERUN_SHARED = False

# Isolated Grok output root.
core.JUDGE_RESULTS_ROOT = RUN_ROOT

# Resume-safe unless explicitly forced.
core.FORCE_RERUN_ALL = ARGS.force


def grok_raw_result_path(
    evaluation_type: str,
    candidate_stem: str,
) -> Path:
    return (
        RUN_ROOT
        / "grok_4_3"
        / core.CATEGORY
        / "raw"
        / evaluation_type
        / f"{candidate_stem}.json"
    )


def grok_merged_result_path(
    candidate_stem: str,
) -> Path:
    return (
        RUN_ROOT
        / "grok_4_3"
        / core.CATEGORY
        / "merged"
        / f"{candidate_stem}_judge.json"
    )


# Replace Nova-specific path helpers.
core.raw_result_path = grok_raw_result_path
core.merged_result_path = grok_merged_result_path


# =============================================================================
# General helpers
# =============================================================================

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


# =============================================================================
# Manifest / deterministic 4-way sharding
# =============================================================================

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

    if len(
        rows
    ) != 72:
        raise RuntimeError(
            f"Expected 72 rows in validation manifest, found {len(rows)}."
        )

    selection_ids = [
        row[
            "selection_id"
        ]
        for row in rows
    ]

    if len(
        set(
            selection_ids
        )
    ) != 72:
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
    result = []

    for row in rows:
        number = (
            selection_number(
                row[
                    "selection_id"
                ]
            )
        )

        assigned_worker = (
            (
                number
                - 1
            )
            % ARGS.workers
            + 1
        )

        if assigned_worker == ARGS.worker:
            result.append(
                row
            )

    return result


# =============================================================================
# Candidate / shared criteria
# =============================================================================

def answer_path_from_manifest(
    row: dict,
) -> Path:
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
    Load the SAME Nova Run-1 report-level shared criteria.

    No model request is made here.
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

    draft = (
        core.load_json(
            draft_path
        )
    )

    verified = (
        core.load_json(
            verified_path
        )
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


# =============================================================================
# Cross-judge metadata / compact CSV
# =============================================================================

def patch_crossjudge_metadata(
    merged: dict,
    selection_id: str,
) -> dict:
    merged[
        "judge"
    ][
        "model"
    ] = GROK_MODEL_NAME

    merged[
        "judge"
    ][
        "model_id"
    ] = GROK_MODEL_ID

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
    ] = SERVICE_TIER

    merged[
        "judge"
    ][
        "reasoning_by_metric"
    ] = {
        "factual_numerical": "low",
        "completeness": "none",
        "coherence": "none",
        "key_information_extract": (
            "reused_from_nova_run1"
        ),
        "key_information_verify": (
            "reused_from_nova_run1"
        ),
    }

    merged[
        "validation"
    ] = {
        "purpose": (
            "cross_judge_agreement"
        ),
        "selection_id": (
            selection_id
        ),
        "cross_judge": (
            GROK_MODEL_NAME
        ),
        "cross_judge_run_number": 1,
        "shared_criteria_regenerated": False,
        "shared_criteria_source": (
            "Nova 2 Lite original verified criteria"
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
        "cross_judge": GROK_MODEL_NAME,
        "report": candidate.get(
            "report"
        ),
        "report_type": candidate.get(
            "report_type"
        ),
        "generation_mode": candidate.get(
            "generation_mode"
        ),
        "generator_key": candidate.get(
            "generator_key"
        ),
        "generator_model": candidate.get(
            "generator_model"
        ),
        "prompt_type": candidate.get(
            "prompt_type"
        ),

        "faithfulness": faith.get(
            "score"
        ),
        "factual_supported": faith.get(
            "supported"
        ),
        "factual_contradicted": faith.get(
            "contradicted"
        ),
        "factual_not_found": faith.get(
            "not_found"
        ),
        "factual_total": faith.get(
            "total"
        ),

        "numerical_accuracy": numerical.get(
            "score"
        ),
        "numerical_correct": numerical.get(
            "correct"
        ),
        "numerical_incorrect": numerical.get(
            "incorrect"
        ),
        "numerical_total": numerical.get(
            "total"
        ),

        "completeness": completeness.get(
            "score"
        ),
        "coherence": coherence.get(
            "score"
        ),

        "judge_input_tokens": usage.get(
            "input_tokens_total"
        ),
        "judge_output_tokens": usage.get(
            "output_tokens_total_including_reasoning"
        ),
        "judge_total_tokens": usage.get(
            "total_tokens"
        ),
        "judge_latency_seconds": usage.get(
            "latency_seconds_total"
        ),

        "merged_file": str(
            merged_path.resolve()
        ),
    }


CSV_FIELDS = [
    "selection_id",
    "cross_judge",
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
# Grok-specific residual numerical coverage recovery
# =============================================================================

GROK_RESIDUAL_NUMERICAL_REPAIR_MAX_SENTENCES = 5


def _existing_grok_numerical_failure(
    candidate_path: Path,
) -> tuple[dict | None, list[str]]:
    """
    Inspect the already-saved Grok Factual/Numerical raw result.

    This does not call the model. It is used only to decide whether a
    previously completed numerical-only pass left a small residual gap
    (4-5 sentences) that can be repaired with one bounded targeted call.
    """
    output_path = (
        core.raw_result_path(
            "factual_numerical",
            candidate_path.stem,
        )
    )

    if not output_path.exists():
        return (
            None,
            [],
        )

    existing = (
        core.load_json(
            output_path
        )
    )

    if existing.get(
        "status"
    ) != "numerical_coverage_failed":
        return (
            existing,
            [],
        )

    coverage = existing.get(
        "numerical_coverage",
        {},
    )

    missing_ids = list(
        coverage.get(
            "missing_sentence_ids",
            [],
        )
        or []
    )

    return (
        existing,
        missing_ids,
    )


def run_grok_candidate_evaluation(
    *,
    evaluation_type: str,
    instructions: str,
    source_report: str,
    candidate_summary: str,
    sentence_map: dict[str, str],
    source_path: Path,
    candidate_path: Path,
    generator_model_name: str,
    final_units,
    key_information_hash_value,
    key_information_file,
):
    """
    Run one Judge metric using the production core.

    Important Grok-only behavior:
    - Fresh calls keep the production numerical repair threshold (3).
    - If a SAVED numerical_coverage_failed result already exists with
      4-5 residual missing sentences, temporarily raise the threshold to 5
      so the core performs ONE residual targeted repair instead of repeating
      the full Factual/Numerical evaluation.
    - If a fresh call ends with such a 4-5 sentence residual gap, immediately
      invoke the same resume path once.

    There is no retry loop and no change to successful existing results.
    """
    old_threshold = (
        core.TARGETED_NUMERICAL_REPAIR_MAX_SENTENCES
    )

    try:
        if (
            evaluation_type
            == "factual_numerical"
        ):
            (
                existing,
                existing_missing,
            ) = (
                _existing_grok_numerical_failure(
                    candidate_path
                )
            )

            if (
                existing is not None
                and 1
                <= len(
                    existing_missing
                )
                <= GROK_RESIDUAL_NUMERICAL_REPAIR_MAX_SENTENCES
            ):
                print(
                    "GROK RESUME residual numerical repair | "
                    f"missing: {existing_missing} | "
                    "using one bounded targeted repair.",
                    flush=True,
                )

                core.TARGETED_NUMERICAL_REPAIR_MAX_SENTENCES = (
                    GROK_RESIDUAL_NUMERICAL_REPAIR_MAX_SENTENCES
                )

        raw = (
            core.run_candidate_evaluation(
                evaluation_type=(
                    evaluation_type
                ),
                instructions=(
                    instructions
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
                    generator_model_name
                ),
                final_units=(
                    final_units
                ),
                key_information_hash_value=(
                    key_information_hash_value
                ),
                key_information_file=(
                    key_information_file
                ),
            )
        )

    finally:
        core.TARGETED_NUMERICAL_REPAIR_MAX_SENTENCES = (
            old_threshold
        )

    # If this was a fresh Grok call and the dedicated numerical-only pass
    # left 4-5 residual sentences, immediately reuse the saved result and
    # perform one targeted residual repair. No full Judge rerun.
    if (
        evaluation_type
        == "factual_numerical"
        and raw.get(
            "status"
        )
        == "numerical_coverage_failed"
    ):
        coverage = raw.get(
            "numerical_coverage",
            {},
        )

        missing_ids = list(
            coverage.get(
                "missing_sentence_ids",
                [],
            )
            or []
        )

        if (
            old_threshold
            < len(
                missing_ids
            )
            <= GROK_RESIDUAL_NUMERICAL_REPAIR_MAX_SENTENCES
        ):
            print(
                "GROK residual numerical coverage after numerical-only pass | "
                f"missing: {missing_ids} -> "
                "running one final targeted repair.",
                flush=True,
            )

            try:
                core.TARGETED_NUMERICAL_REPAIR_MAX_SENTENCES = (
                    GROK_RESIDUAL_NUMERICAL_REPAIR_MAX_SENTENCES
                )

                raw = (
                    core.run_candidate_evaluation(
                        evaluation_type=(
                            evaluation_type
                        ),
                        instructions=(
                            instructions
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
                            generator_model_name
                        ),
                        final_units=None,
                        key_information_hash_value=None,
                        key_information_file=None,
                    )
                )

            finally:
                core.TARGETED_NUMERICAL_REPAIR_MAX_SENTENCES = (
                    old_threshold
                )

    return raw


# =============================================================================
# One selected candidate
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

    # Globals used by the production evaluation machinery.
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
    ) = (
        load_shared_results(
            source_path
        )
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
    log(
        "=" * 100
    )
    log(
        f"{selection_id} | "
        f"Grok 4.3 Cross-Judge | "
        f"{report_type} | "
        f"report {report_number} | "
        f"{generation_mode} | "
        f"{prompt_type} | "
        f"{model_key}"
    )
    log(
        "=" * 100
    )
    log(
        f"Candidate: {candidate_path}"
    )
    log(
        "Shared criteria: REUSED from Nova "
        "(no extraction/verification model calls)"
    )

    raw_results = {}

    for evaluation_type in [
        "factual_numerical",
        "completeness",
        "coherence",
    ]:
        raw = (
            run_grok_candidate_evaluation(
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
        patch_crossjudge_metadata(
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
    Build one final 72-row CSV from saved Grok merged JSON files.

    Makes NO model calls.
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
                "purpose"
            )
            != "cross_judge_agreement"
        ):
            missing.append(
                (
                    selection_id,
                    (
                        f"{merged_path} "
                        "(wrong/missing cross-judge metadata)"
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
            ][
                1:
            ]
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
    print(
        "=" * 100
    )
    print(
        "GROK 4.3 CROSS-JUDGE COLLECT COMPLETE"
    )
    print(
        "=" * 100
    )
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

    log(
        "=" * 100
    )
    log(
        "GROK 4.3 CROSS-JUDGE — 72-ANSWER SUBSET"
    )
    log(
        "=" * 100
    )
    log(
        f"Worker:           {ARGS.worker}/{ARGS.workers}"
    )
    log(
        f"Selected answers: {len(selected_rows)}"
    )
    log(
        f"Model:            {GROK_MODEL_ID}"
    )
    log(
        f"Endpoint:         {GROK_BASE_URL}"
    )
    log(
        f"Service tier:     {SERVICE_TIER}"
    )
    log(
        "Shared criteria:  reused from original Nova run"
    )
    log(
        "Factual/Numeric:  LOW"
    )
    log(
        "Completeness:     NONE"
    )
    log(
        "Coherence:        NONE"
    )
    log(
        "Temperature:      0.0"
    )
    log(
        "=" * 100
    )

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
            ][
                1:
            ]
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
    log(
        "=" * 100
    )
    log(
        "WORKER COMPLETE — GROK 4.3"
    )
    log(
        "=" * 100
    )
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
