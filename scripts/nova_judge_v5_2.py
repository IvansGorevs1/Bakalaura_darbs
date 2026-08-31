from pathlib import Path
import csv
import hashlib
import json
import os
import re
import time
from datetime import datetime

import boto3
from botocore.config import Config
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    ReadTimeoutError,
    ConnectTimeoutError,
    ConnectionClosedError,
)

AWS_REGION = "us-east-1"

JUDGE_MODEL_ID = "us.amazon.nova-2-lite-v1:0"
JUDGE_MODEL_NAME = "Amazon Nova 2 Lite"


JUDGE_REASONING_EFFORT = "high"

SERVICE_TIER = "flex"

CANDIDATE_REASONING_MODES = {
    "factual_numerical": "low",
    "completeness": "off",
    "coherence": "off",
}

CANDIDATE_MAX_TOKENS = {
    "factual_numerical": 20000,
    "completeness": 4000,
    "coherence": 3000,
}

CANDIDATE_TEMPERATURE = 0.0


TARGETED_NUMERICAL_REPAIR_MAX_SENTENCES = 3

PROMPT_VERSION = "v5_2_production_targeted_repair"
RUNNER_IMPLEMENTATION_VERSION = "v5.2.11_trace_and_schema_recovery"


CATEGORY = "annual_report"
REPORT_NUMBER = 1
SUMMARY_PROMPT = "zero_shot"

GENERATOR_MODELS = {
    "openai": "GPT-5.6 Luna",
    "gemini": "Gemini 3.5 Flash-Lite",
    "qwen36": "Qwen3.6-35B-A3B",
}


RUN_MODE = os.getenv("NOVA_JUDGE_RUN_MODE", "normal")

TEST_RUN = False
TEST_TAG = "v4"

FORCE_RERUN_ALL = False
FORCE_RERUN_SHARED = False

REQUEST_DELAY_SECONDS = 0.5

MAX_API_ATTEMPTS = 4
RETRY_DELAYS_SECONDS = [10, 30, 60]


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
REPORTS_DIR = DATA_DIR / "reports"

if RUN_MODE == "normal":
    OUTPUTS_DIR = DATA_DIR / "outputs"

    if TEST_RUN:
        JUDGE_RESULTS_ROOT = (
            DATA_DIR / f"judge_results_test_{TEST_TAG}"
        )
        SCORES_CSV = (
            DATA_DIR
            / "evaluation_results"
            / f"judge_scores_test_{TEST_TAG}.csv"
        )
    else:
        JUDGE_RESULTS_ROOT = DATA_DIR / "judge_results"
        SCORES_CSV = (
            DATA_DIR
            / "evaluation_results"
            / "judge_scores.csv"
        )

elif RUN_MODE == "reasoning_high":
    OUTPUTS_DIR = DATA_DIR / "outputs_reasoning_high"

    if TEST_RUN:
        JUDGE_RESULTS_ROOT = (
            DATA_DIR
            / f"judge_results_reasoning_high_test_{TEST_TAG}"
        )
        SCORES_CSV = (
            DATA_DIR
            / "evaluation_results"
            / f"judge_scores_reasoning_high_test_{TEST_TAG}.csv"
        )
    else:
        JUDGE_RESULTS_ROOT = (
            DATA_DIR / "judge_results_reasoning_high"
        )
        SCORES_CSV = (
            DATA_DIR
            / "evaluation_results"
            / "judge_scores_reasoning_high.csv"
        )

else:
    raise ValueError(
        'RUN_MODE must be "normal" or "reasoning_high".'
    )

if TEST_RUN:
    SHARED_RESULTS_ROOT = (
        DATA_DIR / f"judge_shared_test_{TEST_TAG}"
    )
else:
    SHARED_RESULTS_ROOT = DATA_DIR / "judge_shared"


def find_prompts_dir() -> Path:
    candidates = [
        DATA_DIR / "prompts",
        DATA_DIR / "promts",
    ]

    for path in candidates:
        if (path / "zero_shot.txt").exists():
            return path

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find data/prompts or data/promts."
    )


PROMPTS_DIR = find_prompts_dir()

JUDGE_PROMPT_FILES = {
    "key_information_extract": (
        PROMPTS_DIR / "judge_key_information.txt"
    ),
    "key_information_verify": (
        PROMPTS_DIR / "judge_key_information_verify.txt"
    ),
    "factual_numerical": (
        (
            PROMPTS_DIR / "judge_factual_numerical_v5_2.txt"
        )
        if (
            PROMPTS_DIR / "judge_factual_numerical_v5_2.txt"
        ).exists()
        else (
            PROMPTS_DIR / "judge_factual_numerical.txt"
        )
    ),
    "completeness": (
        PROMPTS_DIR / "judge_completeness.txt"
    ),
    "coherence": (
        PROMPTS_DIR / "judge_coherence.txt"
    ),
}


if not os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
    raise RuntimeError(
        "AWS_BEARER_TOKEN_BEDROCK is not set.\n"
        "PowerShell example:\n"
        '$env:AWS_BEARER_TOKEN_BEDROCK="YOUR_BEDROCK_API_KEY"'
    )


bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    config=Config(
        read_timeout=900,
        connect_timeout=60,
        tcp_keepalive=True,
        retries={
            "max_attempts": 2,
            "mode": "standard",
        },
    ),
)



def read_text(path: Path) -> str:
    return path.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:16]


def json_hash(value) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return text_hash(canonical)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def clean_json_text(text: str) -> str:
    value = text.strip()

    if value.startswith("```"):
        value = re.sub(
            r"^```(?:json)?\s*",
            "",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(
            r"\s*```$",
            "",
            value,
        )

    first = value.find("{")
    last = value.rfind("}")

    if first != -1 and last != -1 and last >= first:
        value = value[first:last + 1]

    return value.strip()


def _escape_literal_control_chars_in_json_strings(
    value: str,
) -> str:
    
    out = []
    in_string = False
    escaped = False

    replacements = {
        "\b": r"\b",
        "\f": r"\f",
        "\n": r"\n",
        "\r": r"\r",
        "\t": r"\t",
    }

    for char in value:
        if in_string:
            if escaped:
                out.append(char)
                escaped = False
                continue

            if char == "\\":
                out.append(char)
                escaped = True
                continue

            if char == '"':
                out.append(char)
                in_string = False
                continue

            if ord(char) < 0x20:
                out.append(
                    replacements.get(
                        char,
                        "\\u%04x" % ord(char),
                    )
                )
                continue

            out.append(char)
            continue

        out.append(char)

        if char == '"':
            in_string = True

    return "".join(out)


def _remove_trailing_commas_outside_json_strings(
    value: str,
) -> str:
   
    out = []
    in_string = False
    escaped = False
    i = 0

    while i < len(value):
        char = value[i]

        if in_string:
            out.append(char)

            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False

            i += 1
            continue

        if char == '"':
            in_string = True
            out.append(char)
            i += 1
            continue

        if char == ",":
            j = i + 1

            while (
                j < len(value)
                and value[j].isspace()
            ):
                j += 1

            if (
                j < len(value)
                and value[j] in "}]"
            ):
                # Skip only the illegal comma; preserve whitespace.
                i += 1
                continue

        out.append(char)
        i += 1

    return "".join(out)



def _repair_invalid_json_escapes_in_strings_v5210(
    value: str,
) -> str:

    out = []
    in_string = False
    i = 0
    valid_simple = set('"\\/bfnrt')
    hex_digits = set("0123456789abcdefABCDEF")

    while i < len(value):
        char = value[i]

        if not in_string:
            out.append(char)
            if char == '"':
                in_string = True
            i += 1
            continue

        if char == '"':
            out.append(char)
            in_string = False
            i += 1
            continue

        if char != "\\":
            out.append(char)
            i += 1
            continue

        if i + 1 >= len(value):
            out.append("\\\\")
            i += 1
            continue

        nxt = value[i + 1]

        if nxt == "'":
            out.append("'")
            i += 2
            continue

        if nxt in valid_simple:
            out.append("\\")
            out.append(nxt)
            i += 2
            continue

        if (
            nxt == "u"
            and i + 5 < len(value)
            and all(
                c in hex_digits
                for c in value[i + 2:i + 6]
            )
        ):
            out.append(
                value[i:i + 6]
            )
            i += 6
            continue

        out.append("\\\\")
        out.append(nxt)
        i += 2

    return "".join(out)


def _repair_one_line_json_string_values_v5210(
    value: str,
) -> str:

    lines = value.splitlines(
        keepends=True
    )
    output = []

    pattern = re.compile(
        r'^(\s*"(?:[^"\\]|\\.)+"\s*:\s*")(.*)$'
    )

    for line in lines:
        newline = ""
        body = line

        if line.endswith("\r\n"):
            body = line[:-2]
            newline = "\r\n"
        elif line.endswith("\n"):
            body = line[:-1]
            newline = "\n"

        match = pattern.match(body)

        if not match:
            output.append(line)
            continue

        prefix = match.group(1)
        rest = match.group(2)

        stripped = rest.rstrip()
        trailing_ws = rest[len(stripped):]

        comma = ""

        if stripped.endswith(","):
            comma = ","
            stripped = stripped[:-1].rstrip()

        quote_positions = []
        escaped = False

        for index, char in enumerate(
            stripped
        ):
            if escaped:
                escaped = False
                continue

            if char == "\\":
                escaped = True
                continue

            if char == '"':
                quote_positions.append(
                    index
                )

        closing_at_end = bool(
            quote_positions
            and quote_positions[-1]
            == len(stripped) - 1
        )

        content = (
            stripped[:-1]
            if closing_at_end
            else stripped
        )

        repaired_content = []
        escaped = False

        for char in content:
            if escaped:
                repaired_content.append(
                    char
                )
                escaped = False
                continue

            if char == "\\":
                repaired_content.append(
                    char
                )
                escaped = True
                continue

            if char == '"':
                repaired_content.append(
                    '\\"'
                )
            else:
                repaired_content.append(
                    char
                )

        repaired_rest = (
            "".join(
                repaired_content
            )
            + '"'
            + comma
            + trailing_ws
        )

        output.append(
            prefix
            + repaired_rest
            + newline
        )

    return "".join(output)


def parse_json_response(text: str) -> dict:

    cleaned = clean_json_text(
        text
    )

    try:
        return json.loads(
            cleaned
        )
    except json.JSONDecodeError as first_error:
        repaired = (
            _repair_invalid_json_escapes_in_strings_v5210(
                cleaned
            )
        )

        repaired = (
            _repair_one_line_json_string_values_v5210(
                repaired
            )
        )

        repaired = (
            _escape_literal_control_chars_in_json_strings(
                repaired
            )
        )

        repaired = (
            _remove_trailing_commas_outside_json_strings(
                repaired
            )
        )

        if repaired == cleaned:
            raise first_error

        return json.loads(
            repaired
        )


def normalize_label(value) -> str:
    return str(value or "").strip().upper()


def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None



SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[.!?])\s+(?=(?:["“‘\'(\[]?[A-Z0-9]))'
)


def build_candidate_sentence_map(
    candidate_summary: str,
) -> dict[str, str]:
    units = []

    paragraphs = [
        part.strip()
        for part in re.split(
            r"\n\s*\n|\n+",
            candidate_summary.strip(),
        )
        if part.strip()
    ]

    for paragraph in paragraphs:
        pieces = [
            piece.strip()
            for piece in SENTENCE_SPLIT_RE.split(
                paragraph
            )
            if piece.strip()
        ]

        if not pieces:
            pieces = [paragraph]

        units.extend(pieces)

    if not units:
        raise ValueError(
            "Candidate summary produced no trace units."
        )

    return {
        f"S{index}": sentence
        for index, sentence in enumerate(
            units,
            start=1,
        )
    }


def format_candidate_with_ids(
    sentence_map: dict[str, str],
) -> str:
    return "\n".join(
        f"[{sentence_id}] {text}"
        for sentence_id, text
        in sentence_map.items()
    )


def validate_sentence_ids(
    sentence_ids,
    valid_sentence_ids: set[str],
    *,
    allow_empty: bool,
    field_name: str,
) -> list[str]:
    errors = []

    if not isinstance(
        sentence_ids,
        list,
    ):
        return [
            f"{field_name} must be a list."
        ]

    if (
        not allow_empty
        and len(sentence_ids) == 0
    ):
        errors.append(
            f"{field_name} must contain at least one sentence ID."
        )

    seen = set()

    for sentence_id in sentence_ids:
        if not isinstance(
            sentence_id,
            str,
        ):
            errors.append(
                f"{field_name} contains a non-string ID."
            )
            continue

        if sentence_id not in valid_sentence_ids:
            errors.append(
                f"{field_name} contains unknown ID {sentence_id!r}."
            )

        if sentence_id in seen:
            errors.append(
                f"{field_name} contains duplicate ID {sentence_id!r}."
            )

        seen.add(sentence_id)

    return errors


def resolve_sentence_ids(
    sentence_ids,
    sentence_map: dict[str, str],
) -> list[dict]:
    if not isinstance(
        sentence_ids,
        list,
    ):
        return []

    result = []

    for sentence_id in sentence_ids:
        if sentence_id in sentence_map:
            result.append(
                {
                    "id": sentence_id,
                    "text": sentence_map[
                        sentence_id
                    ],
                }
            )

    return result


def report_prefix() -> str:
    if CATEGORY == "annual_report":
        return "annual"

    if CATEGORY in {
        "quarterly_report",
        "quartely_report",
    }:
        return "quarterly"

    raise ValueError(
        f"Unsupported CATEGORY: {CATEGORY}"
    )


def category_output_dirs(base: Path) -> list[Path]:
    result = [
        base / CATEGORY
    ]

    if CATEGORY == "quarterly_report":
        result.append(
            base / "quartely_report"
        )

    return result


def find_report_file() -> Path:
    prefix = report_prefix()

    report_dirs = [
        REPORTS_DIR / CATEGORY,
    ]

    if CATEGORY == "quarterly_report":
        report_dirs.append(
            REPORTS_DIR / "quartely_report"
        )

    exact_name = (
        f"report_{prefix}_{REPORT_NUMBER}.md"
    )

    for directory in report_dirs:
        path = directory / exact_name
        if path.exists():
            return path

    fallback = []

    for directory in report_dirs:
        if not directory.exists():
            continue

        fallback.extend(
            directory.glob(
                f"*{prefix}*_{REPORT_NUMBER}.md"
            )
        )

    fallback = sorted(set(fallback))

    if len(fallback) == 1:
        return fallback[0]

    raise FileNotFoundError(
        f"Could not uniquely find report "
        f"{REPORT_NUMBER} for {CATEGORY}."
    )


def find_reference_file() -> Path:
    prefix = report_prefix()

    if prefix == "annual":
        reference_dirs = [
            DATA_DIR
            / "references_txt"
            / "annual_report_reference",
            DATA_DIR
            / "references"
            / "annual_report_reference",
        ]
    else:
        reference_dirs = [
            DATA_DIR
            / "references_txt"
            / "quarterly_report_reference",
            DATA_DIR
            / "references_txt"
            / "quartely_report_reference",
            DATA_DIR
            / "references"
            / "quarterly_report_reference",
        ]

    exact_names = [
        f"reference_{prefix}_{REPORT_NUMBER}_original.txt",
        f"reference_{prefix}_{REPORT_NUMBER}.txt",
        f"report_{prefix}_{REPORT_NUMBER}_reference.txt",
        f"report_{prefix}_{REPORT_NUMBER}.txt",
    ]

    for directory in reference_dirs:
        for filename in exact_names:
            path = directory / filename
            if path.exists():
                return path

    number_pattern = re.compile(
        rf"(?<!\d){REPORT_NUMBER}(?!\d)"
    )

    fallback = []

    for directory in reference_dirs:
        if not directory.exists():
            continue

        for path in directory.glob("*.txt"):
            if number_pattern.search(
                path.stem.lower()
            ):
                fallback.append(path)

    fallback = sorted(set(fallback))

    if len(fallback) == 1:
        return fallback[0]

    if len(fallback) > 1:
        items = "\n".join(
            f"  - {path}"
            for path in fallback
        )
        raise RuntimeError(
            "Several possible reference summaries were found:\n"
            + items
        )

    raise FileNotFoundError(
        f"Reference summary for report "
        f"{REPORT_NUMBER} was not found."
    )


def get_candidate_file(
    model_key: str,
) -> Path:
    prefix = report_prefix()

    filename = (
        f"report_{prefix}_{REPORT_NUMBER}_"
        f"{SUMMARY_PROMPT}_{model_key}.txt"
    )

    for directory in category_output_dirs(
        OUTPUTS_DIR
    ):
        path = directory / filename

        if path.exists():
            return path

    raise FileNotFoundError(
        f"Candidate summary not found: {filename}"
    )


def load_judge_prompts() -> dict[str, str]:
    prompts = {}
    missing = []

    for name, path in JUDGE_PROMPT_FILES.items():
        if not path.exists():
            missing.append(path)
        else:
            prompts[name] = read_text(
                path
            )

    if missing:
        raise FileNotFoundError(
            "Judge prompt files are missing:\n"
            + "\n".join(
                f"  - {path}"
                for path in missing
            )
        )

    return prompts


def extract_response_content(
    response: dict,
):
    content_blocks = (
        response
        .get("output", {})
        .get("message", {})
        .get("content", [])
    )

    visible_text_parts = []
    reasoning_text_parts = []

    for block in content_blocks:
        if "text" in block:
            visible_text_parts.append(
                block["text"]
            )

        reasoning_content = block.get(
            "reasoningContent"
        )

        if reasoning_content:
            reasoning_text = (
                reasoning_content
                .get("reasoningText", {})
                .get("text")
            )

            if reasoning_text is not None:
                reasoning_text_parts.append(
                    reasoning_text
                )

    return (
        "\n".join(
            visible_text_parts
        ).strip(),
        reasoning_text_parts,
    )


def call_nova_once(
    full_prompt: str,
) -> dict:
    start = time.perf_counter()

    response = bedrock.converse(
        modelId=JUDGE_MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": full_prompt
                    }
                ],
            }
        ],
        # IMPORTANT:
        # no inferenceConfig with high reasoning.
        additionalModelRequestFields={
            "reasoningConfig": {
                "type": "enabled",
                "maxReasoningEffort": (
                    JUDGE_REASONING_EFFORT
                ),
            }
        },
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    visible_text, reasoning_texts = (
        extract_response_content(
            response
        )
    )

    usage = response.get(
        "usage",
        {}
    )

    return {
        "visible_text": visible_text,
        "reasoning_texts": reasoning_texts,
        "usage": {
            "input_tokens": usage.get(
                "inputTokens"
            ),
            "output_tokens_including_reasoning": (
                usage.get("outputTokens")
            ),
            "total_tokens": usage.get(
                "totalTokens"
            ),
        },
        "stop_reason": response.get(
            "stopReason"
        ),
        "latency_seconds": round(
            elapsed,
            2,
        ),
    }


TRANSIENT_ERROR_CODES = {
    "ModelErrorException",
    "InternalServerException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
    "ThrottlingException",
    "ModelNotReadyException",
}


def call_nova(
    full_prompt: str,
) -> dict:
    last_exception = None

    for attempt in range(
        1,
        MAX_API_ATTEMPTS + 1,
    ):
        if attempt > 1:
            print(
                f"Retry attempt {attempt}/"
                f"{MAX_API_ATTEMPTS}..."
            )

        try:
            return call_nova_once(
                full_prompt
            )

        except ClientError as exc:
            last_exception = exc

            error = exc.response.get(
                "Error",
                {}
            )

            code = error.get(
                "Code",
                "",
            )
            message = error.get(
                "Message",
                "",
            )

            print(
                f"API ERROR | {code}: {message}"
            )

            if (
                code not in TRANSIENT_ERROR_CODES
                or attempt >= MAX_API_ATTEMPTS
            ):
                raise

        except (
            EndpointConnectionError,
            ReadTimeoutError,
            ConnectTimeoutError,
            ConnectionClosedError,
        ) as exc:
            last_exception = exc

            print(
                "NETWORK ERROR | "
                f"{type(exc).__name__}: {exc}"
            )

            if attempt >= MAX_API_ATTEMPTS:
                raise

        delay_index = min(
            attempt - 1,
            len(RETRY_DELAYS_SECONDS) - 1,
        )

        delay = (
            RETRY_DELAYS_SECONDS[
                delay_index
            ]
        )

        print(
            f"Transient failure. Waiting "
            f"{delay} s before retry..."
        )

        time.sleep(
            delay
        )

    if last_exception:
        raise last_exception

    raise RuntimeError(
        "Nova request failed unexpectedly."
    )


def build_key_extract_input(
    instructions: str,
    source_report: str,
    reference_summary: str,
) -> str:
    return (
        f"{instructions}\n\n"
        "SOURCE FINANCIAL REPORT:\n"
        "<source_report>\n"
        f"{source_report}\n"
        "</source_report>\n\n"
        "REFERENCE SUMMARY:\n"
        "<reference_summary>\n"
        f"{reference_summary}\n"
        "</reference_summary>"
    )


def build_key_verify_input(
    instructions: str,
    source_report: str,
    draft_evaluation: dict,
) -> str:
    draft_json = json.dumps(
        draft_evaluation,
        ensure_ascii=False,
        indent=2,
    )

    return (
        f"{instructions}\n\n"
        "SOURCE FINANCIAL REPORT:\n"
        "<source_report>\n"
        f"{source_report}\n"
        "</source_report>\n\n"
        "DRAFT KEY INFORMATION:\n"
        "<draft_key_information>\n"
        f"{draft_json}\n"
        "</draft_key_information>"
    )


def compact_fixed_criteria(
    final_units: list[dict],
) -> list[dict]:
    return [
        {
            "id": item["id"],
            "information": item[
                "information"
            ],
            "importance": item[
                "importance"
            ],
        }
        for item in final_units
    ]


def build_candidate_input(
    evaluation_type: str,
    instructions: str,
    source_report: str,
    sentence_map: dict[str, str],
    final_units: list[dict] | None = None,
) -> str:

    candidate_with_ids = (
        format_candidate_with_ids(
            sentence_map
        )
    )

    if evaluation_type == "factual_numerical":
        return (
            f"{instructions}\n\n"
            "SOURCE FINANCIAL REPORT:\n"
            "<source_report>\n"
            f"{source_report}\n"
            "</source_report>\n\n"
            "CANDIDATE SUMMARY WITH TRACE IDS:\n"
            "<candidate_summary>\n"
            f"{candidate_with_ids}\n"
            "</candidate_summary>"
        )

    if evaluation_type == "completeness":
        if not final_units:
            raise ValueError(
                "Completeness requires verified fixed criteria."
            )

        fixed_json = json.dumps(
            {
                "key_information_units": (
                    compact_fixed_criteria(
                        final_units
                    )
                )
            },
            ensure_ascii=False,
            indent=2,
        )

        # NO source, reference, source evidence.
        return (
            f"{instructions}\n\n"
            "FIXED KEY INFORMATION CRITERIA:\n"
            "<fixed_key_information>\n"
            f"{fixed_json}\n"
            "</fixed_key_information>\n\n"
            "CANDIDATE SUMMARY WITH TRACE IDS:\n"
            "<candidate_summary>\n"
            f"{candidate_with_ids}\n"
            "</candidate_summary>"
        )

    if evaluation_type == "coherence":
        return (
            f"{instructions}\n\n"
            "CANDIDATE SUMMARY WITH TRACE IDS:\n"
            "<candidate_summary>\n"
            f"{candidate_with_ids}\n"
            "</candidate_summary>"
        )

    raise ValueError(
        f"Unknown evaluation type: "
        f"{evaluation_type}"
    )


def shared_category_root() -> Path:
    return (
        SHARED_RESULTS_ROOT
        / "nova_2_lite"
        / CATEGORY
    )


def key_draft_path(
    source_path: Path,
) -> Path:
    return (
        shared_category_root()
        / "key_information_draft"
        / f"{source_path.stem}.json"
    )


def key_verified_path(
    source_path: Path,
) -> Path:
    return (
        shared_category_root()
        / "key_information_verified"
        / f"{source_path.stem}.json"
    )


def validate_draft_key_information(
    evaluation: dict,
) -> list[str]:
    errors = []

    units = evaluation.get(
        "draft_key_information_units"
    )

    if not isinstance(
        units,
        list,
    ) or not units:
        return [
            "draft_key_information_units "
            "must be a non-empty list."
        ]

    if len(units) > 12:
        errors.append(
            "More than 12 draft units were returned."
        )

    ids = []

    for index, item in enumerate(
        units,
        start=1,
    ):
        if not isinstance(
            item,
            dict,
        ):
            errors.append(
                f"Draft unit {index} is not an object."
            )
            continue

        item_id = str(
            item.get(
                "id",
                "",
            )
        ).strip()

        information = str(
            item.get(
                "information",
                "",
            )
        ).strip()

        importance = normalize_label(
            item.get(
                "importance"
            )
        )

        source_evidence = str(
            item.get(
                "source_evidence",
                "",
            )
        ).strip()

        ids.append(
            item_id
        )

        if item_id != f"D{index}":
            errors.append(
                f"Expected draft id D{index}, "
                f"got {item_id!r}."
            )

        if not information:
            errors.append(
                f"{item_id or index}: "
                "information is empty."
            )

        if importance not in {
            "HIGH",
            "MEDIUM",
        }:
            errors.append(
                f"{item_id or index}: "
                "importance must be HIGH or MEDIUM."
            )

        if not source_evidence:
            errors.append(
                f"{item_id or index}: "
                "source_evidence is empty."
            )

        if not isinstance(
            item.get(
                "reference_support"
            ),
            bool,
        ):
            errors.append(
                f"{item_id or index}: "
                "reference_support must be boolean."
            )

    if len(ids) != len(
        set(ids)
    ):
        errors.append(
            "Draft IDs are not unique."
        )

    return errors


def validate_verified_key_information(
    evaluation: dict,
) -> list[str]:
    errors = []

    units = evaluation.get(
        "key_information_units"
    )

    if not isinstance(
        units,
        list,
    ) or not units:
        return [
            "key_information_units must be "
            "a non-empty list."
        ]

    if len(units) > 12:
        errors.append(
            "More than 12 verified units were returned."
        )

    ids = []

    for index, item in enumerate(
        units,
        start=1,
    ):
        if not isinstance(
            item,
            dict,
        ):
            errors.append(
                f"Verified unit {index} is not an object."
            )
            continue

        item_id = str(
            item.get(
                "id",
                "",
            )
        ).strip()

        information = str(
            item.get(
                "information",
                "",
            )
        ).strip()

        importance = normalize_label(
            item.get(
                "importance"
            )
        )

        source_evidence = str(
            item.get(
                "source_evidence",
                "",
            )
        ).strip()

        ids.append(
            item_id
        )

        if not re.fullmatch(r"K\d+", item_id):
            errors.append(
                f"Invalid final unit id {item_id!r}; "
                "expected format K<number>."
            )

        if not information:
            errors.append(
                f"{item_id or index}: "
                "information is empty."
            )

        if importance not in {
            "HIGH",
            "MEDIUM",
        }:
            errors.append(
                f"{item_id or index}: "
                "importance must be HIGH or MEDIUM."
            )

        if not source_evidence:
            errors.append(
                f"{item_id or index}: "
                "source_evidence is empty."
            )

        origins = item.get(
            "origin_draft_ids"
        )

        if isinstance(
            origins,
            str,
        ):
            raw_origins = (
                origins.strip()
            )

            parsed_origins = (
                re.findall(
                    r"D[0-9]+",
                    raw_origins,
                )
            )

            remainder = re.sub(
                r"D[0-9]+|[\s,;|/]+",
                "",
                raw_origins,
            )

            if (
                parsed_origins
                and not remainder
            ):
                item[
                    "origin_draft_ids"
                ] = parsed_origins
                origins = (
                    parsed_origins
                )

        if not isinstance(
            origins,
            list,
        ):
            errors.append(
                f"{item_id or index}: "
                "origin_draft_ids must be a list."
            )
        else:
            if not origins:
                errors.append(
                    f"{item_id or index}: "
                    "origin_draft_ids must not be empty."
                )

            for origin in origins:
                if not re.fullmatch(
                    r"D[0-9]+",
                    str(origin).strip(),
                ):
                    errors.append(
                        f"{item_id or index}: "
                        f"invalid origin_draft_id {origin!r}."
                    )

    if len(ids) != len(
        set(ids)
    ):
        errors.append(
            "Verified IDs are not unique."
        )

    redundancy_check = evaluation.get(
        "redundancy_check"
    )

    if not isinstance(
        redundancy_check,
        dict,
    ):
        errors.append(
            "redundancy_check must be an object."
        )
    else:
        if redundancy_check.get(
            "passed"
        ) is not True:
            errors.append(
                "Verifier did not confirm a passed "
                "cross-unit redundancy check."
            )

        overlapping_pairs = (
            redundancy_check.get(
                "overlapping_pairs"
            )
        )

        if overlapping_pairs != []:
            errors.append(
                "redundancy_check.overlapping_pairs "
                "must be an empty list in the final output."
            )

    return errors



def normalize_verified_unit_ids(
    evaluation: dict,
) -> tuple[dict, dict[str, str]]:

    units = evaluation.get(
        "key_information_units",
        [],
    )

    id_map = {}

    for index, item in enumerate(
        units,
        start=1,
    ):
        old_id = str(
            item.get("id", "")
        ).strip()

        new_id = f"K{index}"

        id_map[
            old_id
        ] = new_id

        item[
            "original_model_id"
        ] = old_id

        item[
            "id"
        ] = new_id

    evaluation[
        "id_normalization"
    ] = {
        "performed": any(
            old != new
            for old, new in id_map.items()
        ),
        "mapping": id_map,
        "note": (
            "Python normalized final key-information IDs "
            "to sequential K1..Kn without changing criterion content."
        ),
    }

    return evaluation, id_map



def request_and_save_json(
    full_prompt: str,
    output_path: Path,
    metadata: dict,
    validator,
) -> dict:

    try:
        api_result = call_nova(
            full_prompt
        )

        try:
            parsed = parse_json_response(
                api_result[
                    "visible_text"
                ]
            )

            validation_errors = (
                validator(
                    parsed
                )
            )

            parse_error = None

            status = (
                "success"
                if not validation_errors
                else "validation_failed"
            )

        except Exception as exc:
            parsed = None
            validation_errors = []
            parse_error = (
                f"{type(exc).__name__}: {exc}"
            )
            status = (
                "json_parse_failed"
            )

        result = {
            "status": status,
            "metadata": metadata,
            "usage": api_result[
                "usage"
            ],
            "latency_seconds": (
                api_result[
                    "latency_seconds"
                ]
            ),
            "stop_reason": (
                api_result[
                    "stop_reason"
                ]
            ),
            "reasoning": {
                "returned": bool(
                    api_result[
                        "reasoning_texts"
                    ]
                ),
                "content": (
                    api_result[
                        "reasoning_texts"
                    ]
                ),
            },
            "model_output_text": (
                api_result[
                    "visible_text"
                ]
            ),
            "evaluation": parsed,
            "validation_errors": (
                validation_errors
            ),
            "parse_error": (
                parse_error
            ),
        }

        save_json(
            output_path,
            result,
        )

        print(
            f"Input tokens:  "
            f"{api_result['usage']['input_tokens']:,}"
        )
        print(
            f"Output tokens: "
            f"{api_result['usage']['output_tokens_including_reasoning']:,} "
            "(includes hidden reasoning)"
        )
        print(
            f"Latency:       "
            f"{api_result['latency_seconds']:.2f} s"
        )
        print(
            f"STATUS:        {status}"
        )
        print(
            f"Saved:         {output_path}"
        )

        if validation_errors:
            print(
                "Validation errors:"
            )
            for error in validation_errors:
                print(
                    f"  - {error}"
                )

        return result

    except ClientError as exc:
        error = exc.response.get(
            "Error",
            {}
        )

        result = {
            "status": "api_failed",
            "metadata": metadata,
            "error_code": error.get(
                "Code"
            ),
            "error_message": error.get(
                "Message"
            ),
        }

        save_json(
            output_path,
            result,
        )

        print(
            "API FAILED | "
            f"{error.get('Code')}: "
            f"{error.get('Message')}"
        )

        return result

    except Exception as exc:
        result = {
            "status": "failed",
            "metadata": metadata,
            "error_type": type(
                exc
            ).__name__,
            "error_message": str(
                exc
            ),
        }

        save_json(
            output_path,
            result,
        )

        print(
            "FAILED | "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return result


def get_or_create_draft_key_information(
    instructions: str,
    source_report: str,
    reference_summary: str,
    source_path: Path,
    reference_path: Path,
) -> dict:

    output_path = key_draft_path(
        source_path
    )

    p_hash = text_hash(
        instructions
    )
    s_hash = text_hash(
        source_report
    )
    r_hash = text_hash(
        reference_summary
    )

    if (
        output_path.exists()
        and not FORCE_RERUN_SHARED
    ):
        existing = load_json(
            output_path
        )

        meta = existing.get(
            "metadata",
            {}
        )

        if (
            existing.get(
                "status"
            ) == "success"
            and meta.get(
                "prompt_version"
            ) == PROMPT_VERSION
            and meta.get(
                "prompt_hash"
            ) == p_hash
            and meta.get(
                "source_hash"
            ) == s_hash
            and meta.get(
                "reference_hash"
            ) == r_hash
        ):
            print()
            print(
                "SKIP existing successful "
                "V4 draft criteria."
            )
            return existing

    full_prompt = (
        build_key_extract_input(
            instructions=(
                instructions
            ),
            source_report=(
                source_report
            ),
            reference_summary=(
                reference_summary
            ),
        )
    )

    print()
    print("=" * 100)
    print(
        "SHARED STEP 1/2 — DRAFT "
        "KEY-INFORMATION EXTRACTION"
    )
    print("=" * 100)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "judge_model": (
            JUDGE_MODEL_NAME
        ),
        "judge_model_id": (
            JUDGE_MODEL_ID
        ),
        "judge_reasoning_enabled": True,
        "judge_reasoning_effort": (
            JUDGE_REASONING_EFFORT
        ),
        "evaluation_type": (
            "key_information_draft"
        ),
        "prompt_version": (
            PROMPT_VERSION
        ),
        "prompt_hash": (
            p_hash
        ),
        "source_hash": (
            s_hash
        ),
        "reference_hash": (
            r_hash
        ),
        "source_file": str(
            source_path
        ),
        "reference_file": str(
            reference_path
        ),
    }

    return request_and_save_json(
        full_prompt=(
            full_prompt
        ),
        output_path=(
            output_path
        ),
        metadata=(
            metadata
        ),
        validator=(
            validate_draft_key_information
        ),
    )


def get_or_create_verified_key_information(
    instructions: str,
    source_report: str,
    source_path: Path,
    draft_result: dict,
) -> dict:

    output_path = (
        key_verified_path(
            source_path
        )
    )

    draft_evaluation = (
        draft_result[
            "evaluation"
        ]
    )

    p_hash = text_hash(
        instructions
    )
    s_hash = text_hash(
        source_report
    )
    d_hash = json_hash(
        draft_evaluation
    )

    if (
        output_path.exists()
        and not FORCE_RERUN_SHARED
    ):
        existing = load_json(
            output_path
        )

        meta = existing.get(
            "metadata",
            {}
        )

        same_inputs = (
            meta.get(
                "prompt_version"
            ) == PROMPT_VERSION
            and meta.get(
                "prompt_hash"
            ) == p_hash
            and meta.get(
                "source_hash"
            ) == s_hash
            and meta.get(
                "draft_hash"
            ) == d_hash
        )

        if (
            existing.get(
                "status"
            ) == "success"
            and same_inputs
        ):
            print()
            print(
                "SKIP existing successful "
                "V4 verified criteria."
            )
            return existing

        if (
            existing.get(
                "status"
            ) == "validation_failed"
            and same_inputs
            and isinstance(
                existing.get(
                    "evaluation"
                ),
                dict,
            )
        ):
            recovered_evaluation = existing[
                "evaluation"
            ]

            recovery_errors = (
                validate_verified_key_information(
                    recovered_evaluation
                )
            )

            if not recovery_errors:
                normalized_evaluation, id_map = (
                    normalize_verified_unit_ids(
                        recovered_evaluation
                    )
                )

                post_normalization_errors = (
                    validate_verified_key_information(
                        normalized_evaluation
                    )
                )

                if not post_normalization_errors:
                    existing[
                        "status"
                    ] = "success"

                    existing[
                        "evaluation"
                    ] = normalized_evaluation

                    existing[
                        "validation_errors"
                    ] = []

                    existing[
                        "recovery"
                    ] = {
                        "performed": True,
                        "reason": (
                            "Recovered locally after applying deterministic "
                            "schema normalization to an already-paid "
                            "verifier response. No additional Bedrock "
                            "request was made."
                        ),
                    }

                    final_units = (
                        normalized_evaluation[
                            "key_information_units"
                        ]
                    )

                    existing[
                        "key_information_hash"
                    ] = json_hash(
                        compact_fixed_criteria(
                            final_units
                        )
                    )

                    save_json(
                        output_path,
                        existing,
                    )

                    print()
                    print(
                        "RECOVERED existing verifier result "
                        "locally — no new API request."
                    )

                    if normalized_evaluation[
                        "id_normalization"
                    ][
                        "performed"
                    ]:
                        print(
                            "Final key-information IDs were "
                            "normalized by Python:"
                        )
                        for old_id, new_id in id_map.items():
                            if old_id != new_id:
                                print(
                                    f"  {old_id} -> {new_id}"
                                )

                    return existing

            print()
            print(
                "Stored verifier result could not be "
                "recovered locally; a new API request "
                "will be made."
            )
            if recovery_errors:
                for error in recovery_errors:
                    print(
                        f"  - {error}"
                    )

    full_prompt = (
        build_key_verify_input(
            instructions=(
                instructions
            ),
            source_report=(
                source_report
            ),
            draft_evaluation=(
                draft_evaluation
            ),
        )
    )

    print()
    print("=" * 100)
    print(
        "SHARED STEP 2/2 — SOURCE VERIFICATION / "
        "NON-OVERLAP FINALIZATION"
    )
    print("=" * 100)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "judge_model": (
            JUDGE_MODEL_NAME
        ),
        "judge_model_id": (
            JUDGE_MODEL_ID
        ),
        "judge_reasoning_enabled": True,
        "judge_reasoning_effort": (
            JUDGE_REASONING_EFFORT
        ),
        "evaluation_type": (
            "key_information_verified"
        ),
        "prompt_version": (
            PROMPT_VERSION
        ),
        "prompt_hash": (
            p_hash
        ),
        "source_hash": (
            s_hash
        ),
        "draft_hash": (
            d_hash
        ),
        "source_file": str(
            source_path
        ),
        "draft_file": str(
            key_draft_path(
                source_path
            )
        ),
    }

    result = (
        request_and_save_json(
            full_prompt=(
                full_prompt
            ),
            output_path=(
                output_path
            ),
            metadata=(
                metadata
            ),
            validator=(
                validate_verified_key_information
            ),
        )
    )

    if result.get(
        "status"
    ) == "success":
        normalized_evaluation, id_map = (
            normalize_verified_unit_ids(
                result[
                    "evaluation"
                ]
            )
        )

        post_normalization_errors = (
            validate_verified_key_information(
                normalized_evaluation
            )
        )

        if post_normalization_errors:
            result[
                "status"
            ] = "validation_failed"
            result[
                "validation_errors"
            ] = (
                result.get(
                    "validation_errors",
                    []
                )
                + post_normalization_errors
            )

            save_json(
                output_path,
                result,
            )

            print(
                "Post-normalization validation failed:"
            )
            for error in post_normalization_errors:
                print(
                    f"  - {error}"
                )

            return result

        result[
            "evaluation"
        ] = normalized_evaluation

        final_units = (
            normalized_evaluation[
                "key_information_units"
            ]
        )

        result[
            "key_information_hash"
        ] = json_hash(
            compact_fixed_criteria(
                final_units
            )
        )

        save_json(
            output_path,
            result,
        )

        if normalized_evaluation[
            "id_normalization"
        ][
            "performed"
        ]:
            print(
                "Final key-information IDs were "
                "normalized by Python:"
            )
            for old_id, new_id in id_map.items():
                if old_id != new_id:
                    print(
                        f"  {old_id} -> {new_id}"
                    )

    return result


def normalize_candidate_claim_ids_v526(
    evaluation: dict,
) -> dict:

    for field_name, prefix in [
        ("factual_claims", "F"),
        ("numerical_claims", "N"),
    ]:
        claims = evaluation.get(
            field_name,
            [],
        )

        if not isinstance(
            claims,
            list,
        ):
            continue

        for index, claim in enumerate(
            claims,
            start=1,
        ):
            if not isinstance(
                claim,
                dict,
            ):
                continue

            old_id = str(
                claim.get(
                    "id",
                    "",
                )
            ).strip()

            new_id = (
                f"{prefix}{index}"
            )

            if (
                old_id
                and old_id != new_id
            ):
                claim[
                    "original_model_id"
                ] = old_id

            claim[
                "id"
            ] = new_id

    return evaluation




def normalize_factual_schema_labels_v527(
    evaluation: dict,
) -> dict:

    factual_claims = evaluation.get(
        "factual_claims",
        [],
    )

    if not isinstance(
        factual_claims,
        list,
    ):
        return evaluation

    for claim in factual_claims:
        if not isinstance(
            claim,
            dict,
        ):
            continue

        label = normalize_label(
            claim.get(
                "label"
            )
        )

        if label == "CORRECT":
            claim[
                "original_model_label"
            ] = claim.get(
                "label"
            )
            claim[
                "label"
            ] = "SUPPORTED"


        elif label in {
            "INCORRECT_VALUE",
            "INCORRECT_UNIT_OR_CURRENCY",
            "INCORRECT_PERIOD",
            "INCORRECT_DIRECTION_OR_COMPARISON",
        }:
            claim[
                "original_model_label"
            ] = claim.get(
                "label"
            )
            claim[
                "label"
            ] = "CONTRADICTED"
            claim[
                "schema_normalization_reason"
            ] = (
                "Specific numerical error label inside factual_claims "
                "was normalized to factual CONTRADICTED."
            )

    return evaluation


def normalize_numerical_schema_labels_v529(
    evaluation: dict,
) -> dict:

    numerical_claims = evaluation.get(
        "numerical_claims",
        [],
    )

    if not isinstance(
        numerical_claims,
        list,
    ):
        return evaluation

    for claim in numerical_claims:
        if not isinstance(
            claim,
            dict,
        ):
            continue

        label = normalize_label(
            claim.get(
                "label"
            )
        )

        if label == "SUPPORTED":
            claim[
                "original_model_label"
            ] = claim.get(
                "label"
            )
            claim[
                "label"
            ] = "CORRECT"

        elif label in {
            "INCORRECT_VALUE",
            "INCORRECT_UNIT_OR_CURRENCY",
            "INCORRECT_PERIOD",
            "INCORRECT_DIRECTION_OR_COMPARISON",
            "UNSUPPORTED_NUMBER",
            "OTHER",
        }:
            claim[
                "original_model_label"
            ] = claim.get(
                "label"
            )
            claim[
                "label"
            ] = "INCORRECT"

            if not claim.get(
                "error_type"
            ):
                claim[
                    "error_type"
                ] = label

    return evaluation


def ambiguous_factual_claim_ids_v529(
    evaluation: dict,
) -> list[str]:

    result = []

    for claim in evaluation.get(
        "factual_claims",
        [],
    ):
        if not isinstance(
            claim,
            dict,
        ):
            continue

        if normalize_label(
            claim.get(
                "label"
            )
        ) == "INCORRECT":
            claim_id = str(
                claim.get(
                    "id",
                    "",
                )
            ).strip()

            if claim_id:
                result.append(
                    claim_id
                )

    return result


def build_targeted_factual_label_repair_prompt_v529(
    source_report: str,
    evaluation: dict,
    sentence_map: dict[str, str],
    claim_ids: list[str],
) -> str:

    wanted = set(
        claim_ids
    )

    blocks = []

    for claim in evaluation.get(
        "factual_claims",
        [],
    ):
        if not isinstance(
            claim,
            dict,
        ):
            continue

        claim_id = str(
            claim.get(
                "id",
                "",
            )
        ).strip()

        if claim_id not in wanted:
            continue

        sentence_ids = claim.get(
            "candidate_sentence_ids",
            [],
        )

        sentence_text = "\n".join(
            f"[{sid}] {sentence_map.get(sid, '')}"
            for sid in sentence_ids
            if isinstance(
                sid,
                str,
            )
        )

        blocks.append(
            "\n".join(
                [
                    f"CLAIM_ID: {claim_id}",
                    f"CLAIM: {claim.get('claim', '')}",
                    "CANDIDATE SENTENCE(S):",
                    sentence_text,
                    f"PREVIOUS LABEL: {claim.get('label', '')}",
                ]
            )
        )

    claims_text = "\n\n".join(
        blocks
    )

    return (
        "You are repairing ONLY factual-label taxonomy errors in an already "
        "completed financial-summary evaluation.\n\n"
        "Use SOURCE as the only evidence. Do NOT re-evaluate any factual "
        "claim not listed below and do NOT evaluate numerical accuracy, "
        "completeness, coherence, or style.\n\n"
        "For every listed claim choose exactly one factual label:\n"
        "SUPPORTED — the claim is supported by SOURCE.\n"
        "CONTRADICTED — SOURCE directly conflicts with a material part of "
        "the claim.\n"
        "NOT_FOUND — SOURCE does not provide enough evidence to verify the "
        "claim and does not directly contradict it.\n\n"
        "Return only valid JSON using exactly:\n"
        "{\n"
        '  "claim_checks": [\n'
        "    {\n"
        '      "id": "F17",\n'
        '      "label": "CONTRADICTED",\n'
        '      "source_evidence": "...",\n'
        '      "reason": "..."\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "SOURCE FINANCIAL REPORT:\n"
        "<source_report>\n"
        f"{source_report}\n"
        "</source_report>\n\n"
        "ONLY THESE FACTUAL CLAIMS REQUIRE LABEL REPAIR:\n"
        "<claims>\n"
        f"{claims_text}\n"
        "</claims>"
    )


def apply_targeted_factual_label_repair_v529(
    evaluation: dict,
    repair_evaluation: dict,
    claim_ids: list[str],
) -> tuple[dict, list[str]]:

    allowed = set(
        claim_ids
    )

    checks = repair_evaluation.get(
        "claim_checks",
        [],
    )

    if not isinstance(
        checks,
        list,
    ):
        checks = []

    by_id = {
        str(item.get("id", "")).strip(): item
        for item in checks
        if isinstance(
            item,
            dict,
        )
    }

    resolved = []

    for claim in evaluation.get(
        "factual_claims",
        [],
    ):
        if not isinstance(
            claim,
            dict,
        ):
            continue

        claim_id = str(
            claim.get(
                "id",
                "",
            )
        ).strip()

        if claim_id not in allowed:
            continue

        repair = by_id.get(
            claim_id
        )

        if not isinstance(
            repair,
            dict,
        ):
            continue

        label = normalize_label(
            repair.get(
                "label"
            )
        )

        if label not in {
            "SUPPORTED",
            "CONTRADICTED",
            "NOT_FOUND",
        }:
            continue

        claim[
            "original_model_label"
        ] = claim.get(
            "label"
        )
        claim[
            "label"
        ] = label

        evidence = str(
            repair.get(
                "source_evidence",
                "",
            )
        ).strip()

        if evidence:
            claim[
                "source_evidence"
            ] = evidence

        claim[
            "label_repair_reason"
        ] = repair.get(
            "reason"
        )

        resolved.append(
            claim_id
        )

    unresolved = sorted(
        allowed - set(resolved)
    )

    return (
        evaluation,
        unresolved,
    )


def normalize_factual_numerical_schema_v527(
    evaluation: dict,
) -> dict:
    """
    Apply only deterministic cosmetic/schema normalizations before
    semantic validation.
    """
    evaluation = (
        normalize_candidate_claim_ids_v526(
            evaluation
        )
    )

    evaluation = (
        normalize_factual_schema_labels_v527(
            evaluation
        )
    )

    evaluation = (
        normalize_numerical_schema_labels_v529(
            evaluation
        )
    )

    return evaluation



def factual_claim_ids_with_invalid_trace_v5211(
    evaluation: dict,
    sentence_map: dict[str, str],
) -> list[str]:

    valid_ids = set(
        sentence_map
    )
    result = []

    for claim in evaluation.get(
        "factual_claims",
        [],
    ):
        if not isinstance(
            claim,
            dict,
        ):
            continue

        claim_id = str(
            claim.get(
                "id",
                "",
            )
        ).strip()

        ids = claim.get(
            "candidate_sentence_ids"
        )

        bad = (
            not isinstance(
                ids,
                list,
            )
            or not ids
            or any(
                not isinstance(
                    sid,
                    str,
                )
                or sid not in valid_ids
                for sid in (
                    ids
                    if isinstance(
                        ids,
                        list,
                    )
                    else []
                )
            )
        )

        if (
            bad
            and claim_id
        ):
            result.append(
                claim_id
            )

    return result


def build_factual_trace_id_repair_prompt_v5211(
    evaluation: dict,
    sentence_map: dict[str, str],
) -> str:
    """
    Candidate-only trace alignment. No source report is needed because
    this step does not judge correctness; it only maps already-created
    factual claims back to candidate sentences.
    """
    candidate_text = format_candidate_with_ids(
        sentence_map
    )

    claim_blocks = []

    for claim in evaluation.get(
        "factual_claims",
        [],
    ):
        if not isinstance(
            claim,
            dict,
        ):
            continue

        claim_blocks.append(
            "\n".join(
                [
                    f"CLAIM_ID: {claim.get('id', '')}",
                    f"CURRENT_SENTENCE_IDS: "
                    f"{claim.get('candidate_sentence_ids', [])}",
                    f"CLAIM: {claim.get('claim', '')}",
                ]
            )
        )

    claims_text = "\n\n".join(
        claim_blocks
    )

    return (
        "You are repairing ONLY traceability links between already-created "
        "factual claims and the candidate summary.\n\n"

        "IMPORTANT:\n"
        "- Do NOT judge whether a claim is true or false.\n"
        "- Do NOT use outside knowledge.\n"
        "- Do NOT change claim text, labels, evidence, or numerical claims.\n"
        "- Use ONLY the candidate summary sentences below.\n"
        "- For every factual claim, assign the candidate sentence ID or IDs "
        "that contain the information summarized by that claim.\n"
        "- If one claim combines information from multiple candidate "
        "sentences, return all relevant IDs.\n"
        "- Multiple factual claims may legitimately refer to the same "
        "candidate sentence.\n"
        "- Every returned ID must be one of the supplied [S...] IDs.\n"
        "- Return a mapping for EVERY factual claim listed below.\n\n"

        "Return only valid JSON using exactly:\n"
        "{\n"
        '  "claim_sentence_mappings": [\n'
        "    {\n"
        '      "id": "F1",\n'
        '      "candidate_sentence_ids": ["S1"]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"

        "CANDIDATE SUMMARY:\n"
        "<candidate_summary>\n"
        f"{candidate_text}\n"
        "</candidate_summary>\n\n"

        "FACTUAL CLAIMS TO REMAP:\n"
        "<factual_claims>\n"
        f"{claims_text}\n"
        "</factual_claims>"
    )


def apply_factual_trace_id_repair_v5211(
    evaluation: dict,
    repair_evaluation: dict,
    sentence_map: dict[str, str],
) -> tuple[dict, list[str]]:
    """
    Replace only candidate_sentence_ids in factual_claims.
    """
    valid_sentence_ids = set(
        sentence_map
    )

    mappings = repair_evaluation.get(
        "claim_sentence_mappings",
        [],
    )

    if not isinstance(
        mappings,
        list,
    ):
        mappings = []

    by_id = {
        str(
            item.get(
                "id",
                "",
            )
        ).strip(): item
        for item in mappings
        if isinstance(
            item,
            dict,
        )
    }

    unresolved = []

    for claim in evaluation.get(
        "factual_claims",
        [],
    ):
        if not isinstance(
            claim,
            dict,
        ):
            continue

        claim_id = str(
            claim.get(
                "id",
                "",
            )
        ).strip()

        mapping = by_id.get(
            claim_id
        )

        if not isinstance(
            mapping,
            dict,
        ):
            unresolved.append(
                claim_id
            )
            continue

        ids = mapping.get(
            "candidate_sentence_ids"
        )

        if (
            not isinstance(
                ids,
                list,
            )
            or not ids
            or any(
                not isinstance(
                    sid,
                    str,
                )
                or sid not in valid_sentence_ids
                for sid in ids
            )
        ):
            unresolved.append(
                claim_id
            )
            continue

        old_ids = claim.get(
            "candidate_sentence_ids"
        )

        normalized_ids = list(
            dict.fromkeys(
                ids
            )
        )

        if old_ids != normalized_ids:
            claim[
                "original_candidate_sentence_ids"
            ] = old_ids

        claim[
            "candidate_sentence_ids"
        ] = normalized_ids

    return (
        evaluation,
        unresolved,
    )


def validate_factual_numerical_output(
    evaluation: dict,
    sentence_map: dict[str, str],
) -> list[str]:
    errors = []

    valid_sentence_ids = set(
        sentence_map
    )

    factual_claims = evaluation.get(
        "factual_claims"
    )

    numerical_claims = evaluation.get(
        "numerical_claims"
    )

    if not isinstance(
        factual_claims,
        list,
    ):
        errors.append(
            "factual_claims must be a list."
        )
        factual_claims = []

    if not isinstance(
        numerical_claims,
        list,
    ):
        errors.append(
            "numerical_claims must be a list."
        )
        numerical_claims = []

    allowed_factual = {
        "SUPPORTED",
        "CONTRADICTED",
        "NOT_FOUND",
    }

    allowed_numeric = {
        "CORRECT",
        "INCORRECT",
    }

    allowed_error_types = {
        "INCORRECT_VALUE",
        "INCORRECT_UNIT_OR_CURRENCY",
        "INCORRECT_PERIOD",
        "INCORRECT_DIRECTION_OR_COMPARISON",
        "UNSUPPORTED_NUMBER",
        "OTHER",
    }

    factual_ids = []

    for index, item in enumerate(
        factual_claims,
        start=1,
    ):
        if not isinstance(
            item,
            dict,
        ):
            errors.append(
                f"Factual item {index} "
                "is not an object."
            )
            continue

        item_id = str(
            item.get(
                "id",
                "",
            )
        ).strip()

        factual_ids.append(
            item_id
        )

        if item_id != f"F{index}":
            errors.append(
                f"Expected F{index}, "
                f"got {item_id!r}."
            )

        label = normalize_label(
            item.get(
                "label"
            )
        )

        if label not in allowed_factual:
            errors.append(
                f"{item_id}: invalid factual "
                f"label {label!r}."
            )

        errors.extend(
            f"{item_id}: {error}"
            for error in validate_sentence_ids(
                item.get(
                    "candidate_sentence_ids"
                ),
                valid_sentence_ids,
                allow_empty=False,
                field_name=(
                    "candidate_sentence_ids"
                ),
            )
        )

        if not str(
            item.get(
                "claim",
                "",
            )
        ).strip():
            errors.append(
                f"{item_id}: claim is empty."
            )

    if len(factual_ids) != len(
        set(factual_ids)
    ):
        errors.append(
            "Factual IDs are not unique."
        )

    numerical_ids = []

    for index, item in enumerate(
        numerical_claims,
        start=1,
    ):
        if not isinstance(
            item,
            dict,
        ):
            errors.append(
                f"Numerical item {index} "
                "is not an object."
            )
            continue

        item_id = str(
            item.get(
                "id",
                "",
            )
        ).strip()

        numerical_ids.append(
            item_id
        )

        if item_id != f"N{index}":
            errors.append(
                f"Expected N{index}, "
                f"got {item_id!r}."
            )

        label = normalize_label(
            item.get(
                "label"
            )
        )

        if label not in allowed_numeric:
            errors.append(
                f"{item_id}: invalid numerical "
                f"label {label!r}."
            )

        errors.extend(
            f"{item_id}: {error}"
            for error in validate_sentence_ids(
                item.get(
                    "candidate_sentence_ids"
                ),
                valid_sentence_ids,
                allow_empty=False,
                field_name=(
                    "candidate_sentence_ids"
                ),
            )
        )

        if not str(
            item.get(
                "claim",
                "",
            )
        ).strip():
            errors.append(
                f"{item_id}: claim is empty."
            )

        error_type = item.get(
            "error_type"
        )

        if label == "CORRECT":
            if error_type not in {
                None,
                "",
            }:
                errors.append(
                    f"{item_id}: CORRECT claim "
                    "must have error_type=null."
                )

        if label == "INCORRECT":
            if normalize_label(
                error_type
            ) not in allowed_error_types:
                errors.append(
                    f"{item_id}: INCORRECT claim "
                    f"has invalid error_type "
                    f"{error_type!r}."
                )

    if len(numerical_ids) != len(
        set(numerical_ids)
    ):
        errors.append(
            "Numerical IDs are not unique."
        )

    return errors


def validate_completeness_output(
    evaluation: dict,
    final_units: list[dict],
    sentence_map: dict[str, str],
) -> list[str]:
    errors = []

    valid_sentence_ids = set(
        sentence_map
    )

    coverage_results = evaluation.get(
        "coverage_results"
    )

    if not isinstance(
        coverage_results,
        list,
    ):
        return [
            "coverage_results must be a list."
        ]

    expected_ids = [
        item["id"]
        for item in final_units
    ]

    returned_ids = [
        str(
            item.get(
                "id"
            )
        )
        for item in coverage_results
        if isinstance(
            item,
            dict,
        )
    ]

    if returned_ids != expected_ids:
        errors.append(
            "coverage_results IDs/order "
            "do not exactly match verified criteria."
        )

    allowed = {
        "PRESENT",
        "PARTIALLY_PRESENT",
        "MISSING",
    }

    for item in coverage_results:
        if not isinstance(
            item,
            dict,
        ):
            errors.append(
                "A coverage result "
                "is not an object."
            )
            continue

        item_id = str(
            item.get(
                "id",
                "",
            )
        )

        coverage = normalize_label(
            item.get(
                "coverage"
            )
        )

        if coverage not in allowed:
            errors.append(
                f"{item_id}: invalid coverage label."
            )
            continue

        sentence_ids = item.get(
            "candidate_sentence_ids"
        )

        errors.extend(
            f"{item_id}: {error}"
            for error in validate_sentence_ids(
                sentence_ids,
                valid_sentence_ids,
                allow_empty=(
                    coverage == "MISSING"
                ),
                field_name=(
                    "candidate_sentence_ids"
                ),
            )
        )

        if (
            coverage == "MISSING"
            and sentence_ids not in (
                [],
                None,
            )
        ):
            if isinstance(
                sentence_ids,
                list,
            ) and len(
                sentence_ids
            ) > 0:
                errors.append(
                    f"{item_id}: MISSING must "
                    "use an empty sentence ID list."
                )

    score = safe_int(
        evaluation.get(
            "score"
        )
    )

    if score not in {
        0,
        1,
        2,
        3,
        4,
    }:
        errors.append(
            "Completeness score must be "
            "an integer 0-4."
        )

    expected_label = {
        0: "Not at all",
        1: "Not generally",
        2: "Neutral/Mixed",
        3: "Generally yes",
        4: "Yes",
    }.get(
        score
    )

    if (
        expected_label is not None
        and evaluation.get(
            "label"
        ) != expected_label
    ):
        errors.append(
            "Completeness label does not "
            "match the score."
        )

    return errors


def validate_coherence_output(
    evaluation: dict,
    sentence_map: dict[str, str],
) -> list[str]:
    errors = []

    valid_sentence_ids = set(
        sentence_map
    )

    score = safe_int(
        evaluation.get(
            "score"
        )
    )

    if score not in {
        0,
        1,
        2,
        3,
        4,
    }:
        errors.append(
            "Coherence score must be "
            "integer 0-4."
        )

    expected_label = {
        0: "Not at all",
        1: "Not generally",
        2: "Neutral/Mixed",
        3: "Generally yes",
        4: "Yes",
    }.get(
        score
    )

    if (
        expected_label is not None
        and evaluation.get(
            "label"
        ) != expected_label
    ):
        errors.append(
            "Coherence label does not "
            "match the score."
        )

    issues = evaluation.get(
        "issues"
    )

    if not isinstance(
        issues,
        list,
    ):
        errors.append(
            "issues must be a list."
        )
        return errors

    allowed_issue_types = {
        "SELF_CONTRADICTION",
        "LOGICAL_GAP",
        "ORGANIZATION",
        "REPETITION",
        "OTHER",
    }

    for index, issue in enumerate(
        issues,
        start=1,
    ):
        if not isinstance(
            issue,
            dict,
        ):
            errors.append(
                f"Coherence issue {index} "
                "is not an object."
            )
            continue

        issue_type = normalize_label(
            issue.get(
                "type"
            )
        )

        if issue_type not in allowed_issue_types:
            errors.append(
                f"Coherence issue {index} "
                f"has invalid type {issue_type!r}."
            )

        errors.extend(
            f"Coherence issue {index}: {error}"
            for error in validate_sentence_ids(
                issue.get(
                    "candidate_sentence_ids"
                ),
                valid_sentence_ids,
                allow_empty=False,
                field_name=(
                    "candidate_sentence_ids"
                ),
            )
        )

    return errors



def raw_result_path(
    evaluation_type: str,
    candidate_stem: str,
) -> Path:
    return (
        JUDGE_RESULTS_ROOT
        / "nova_2_lite"
        / CATEGORY
        / "raw"
        / evaluation_type
        / f"{candidate_stem}.json"
    )


def can_reuse_candidate_result(
    existing: dict,
    prompt_hash_value: str,
    source_hash_value: str | None,
    candidate_hash_value: str,
    sentence_map_hash_value: str,
    key_information_hash_value: str | None,
) -> bool:

    if existing.get(
        "status"
    ) != "success":
        return False

    meta = existing.get(
        "metadata",
        {}
    )

    if meta.get(
        "prompt_version"
    ) != PROMPT_VERSION:
        return False

    if meta.get(
        "prompt_hash"
    ) != prompt_hash_value:
        return False

    if meta.get(
        "candidate_hash"
    ) != candidate_hash_value:
        return False

    if meta.get(
        "sentence_map_hash"
    ) != sentence_map_hash_value:
        return False

    if (
        source_hash_value is not None
        and meta.get(
            "source_hash"
        ) != source_hash_value
    ):
        return False

    if (
        key_information_hash_value is not None
        and meta.get(
            "key_information_hash"
        ) != key_information_hash_value
    ):
        return False

    return True


def run_candidate_evaluation(
    evaluation_type: str,
    instructions: str,
    source_report: str,
    candidate_summary: str,
    sentence_map: dict[str, str],
    source_path: Path,
    candidate_path: Path,
    generator_model_name: str,
    final_units: list[dict] | None = None,
    key_information_hash_value: str | None = None,
    key_information_file: Path | None = None,
) -> dict:

    output_path = raw_result_path(
        evaluation_type,
        candidate_path.stem,
    )

    p_hash = text_hash(
        instructions
    )
    c_hash = text_hash(
        candidate_summary
    )
    sentence_map_hash_value = json_hash(
        sentence_map
    )

    s_hash = (
        text_hash(
            source_report
        )
        if evaluation_type
        == "factual_numerical"
        else None
    )

    if (
        output_path.exists()
        and not FORCE_RERUN_ALL
    ):
        existing = load_json(
            output_path
        )

        if can_reuse_candidate_result(
            existing=(
                existing
            ),
            prompt_hash_value=(
                p_hash
            ),
            source_hash_value=(
                s_hash
            ),
            candidate_hash_value=(
                c_hash
            ),
            sentence_map_hash_value=(
                sentence_map_hash_value
            ),
            key_information_hash_value=(
                key_information_hash_value
                if evaluation_type
                == "completeness"
                else None
            ),
        ):
            print(
                f"SKIP existing current V4 "
                f"result: {evaluation_type}"
            )
            return existing

    full_prompt = (
        build_candidate_input(
            evaluation_type=(
                evaluation_type
            ),
            instructions=(
                instructions
            ),
            source_report=(
                source_report
            ),
            sentence_map=(
                sentence_map
            ),
            final_units=(
                final_units
            ),
        )
    )

    print()
    print("-" * 90)
    print(
        f"Evaluation: {evaluation_type}"
    )
    print("-" * 90)

    try:
        api_result = call_nova(
            full_prompt
        )

        try:
            parsed = parse_json_response(
                api_result[
                    "visible_text"
                ]
            )

            if (
                evaluation_type
                == "factual_numerical"
            ):
                validation_errors = (
                    validate_factual_numerical_output(
                        parsed,
                        sentence_map,
                    )
                )

            elif (
                evaluation_type
                == "completeness"
            ):
                validation_errors = (
                    validate_completeness_output(
                        parsed,
                        final_units or [],
                        sentence_map,
                    )
                )

            elif (
                evaluation_type
                == "coherence"
            ):
                validation_errors = (
                    validate_coherence_output(
                        parsed,
                        sentence_map,
                    )
                )

            else:
                validation_errors = [
                    "Unknown evaluation type."
                ]

            parse_error = None

            status = (
                "success"
                if not validation_errors
                else "validation_failed"
            )

        except Exception as exc:
            parsed = None
            validation_errors = []
            status = (
                "json_parse_failed"
            )
            parse_error = (
                f"{type(exc).__name__}: {exc}"
            )

        result = {
            "status": status,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "judge_model": JUDGE_MODEL_NAME,
                "judge_model_id": JUDGE_MODEL_ID,
                "judge_reasoning_enabled": True,
                "judge_reasoning_effort": (
                    JUDGE_REASONING_EFFORT
                ),
                "candidate_generation_mode": (
                    RUN_MODE
                ),
                "generator_model": (
                    generator_model_name
                ),
                "candidate_file": str(
                    candidate_path
                ),
                "evaluation_type": (
                    evaluation_type
                ),
                "prompt_version": (
                    PROMPT_VERSION
                ),
                "prompt_hash": (
                    p_hash
                ),
                "candidate_hash": (
                    c_hash
                ),
                "sentence_map_hash": (
                    sentence_map_hash_value
                ),
                "source_file": (
                    str(
                        source_path
                    )
                    if evaluation_type
                    == "factual_numerical"
                    else None
                ),
                "source_hash": (
                    s_hash
                ),
                "key_information_hash": (
                    key_information_hash_value
                    if evaluation_type
                    == "completeness"
                    else None
                ),
                "key_information_file": (
                    str(
                        key_information_file
                    )
                    if (
                        evaluation_type
                        == "completeness"
                        and key_information_file
                        is not None
                    )
                    else None
                ),
            },
            "candidate_sentence_map": (
                sentence_map
            ),
            "usage": api_result[
                "usage"
            ],
            "latency_seconds": (
                api_result[
                    "latency_seconds"
                ]
            ),
            "stop_reason": (
                api_result[
                    "stop_reason"
                ]
            ),
            "reasoning": {
                "returned": bool(
                    api_result[
                        "reasoning_texts"
                    ]
                ),
                "content": (
                    api_result[
                        "reasoning_texts"
                    ]
                ),
            },
            "model_output_text": (
                api_result[
                    "visible_text"
                ]
            ),
            "evaluation": (
                parsed
            ),
            "validation_errors": (
                validation_errors
            ),
            "parse_error": (
                parse_error
            ),
        }

        save_json(
            output_path,
            result,
        )

        print(
            f"Input tokens:  "
            f"{api_result['usage']['input_tokens']:,}"
        )
        print(
            f"Output tokens: "
            f"{api_result['usage']['output_tokens_including_reasoning']:,} "
            "(includes hidden reasoning)"
        )
        print(
            f"Stop reason:   "
            f"{api_result['stop_reason']}"
        )
        print(
            f"Latency:       "
            f"{api_result['latency_seconds']:.2f} s"
        )
        print(
            f"STATUS:        {status}"
        )
        print(
            f"Saved:         {output_path}"
        )

        if validation_errors:
            print(
                "Validation errors:"
            )
            for error in validation_errors:
                print(
                    f"  - {error}"
                )

        return result

    except ClientError as exc:
        error = exc.response.get(
            "Error",
            {}
        )

        result = {
            "status": "api_failed",
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "judge_model": (
                    JUDGE_MODEL_NAME
                ),
                "judge_model_id": (
                    JUDGE_MODEL_ID
                ),
                "candidate_generation_mode": (
                    RUN_MODE
                ),
                "generator_model": (
                    generator_model_name
                ),
                "candidate_file": str(
                    candidate_path
                ),
                "evaluation_type": (
                    evaluation_type
                ),
                "prompt_version": (
                    PROMPT_VERSION
                ),
                "prompt_hash": (
                    p_hash
                ),
                "candidate_hash": (
                    c_hash
                ),
                "sentence_map_hash": (
                    sentence_map_hash_value
                ),
                "source_hash": (
                    s_hash
                ),
                "key_information_hash": (
                    key_information_hash_value
                ),
            },
            "candidate_sentence_map": (
                sentence_map
            ),
            "error_code": error.get(
                "Code"
            ),
            "error_message": error.get(
                "Message"
            ),
        }

        save_json(
            output_path,
            result,
        )

        print(
            "API FAILED | "
            f"{error.get('Code')}: "
            f"{error.get('Message')}"
        )

        return result

    except Exception as exc:
        result = {
            "status": "failed",
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "judge_model": (
                    JUDGE_MODEL_NAME
                ),
                "judge_model_id": (
                    JUDGE_MODEL_ID
                ),
                "candidate_generation_mode": (
                    RUN_MODE
                ),
                "generator_model": (
                    generator_model_name
                ),
                "candidate_file": str(
                    candidate_path
                ),
                "evaluation_type": (
                    evaluation_type
                ),
                "prompt_version": (
                    PROMPT_VERSION
                ),
                "prompt_hash": (
                    p_hash
                ),
                "candidate_hash": (
                    c_hash
                ),
                "sentence_map_hash": (
                    sentence_map_hash_value
                ),
                "source_hash": (
                    s_hash
                ),
                "key_information_hash": (
                    key_information_hash_value
                ),
            },
            "candidate_sentence_map": (
                sentence_map
            ),
            "error_type": type(
                exc
            ).__name__,
            "error_message": str(
                exc
            ),
        }

        save_json(
            output_path,
            result,
        )

        print(
            "FAILED | "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return result


def calculate_factual_score(
    evaluation: dict,
) -> dict:
    claims = evaluation.get(
        "factual_claims",
        [],
    )

    supported = sum(
        normalize_label(
            item.get(
                "label"
            )
        ) == "SUPPORTED"
        for item in claims
    )

    contradicted = sum(
        normalize_label(
            item.get(
                "label"
            )
        ) == "CONTRADICTED"
        for item in claims
    )

    not_found = sum(
        normalize_label(
            item.get(
                "label"
            )
        ) == "NOT_FOUND"
        for item in claims
    )

    total = (
        supported
        + contradicted
        + not_found
    )

    score = (
        supported / total
        if total > 0
        else None
    )

    return {
        "score": (
            round(
                score,
                6,
            )
            if score is not None
            else None
        ),
        "supported": (
            supported
        ),
        "contradicted": (
            contradicted
        ),
        "not_found": (
            not_found
        ),
        "total": (
            total
        ),
    }


def calculate_numerical_score(
    evaluation: dict,
) -> dict:
    claims = evaluation.get(
        "numerical_claims",
        [],
    )

    correct = sum(
        normalize_label(
            item.get(
                "label"
            )
        ) == "CORRECT"
        for item in claims
    )

    incorrect = sum(
        normalize_label(
            item.get(
                "label"
            )
        ) == "INCORRECT"
        for item in claims
    )

    total = (
        correct
        + incorrect
    )

    score = (
        correct / total
        if total > 0
        else None
    )

    return {
        "score": (
            round(
                score,
                6,
            )
            if score is not None
            else None
        ),
        "correct": (
            correct
        ),
        "incorrect": (
            incorrect
        ),
        "total": (
            total
        ),
    }


def parse_rubric_score(
    evaluation: dict,
):
    score = safe_int(
        evaluation.get(
            "score"
        )
    )

    if score not in {
        0,
        1,
        2,
        3,
        4,
    }:
        return None

    return score


def sum_usage(
    raw_results: dict,
) -> dict:
    input_total = 0
    output_total = 0
    token_total = 0
    latency_total = 0.0

    for raw in raw_results.values():
        usage = raw.get(
            "usage",
            {},
        )

        input_total += (
            usage.get(
                "input_tokens"
            )
            or 0
        )

        output_total += (
            usage.get(
                "output_tokens_including_reasoning"
            )
            or 0
        )

        token_total += (
            usage.get(
                "total_tokens"
            )
            or 0
        )

        latency_total += (
            raw.get(
                "latency_seconds"
            )
            or 0
        )

    return {
        "input_tokens_total": (
            input_total
        ),
        "output_tokens_total_including_reasoning": (
            output_total
        ),
        "total_tokens": (
            token_total
        ),
        "latency_seconds_total": round(
            latency_total,
            2,
        ),
    }


def sum_shared_usage(
    draft_result: dict,
    verified_result: dict,
) -> dict:
    return sum_usage(
        {
            "draft": (
                draft_result
            ),
            "verification": (
                verified_result
            ),
        }
    )


def merged_result_path(
    candidate_stem: str,
) -> Path:
    return (
        JUDGE_RESULTS_ROOT
        / "nova_2_lite"
        / CATEGORY
        / "merged"
        / f"{candidate_stem}_judge.json"
    )


def enrich_claims_with_sentences(
    claims: list[dict],
    sentence_map: dict[str, str],
) -> list[dict]:
    enriched = []

    for claim in claims:
        item = dict(
            claim
        )

        item[
            "candidate_sentences"
        ] = resolve_sentence_ids(
            claim.get(
                "candidate_sentence_ids"
            ),
            sentence_map,
        )

        enriched.append(
            item
        )

    return enriched


def combine_units_with_coverage(
    final_units: list[dict],
    completeness_evaluation: dict,
    sentence_map: dict[str, str],
) -> list[dict]:

    coverage_map = {
        str(
            item.get(
                "id"
            )
        ): item
        for item
        in completeness_evaluation.get(
            "coverage_results",
            [],
        )
        if isinstance(
            item,
            dict,
        )
    }

    combined = []

    for unit in final_units:
        item_id = unit[
            "id"
        ]

        coverage = coverage_map.get(
            item_id,
            {},
        )

        sentence_ids = coverage.get(
            "candidate_sentence_ids",
            [],
        )

        combined.append(
            {
                "id": (
                    item_id
                ),
                "information": unit.get(
                    "information"
                ),
                "importance": unit.get(
                    "importance"
                ),
                "source_evidence": unit.get(
                    "source_evidence"
                ),
                "origin_draft_ids": unit.get(
                    "origin_draft_ids",
                    [],
                ),
                "coverage": coverage.get(
                    "coverage"
                ),
                "candidate_sentence_ids": (
                    sentence_ids
                ),
                "candidate_sentences": (
                    resolve_sentence_ids(
                        sentence_ids,
                        sentence_map,
                    )
                ),
            }
        )

    return combined


def enrich_coherence_issues(
    issues: list[dict],
    sentence_map: dict[str, str],
) -> list[dict]:
    enriched = []

    for issue in issues:
        item = dict(
            issue
        )

        item[
            "candidate_sentences"
        ] = resolve_sentence_ids(
            issue.get(
                "candidate_sentence_ids"
            ),
            sentence_map,
        )

        enriched.append(
            item
        )

    return enriched


def merge_candidate_results(
    generator_key: str,
    generator_model_name: str,
    candidate_path: Path,
    source_path: Path,
    reference_path: Path,
    draft_result: dict,
    verified_result: dict,
    sentence_map: dict[str, str],
    raw_results: dict,
):

    required = {
        "factual_numerical",
        "completeness",
        "coherence",
    }

    all_success = all(
        raw_results.get(
            evaluation_type,
            {},
        ).get(
            "status"
        ) == "success"
        for evaluation_type
        in required
    )

    if not all_success:
        print(
            "Merged result was not created "
            "because not all three Judge "
            "requests succeeded."
        )
        return None

    factual_numerical = (
        raw_results[
            "factual_numerical"
        ][
            "evaluation"
        ]
    )

    completeness_eval = (
        raw_results[
            "completeness"
        ][
            "evaluation"
        ]
    )

    coherence_eval = (
        raw_results[
            "coherence"
        ][
            "evaluation"
        ]
    )

    factual = (
        calculate_factual_score(
            factual_numerical
        )
    )

    numerical = (
        calculate_numerical_score(
            factual_numerical
        )
    )

    completeness_score = (
        parse_rubric_score(
            completeness_eval
        )
    )

    coherence_score = (
        parse_rubric_score(
            coherence_eval
        )
    )

    if completeness_score is None:
        raise ValueError(
            "Completeness returned "
            "an invalid score."
        )

    if coherence_score is None:
        raise ValueError(
            "Coherence returned "
            "an invalid score."
        )

    final_units = (
        verified_result[
            "evaluation"
        ][
            "key_information_units"
        ]
    )

    completeness_combined = (
        combine_units_with_coverage(
            final_units,
            completeness_eval,
            sentence_map,
        )
    )

    present = sum(
        normalize_label(
            item.get(
                "coverage"
            )
        ) == "PRESENT"
        for item
        in completeness_combined
    )

    partial = sum(
        normalize_label(
            item.get(
                "coverage"
            )
        ) == "PARTIALLY_PRESENT"
        for item
        in completeness_combined
    )

    missing = sum(
        normalize_label(
            item.get(
                "coverage"
            )
        ) == "MISSING"
        for item
        in completeness_combined
    )

    merged = {
        "status": "success",

        "candidate": {
            "file": str(
                candidate_path
            ),
            "report": (
                source_path.name
            ),
            "report_type": (
                CATEGORY
            ),
            "generator_key": (
                generator_key
            ),
            "generator_model": (
                generator_model_name
            ),
            "prompt_type": (
                SUMMARY_PROMPT
            ),
            "generation_mode": (
                RUN_MODE
            ),
        },

        "judge": {
            "model": (
                JUDGE_MODEL_NAME
            ),
            "model_id": (
                JUDGE_MODEL_ID
            ),
            "reasoning_enabled": (
                True
            ),
            "reasoning_effort": (
                JUDGE_REASONING_EFFORT
            ),
            "prompt_version": (
                PROMPT_VERSION
            ),
        },

        "evaluation_design": {
            "pointwise_candidate_evaluation": (
                True
            ),
            "traceability_method": (
                "python_generated_sentence_ids"
            ),
            "faithfulness_numerical_uses_source": (
                True
            ),
            "completeness_uses_source": (
                False
            ),
            "completeness_uses_reference": (
                False
            ),
            "completeness_uses_verified_fixed_criteria": (
                True
            ),
            "coherence_uses_source": (
                False
            ),
            "coherence_uses_reference": (
                False
            ),
        },

        "candidate_sentence_map": (
            sentence_map
        ),

        "completeness_method": {
            "criteria_generated_once_per_report": (
                True
            ),
            "criteria_verified_against_source": (
                True
            ),
            "criteria_cross_unit_redundancy_checked": (
                True
            ),
            "draft_file": str(
                key_draft_path(
                    source_path
                )
            ),
            "verified_file": str(
                key_verified_path(
                    source_path
                )
            ),
            "key_information_hash": (
                verified_result[
                    "key_information_hash"
                ]
            ),
            "key_information_units_total": len(
                final_units
            ),
        },

        "scores": {
            "faithfulness": (
                factual
            ),
            "numerical_accuracy": (
                numerical
            ),
            "completeness": {
                "score": (
                    completeness_score
                ),
                "scale_max": 4,
                "label": completeness_eval.get(
                    "label"
                ),
                "present": (
                    present
                ),
                "partially_present": (
                    partial
                ),
                "missing": (
                    missing
                ),
                "total_units": len(
                    final_units
                ),
            },
            "coherence": {
                "score": (
                    coherence_score
                ),
                "scale_max": 4,
                "label": coherence_eval.get(
                    "label"
                ),
            },
        },

        "details": {
            "factual_claims": (
                enrich_claims_with_sentences(
                    factual_numerical.get(
                        "factual_claims",
                        [],
                    ),
                    sentence_map,
                )
            ),
            "numerical_claims": (
                enrich_claims_with_sentences(
                    factual_numerical.get(
                        "numerical_claims",
                        [],
                    ),
                    sentence_map,
                )
            ),
            "verified_key_information_units": (
                completeness_combined
            ),
            "completeness_missing_or_partial": (
                completeness_eval.get(
                    "missing_or_partial_key_information",
                    [],
                )
            ),
            "completeness_justification": (
                completeness_eval.get(
                    "justification"
                )
            ),
            "coherence_issues": (
                enrich_coherence_issues(
                    coherence_eval.get(
                        "issues",
                        [],
                    ),
                    sentence_map,
                )
            ),
            "coherence_justification": (
                coherence_eval.get(
                    "justification"
                )
            ),
        },

        "usage": (
            sum_usage(
                raw_results
            )
        ),

        "shared_setup_usage": (
            sum_shared_usage(
                draft_result,
                verified_result,
            )
        ),

        "files": {
            "source_report": str(
                source_path
            ),
            "reference_summary": str(
                reference_path
            ),
            "key_information_draft": str(
                key_draft_path(
                    source_path
                )
            ),
            "key_information_verified": str(
                key_verified_path(
                    source_path
                )
            ),
            "raw_results": {
                evaluation_type: str(
                    raw_result_path(
                        evaluation_type,
                        candidate_path.stem,
                    )
                )
                for evaluation_type
                in required
            },
        },
    }

    output_path = (
        merged_result_path(
            candidate_path.stem
        )
    )

    save_json(
        output_path,
        merged,
    )

    print(
        f"Merged result saved: "
        f"{output_path}"
    )

    return merged


CSV_FIELDS = [
    "report",
    "report_type",
    "generator_model",
    "generator_key",
    "prompt_type",
    "generation_mode",
    "judge_model",
    "judge_reasoning_effort",

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
    "completeness_label",
    "key_information_total",
    "completeness_present",
    "completeness_partially_present",
    "completeness_missing",

    "coherence",
    "coherence_label",

    "judge_input_tokens_total",
    "judge_output_tokens_total_including_reasoning",
    "judge_total_tokens",
    "judge_latency_seconds_total",

    "merged_file",
]


def merged_to_csv_row(
    merged: dict,
    merged_path: Path,
) -> dict:

    faith = (
        merged[
            "scores"
        ][
            "faithfulness"
        ]
    )

    numerical = (
        merged[
            "scores"
        ][
            "numerical_accuracy"
        ]
    )

    completeness = (
        merged[
            "scores"
        ][
            "completeness"
        ]
    )

    coherence = (
        merged[
            "scores"
        ][
            "coherence"
        ]
    )

    usage = merged.get(
        "usage",
        {},
    )

    return {
        "report": (
            merged[
                "candidate"
            ][
                "report"
            ]
        ),
        "report_type": (
            merged[
                "candidate"
            ][
                "report_type"
            ]
        ),
        "generator_model": (
            merged[
                "candidate"
            ][
                "generator_model"
            ]
        ),
        "generator_key": (
            merged[
                "candidate"
            ][
                "generator_key"
            ]
        ),
        "prompt_type": (
            merged[
                "candidate"
            ][
                "prompt_type"
            ]
        ),
        "generation_mode": (
            merged[
                "candidate"
            ][
                "generation_mode"
            ]
        ),
        "judge_model": (
            merged[
                "judge"
            ][
                "model"
            ]
        ),
        "judge_reasoning_effort": (
            merged[
                "judge"
            ][
                "reasoning_effort"
            ]
        ),

        "faithfulness": (
            faith[
                "score"
            ]
        ),
        "factual_supported": (
            faith[
                "supported"
            ]
        ),
        "factual_contradicted": (
            faith[
                "contradicted"
            ]
        ),
        "factual_not_found": (
            faith[
                "not_found"
            ]
        ),
        "factual_total": (
            faith[
                "total"
            ]
        ),

        "numerical_accuracy": (
            numerical[
                "score"
            ]
        ),
        "numerical_correct": (
            numerical[
                "correct"
            ]
        ),
        "numerical_incorrect": (
            numerical[
                "incorrect"
            ]
        ),
        "numerical_total": (
            numerical[
                "total"
            ]
        ),

        "completeness": (
            completeness[
                "score"
            ]
        ),
        "completeness_label": (
            completeness[
                "label"
            ]
        ),
        "key_information_total": (
            completeness[
                "total_units"
            ]
        ),
        "completeness_present": (
            completeness[
                "present"
            ]
        ),
        "completeness_partially_present": (
            completeness[
                "partially_present"
            ]
        ),
        "completeness_missing": (
            completeness[
                "missing"
            ]
        ),

        "coherence": (
            coherence[
                "score"
            ]
        ),
        "coherence_label": (
            coherence[
                "label"
            ]
        ),

        "judge_input_tokens_total": (
            usage.get(
                "input_tokens_total"
            )
        ),
        "judge_output_tokens_total_including_reasoning": (
            usage.get(
                "output_tokens_total_including_reasoning"
            )
        ),
        "judge_total_tokens": (
            usage.get(
                "total_tokens"
            )
        ),
        "judge_latency_seconds_total": (
            usage.get(
                "latency_seconds_total"
            )
        ),

        "merged_file": str(
            merged_path
        ),
    }


def write_csv(
    merged_results: list[dict],
) -> None:

    rows = []

    for merged in merged_results:
        if merged is None:
            continue

        path = merged_result_path(
            Path(
                merged[
                    "candidate"
                ][
                    "file"
                ]
            ).stem
        )

        rows.append(
            merged_to_csv_row(
                merged,
                path,
            )
        )

    SCORES_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SCORES_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                CSV_FIELDS
            ),
        )
        writer.writeheader()
        writer.writerows(
            rows
        )

    print()
    print(
        f"CSV saved: {SCORES_CSV}"
    )



def main():

    print("=" * 100)
    print(
        "NOVA 2 LITE — LLM-AS-A-JUDGE TEST V4"
    )
    print(
        "SENTENCE IDS + NON-OVERLAPPING COMPLETENESS + "
        "ACCOUNTING-SAFE COHERENCE"
    )
    print("=" * 100)

    print(
        f"Region:                 "
        f"{AWS_REGION}"
    )
    print(
        f"Judge model:            "
        f"{JUDGE_MODEL_ID}"
    )
    print(
        f"Judge reasoning:        "
        f"{JUDGE_REASONING_EFFORT}"
    )
    print(
        f"Candidate mode:         "
        f"{RUN_MODE}"
    )
    print(
        f"Category:               "
        f"{CATEGORY}"
    )
    print(
        f"Report:                 "
        f"{REPORT_NUMBER}"
    )
    print(
        f"Summary prompt:         "
        f"{SUMMARY_PROMPT}"
    )
    print(
        f"Generator models:       "
        f"{', '.join(GENERATOR_MODELS.keys())}"
    )
    print(
        "Shared setup requests:  2 "
        "(extract + verify)"
    )
    print(
        f"Candidate requests:     "
        f"{len(GENERATOR_MODELS) * 3}"
    )
    print(
        f"Maximum new requests:   "
        f"{2 + len(GENERATOR_MODELS) * 3}"
    )
    print(
        f"Prompts directory:      "
        f"{PROMPTS_DIR}"
    )
    print(
        f"Shared results root:    "
        f"{SHARED_RESULTS_ROOT}"
    )
    print(
        f"Candidate results root: "
        f"{JUDGE_RESULTS_ROOT}"
    )
    print("=" * 100)

    prompts = load_judge_prompts()

    source_path = find_report_file()
    reference_path = find_reference_file()

    source_report = read_text(
        source_path
    )
    reference_summary = read_text(
        reference_path
    )

    print()
    print(
        f"Source report:     "
        f"{source_path}"
    )
    print(
        f"Reference summary: "
        f"{reference_path}"
    )


    draft_result = (
        get_or_create_draft_key_information(
            instructions=(
                prompts[
                    "key_information_extract"
                ]
            ),
            source_report=(
                source_report
            ),
            reference_summary=(
                reference_summary
            ),
            source_path=(
                source_path
            ),
            reference_path=(
                reference_path
            ),
        )
    )

    if draft_result.get(
        "status"
    ) != "success":
        print()
        print(
            "STOP: draft key-information "
            "extraction failed."
        )
        return


    verified_result = (
        get_or_create_verified_key_information(
            instructions=(
                prompts[
                    "key_information_verify"
                ]
            ),
            source_report=(
                source_report
            ),
            source_path=(
                source_path
            ),
            draft_result=(
                draft_result
            ),
        )
    )

    if verified_result.get(
        "status"
    ) != "success":
        print()
        print(
            "STOP: verification/non-overlap "
            "finalization failed."
        )
        return

    final_units = (
        verified_result[
            "evaluation"
        ][
            "key_information_units"
        ]
    )

    key_information_hash_value = (
        verified_result[
            "key_information_hash"
        ]
    )

    verified_file = (
        key_verified_path(
            source_path
        )
    )

    print()
    print("=" * 100)
    print(
        "VERIFIED NON-OVERLAPPING "
        "KEY INFORMATION UNITS"
    )
    print("=" * 100)

    for unit in final_units:
        print(
            f"{unit.get('id')} "
            f"[{unit.get('importance')}]: "
            f"{unit.get('information')}"
        )

    print(
        f"\nVerified criteria hash: "
        f"{key_information_hash_value}"
    )

    redundancy_check = (
        verified_result[
            "evaluation"
        ].get(
            "redundancy_check",
            {},
        )
    )

    print(
        "Redundancy check:       "
        f"{redundancy_check.get('passed')}"
    )

    # --------------------------------------------------------
    # Candidate evaluation
    # --------------------------------------------------------

    evaluation_order = [
        "factual_numerical",
        "completeness",
        "coherence",
    ]

    merged_results = []

    for model_index, (
        generator_key,
        generator_model_name,
    ) in enumerate(
        GENERATOR_MODELS.items(),
        start=1,
    ):

        candidate_path = (
            get_candidate_file(
                generator_key
            )
        )

        candidate_summary = read_text(
            candidate_path
        )

        sentence_map = (
            build_candidate_sentence_map(
                candidate_summary
            )
        )

        print()
        print()
        print("=" * 100)
        print(
            f"[{model_index}/"
            f"{len(GENERATOR_MODELS)}] "
            f"Candidate model: "
            f"{generator_model_name}"
        )
        print(
            f"Candidate file:  "
            f"{candidate_path}"
        )
        print(
            f"Trace sentences: "
            f"{len(sentence_map)}"
        )
        print("=" * 100)

        raw_results = {}

        for (
            evaluation_index,
            evaluation_type,
        ) in enumerate(
            evaluation_order,
            start=1,
        ):

            if (
                evaluation_type
                == "factual_numerical"
            ):
                data_note = (
                    "SOURCE + CANDIDATE IDs"
                )

            elif (
                evaluation_type
                == "completeness"
            ):
                data_note = (
                    "VERIFIED FIXED CRITERIA + "
                    "CANDIDATE IDs "
                    "(NO SOURCE/REFERENCE)"
                )

            else:
                data_note = (
                    "CANDIDATE IDs ONLY "
                    "(NO SOURCE/REFERENCE)"
                )

            print()
            print(
                f"Request "
                f"{evaluation_index}/3 "
                f"for {generator_key}"
            )
            print(
                f"Input design: "
                f"{data_note}"
            )

            raw = (
                run_candidate_evaluation(
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
                        generator_model_name
                    ),
                    final_units=(
                        final_units
                        if evaluation_type
                        == "completeness"
                        else None
                    ),
                    key_information_hash_value=(
                        key_information_hash_value
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
                REQUEST_DELAY_SECONDS > 0
                and evaluation_index < 3
            ):
                time.sleep(
                    REQUEST_DELAY_SECONDS
                )

        try:
            merged = (
                merge_candidate_results(
                    generator_key=(
                        generator_key
                    ),
                    generator_model_name=(
                        generator_model_name
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

        except Exception as exc:
            print(
                "MERGE FAILED | "
                f"{type(exc).__name__}: "
                f"{exc}"
            )
            merged = None

        merged_results.append(
            merged
        )

        if (
            REQUEST_DELAY_SECONDS > 0
            and model_index
            < len(
                GENERATOR_MODELS
            )
        ):
            time.sleep(
                REQUEST_DELAY_SECONDS
            )

    write_csv(
        merged_results
    )

    print()
    print("=" * 100)
    print("TEST COMPLETE")
    print("=" * 100)

    for merged in merged_results:
        if merged is None:
            continue

        candidate = (
            merged[
                "candidate"
            ]
        )

        scores = (
            merged[
                "scores"
            ]
        )

        print()
        print(
            candidate[
                "generator_model"
            ]
        )

        print(
            "  Faithfulness:       "
            f"{scores['faithfulness']['score']} "
            f"({scores['faithfulness']['supported']}/"
            f"{scores['faithfulness']['total']} supported)"
        )

        print(
            "  Numerical accuracy: "
            f"{scores['numerical_accuracy']['score']} "
            f"({scores['numerical_accuracy']['correct']}/"
            f"{scores['numerical_accuracy']['total']} correct)"
        )

        print(
            "  Completeness:        "
            f"{scores['completeness']['score']}/4 "
            f"({scores['completeness']['label']}) "
            f"| present={scores['completeness']['present']}, "
            f"partial={scores['completeness']['partially_present']}, "
            f"missing={scores['completeness']['missing']}"
        )

        print(
            "  Coherence:           "
            f"{scores['coherence']['score']}/4 "
            f"({scores['coherence']['label']})"
        )

        print(
            "  Candidate Judge input tokens: "
            f"{merged['usage']['input_tokens_total']:,}"
        )

    print()
    print(
        f"Shared draft:     "
        f"{key_draft_path(source_path)}"
    )
    print(
        f"Shared verified:  "
        f"{key_verified_path(source_path)}"
    )
    print(
        f"Detailed results: "
        f"{JUDGE_RESULTS_ROOT}"
    )
    print(
        f"CSV:              "
        f"{SCORES_CSV}"
    )


if __name__ == "__main__":
    print(
        "This is the production core module. "
        "Run run_nova_judge_flex_production.py instead."
    )



def call_nova_once(
    full_prompt: str,
    reasoning_mode: str = "high",
    max_tokens: int | None = None,
) -> dict:
    """
    General Nova 2 Lite Converse call for production.

    HIGH:
      - reasoningConfig=high
      - inferenceConfig intentionally omitted

    LOW:
      - reasoningConfig=low
      - temperature=0
      - maxTokens set

    OFF:
      - reasoningConfig omitted
      - temperature=0
      - maxTokens set

    All calls use Flex service tier.
    """
    kwargs = {
        "modelId": JUDGE_MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": full_prompt
                    }
                ],
            }
        ],
        "serviceTier": {
            "type": SERVICE_TIER,
        },
    }

    if reasoning_mode == "high":
        kwargs[
            "additionalModelRequestFields"
        ] = {
            "reasoningConfig": {
                "type": "enabled",
                "maxReasoningEffort": "high",
            }
        }

    elif reasoning_mode == "low":
        if max_tokens is None:
            raise ValueError(
                "LOW reasoning requires max_tokens."
            )

        kwargs["inferenceConfig"] = {
            "maxTokens": max_tokens,
            "temperature": CANDIDATE_TEMPERATURE,
        }

        kwargs[
            "additionalModelRequestFields"
        ] = {
            "reasoningConfig": {
                "type": "enabled",
                "maxReasoningEffort": "low",
            }
        }

    elif reasoning_mode == "off":
        if max_tokens is None:
            raise ValueError(
                "OFF reasoning requires max_tokens."
            )

        kwargs["inferenceConfig"] = {
            "maxTokens": max_tokens,
            "temperature": CANDIDATE_TEMPERATURE,
        }

    else:
        raise ValueError(
            f"Unknown reasoning mode: {reasoning_mode}"
        )

    start = time.perf_counter()

    response = bedrock.converse(
        **kwargs
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    visible_text, reasoning_texts = (
        extract_response_content(
            response
        )
    )

    usage = response.get(
        "usage",
        {}
    )

    return {
        "visible_text": visible_text,
        "reasoning_texts": reasoning_texts,
        "usage": {
            "input_tokens": (
                usage.get("inputTokens")
                or 0
            ),
            "output_tokens_including_reasoning": (
                usage.get("outputTokens")
                or 0
            ),
            "total_tokens": (
                usage.get("totalTokens")
                or 0
            ),
        },
        "stop_reason": response.get(
            "stopReason"
        ),
        "latency_seconds": round(
            elapsed,
            2,
        ),
        "resolved_service_tier": (
            response.get("serviceTier")
        ),
    }


def call_nova(
    full_prompt: str,
    reasoning_mode: str = "high",
    max_tokens: int | None = None,
) -> dict:
    """
    Logical request with retries only for transient AWS/network errors.
    These retries are separate from the one optional targeted numerical
    coverage repair.
    """
    last_exception = None

    for attempt in range(
        1,
        MAX_API_ATTEMPTS + 1,
    ):
        if attempt > 1:
            print(
                f"Transient retry {attempt}/"
                f"{MAX_API_ATTEMPTS}..."
            )

        try:
            return call_nova_once(
                full_prompt=full_prompt,
                reasoning_mode=reasoning_mode,
                max_tokens=max_tokens,
            )

        except ClientError as exc:
            last_exception = exc

            error = exc.response.get(
                "Error",
                {},
            )

            code = error.get(
                "Code",
                "",
            )
            message = error.get(
                "Message",
                "",
            )

            print(
                f"API ERROR | {code}: {message}"
            )

            if (
                code not in TRANSIENT_ERROR_CODES
                or attempt >= MAX_API_ATTEMPTS
            ):
                raise

        except (
            EndpointConnectionError,
            ReadTimeoutError,
            ConnectTimeoutError,
            ConnectionClosedError,
        ) as exc:
            last_exception = exc

            print(
                "NETWORK ERROR | "
                f"{type(exc).__name__}: {exc}"
            )

            if attempt >= MAX_API_ATTEMPTS:
                raise

        delay_index = min(
            attempt - 1,
            len(RETRY_DELAYS_SECONDS) - 1,
        )

        delay = (
            RETRY_DELAYS_SECONDS[
                delay_index
            ]
        )

        print(
            f"Transient failure. "
            f"Waiting {delay} s..."
        )
        time.sleep(delay)

    if last_exception:
        raise last_exception

    raise RuntimeError(
        "Nova request failed unexpectedly."
    )



def repair_json_format_with_nova_v523(
    malformed_text: str,
    context_label: str,
) -> dict:
    """
    Cheap syntax-only repair.

    The repair model receives ONLY the already-produced malformed output,
    never the source report or candidate. It is explicitly prohibited
    from adding/removing/re-evaluating claims.

    This is therefore a serialization recovery step, not another Judge
    evaluation.
    """
    repair_prompt = (
        "You are a strict JSON syntax formatter.\n\n"
        "The text below is the output of another model. It was intended "
        "to be valid JSON but Python could not parse it.\n\n"
        "TASK:\n"
        "- Return the SAME JSON content as valid JSON.\n"
        "- Preserve every key, value, ID, label, list item, number, claim, "
        "evidence string and null value.\n"
        "- Do NOT add, remove, merge, split, summarize, reinterpret, "
        "re-evaluate, or correct any semantic content.\n"
        "- Only repair serialization/syntax problems such as code fences, "
        "literal control characters, quote escaping, commas, colons, and "
        "closing brackets/braces.\n"
        "- If valid JSON cannot be produced without inventing missing "
        "semantic content, return exactly: "
        '{"json_repair_failed": true}\n\n'
        f"CONTEXT LABEL: {context_label}\n\n"
        "MALFORMED MODEL OUTPUT:\n"
        "<malformed_output>\n"
        f"{malformed_text}\n"
        "</malformed_output>\n\n"
        "Return valid JSON only."
    )

    return call_nova(
        full_prompt=repair_prompt,
        reasoning_mode="off",
        max_tokens=12000,
    )


def parse_json_with_format_recovery_v523(
    raw_text: str,
    context_label: str,
) -> tuple[dict, dict | None, dict]:
    """
    Parse order:
      1) normal JSON parse;
      2) deterministic local control-character repair;
      3) one cheap Nova OFF format-only repair, if still needed.

    Returns:
      parsed_json,
      format_repair_api_or_none,
      diagnostic
    """
    try:
        parsed = parse_json_response(
            raw_text
        )

        return (
            parsed,
            None,
            {
                "performed": False,
                "context": context_label,
                "original_parse_error": None,
            },
        )

    except json.JSONDecodeError as exc:
        original_error = (
            f"{type(exc).__name__}: {exc}"
        )

    format_api = (
        repair_json_format_with_nova_v523(
            malformed_text=raw_text,
            context_label=context_label,
        )
    )

    repaired_text = (
        format_api[
            "visible_text"
        ]
    )

    parsed = parse_json_response(
        repaired_text
    )

    if (
        isinstance(parsed, dict)
        and parsed.get(
            "json_repair_failed"
        ) is True
    ):
        raise ValueError(
            "Format-only JSON repair reported "
            "json_repair_failed=true."
        )

    return (
        parsed,
        format_api,
        {
            "performed": True,
            "context": context_label,
            "original_parse_error": (
                original_error
            ),
            "repaired_model_output_text": (
                repaired_text
            ),
        },
    )


# -------------------------------------------------------------------
# Conservative numerical sentence coverage guard
# -------------------------------------------------------------------

CURRENCY_NUMBER_RE_V52 = re.compile(
    r"\b(?:SEK|EUR|USD|GBP|NOK|DKK|CHF|JPY|CNY|RMB)"
    r"\s*[-+]?\d",
    flags=re.IGNORECASE,
)

PERCENT_RE_V52 = re.compile(
    r"(?:[-+]?\d+(?:[.,]\d+)?)\s*"
    r"(?:%|percent\b|percentage\s+points?\b)",
    flags=re.IGNORECASE,
)

SCALED_NUMBER_RE_V52 = re.compile(
    r"(?:[-+]?\d+(?:[.,]\d+)?)\s*"
    r"(?:million|billion|thousand|m\b|bn\b)",
    flags=re.IGNORECASE,
)

COUNT_QUANTITY_RE_V52 = re.compile(
    r"(?<![\w-])[-+]?\d+(?:[.,]\d+)?\s+"
    r"(?:(?:[A-Za-z][A-Za-z-]*\s+){0,2})?"
    r"(?:shares?|employees?|customers?|satellites?|radiometers?|"
    r"units?|stores?|locations?|orders?|contracts?|projects?|"
    r"facilities?|subsidiaries?|countries?|vehicles?|vessels?|"
    r"aircraft|sites?)\b",
    flags=re.IGNORECASE,
)

IDENTIFIER_PREFIX_RE_V52 = re.compile(
    r"(?:phase|stage|version|section|part|note|page)\s*$",
    flags=re.IGNORECASE,
)


def has_explicit_count_quantity_v52(
    text: str,
) -> bool:
    for match in (
        COUNT_QUANTITY_RE_V52.finditer(
            text
        )
    ):
        prefix = text[
            :match.start()
        ]

        if (
            IDENTIFIER_PREFIX_RE_V52.search(
                prefix
            )
        ):
            continue

        # Prevent false positives such as:
        #   "in 2025 as customer orders were delayed"
        # where the permissive count regex can read
        # "2025 as customer orders" as a quantity.
        matched_text = match.group(
            0
        )

        number_match = re.match(
            r"[-+]?\d+(?:[.,]\d+)?",
            matched_text,
        )

        if number_match:
            raw_number = (
                number_match.group(
                    0
                )
            )
            if (
                raw_number.isdigit()
                and 1900
                <= int(
                    raw_number
                )
                <= 2100
            ):
                continue

        return True

    return False


def strong_numerical_sentence_ids_v52(
    sentence_map: dict[str, str],
) -> set[str]:
    """
    Safety net only. It does NOT define how many N-claims Nova must
    produce. It only identifies sentences with strong quantitative
    signals that should normally be represented by at least one
    numerical claim.
    """
    result = set()

    for (
        sentence_id,
        sentence,
    ) in sentence_map.items():

        if (
            CURRENCY_NUMBER_RE_V52.search(
                sentence
            )
            or PERCENT_RE_V52.search(
                sentence
            )
            or SCALED_NUMBER_RE_V52.search(
                sentence
            )
            or has_explicit_count_quantity_v52(
                sentence
            )
        ):
            result.add(
                sentence_id
            )

    return result


def numerical_claim_sentence_ids_v52(
    evaluation: dict,
) -> set[str]:
    result = set()

    for claim in evaluation.get(
        "numerical_claims",
        [],
    ):
        ids = claim.get(
            "candidate_sentence_ids",
            [],
        )

        if not isinstance(
            ids,
            list,
        ):
            continue

        for sentence_id in ids:
            if isinstance(
                sentence_id,
                str,
            ):
                result.add(
                    sentence_id
                )

    return result


def numerical_coverage_check_v52(
    evaluation: dict,
    sentence_map: dict[str, str],
    accepted_non_numerical_sentence_ids=None,
) -> dict:

    expected = (
        strong_numerical_sentence_ids_v52(
            sentence_map
        )
    )

    covered = (
        numerical_claim_sentence_ids_v52(
            evaluation
        )
    )

    accepted = set(
        accepted_non_numerical_sentence_ids
        or []
    )

    missing = sorted(
        expected
        - covered
        - accepted,
        key=lambda value: (
            int(value[1:])
            if value[1:].isdigit()
            else 999999
        ),
    )

    return {
        "ok": not missing,
        "expected_sentence_ids": sorted(
            expected,
            key=lambda value: (
                int(value[1:])
                if value[1:].isdigit()
                else 999999
            ),
        ),
        "covered_sentence_ids": sorted(
            covered,
            key=lambda value: (
                int(value[1:])
                if value[1:].isdigit()
                else 999999
            ),
        ),
        "accepted_non_numerical_sentence_ids": (
            sorted(
                accepted,
                key=lambda value: (
                    int(value[1:])
                    if value[1:].isdigit()
                    else 999999
                ),
            )
        ),
        "missing_sentence_ids": missing,
    }


def build_targeted_numerical_repair_prompt_v52(
    source_report: str,
    missing_sentence_ids: list[str],
    sentence_map: dict[str, str],
) -> str:

    missing_text = "\n".join(
        f"[{sentence_id}] "
        f"{sentence_map[sentence_id]}"
        for sentence_id
        in missing_sentence_ids
        if sentence_id in sentence_map
    )

    return (
        "You are repairing ONLY missing numerical coverage in a "
        "financial-summary evaluation.\n\n"

        "Use SOURCE as the only evidence.\n"

        "For EVERY listed candidate sentence, first decide whether "
        "it actually contains an independently verifiable numerical "
        "proposition.\n\n"

        "Classifications:\n"

        "HAS_NUMERICAL_PROPOSITION — one or more quantitative "
        "propositions should be numerically verified.\n"

        "NO_NUMERICAL_PROPOSITION — the detected number is only an "
        "identifier, ordinal, label, product name, phase/stage number, "
        "section/note/page number, or otherwise not a quantitative "
        "proposition for numerical-accuracy scoring.\n\n"

        "Examples of NO_NUMERICAL_PROPOSITION:\n"
        "- Phase 1 contract\n"
        "- Stage 2 programme\n"
        "- Sedna-1 / Sedna-2 identifiers\n"
        "- Section 3 / Note 5 / Page 6\n\n"

        "For HAS_NUMERICAL_PROPOSITION return one or more numerical "
        "claims. Do not evaluate candidate sentences that are not "
        "listed below.\n\n"

        "Numerical labels:\n"
        "CORRECT\n"
        "INCORRECT\n\n"

        "For INCORRECT choose exactly one primary error_type:\n"
        "INCORRECT_VALUE\n"
        "INCORRECT_UNIT_OR_CURRENCY\n"
        "INCORRECT_PERIOD\n"
        "INCORRECT_DIRECTION_OR_COMPARISON\n"
        "UNSUPPORTED_NUMBER\n"
        "OTHER\n\n"

        "Return only valid JSON:\n"
        "{\n"
        '  "sentence_checks": [\n'
        "    {\n"
        '      "candidate_sentence_id": "S7",\n'
        '      "classification": "HAS_NUMERICAL_PROPOSITION",\n'
        '      "reason": "...",\n'
        '      "numerical_claims": [\n'
        "        {\n"
        '          "claim": "...",\n'
        '          "candidate_value": "...",\n'
        '          "label": "CORRECT",\n'
        '          "error_type": null,\n'
        '          "source_value": "...",\n'
        '          "source_evidence": "...",\n'
        '          "error_explanation": null\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"

        "SOURCE FINANCIAL REPORT:\n"
        "<source_report>\n"
        f"{source_report}\n"
        "</source_report>\n\n"

        "ONLY THESE CANDIDATE SENTENCES REQUIRE A COVERAGE CHECK:\n"
        "<candidate_sentences>\n"
        f"{missing_text}\n"
        "</candidate_sentences>"
    )


def apply_targeted_numerical_repair_v52(
    evaluation: dict,
    repair_evaluation: dict,
    missing_sentence_ids: list[str],
):
    """
    Keep the FIRST full evaluation. Only supplemental numerical claims
    are merged. Factual claims are never regenerated/replaced.
    """
    allowed = set(
        missing_sentence_ids
    )

    accepted_non_numerical = []
    resolved_with_claim = set()

    existing_claims = list(
        evaluation.get(
            "numerical_claims",
            [],
        )
    )

    checks = repair_evaluation.get(
        "sentence_checks",
        [],
    )

    if not isinstance(
        checks,
        list,
    ):
        checks = []

    for check in checks:
        if not isinstance(
            check,
            dict,
        ):
            continue

        sentence_id = str(
            check.get(
                "candidate_sentence_id",
                "",
            )
        ).strip()

        if sentence_id not in allowed:
            continue

        classification = normalize_label(
            check.get(
                "classification"
            )
        )

        if (
            classification
            == "NO_NUMERICAL_PROPOSITION"
        ):
            accepted_non_numerical.append(
                sentence_id
            )
            continue

        if (
            classification
            != "HAS_NUMERICAL_PROPOSITION"
        ):
            continue

        claims = check.get(
            "numerical_claims",
            [],
        )

        if (
            not isinstance(
                claims,
                list,
            )
            or not claims
        ):
            continue

        added_any = False

        for claim in claims:
            if not isinstance(
                claim,
                dict,
            ):
                continue

            item = dict(
                claim
            )

            claim_text = str(
                item.get(
                    "claim",
                    "",
                )
            ).strip()

            label = normalize_label(
                item.get(
                    "label"
                )
            )

            error_type = item.get(
                "error_type"
            )

            if not claim_text:
                continue

            if label not in {
                "CORRECT",
                "INCORRECT",
            }:
                continue

            if (
                label == "CORRECT"
                and error_type not in {
                    None,
                    "",
                }
            ):
                continue

            if (
                label == "INCORRECT"
                and normalize_label(
                    error_type
                ) not in {
                    "INCORRECT_VALUE",
                    "INCORRECT_UNIT_OR_CURRENCY",
                    "INCORRECT_PERIOD",
                    "INCORRECT_DIRECTION_OR_COMPARISON",
                    "UNSUPPORTED_NUMBER",
                    "OTHER",
                }
            ):
                continue

            item[
                "candidate_sentence_ids"
            ] = [
                sentence_id
            ]

            existing_claims.append(
                item
            )

            added_any = True

        if added_any:
            resolved_with_claim.add(
                sentence_id
            )

    for index, claim in enumerate(
        existing_claims,
        start=1,
    ):
        claim["id"] = f"N{index}"

    merged = dict(
        evaluation
    )

    merged[
        "numerical_claims"
    ] = existing_claims

    unresolved = sorted(
        allowed
        - set(
            accepted_non_numerical
        )
        - resolved_with_claim,
        key=lambda value: (
            int(value[1:])
            if value[1:].isdigit()
            else 999999
        ),
    )

    return (
        merged,
        sorted(
            set(
                accepted_non_numerical
            ),
            key=lambda value: (
                int(value[1:])
                if value[1:].isdigit()
                else 999999
            ),
        ),
        unresolved,
    )


def build_numerical_only_recovery_prompt_v524(
    source_report: str,
    sentence_map: dict[str, str],
) -> str:
    """
    Full candidate, NUMERICAL-ONLY recovery.

    Used only when the first combined Factual/Numerical evaluation
    missed many strongly numerical sentences. It rebuilds ONLY the
    numerical layer; the first factual_claims remain untouched.
    """
    candidate_text = format_candidate_with_ids(
        sentence_map
    )

    return (
        "You are evaluating ONLY numerical accuracy in a financial-report "
        "summary.\n\n"

        "You are given:\n"
        "1. SOURCE FINANCIAL REPORT — the only evidence source.\n"
        "2. CANDIDATE SUMMARY with immutable sentence IDs [S1], [S2], ...\n\n"

        "Do NOT evaluate non-numerical factual claims, completeness, "
        "coherence, or style.\n\n"

        "TASK:\n"
        "- Identify every independently verifiable numerical proposition "
        "in the candidate.\n"
        "- Ensure every candidate sentence containing an explicit "
        "financial value, percentage, ratio, quantity, numerical change, "
        "or quantitative comparison is represented by at least one "
        "numerical_claim.\n"
        "- One sentence may produce one or several numerical claims.\n"
        "- There is NO fixed required number of claims and no requirement "
        "to create one claim per number token.\n"
        "- Do not create numerical claims for identifiers such as Phase 1, "
        "Stage 2, Sedna-1, Sedna-2, section numbers, page numbers, or "
        "similar labels unless the number itself is a quantitative "
        "proposition.\n\n"

        "For each numerical proposition verify:\n"
        "- value;\n"
        "- sign/direction;\n"
        "- unit/currency;\n"
        "- indicator/object;\n"
        "- reporting period;\n"
        "- comparison basis.\n\n"

        "Labels:\n"
        "CORRECT — the complete numerical proposition is consistent with SOURCE.\n"
        "INCORRECT — at least one material component is inconsistent with "
        "or unsupported by SOURCE.\n\n"

        "For INCORRECT choose exactly one primary error_type:\n"
        "INCORRECT_VALUE\n"
        "INCORRECT_UNIT_OR_CURRENCY\n"
        "INCORRECT_PERIOD\n"
        "INCORRECT_DIRECTION_OR_COMPARISON\n"
        "UNSUPPORTED_NUMBER\n"
        "OTHER\n\n"

        "Return only valid JSON using exactly:\n"
        "{\n"
        '  "numerical_claims": [\n'
        "    {\n"
        '      "id": "N1",\n'
        '      "candidate_sentence_ids": ["S2"],\n'
        '      "claim": "...",\n'
        '      "candidate_value": "...",\n'
        '      "label": "CORRECT",\n'
        '      "error_type": null,\n'
        '      "source_value": "...",\n'
        '      "source_evidence": "...",\n'
        '      "error_explanation": null\n'
        "    }\n"
        "  ]\n"
        "}\n\n"

        "Use sequential IDs N1, N2, N3...\n\n"

        "SOURCE FINANCIAL REPORT:\n"
        "<source_report>\n"
        f"{source_report}\n"
        "</source_report>\n\n"

        "CANDIDATE SUMMARY WITH TRACE IDS:\n"
        "<candidate_summary>\n"
        f"{candidate_text}\n"
        "</candidate_summary>"
    )


def validate_numerical_only_recovery_v524(
    evaluation: dict,
    sentence_map: dict[str, str],
) -> list[str]:
    """
    Reuse the production Factual/Numerical validator while supplying an
    intentionally empty factual layer.
    """
    claims = evaluation.get(
        "numerical_claims"
    )

    if not isinstance(
        claims,
        list,
    ):
        return [
            "numerical_claims must be a list."
        ]

    wrapper = {
        "factual_claims": [],
        "numerical_claims": claims,
    }

    return validate_factual_numerical_output(
        wrapper,
        sentence_map,
    )


def merge_numerical_only_recovery_v524(
    original_evaluation: dict,
    numerical_only_evaluation: dict,
) -> dict:
    """
    Preserve the first full factual_claims exactly.
    Replace only numerical_claims with the dedicated numerical-only pass.
    """
    merged = dict(
        original_evaluation
    )

    claims = [
        dict(item)
        for item in numerical_only_evaluation.get(
            "numerical_claims",
            [],
        )
        if isinstance(
            item,
            dict,
        )
    ]

    for index, claim in enumerate(
        claims,
        start=1,
    ):
        claim["id"] = f"N{index}"

    merged[
        "numerical_claims"
    ] = claims

    return merged



def _combined_api_usage_v52(
    api_results: list[dict],
) -> dict:
    return {
        "input_tokens": sum(
            (
                result.get(
                    "usage",
                    {},
                ).get(
                    "input_tokens"
                )
                or 0
            )
            for result in api_results
        ),
        "output_tokens_including_reasoning": sum(
            (
                result.get(
                    "usage",
                    {},
                ).get(
                    "output_tokens_including_reasoning"
                )
                or 0
            )
            for result in api_results
        ),
        "total_tokens": sum(
            (
                result.get(
                    "usage",
                    {},
                ).get(
                    "total_tokens"
                )
                or 0
            )
            for result in api_results
        ),
        "latency_seconds": round(
            sum(
                result.get(
                    "latency_seconds",
                    0.0,
                )
                or 0.0
                for result in api_results
            ),
            2,
        ),
    }


def run_candidate_evaluation(
    evaluation_type: str,
    instructions: str,
    source_report: str,
    candidate_summary: str,
    sentence_map: dict[str, str],
    source_path: Path,
    candidate_path: Path,
    generator_model_name: str,
    final_units: list[dict] | None = None,
    key_information_hash_value: str | None = None,
    key_information_file: Path | None = None,
) -> dict:
    """
    V5.2 production candidate Judge call.

    Factual/Numerical:
      one full LOW call
      + at most one targeted LOW repair.

    Completeness:
      OFF.

    Coherence:
      OFF.
    """

    output_path = raw_result_path(
        evaluation_type,
        candidate_path.stem,
    )

    p_hash = text_hash(
        instructions
    )
    c_hash = text_hash(
        candidate_summary
    )
    sentence_map_hash_value = (
        json_hash(
            sentence_map
        )
    )

    s_hash = (
        text_hash(
            source_report
        )
        if evaluation_type
        == "factual_numerical"
        else None
    )

    if (
        output_path.exists()
        and not FORCE_RERUN_ALL
    ):
        existing = load_json(
            output_path
        )

        if can_reuse_candidate_result(
            existing=existing,
            prompt_hash_value=p_hash,
            source_hash_value=s_hash,
            candidate_hash_value=c_hash,
            sentence_map_hash_value=(
                sentence_map_hash_value
            ),
            key_information_hash_value=(
                key_information_hash_value
                if evaluation_type
                == "completeness"
                else None
            ),
        ):
            print(
                f"SKIP existing successful: "
                f"{evaluation_type}"
            )
            return existing

        if (
            evaluation_type == "factual_numerical"
            and existing.get("status") == "failed"
            and existing.get("error_type") == "JSONDecodeError"
            and isinstance(
                existing.get("model_output_text"),
                str,
            )
            and existing.get(
                "model_output_text"
            ).strip()
        ):
            meta = existing.get(
                "metadata",
                {},
            )

            same_inputs = (
                meta.get("prompt_version") == PROMPT_VERSION
                and meta.get("prompt_hash") == p_hash
                and meta.get("candidate_hash") == c_hash
                and meta.get("sentence_map_hash")
                == sentence_map_hash_value
                and meta.get("source_hash") == s_hash
            )

            if same_inputs:
                try:
                    recovered_evaluation = (
                        parse_json_response(
                            existing[
                                "model_output_text"
                            ]
                        )
                    )

                    recovered_evaluation = (
                        normalize_factual_numerical_schema_v527(
                            recovered_evaluation
                        )
                    )

                    recovery_errors = (
                        validate_factual_numerical_output(
                            recovered_evaluation,
                            sentence_map,
                        )
                    )

                    if not recovery_errors:
                        recovery_coverage = (
                            numerical_coverage_check_v52(
                                recovered_evaluation,
                                sentence_map,
                            )
                        )

                        existing[
                            "evaluation"
                        ] = recovered_evaluation
                        existing[
                            "validation_errors"
                        ] = []
                        existing[
                            "numerical_coverage"
                        ] = recovery_coverage
                        existing[
                            "parse_error"
                        ] = None
                        existing[
                            "local_json_recovery"
                        ] = {
                            "performed": True,
                            "implementation": (
                                "v5.2.10"
                            ),
                            "reason": (
                                "Recovered the already-paid malformed "
                                "Judge JSON deterministically without "
                                "a new Bedrock request."
                            ),
                            "original_error": existing.get(
                                "error_message"
                            ),
                        }

                        if recovery_coverage[
                            "ok"
                        ]:
                            existing[
                                "status"
                            ] = "success"

                            save_json(
                                output_path,
                                existing,
                            )

                            print(
                                "RECOVERED existing malformed "
                                "Factual/Numerical JSON locally — "
                                "no new API request."
                            )

                            return existing

                        # Keep the parsed result and hand any small residual
                        # gap to the normal bounded numerical recovery path
                        # below rather than rerunning the full Judge call.
                        existing[
                            "status"
                        ] = (
                            "numerical_coverage_failed"
                        )

                        save_json(
                            output_path,
                            existing,
                        )

                        print(
                            "RECOVERED malformed JSON locally; "
                            "numerical coverage is incomplete, "
                            "continuing with bounded coverage repair."
                        )

                except Exception as recovery_exc:
                    print(
                        "Local malformed-JSON recovery was not "
                        "sufficient; normal rerun path remains available | "
                        f"{type(recovery_exc).__name__}: "
                        f"{recovery_exc}"
                    )

        if (
            evaluation_type
            == "factual_numerical"
            and existing.get(
                "status"
            ) == "validation_failed"
            and isinstance(
                existing.get(
                    "evaluation"
                ),
                dict,
            )
        ):
            meta = existing.get(
                "metadata",
                {},
            )

            same_inputs = (
                meta.get(
                    "prompt_version"
                ) == PROMPT_VERSION
                and meta.get(
                    "prompt_hash"
                ) == p_hash
                and meta.get(
                    "candidate_hash"
                ) == c_hash
                and meta.get(
                    "sentence_map_hash"
                ) == sentence_map_hash_value
                and meta.get(
                    "source_hash"
                ) == s_hash
            )

            if same_inputs:
                resume_trace_api = None
                resume_trace_format_api = None
                resume_trace_evaluation = None
                resume_trace_format_info = {
                    "performed": False,
                    "context": None,
                    "original_parse_error": None,
                }
                resume_trace_unresolved = []

                recovered_evaluation = (
                    normalize_factual_numerical_schema_v527(
                        existing[
                            "evaluation"
                        ]
                    )
                )

                invalid_trace_claim_ids = (
                    factual_claim_ids_with_invalid_trace_v5211(
                        recovered_evaluation,
                        sentence_map,
                    )
                )

                if invalid_trace_claim_ids:
                    print(
                        "RESUME factual trace-ID repair only | "
                        f"triggered by: {invalid_trace_claim_ids}"
                    )

                    resume_trace_prompt = (
                        build_factual_trace_id_repair_prompt_v5211(
                            evaluation=recovered_evaluation,
                            sentence_map=sentence_map,
                        )
                    )

                    resume_trace_api = call_nova(
                        full_prompt=resume_trace_prompt,
                        reasoning_mode="off",
                        max_tokens=5000,
                    )

                    (
                        resume_trace_evaluation,
                        resume_trace_format_api,
                        resume_trace_format_info,
                    ) = (
                        parse_json_with_format_recovery_v523(
                            raw_text=(
                                resume_trace_api[
                                    "visible_text"
                                ]
                            ),
                            context_label=(
                                "resume factual trace-ID repair"
                            ),
                        )
                    )

                    (
                        recovered_evaluation,
                        resume_trace_unresolved,
                    ) = (
                        apply_factual_trace_id_repair_v5211(
                            evaluation=recovered_evaluation,
                            repair_evaluation=(
                                resume_trace_evaluation
                            ),
                            sentence_map=sentence_map,
                        )
                    )

                recovery_errors = (
                    validate_factual_numerical_output(
                        recovered_evaluation,
                        sentence_map,
                    )
                )

                ambiguous_ids = (
                    ambiguous_factual_claim_ids_v529(
                        recovered_evaluation
                    )
                )

                expected_ambiguous_errors = {
                    f"{claim_id}: invalid factual label 'INCORRECT'."
                    for claim_id in ambiguous_ids
                }

                if (
                    ambiguous_ids
                    and set(recovery_errors)
                    == expected_ambiguous_errors
                ):
                    print(
                        "RESUME factual label repair only | "
                        f"claims: {ambiguous_ids}"
                    )

                    factual_prompt = (
                        build_targeted_factual_label_repair_prompt_v529(
                            source_report=source_report,
                            evaluation=recovered_evaluation,
                            sentence_map=sentence_map,
                            claim_ids=ambiguous_ids,
                        )
                    )

                    factual_api = call_nova(
                        full_prompt=factual_prompt,
                        reasoning_mode="low",
                        max_tokens=4000,
                    )

                    (
                        factual_eval,
                        factual_format_api,
                        factual_format_info,
                    ) = (
                        parse_json_with_format_recovery_v523(
                            raw_text=factual_api[
                                "visible_text"
                            ],
                            context_label=(
                                "resume targeted factual label repair"
                            ),
                        )
                    )

                    (
                        recovered_evaluation,
                        factual_unresolved,
                    ) = (
                        apply_targeted_factual_label_repair_v529(
                            evaluation=recovered_evaluation,
                            repair_evaluation=factual_eval,
                            claim_ids=ambiguous_ids,
                        )
                    )

                    recovery_errors = (
                        validate_factual_numerical_output(
                            recovered_evaluation,
                            sentence_map,
                        )
                    )

                    if not recovery_errors:
                        recovery_coverage = (
                            numerical_coverage_check_v52(
                                recovered_evaluation,
                                sentence_map,
                            )
                        )

                        if recovery_coverage[
                            "ok"
                        ]:
                            old_usage = existing.get(
                                "usage",
                                {},
                            )

                            extra_apis = [
                                api
                                for api in [
                                    resume_trace_api,
                                    resume_trace_format_api,
                                    factual_api,
                                    factual_format_api,
                                ]
                                if api is not None
                            ]

                            extra_usage = (
                                _combined_api_usage_v52(
                                    extra_apis
                                )
                            )

                            existing[
                                "status"
                            ] = "success"
                            existing[
                                "evaluation"
                            ] = recovered_evaluation
                            existing[
                                "validation_errors"
                            ] = []
                            existing[
                                "numerical_coverage"
                            ] = recovery_coverage

                            existing[
                                "usage"
                            ] = {
                                "input_tokens": (
                                    old_usage.get(
                                        "input_tokens",
                                        0,
                                    )
                                    + extra_usage[
                                        "input_tokens"
                                    ]
                                ),
                                "output_tokens_including_reasoning": (
                                    old_usage.get(
                                        "output_tokens_including_reasoning",
                                        0,
                                    )
                                    + extra_usage[
                                        "output_tokens_including_reasoning"
                                    ]
                                ),
                                "total_tokens": (
                                    old_usage.get(
                                        "total_tokens",
                                        0,
                                    )
                                    + extra_usage[
                                        "total_tokens"
                                    ]
                                ),
                            }

                            existing[
                                "latency_seconds"
                            ] = round(
                                (
                                    existing.get(
                                        "latency_seconds",
                                        0.0,
                                    )
                                    or 0.0
                                )
                                + extra_usage[
                                    "latency_seconds"
                                ],
                                2,
                            )

                            existing[
                                "logical_requests"
                            ] = (
                                existing.get(
                                    "logical_requests",
                                    0,
                                )
                                + len(extra_apis)
                            )

                            existing[
                                "factual_label_repair"
                            ] = {
                                "performed": True,
                                "claim_ids": ambiguous_ids,
                                "unresolved_claim_ids": (
                                    factual_unresolved
                                ),
                                "evaluation": factual_eval,
                                "json_format_recovery": (
                                    factual_format_info
                                ),
                                "resume_only": True,
                            }

                            if resume_trace_api is not None:
                                existing[
                                    "trace_id_repair"
                                ] = {
                                    "performed": True,
                                    "unresolved_claim_ids": (
                                        resume_trace_unresolved
                                    ),
                                    "evaluation": (
                                        resume_trace_evaluation
                                    ),
                                    "json_format_recovery": (
                                        resume_trace_format_info
                                    ),
                                    "resume_only": True,
                                }

                            existing[
                                "local_recovery"
                            ] = {
                                "performed": True,
                                "reason": (
                                    "Repaired ambiguous factual "
                                    "INCORRECT taxonomy labels with one "
                                    "targeted source-grounded request; "
                                    "the prior full Judge evaluation was reused."
                                ),
                            }

                            save_json(
                                output_path,
                                existing,
                            )

                            print(
                                "RECOVERED existing factual-label "
                                "validation failure with one targeted "
                                "repair — no full Judge rerun."
                            )

                            return existing

                if not recovery_errors:
                    recovery_coverage = (
                        numerical_coverage_check_v52(
                            recovered_evaluation,
                            sentence_map,
                        )
                    )

                    if recovery_coverage[
                        "ok"
                    ]:
                        if resume_trace_api is not None:
                            old_usage = existing.get(
                                "usage",
                                {},
                            )

                            trace_apis = [
                                api
                                for api in [
                                    resume_trace_api,
                                    resume_trace_format_api,
                                ]
                                if api is not None
                            ]

                            trace_usage = (
                                _combined_api_usage_v52(
                                    trace_apis
                                )
                            )

                            existing[
                                "usage"
                            ] = {
                                "input_tokens": (
                                    old_usage.get(
                                        "input_tokens",
                                        0,
                                    )
                                    + trace_usage[
                                        "input_tokens"
                                    ]
                                ),
                                "output_tokens_including_reasoning": (
                                    old_usage.get(
                                        "output_tokens_including_reasoning",
                                        0,
                                    )
                                    + trace_usage[
                                        "output_tokens_including_reasoning"
                                    ]
                                ),
                                "total_tokens": (
                                    old_usage.get(
                                        "total_tokens",
                                        0,
                                    )
                                    + trace_usage[
                                        "total_tokens"
                                    ]
                                ),
                            }

                            existing[
                                "latency_seconds"
                            ] = round(
                                (
                                    existing.get(
                                        "latency_seconds",
                                        0.0,
                                    )
                                    or 0.0
                                )
                                + trace_usage[
                                    "latency_seconds"
                                ],
                                2,
                            )

                            existing[
                                "logical_requests"
                            ] = (
                                existing.get(
                                    "logical_requests",
                                    0,
                                )
                                + len(
                                    trace_apis
                                )
                            )

                            existing[
                                "trace_id_repair"
                            ] = {
                                "performed": True,
                                "unresolved_claim_ids": (
                                    resume_trace_unresolved
                                ),
                                "evaluation": (
                                    resume_trace_evaluation
                                ),
                                "json_format_recovery": (
                                    resume_trace_format_info
                                ),
                                "resume_only": True,
                            }

                        existing[
                            "status"
                        ] = "success"
                        existing[
                            "evaluation"
                        ] = (
                            recovered_evaluation
                        )
                        existing[
                            "validation_errors"
                        ] = []
                        existing[
                            "numerical_coverage"
                        ] = (
                            recovery_coverage
                        )
                        existing[
                            "local_recovery"
                        ] = {
                            "performed": True,
                            "reason": (
                                "Recovered deterministic F/N schema details "
                                "(claim IDs, taxonomy labels, and/or factual "
                                "candidate trace IDs). Only a cheap candidate-only "
                                "trace request is used when trace realignment is needed."
                            ),
                        }

                        save_json(
                            output_path,
                            existing,
                        )

                        print(
                            "RECOVERED existing Factual/Numerical "
                            "result locally — deterministic schema "
                            "normalization applied, no new API request."
                        )

                        return existing

        if (
            evaluation_type == "factual_numerical"
            and existing.get("status") == "numerical_coverage_failed"
            and isinstance(existing.get("evaluation"), dict)
        ):
            meta = existing.get("metadata", {})

            same_inputs = (
                meta.get("prompt_version") == PROMPT_VERSION
                and meta.get("prompt_hash") == p_hash
                and meta.get("candidate_hash") == c_hash
                and meta.get("sentence_map_hash") == sentence_map_hash_value
                and meta.get("source_hash") == s_hash
            )

            if same_inputs:
                recovered_evaluation = (
                    normalize_factual_numerical_schema_v527(
                        existing["evaluation"]
                    )
                )

                recovery_errors = (
                    validate_factual_numerical_output(
                        recovered_evaluation,
                        sentence_map,
                    )
                )

                if not recovery_errors:
                    recovery_coverage = (
                        numerical_coverage_check_v52(
                            recovered_evaluation,
                            sentence_map,
                        )
                    )

                    missing_ids = list(
                        recovery_coverage["missing_sentence_ids"]
                    )
                    if recovery_coverage[
                        "ok"
                    ]:
                        existing[
                            "status"
                        ] = "success"
                        existing[
                            "evaluation"
                        ] = recovered_evaluation
                        existing[
                            "validation_errors"
                        ] = []
                        existing[
                            "numerical_coverage"
                        ] = recovery_coverage
                        existing[
                            "local_recovery"
                        ] = {
                            "performed": True,
                            "reason": (
                                "Numerical coverage became complete after "
                                "deterministic guard correction; no new "
                                "Bedrock request was needed."
                            ),
                        }

                        save_json(
                            output_path,
                            existing,
                        )

                        print(
                            "RECOVERED existing numerical coverage "
                            "failure locally — corrected guard no longer "
                            "flags a false-positive sentence."
                        )

                        return existing

                    if (
                        not recovery_coverage["ok"]
                        and 1 <= len(missing_ids)
                        <= TARGETED_NUMERICAL_REPAIR_MAX_SENTENCES
                    ):
                        print(
                            "RESUME residual numerical repair only | "
                            f"missing: {missing_ids}"
                        )

                        residual_prompt = (
                            build_targeted_numerical_repair_prompt_v52(
                                source_report=source_report,
                                missing_sentence_ids=missing_ids,
                                sentence_map=sentence_map,
                            )
                        )

                        residual_api = call_nova(
                            full_prompt=residual_prompt,
                            reasoning_mode="low",
                            max_tokens=4000,
                        )

                        (
                            residual_evaluation,
                            residual_format_api,
                            residual_format_info,
                        ) = (
                            parse_json_with_format_recovery_v523(
                                raw_text=residual_api["visible_text"],
                                context_label=(
                                    "resume residual targeted numerical repair"
                                ),
                            )
                        )

                        (
                            recovered_evaluation,
                            accepted_non_numerical,
                            unresolved,
                        ) = (
                            apply_targeted_numerical_repair_v52(
                                evaluation=recovered_evaluation,
                                repair_evaluation=residual_evaluation,
                                missing_sentence_ids=missing_ids,
                            )
                        )

                        final_errors = (
                            validate_factual_numerical_output(
                                recovered_evaluation,
                                sentence_map,
                            )
                        )

                        final_coverage = (
                            numerical_coverage_check_v52(
                                recovered_evaluation,
                                sentence_map,
                                accepted_non_numerical,
                            )
                        )

                        if (
                            not final_errors
                            and final_coverage["ok"]
                        ):
                            old_usage = existing.get("usage", {})

                            extra_apis = [
                                api
                                for api in [
                                    residual_api,
                                    residual_format_api,
                                ]
                                if api is not None
                            ]

                            extra_usage = (
                                _combined_api_usage_v52(
                                    extra_apis
                                )
                            )

                            existing["status"] = "success"
                            existing["evaluation"] = recovered_evaluation
                            existing["validation_errors"] = []
                            existing["numerical_coverage"] = final_coverage
                            existing[
                                "numerical_recovery_strategy"
                            ] = (
                                "numerical_only_full_pass"
                                "_plus_targeted_repair"
                            )

                            existing["usage"] = {
                                "input_tokens": (
                                    old_usage.get("input_tokens", 0)
                                    + extra_usage["input_tokens"]
                                ),
                                "output_tokens_including_reasoning": (
                                    old_usage.get(
                                        "output_tokens_including_reasoning",
                                        0,
                                    )
                                    + extra_usage[
                                        "output_tokens_including_reasoning"
                                    ]
                                ),
                                "total_tokens": (
                                    old_usage.get("total_tokens", 0)
                                    + extra_usage["total_tokens"]
                                ),
                            }

                            existing["latency_seconds"] = round(
                                (
                                    existing.get("latency_seconds", 0.0)
                                    or 0.0
                                )
                                + extra_usage["latency_seconds"],
                                2,
                            )

                            existing["logical_requests"] = (
                                existing.get("logical_requests", 0)
                                + len(extra_apis)
                            )

                            existing["targeted_repair_requests"] = (
                                existing.get("targeted_repair_requests", 0)
                                + 1
                            )

                            existing["json_format_repair_requests"] = (
                                existing.get(
                                    "json_format_repair_requests",
                                    0,
                                )
                                + int(residual_format_api is not None)
                            )

                            existing["targeted_repair"] = {
                                "performed": True,
                                "accepted_non_numerical_sentence_ids": (
                                    accepted_non_numerical
                                ),
                                "unresolved_sentence_ids": unresolved,
                                "model_output_text": (
                                    residual_api["visible_text"]
                                ),
                                "evaluation": residual_evaluation,
                                "json_format_recovery": residual_format_info,
                                "usage": residual_api["usage"],
                                "resume_only": True,
                            }

                            existing["local_recovery"] = {
                                "performed": True,
                                "reason": (
                                    "Completed only the residual numerical "
                                    "coverage gap with one targeted Flex "
                                    "repair; prior full Judge calls were reused."
                                ),
                            }

                            save_json(
                                output_path,
                                existing,
                            )

                            print(
                                "RECOVERED existing numerical coverage "
                                "failure with one residual targeted repair — "
                                "no full Judge rerun."
                            )

                            return existing

    full_prompt = build_candidate_input(
        evaluation_type=(
            evaluation_type
        ),
        instructions=(
            instructions
        ),
        source_report=(
            source_report
        ),
        sentence_map=(
            sentence_map
        ),
        final_units=(
            final_units
        ),
    )

    reasoning_mode = (
        CANDIDATE_REASONING_MODES[
            evaluation_type
        ]
    )

    max_tokens = (
        CANDIDATE_MAX_TOKENS[
            evaluation_type
        ]
    )

    print()
    print(
        f"{evaluation_type} | "
        f"reasoning={reasoning_mode} | "
        f"tier={SERVICE_TIER}"
    )

    first_api = None
    repair_api = None
    repair_evaluation = None

    numerical_only_api = None
    numerical_only_evaluation = None

    factual_label_repair_api = None
    factual_label_repair_evaluation = None
    factual_label_repair_unresolved = []

    trace_id_repair_api = None
    trace_id_repair_evaluation = None
    trace_id_repair_unresolved = []

    first_json_format_api = None
    repair_json_format_api = None
    numerical_only_json_format_api = None

    first_json_format_info = {
        "performed": False,
        "context": None,
        "original_parse_error": None,
    }
    repair_json_format_info = {
        "performed": False,
        "context": None,
        "original_parse_error": None,
    }
    numerical_only_json_format_info = {
        "performed": False,
        "context": None,
        "original_parse_error": None,
    }
    factual_label_repair_json_format_api = None
    factual_label_repair_json_format_info = {
        "performed": False,
        "context": None,
        "original_parse_error": None,
    }

    trace_id_repair_json_format_api = None
    trace_id_repair_json_format_info = {
        "performed": False,
        "context": None,
        "original_parse_error": None,
    }

    numerical_recovery_strategy = None
    accepted_non_numerical = []
    unresolved_after_repair = []
    coverage = None

    try:
        first_api = call_nova(
            full_prompt=full_prompt,
            reasoning_mode=(
                reasoning_mode
            ),
            max_tokens=(
                max_tokens
            ),
        )

        (
            parsed,
            first_json_format_api,
            first_json_format_info,
        ) = parse_json_with_format_recovery_v523(
            raw_text=(
                first_api[
                    "visible_text"
                ]
            ),
            context_label=(
                f"{evaluation_type}: full Judge response"
            ),
        )

        if first_json_format_api is not None:
            print(
                "JSON syntax recovery succeeded for "
                f"{evaluation_type} full response."
            )

        if (
            evaluation_type
            == "factual_numerical"
        ):
            parsed = (
                normalize_factual_numerical_schema_v527(
                    parsed
                )
            )

            invalid_trace_claim_ids = (
                factual_claim_ids_with_invalid_trace_v5211(
                    parsed,
                    sentence_map,
                )
            )

            if invalid_trace_claim_ids:
                print(
                    "Factual trace-ID repair | "
                    f"triggered by: {invalid_trace_claim_ids}"
                )

                trace_id_prompt = (
                    build_factual_trace_id_repair_prompt_v5211(
                        evaluation=parsed,
                        sentence_map=sentence_map,
                    )
                )

                trace_id_repair_api = call_nova(
                    full_prompt=trace_id_prompt,
                    reasoning_mode="off",
                    max_tokens=5000,
                )

                (
                    trace_id_repair_evaluation,
                    trace_id_repair_json_format_api,
                    trace_id_repair_json_format_info,
                ) = (
                    parse_json_with_format_recovery_v523(
                        raw_text=(
                            trace_id_repair_api[
                                "visible_text"
                            ]
                        ),
                        context_label=(
                            "factual trace-ID repair"
                        ),
                    )
                )

                (
                    parsed,
                    trace_id_repair_unresolved,
                ) = (
                    apply_factual_trace_id_repair_v5211(
                        evaluation=parsed,
                        repair_evaluation=(
                            trace_id_repair_evaluation
                        ),
                        sentence_map=sentence_map,
                    )
                )

            ambiguous_factual_ids = (
                ambiguous_factual_claim_ids_v529(
                    parsed
                )
            )

            if ambiguous_factual_ids:
                print(
                    "Ambiguous factual label repair | "
                    f"claims: {ambiguous_factual_ids}"
                )

                factual_label_prompt = (
                    build_targeted_factual_label_repair_prompt_v529(
                        source_report=source_report,
                        evaluation=parsed,
                        sentence_map=sentence_map,
                        claim_ids=ambiguous_factual_ids,
                    )
                )

                factual_label_repair_api = call_nova(
                    full_prompt=factual_label_prompt,
                    reasoning_mode="low",
                    max_tokens=4000,
                )

                (
                    factual_label_repair_evaluation,
                    factual_label_repair_json_format_api,
                    factual_label_repair_json_format_info,
                ) = (
                    parse_json_with_format_recovery_v523(
                        raw_text=(
                            factual_label_repair_api[
                                "visible_text"
                            ]
                        ),
                        context_label=(
                            "targeted factual label repair"
                        ),
                    )
                )

                (
                    parsed,
                    factual_label_repair_unresolved,
                ) = (
                    apply_targeted_factual_label_repair_v529(
                        evaluation=parsed,
                        repair_evaluation=(
                            factual_label_repair_evaluation
                        ),
                        claim_ids=ambiguous_factual_ids,
                    )
                )

            validation_errors = (
                validate_factual_numerical_output(
                    parsed,
                    sentence_map,
                )
            )

            if not validation_errors:
                coverage = (
                    numerical_coverage_check_v52(
                        parsed,
                        sentence_map,
                    )
                )

                if not coverage["ok"]:
                    missing_ids = list(
                        coverage[
                            "missing_sentence_ids"
                        ]
                    )

                    print(
                        "Numerical coverage warning | "
                        f"missing: {missing_ids}"
                    )

                    if (
                        len(missing_ids)
                        <= TARGETED_NUMERICAL_REPAIR_MAX_SENTENCES
                    ):
                        numerical_recovery_strategy = (
                            "targeted_repair"
                        )

                        repair_prompt = (
                            build_targeted_numerical_repair_prompt_v52(
                                source_report=(
                                    source_report
                                ),
                                missing_sentence_ids=(
                                    missing_ids
                                ),
                                sentence_map=(
                                    sentence_map
                                ),
                            )
                        )

                        repair_api = call_nova(
                            full_prompt=(
                                repair_prompt
                            ),
                            reasoning_mode="low",
                            max_tokens=4000,
                        )

                        try:
                            (
                                repair_evaluation,
                                repair_json_format_api,
                                repair_json_format_info,
                            ) = (
                                parse_json_with_format_recovery_v523(
                                    raw_text=(
                                        repair_api[
                                            "visible_text"
                                        ]
                                    ),
                                    context_label=(
                                        "targeted numerical repair"
                                    ),
                                )
                            )

                            if (
                                repair_json_format_api
                                is not None
                            ):
                                print(
                                    "JSON syntax recovery succeeded "
                                    "for targeted numerical repair."
                                )

                            (
                                parsed,
                                accepted_non_numerical,
                                unresolved_after_repair,
                            ) = (
                                apply_targeted_numerical_repair_v52(
                                    evaluation=parsed,
                                    repair_evaluation=(
                                        repair_evaluation
                                    ),
                                    missing_sentence_ids=(
                                        missing_ids
                                    ),
                                )
                            )

                            validation_errors = (
                                validate_factual_numerical_output(
                                    parsed,
                                    sentence_map,
                                )
                            )

                            if not validation_errors:
                                coverage = (
                                    numerical_coverage_check_v52(
                                        parsed,
                                        sentence_map,
                                        accepted_non_numerical,
                                    )
                                )

                        except Exception as exc:
                            unresolved_after_repair = list(
                                missing_ids
                            )

                            coverage = dict(
                                coverage
                            )
                            coverage[
                                "ok"
                            ] = False

                            validation_errors = []

                            print(
                                "Targeted numerical repair could not "
                                "be parsed/applied: "
                                f"{type(exc).__name__}: {exc}"
                            )

                    else:
                        numerical_recovery_strategy = (
                            "numerical_only_full_pass"
                        )

                        print(
                            "Large numerical coverage gap "
                            f"({len(missing_ids)} sentences) -> "
                            "running one dedicated numerical-only pass."
                        )

                        numerical_only_prompt = (
                            build_numerical_only_recovery_prompt_v524(
                                source_report=(
                                    source_report
                                ),
                                sentence_map=(
                                    sentence_map
                                ),
                            )
                        )

                        numerical_only_api = call_nova(
                            full_prompt=(
                                numerical_only_prompt
                            ),
                            reasoning_mode="low",
                            max_tokens=16000,
                        )

                        try:
                            (
                                numerical_only_evaluation,
                                numerical_only_json_format_api,
                                numerical_only_json_format_info,
                            ) = (
                                parse_json_with_format_recovery_v523(
                                    raw_text=(
                                        numerical_only_api[
                                            "visible_text"
                                        ]
                                    ),
                                    context_label=(
                                        "dedicated numerical-only recovery"
                                    ),
                                )
                            )

                            if (
                                numerical_only_json_format_api
                                is not None
                            ):
                                print(
                                    "JSON syntax recovery succeeded "
                                    "for dedicated numerical-only pass."
                                )

                            numerical_only_evaluation = (
                                normalize_factual_numerical_schema_v527(
                                    numerical_only_evaluation
                                )
                            )

                            numerical_only_errors = (
                                validate_numerical_only_recovery_v524(
                                    numerical_only_evaluation,
                                    sentence_map,
                                )
                            )

                            if numerical_only_errors:
                                validation_errors = [
                                    "Numerical-only recovery: "
                                    + error
                                    for error in numerical_only_errors
                                ]

                            else:
                                parsed = (
                                    merge_numerical_only_recovery_v524(
                                        original_evaluation=(
                                            parsed
                                        ),
                                        numerical_only_evaluation=(
                                            numerical_only_evaluation
                                        ),
                                    )
                                )

                                validation_errors = (
                                    validate_factual_numerical_output(
                                        parsed,
                                        sentence_map,
                                    )
                                )

                                if not validation_errors:
                                    coverage = (
                                        numerical_coverage_check_v52(
                                            parsed,
                                            sentence_map,
                                        )
                                    )

                                    if (
                                        not coverage["ok"]
                                        and 1
                                        <= len(
                                            coverage[
                                                "missing_sentence_ids"
                                            ]
                                        )
                                        <= TARGETED_NUMERICAL_REPAIR_MAX_SENTENCES
                                    ):
                                        residual_ids = list(
                                            coverage[
                                                "missing_sentence_ids"
                                            ]
                                        )

                                        print(
                                            "Residual numerical coverage "
                                            "after numerical-only pass | "
                                            f"missing: {residual_ids} -> "
                                            "running one final targeted repair."
                                        )

                                        numerical_recovery_strategy = (
                                            "numerical_only_full_pass"
                                            "_plus_targeted_repair"
                                        )

                                        repair_prompt = (
                                            build_targeted_numerical_repair_prompt_v52(
                                                source_report=source_report,
                                                missing_sentence_ids=(
                                                    residual_ids
                                                ),
                                                sentence_map=sentence_map,
                                            )
                                        )

                                        repair_api = call_nova(
                                            full_prompt=repair_prompt,
                                            reasoning_mode="low",
                                            max_tokens=4000,
                                        )

                                        (
                                            repair_evaluation,
                                            repair_json_format_api,
                                            repair_json_format_info,
                                        ) = (
                                            parse_json_with_format_recovery_v523(
                                                raw_text=(
                                                    repair_api[
                                                        "visible_text"
                                                    ]
                                                ),
                                                context_label=(
                                                    "residual targeted "
                                                    "numerical repair"
                                                ),
                                            )
                                        )

                                        (
                                            parsed,
                                            accepted_non_numerical,
                                            unresolved_after_repair,
                                        ) = (
                                            apply_targeted_numerical_repair_v52(
                                                evaluation=parsed,
                                                repair_evaluation=(
                                                    repair_evaluation
                                                ),
                                                missing_sentence_ids=(
                                                    residual_ids
                                                ),
                                            )
                                        )

                                        validation_errors = (
                                            validate_factual_numerical_output(
                                                parsed,
                                                sentence_map,
                                            )
                                        )

                                        if not validation_errors:
                                            coverage = (
                                                numerical_coverage_check_v52(
                                                    parsed,
                                                    sentence_map,
                                                    accepted_non_numerical,
                                                )
                                            )

                        except Exception as exc:
                            coverage = dict(
                                coverage
                            )
                            coverage[
                                "ok"
                            ] = False

                            validation_errors = []

                            print(
                                "Dedicated numerical-only recovery "
                                "could not be parsed/applied: "
                                f"{type(exc).__name__}: {exc}"
                            )

            parse_error = None

            if validation_errors:
                status = (
                    "validation_failed"
                )
            elif (
                coverage is not None
                and not coverage["ok"]
            ):
                status = (
                    "numerical_coverage_failed"
                )
            else:
                status = "success"

        elif (
            evaluation_type
            == "completeness"
        ):
            validation_errors = (
                validate_completeness_output(
                    parsed,
                    final_units or [],
                    sentence_map,
                )
            )
            parse_error = None
            status = (
                "success"
                if not validation_errors
                else "validation_failed"
            )

        elif evaluation_type == "coherence":
            validation_errors = (
                validate_coherence_output(
                    parsed,
                    sentence_map,
                )
            )
            parse_error = None
            status = (
                "success"
                if not validation_errors
                else "validation_failed"
            )

        else:
            raise ValueError(
                evaluation_type
            )

    except Exception as exc:
        result = {
            "status": "failed",
            "metadata": {
                "timestamp": (
                    datetime.now().isoformat()
                ),
                "judge_model": (
                    JUDGE_MODEL_NAME
                ),
                "judge_model_id": (
                    JUDGE_MODEL_ID
                ),
                "requested_service_tier": (
                    SERVICE_TIER
                ),
                "candidate_generation_mode": (
                    RUN_MODE
                ),
                "generator_model": (
                    generator_model_name
                ),
                "candidate_file": str(
                    candidate_path
                ),
                "evaluation_type": (
                    evaluation_type
                ),
                "prompt_version": (
                    PROMPT_VERSION
                ),
                "prompt_hash": p_hash,
                "candidate_hash": c_hash,
                "sentence_map_hash": (
                    sentence_map_hash_value
                ),
                "source_hash": s_hash,
                "key_information_hash": (
                    key_information_hash_value
                ),
            },
            "candidate_sentence_map": (
                sentence_map
            ),
            "model_output_text": (
                first_api[
                    "visible_text"
                ]
                if first_api is not None
                else None
            ),
            "json_format_recovery": {
                "full_response": (
                    first_json_format_info
                ),
                "targeted_repair": (
                    repair_json_format_info
                ),
            },
            "usage": (
                _combined_api_usage_v52(
                    [
                        api
                        for api in [
                            first_api,
                            first_json_format_api,
                            repair_api,
                            repair_json_format_api,
                            numerical_only_api,
                            numerical_only_json_format_api,
                        ]
                        if api is not None
                    ]
                )
                if first_api is not None
                else None
            ),
            "error_type": (
                type(exc).__name__
            ),
            "error_message": str(exc),
        }

        save_json(
            output_path,
            result,
        )

        print(
            "FAILED | "
            f"{type(exc).__name__}: {exc}"
        )

        return result

    api_results = [
        api
        for api in [
            first_api,
            first_json_format_api,
            repair_api,
            repair_json_format_api,
            numerical_only_api,
            numerical_only_json_format_api,
            factual_label_repair_api,
            factual_label_repair_json_format_api,
            trace_id_repair_api,
            trace_id_repair_json_format_api,
        ]
        if api is not None
    ]

    usage = (
        _combined_api_usage_v52(
            api_results
        )
    )

    result = {
        "status": status,
        "metadata": {
            "timestamp": (
                datetime.now().isoformat()
            ),
            "judge_model": (
                JUDGE_MODEL_NAME
            ),
            "judge_model_id": (
                JUDGE_MODEL_ID
            ),
            "requested_service_tier": (
                SERVICE_TIER
            ),
            "resolved_service_tier": (
                first_api.get(
                    "resolved_service_tier"
                )
            ),
            "judge_reasoning_enabled": (
                reasoning_mode != "off"
            ),
            "judge_reasoning_effort": (
                reasoning_mode
            ),
            "candidate_generation_mode": (
                RUN_MODE
            ),
            "generator_model": (
                generator_model_name
            ),
            "candidate_file": str(
                candidate_path
            ),
            "evaluation_type": (
                evaluation_type
            ),
            "prompt_version": (
                PROMPT_VERSION
            ),
            "prompt_hash": p_hash,
            "candidate_hash": c_hash,
            "sentence_map_hash": (
                sentence_map_hash_value
            ),
            "source_file": (
                str(source_path)
                if evaluation_type
                == "factual_numerical"
                else None
            ),
            "source_hash": s_hash,
            "key_information_hash": (
                key_information_hash_value
                if evaluation_type
                == "completeness"
                else None
            ),
            "key_information_file": (
                str(key_information_file)
                if (
                    evaluation_type
                    == "completeness"
                    and key_information_file
                    is not None
                )
                else None
            ),
        },

        "candidate_sentence_map": (
            sentence_map
        ),

        "usage": {
            "input_tokens": (
                usage[
                    "input_tokens"
                ]
            ),
            "output_tokens_including_reasoning": (
                usage[
                    "output_tokens_including_reasoning"
                ]
            ),
            "total_tokens": (
                usage[
                    "total_tokens"
                ]
            ),
        },

        "latency_seconds": (
            usage[
                "latency_seconds"
            ]
        ),

        "logical_requests": len(
            api_results
        ),

        "full_evaluation_requests": 1,

        "targeted_repair_requests": (
            1
            if repair_api is not None
            else 0
        ),

        "json_format_repair_requests": (
            int(
                first_json_format_api
                is not None
            )
            + int(
                repair_json_format_api
                is not None
            )
            + int(
                numerical_only_json_format_api
                is not None
            )
            + int(
                factual_label_repair_json_format_api
                is not None
            )
            + int(
                trace_id_repair_json_format_api
                is not None
            )
        ),

        "json_format_recovery": {
            "full_response": (
                first_json_format_info
            ),
            "targeted_repair": (
                repair_json_format_info
            ),
            "numerical_only_recovery": (
                numerical_only_json_format_info
            ),
            "factual_label_repair": (
                factual_label_repair_json_format_info
            ),
            "trace_id_repair": (
                trace_id_repair_json_format_info
            ),
        },

        "numerical_recovery_strategy": (
            numerical_recovery_strategy
        ),

        "stop_reason": (
            first_api[
                "stop_reason"
            ]
        ),

        "reasoning": {
            "returned": bool(
                first_api[
                    "reasoning_texts"
                ]
            ),
            "content": (
                first_api[
                    "reasoning_texts"
                ]
            ),
        },

        "model_output_text": (
            first_api[
                "visible_text"
            ]
        ),

        "evaluation": parsed,

        "numerical_coverage": (
            coverage
        ),

        "targeted_repair": {
            "performed": (
                repair_api is not None
            ),
            "accepted_non_numerical_sentence_ids": (
                accepted_non_numerical
            ),
            "unresolved_sentence_ids": (
                unresolved_after_repair
            ),
            "model_output_text": (
                repair_api[
                    "visible_text"
                ]
                if repair_api is not None
                else None
            ),
            "evaluation": (
                repair_evaluation
            ),
            "json_format_recovery": (
                repair_json_format_info
            ),
            "usage": (
                repair_api[
                    "usage"
                ]
                if repair_api is not None
                else None
            ),
        },

        "factual_label_repair": {
            "performed": (
                factual_label_repair_api
                is not None
            ),
            "unresolved_claim_ids": (
                factual_label_repair_unresolved
            ),
            "model_output_text": (
                factual_label_repair_api[
                    "visible_text"
                ]
                if factual_label_repair_api
                is not None
                else None
            ),
            "evaluation": (
                factual_label_repair_evaluation
            ),
            "json_format_recovery": (
                factual_label_repair_json_format_info
            ),
            "usage": (
                factual_label_repair_api[
                    "usage"
                ]
                if factual_label_repair_api
                is not None
                else None
            ),
        },

        "trace_id_repair": {
            "performed": (
                trace_id_repair_api
                is not None
            ),
            "unresolved_claim_ids": (
                trace_id_repair_unresolved
            ),
            "model_output_text": (
                trace_id_repair_api[
                    "visible_text"
                ]
                if trace_id_repair_api
                is not None
                else None
            ),
            "evaluation": (
                trace_id_repair_evaluation
            ),
            "json_format_recovery": (
                trace_id_repair_json_format_info
            ),
            "usage": (
                trace_id_repair_api[
                    "usage"
                ]
                if trace_id_repair_api
                is not None
                else None
            ),
        },

        "numerical_only_recovery": {
            "performed": (
                numerical_only_api
                is not None
            ),
            "model_output_text": (
                numerical_only_api[
                    "visible_text"
                ]
                if numerical_only_api is not None
                else None
            ),
            "evaluation": (
                numerical_only_evaluation
            ),
            "json_format_recovery": (
                numerical_only_json_format_info
            ),
            "usage": (
                numerical_only_api[
                    "usage"
                ]
                if numerical_only_api is not None
                else None
            ),
        },

        "validation_errors": (
            validation_errors
        ),

        "parse_error": (
            parse_error
        ),
    }

    save_json(
        output_path,
        result,
    )

    print(
        f"STATUS: {status} | "
        f"input={usage['input_tokens']:,} | "
        f"output={usage['output_tokens_including_reasoning']:,} | "
        f"latency={usage['latency_seconds']:.2f}s | "
        f"logical_requests={len(api_results)}"
    )

    if validation_errors:
        print(
            "Validation errors:"
        )
        for error in validation_errors:
            print(
                f"  - {error}"
            )

    if (
        coverage is not None
        and not coverage["ok"]
    ):
        print(
            "Unresolved numerical coverage: "
            f"{coverage['missing_sentence_ids']}"
        )

    return result

