from google import genai
from google.genai import types
from openai import OpenAI
from pathlib import Path
from botocore.config import Config
import boto3
import os
import time
import re



# =========================================================
# API KEYS
# =========================================================

GEMINI_API_KEY = "AIzaSyAR7DDbIftxLACeb8QiENSeJ2kLm_3qVSU"
OPENAI_API_KEY = "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"
AWS_BEDROCK_API_KEY = "ABSKQmVkcm9ja0FQSUtleS1jOHZ4LWF0LTM3ODk4ODE4ODY1Njpudlc2VmthYTJVREFkcUxpYllmZW5oRUpoM0kvelZ6eGVUM0dvQWZTV3pIVndiUVJtWDgzOUZ2MXBicz0="

## =========================================================
# MODELS
# =========================================================

GEMINI_MODEL = "gemini-3.5-flash-lite"
OPENAI_MODEL = "gpt-5.6-luna"
NOVA_MODEL = "eu.amazon.nova-2-lite-v1:0"
AWS_REGION = "eu-central-1"


# =========================================================
# MODEL SELECTION
# =========================================================

# Choose which model to run:
#
# "gemini"
# "openai"
# "nova"

SELECTED_MODEL = "openai"


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
PROMPTS_DIR = DATA_DIR / "prompts"
REPORTS_DIR = DATA_DIR / "reports"
OUTPUTS_DIR = DATA_DIR / "outputs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# EXPERIMENT SETTINGS
# =========================================================

MAX_OUTPUT_TOKENS = 1000

PROMPT_NAMES = [
    "zero_shot",
    "few_shot",
    "chain_of_event",
]

REPORT_CATEGORIES = [
    "annual_report",
    "quarterly_report",
]


# =========================================================
# RUN MODE
# =========================================================

# "full"   -> normal experiment run
# "single" -> rerun one specific category/report/prompt
RUN_MODE = "single"

# Settings used only when RUN_MODE = "single".
# The selected result is regenerated even if the output file already exists,
# and the existing file is overwritten with the new response.
MANUAL_CATEGORY = "annual_report"
MANUAL_REPORT_FILENAME = "report_annual_8.md"
MANUAL_PROMPT = "few_shot"


# =========================================================
# TEMPORARY TEST LIMIT
# =========================================================

# Number of reports processed from EACH category.
#
# 2 means:
# 2 annual reports
# 2 quarterly reports
#
# Set to None later to process all reports.

REPORT_LIMIT = None


# =========================================================
# DELAYS
# =========================================================

# Pause after each API request.
REQUEST_DELAY_SECONDS = 15

# Additional pause before processing the next report.
REPORT_DELAY_SECONDS = 25


# =========================================================
# RETRY SETTINGS
# =========================================================

MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 30


# =========================================================
# FILE FUNCTIONS
# =========================================================

def load_prompt(prompt_name: str) -> str:

    prompt_path = PROMPTS_DIR / f"{prompt_name}.txt"

    with open(prompt_path, "r", encoding="utf-8") as file:
        return file.read().strip()


def load_report(category: str, report_filename: str) -> str:

    report_path = REPORTS_DIR / category / report_filename

    with open(report_path, "r", encoding="utf-8") as file:
        return file.read().strip()


def natural_sort_key(path: Path):

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def get_reports_in_category(category: str) -> list[str]:

    category_dir = REPORTS_DIR / category

    report_files = sorted(
        category_dir.glob("*.md"),
        key=natural_sort_key
    )

    # Temporary limitation for testing
    if REPORT_LIMIT is not None:
        report_files = report_files[:REPORT_LIMIT]

    return [file.name for file in report_files]


def build_full_prompt(
    prompt_text: str,
    report_text: str
) -> str:

    return (
        f"{prompt_text}\n\n"
        f"Financial report:\n\n"
        f"{report_text}"
    )


# =========================================================
# OUTPUT
# =========================================================

def save_result(
    model_name: str,
    category: str,
    report_filename: str,
    prompt_name: str,
    summary_text: str
) -> None:

    category_output_dir = OUTPUTS_DIR / category
    category_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    report_stem = Path(report_filename).stem

    output_filename = (
        f"{report_stem}_{prompt_name}_{model_name}.txt"
    )

    output_path = (
        category_output_dir
        / output_filename
    )

    output_path.write_text(
        summary_text,
        encoding="utf-8"
    )

    print(f"Saved result to: {output_path}")


# =========================================================
# API CLIENTS
# =========================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)

# Amazon Bedrock long-term API key authentication.
# If AWS_BEARER_TOKEN_BEDROCK is already configured in the OS,
# leave AWS_BEDROCK_API_KEY empty.
if AWS_BEDROCK_API_KEY:
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = AWS_BEDROCK_API_KEY

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    config=Config(
        read_timeout=3600,
        connect_timeout=60,
    ),
)


# =========================================================
# GEMINI
# =========================================================

def generate_with_gemini(
    prompt_text: str
) -> str:

    response = gemini_client.models.generate_content(

        model=GEMINI_MODEL,

        contents=prompt_text,

        config=types.GenerateContentConfig(

            max_output_tokens=MAX_OUTPUT_TOKENS,

            thinking_config=types.ThinkingConfig(
                thinking_level="minimal"
            ),

            temperature=0,
            top_p=1,
            candidate_count=1,
        )
    )

    # Token statistics
    usage = response.usage_metadata

    if usage:

        print(
            "Gemini tokens | "
            f"input: "
            f"{getattr(usage, 'prompt_token_count', None)} | "
            f"output: "
            f"{getattr(usage, 'candidates_token_count', None)} | "
            f"thinking: "
            f"{getattr(usage, 'thoughts_token_count', None)} | "
            f"total: "
            f"{getattr(usage, 'total_token_count', None)}"
        )

    # Finish reason
    if response.candidates:

        finish_reason = (
            response.candidates[0]
            .finish_reason
        )

        print(
            f"Gemini finish reason: {finish_reason}"
        )

        # Do not save truncated or otherwise incomplete Gemini responses.
        # This is especially important for manual reruns of responses that
        # previously ended because MAX_OUTPUT_TOKENS was reached.
        finish_reason_text = str(finish_reason).upper()

        if "MAX_TOKENS" in finish_reason_text:
            raise RuntimeError(
                "Gemini response reached MAX_OUTPUT_TOKENS "
                "and was truncated."
            )

        if "STOP" not in finish_reason_text:
            raise RuntimeError(
                "Gemini response did not finish normally. "
                f"Finish reason: {finish_reason}"
            )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return response.text


# =========================================================
# OPENAI
# =========================================================

def generate_with_openai(
    prompt_text: str
) -> str:

    response = openai_client.responses.create(

        model=OPENAI_MODEL,

        input=prompt_text,

        reasoning={
            "effort": "none"
        },

        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    # Token statistics
    if response.usage:

        reasoning_tokens = None

        if response.usage.output_tokens_details:

            reasoning_tokens = (
                response.usage
                .output_tokens_details
                .reasoning_tokens
            )

        print(
            "OpenAI tokens | "
            f"input: {response.usage.input_tokens} | "
            f"output: {response.usage.output_tokens} | "
            f"reasoning: {reasoning_tokens} | "
            f"total: {response.usage.total_tokens}"
        )

    # Check if generation was truncated
    if response.status == "incomplete":

        reason = None

        if response.incomplete_details:
            reason = (
                response
                .incomplete_details
                .reason
            )

        raise RuntimeError(
            f"OpenAI response incomplete. "
            f"Reason: {reason}"
        )

    if not response.output_text:

        raise RuntimeError(
            "OpenAI returned an empty response."
        )

    return response.output_text


# =========================================================
# AMAZON NOVA 2 LITE
# =========================================================

def generate_with_nova(
    prompt_text: str
) -> str:

    response = bedrock_client.converse(

        modelId=NOVA_MODEL,

        messages=[
            {
                "role": "user",
                "content": [
                    {"text": prompt_text}
                ],
            }
        ],

        inferenceConfig={
            "maxTokens": MAX_OUTPUT_TOKENS,
            # Nova 2 Lite documents 0.00001 as the minimum
            # temperature value, so this is effectively deterministic.
            "temperature": 0.00001,
        },

        # Extended thinking is disabled explicitly for reproducibility
        # and to keep the experiment comparable with the other models.
        additionalModelRequestFields={
            "reasoningConfig": {
                "type": "disabled"
            }
        },
    )

    # Token statistics
    usage = response.get("usage", {})

    print(
        "Amazon Nova tokens | "
        f"input: {usage.get('inputTokens')} | "
        f"output: {usage.get('outputTokens')} | "
        f"total: {usage.get('totalTokens')}"
    )

    # Latency
    metrics = response.get("metrics", {})

    if metrics.get("latencyMs") is not None:
        print(
            "Amazon Nova latency: "
            f"{metrics.get('latencyMs')} ms"
        )

    # Finish reason
    stop_reason = response.get("stopReason")

    print(
        f"Amazon Nova stop reason: {stop_reason}"
    )

    if stop_reason != "end_turn":
        raise RuntimeError(
            "Amazon Nova response did not finish normally. "
            f"Stop reason: {stop_reason}"
        )

    # Extract only final text blocks. This also keeps the code safe if
    # reasoning content is ever enabled later.
    content_blocks = (
        response
        .get("output", {})
        .get("message", {})
        .get("content", [])
    )

    text_parts = [
        block["text"]
        for block in content_blocks
        if "text" in block
    ]

    summary_text = "\n".join(text_parts).strip()

    if not summary_text:
        raise RuntimeError(
            "Amazon Nova returned an empty response."
        )

    return summary_text


# =========================================================
# MODEL ROUTER
# =========================================================

def generate_summary(
    model_name: str,
    prompt_text: str
) -> str:

    if model_name == "gemini":

        return generate_with_gemini(
            prompt_text
        )

    elif model_name == "openai":

        return generate_with_openai(
            prompt_text
        )

    elif model_name == "nova":

        return generate_with_nova(
            prompt_text
        )

    else:

        raise ValueError(
            f"Unknown model name: {model_name}"
        )


# =========================================================
# RETRY LOGIC
# =========================================================

def generate_with_retry(
    model_name: str,
    prompt_text: str
) -> str:

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            return generate_summary(
                model_name,
                prompt_text
            )

        except Exception as error:

            print(
                f"Attempt "
                f"{attempt}/{MAX_RETRIES} failed "
                f"for {model_name}: "
                f"{error}"
            )

            if attempt == MAX_RETRIES:
                raise

            wait_time = (
                RETRY_BASE_DELAY_SECONDS
                * (2 ** (attempt - 1))
            )

            print(
                f"Waiting {wait_time} seconds "
                f"before retry..."
            )

            time.sleep(wait_time)


# =========================================================
# REPORT PROCESSING
# =========================================================

def process_report(
    category: str,
    report_filename: str,
    prompt_names: list[str] | None = None,
    overwrite_existing: bool = False,
) -> None:

    if prompt_names is None:
        prompt_names = PROMPT_NAMES

    report_text = load_report(
        category,
        report_filename
    )

    print("\n" + "=" * 70)

    print(
        f"Category: {category} | "
        f"Report: {report_filename}"
    )

    print("=" * 70)

    for prompt_name in prompt_names:

        prompt_text = load_prompt(
            prompt_name
        )

        full_prompt = build_full_prompt(
            prompt_text,
            report_text
        )

        print(
            f"\nPrompt: {prompt_name}"
        )

        print(
            f"Running model: {SELECTED_MODEL}"
        )


        try:

            if result_exists(
                SELECTED_MODEL,
                category,
                report_filename,
                prompt_name
            ):
                if overwrite_existing:
                    print(
                        f"Existing result will be overwritten: "
                        f"{report_filename} | {prompt_name} | {SELECTED_MODEL}"
                    )
                else:
                    print(
                        f"Skipping existing result: "
                        f"{report_filename} | {prompt_name} | {SELECTED_MODEL}"
                    )
                    continue

            result = generate_with_retry(
                SELECTED_MODEL,
                full_prompt
            )

            save_result(
                model_name=SELECTED_MODEL,
                category=category,
                report_filename=report_filename,
                prompt_name=prompt_name,
                summary_text=result,
            )

        except Exception as error:

            print(
                f"FINAL ERROR | "
                f"{SELECTED_MODEL} | "
                f"{category} | "
                f"{report_filename} | "
                f"{prompt_name}: "
                f"{error}"
            )

        print(
            f"Waiting "
            f"{REQUEST_DELAY_SECONDS} seconds "
            f"before next API request..."
        )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

def result_exists(
    model_name: str,
    category: str,
    report_filename: str,
    prompt_name: str
) -> bool:

    category_output_dir = OUTPUTS_DIR / category

    report_stem = Path(report_filename).stem

    output_filename = (
        f"{report_stem}_{prompt_name}_{model_name}.txt"
    )

    output_path = category_output_dir / output_filename

    return output_path.exists()
# =========================================================
# MAIN EXPERIMENT
# =========================================================

def main():

    if SELECTED_MODEL not in [
        "gemini",
        "openai",
        "nova"
    ]:

        raise ValueError(
            "SELECTED_MODEL must be "
            "'gemini', 'openai' or 'nova'"
        )

    if RUN_MODE not in [
        "full",
        "single"
    ]:
        raise ValueError(
            "RUN_MODE must be 'full' or 'single'"
        )

    print("\n" + "=" * 70)

    print("EXPERIMENT SETTINGS")

    print("=" * 70)

    print(
        f"Selected model: {SELECTED_MODEL}"
    )

    print(
        f"Run mode: {RUN_MODE}"
    )

    print(
        f"Max output tokens: "
        f"{MAX_OUTPUT_TOKENS}"
    )

    print(
        f"Report limit per category: "
        f"{REPORT_LIMIT}"
    )

    print(
        f"Prompts: "
        f"{PROMPT_NAMES}"
    )

    print("=" * 70)

    # -----------------------------------------------------
    # MANUAL SINGLE-RERUN MODE
    # -----------------------------------------------------
    if RUN_MODE == "single":

        if MANUAL_CATEGORY not in REPORT_CATEGORIES:
            raise ValueError(
                f"Unknown MANUAL_CATEGORY: {MANUAL_CATEGORY}. "
                f"Choose one of: {REPORT_CATEGORIES}"
            )

        if MANUAL_PROMPT not in PROMPT_NAMES:
            raise ValueError(
                f"Unknown MANUAL_PROMPT: {MANUAL_PROMPT}. "
                f"Choose one of: {PROMPT_NAMES}"
            )

        report_path = (
            REPORTS_DIR
            / MANUAL_CATEGORY
            / MANUAL_REPORT_FILENAME
        )

        if not report_path.exists():
            raise FileNotFoundError(
                f"Manual report not found: {report_path}"
            )

        print("\nMANUAL SINGLE RERUN")
        print(
            f"Category: {MANUAL_CATEGORY}"
        )
        print(
            f"Report: {MANUAL_REPORT_FILENAME}"
        )
        print(
            f"Prompt: {MANUAL_PROMPT}"
        )
        print(
            "Existing output file: WILL BE OVERWRITTEN"
        )

        process_report(
            category=MANUAL_CATEGORY,
            report_filename=MANUAL_REPORT_FILENAME,
            prompt_names=[MANUAL_PROMPT],
            overwrite_existing=True,
        )

        return

    first_report = True

    for category in REPORT_CATEGORIES:

        report_files = (
            get_reports_in_category(
                category
            )
        )

        print(
            f"\nFound "
            f"{len(report_files)} reports "
            f"for processing "
            f"in {category}"
        )

        for report_filename in report_files:

            if not first_report:

                print(
                    f"\nWaiting "
                    f"{REPORT_DELAY_SECONDS} seconds "
                    f"before next report..."
                )

                time.sleep(
                    REPORT_DELAY_SECONDS
                )

            first_report = False

            process_report(
                category,
                report_filename
            )


if __name__ == "__main__":
    main()