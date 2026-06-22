import json
import random
import csv
from pathlib import Path
from openai import OpenAI

JUDGE_MODEL = "gpt-5.4-mini"
MAX_OUTPUT_TOKENS = 1200

random.seed(42)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

REPORTS_DIR = DATA_DIR / "reports"
OUTPUTS_DIR = DATA_DIR / "outputs"
JUDGE_RESULTS_DIR = DATA_DIR / "judge_results"
JUDGE_MAPPINGS_DIR = DATA_DIR / "judge_mappings"

selected_category = "chunked"

report_source_category = selected_category

if selected_category == "chunked":
    report_source_category = "chunked_judge"

# Choose manually what to evaluate
prompt_files = ["chain_of_event"]          # example: ["zero_shot", "few_shot", "chain_of_event"]
report_files = ["report_4.md"]

client = OpenAI(api_key=OPENAI_API_KEY)

JUDGE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
JUDGE_MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)
(JUDGE_RESULTS_DIR / selected_category).mkdir(parents=True, exist_ok=True)
(JUDGE_MAPPINGS_DIR / selected_category).mkdir(parents=True, exist_ok=True)


def load_text(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8").strip()


def build_summary_path(category: str, report_stem: str, prompt_name: str, model_name: str) -> Path:
    return OUTPUTS_DIR / category / f"{report_stem}_{prompt_name}_{model_name}.txt"


def anonymize_summaries(category: str, report_stem: str, prompt_name: str):
    candidates = [
        ("gemini", build_summary_path(category, report_stem, prompt_name, "gemini")),
        ("openai", build_summary_path(category, report_stem, prompt_name, "openai")),
        ("ollama", build_summary_path(category, report_stem, prompt_name, "ollama")),
    ]

    loaded = []
    for model_name, path in candidates:
        if not path.exists():
            raise FileNotFoundError(f"Missing summary file: {path}")
        loaded.append({
            "model": model_name,
            "path": path,
            "text": load_text(path)
        })

    random.shuffle(loaded)

    labels = ["A", "B", "C"]
    anonymized = {}
    mapping_rows = []

    for label, item in zip(labels, loaded):
        anonymized[label] = item["text"]
        mapping_rows.append({
            "label": label,
            "model": item["model"],
            "filename": item["path"].name
        })

    return anonymized, mapping_rows


def build_judge_prompt(report_text: str, summaries: dict[str, str]) -> str:
    return f"""
Your task is to evaluate three summaries of the same financial report: Summary A, Summary B, and Summary C. 

Evaluate each summary only against the source financial report. 
Do not compare the summaries to any reference summary. 
Do not reward stylistic richness or creativity. 
Focus only on factual correctness, numerical correctness, coverage of important information, and coherence. 

Evaluation criteria: 

1. Faithfulness 
- The summary must contain only claims supported by the source report. 
- Penalize unsupported statements, hallucinations, or misleading interpretations. 

2. Numerical accuracy 
- Financial values, percentages, time periods, and comparisons must match the source report. 
- Penalize incorrect numbers, distorted comparisons, wrong periods, or missing numerical precision where important. 

3. Coverage 
- The summary should cover the most important financial indicators, major changes compared with the previous period, and the main factors affecting performance. 
- Penalize omission of important information. 

4. Coherence 
- The summary should be logically organized, easy to follow, and free from contradiction or unnecessary repetition. 

Scoring rules: 
- Score each criterion from 0.00 to 1.00 in increments of 0.05 only. 
- 0.00 means very poor. 1.00 means excellent. 
- For each criterion, provide exactly one sentence justification before the score. 
- Do not provide overall ranking. 
- Do not output any text outside the JSON object. 

Return exactly this JSON format: 
{{
  "A": {{
    "faithfulness": {{"justification": "...", "score": 0.00}},
    "numerical_accuracy": {{"justification": "...", "score": 0.00}},
    "coverage": {{"justification": "...", "score": 0.00}},
    "coherence": {{"justification": "...", "score": 0.00}}
  }},
  "B": {{
    "faithfulness": {{"justification": "...", "score": 0.00}},
    "numerical_accuracy": {{"justification": "...", "score": 0.00}},
    "coverage": {{"justification": "...", "score": 0.00}},
    "coherence": {{"justification": "...", "score": 0.00}}
  }},
  "C": {{
    "faithfulness": {{"justification": "...", "score": 0.00}},
    "numerical_accuracy": {{"justification": "...", "score": 0.00}},
    "coverage": {{"justification": "...", "score": 0.00}},
    "coherence": {{"justification": "...", "score": 0.00}}
  }}
}} 

Source financial report:
{report_text}


Summary A: {summaries["A"]} 
Summary B: {summaries["B"]} 
Summary C: {summaries["C"]}
""".strip()


def judge_summaries(prompt_text: str) -> dict:
    criterion_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "justification": {"type": "string"},
            "score": {"type": "number"}
        },
        "required": ["justification", "score"]
    }

    schema = {
        "name": "judge_scores",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                key: {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "faithfulness": criterion_schema,
                        "numerical_accuracy": criterion_schema,
                        "coverage": criterion_schema,
                        "coherence": criterion_schema
                    },
                    "required": [
                        "faithfulness",
                        "numerical_accuracy",
                        "coverage",
                        "coherence"
                    ]
                }
                for key in ["A", "B", "C"]
            },
            "required": ["A", "B", "C"]
        },
        "type": "json_schema",
        "strict": True
    }

    response = client.responses.create(
        model=JUDGE_MODEL,
        input=prompt_text,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        text={
            "format": schema
        }
    )

    return json.loads(response.output_text)


def recompute_average(scores: dict) -> dict:
    for label in ["A", "B", "C"]:
        vals = [
            scores[label]["faithfulness"]["score"],
            scores[label]["numerical_accuracy"]["score"],
            scores[label]["coverage"]["score"],
            scores[label]["coherence"]["score"],
        ]
        scores[label]["average"] = round(sum(vals) / 4, 4)
    return scores


def save_mapping(category: str, report_stem: str, prompt_name: str, rows: list[dict]) -> None:
    out_path = JUDGE_MAPPINGS_DIR / category / f"{report_stem}_{prompt_name}_mapping.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "model", "filename"])
        writer.writeheader()
        writer.writerows(rows)


def save_result(category: str, report_stem: str, prompt_name: str, result: dict) -> None:
    out_path = JUDGE_RESULTS_DIR / category / f"{report_stem}_{prompt_name}_judge.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


for report_filename in report_files:
    report_stem = Path(report_filename).stem
    report_path = REPORTS_DIR / report_source_category / report_filename

    if not report_path.exists():
        print(f"Skipping missing report: {report_path}")
        continue

    report_text = load_text(report_path)

    for prompt_name in prompt_files:
        print(f"Processing {report_stem} | {prompt_name}")

        try:
            anonymized, mapping_rows = anonymize_summaries(selected_category, report_stem, prompt_name)
            save_mapping(selected_category, report_stem, prompt_name, mapping_rows)

            judge_prompt = build_judge_prompt(report_text, anonymized)
            scores = judge_summaries(judge_prompt)
            scores = recompute_average(scores)

            save_result(selected_category, report_stem, prompt_name, scores)

            print(f"Saved judge result for {report_stem} | {prompt_name}")

        except Exception as error:
            print(f"Error for {report_stem} | {prompt_name}: {error}")