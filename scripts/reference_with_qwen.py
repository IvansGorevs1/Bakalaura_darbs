import requests
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

SOURCE_DIR = DATA_DIR / "reference_sources" / "short"
OUTPUT_DIR = DATA_DIR / "references" / "short"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
QWEN_MODEL = "qwen2.5:1.5b-instruct"


def load_text(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()


def save_text(file_path: Path, text: str) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)


def build_reference_prompt(source_text: str) -> str:
    return f"""
Read the following reference financial-report source and rewrite it into the exact structure below.

Structure:
Main financial indicators
Most significant changes compared with the previous period
Main factors affecting performance
Overall summary

Rules:
- Use bullet points only.
- Use only information explicitly stated in the source text.
- Preserve all numerical values exactly.
- Do not add assumptions, interpretations, or facts that are not present in the source.
- Omit webinar details, contact information, attachments, and full statement tables.
- Keep the result concise and suitable for use as a reference summary for evaluation.

Reference source:

{source_text}
""".strip()


def generate_with_qwen(prompt_text: str) -> str:
    payload = {
        "model": QWEN_MODEL,
        "prompt": prompt_text,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()

    data = response.json()
    return data["response"]


source_filename = "report_1_reference_source.txt"
output_filename = "report_1_reference.txt"

source_path = SOURCE_DIR / source_filename
output_path = OUTPUT_DIR / output_filename

source_text = load_text(source_path)
prompt_text = build_reference_prompt(source_text)
reference_text = generate_with_qwen(prompt_text)

save_text(output_path, reference_text)

print(f"Reference saved to: {output_path}")