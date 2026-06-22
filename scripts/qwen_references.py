import requests
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

ORIGINAL_REFERENCES_DIR = DATA_DIR / "original_references"
REFERENCES_DIR = DATA_DIR / "references"

OLLAMA_URL = "http://localhost:11434/api/generate"
QWEN_MODEL = "qwen2.5:7b-instruct"

MAX_OUTPUT_TOKENS = 800


def load_text(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()


def save_text(file_path: Path, text: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)


def build_reference_prompt(source_text: str) -> str:
    return f"""
Read the following financial reference source and rewrite it into a structured reference summary for evaluation.

Return exactly 15 bullet points in total.
Return exactly 4 bullet points under "Main financial indicators".
Return exactly 4 bullet points under "Most significant changes compared with the previous period".
Return exactly 4 bullet points under "Main factors affecting performance".
Return exactly 3 bullet points under "Overall summary".

Write the section titles exactly as shown below.
Do not add any extra headings, explanations, intro sentences, or concluding text.
Do not add more bullet points.
Do not use fewer bullet points.
If information for a bullet point is missing, write:
- Not explicitly stated in the source text.

Use exactly this structure:

Main financial indicators
- ...
- ...
- ...
- ...

Most significant changes compared with the previous period
- ...
- ...
- ...
- ...

Main factors affecting performance
- ...
- ...
- ...
- ...

Overall summary
- ...
- ...
- ...

Strict rules:
- Use only the four sections shown above.
- Use bullet points only.
- Do not use tables.
- Do not add any extra sections, comments, contact information, company background, links, webinar details, attachment details, or market news.
- Use only information explicitly stated in the source text.
- Preserve all numerical values exactly as written in the source.
- Do not infer, calculate, reinterpret, or generalize values.
- Prefer the most important annual indicators and changes; include quarterly information only if it is explicitly emphasized as important in the source.
- Keep the result concise but sufficiently complete for comparison with model-generated summaries.

Reference source:

{source_text}
""".strip()


def generate_with_qwen(prompt_text: str) -> str:
    payload = {
        "model": QWEN_MODEL,
        "prompt": prompt_text,
        "stream": False,
        "options": {
            "num_ctx": 32768,
            "temperature": 0,
            "top_p": 1,
            "num_predict": MAX_OUTPUT_TOKENS,
            "repeat_penalty": 1.1
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=600)
    response.raise_for_status()

    data = response.json()

    print("prompt_eval_count:", data.get("prompt_eval_count"))
    print("eval_count:", data.get("eval_count"))

    return data["response"]

def build_output_filename(original_filename: str) -> str:
    return original_filename.replace("_original.md", ".txt")


selected_category = "direct"

category_original_dir = ORIGINAL_REFERENCES_DIR / selected_category
category_reference_dir = REFERENCES_DIR / selected_category

original_files = sorted(category_original_dir.glob("*.md"))

print(f"Category: {selected_category}")
print(f"Found {len(original_files)} original reference files")

for original_path in original_files:
    print(f"\nProcessing: {original_path.name}")

    source_text = load_text(original_path)
    prompt_text = build_reference_prompt(source_text)
    reference_text = generate_with_qwen(prompt_text)

    output_filename = build_output_filename(original_path.name)
    output_path = category_reference_dir / output_filename

    save_text(output_path, reference_text)

    print(f"Saved to: {output_path}")