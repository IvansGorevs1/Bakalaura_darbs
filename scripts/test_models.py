from google import genai
from google.genai import types
from openai import OpenAI
import requests
from pathlib import Path
import time



GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
OPENAI_MODEL = "gpt-5.4-nano"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
PROMPTS_DIR = DATA_DIR / "prompts"
REPORTS_DIR = DATA_DIR / "reports"
OUTPUTS_DIR = DATA_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_OUTPUT_TOKENS = 800

def load_prompt(prompt_filename: str) -> str:
    prompt_path = PROMPTS_DIR / prompt_filename

    with open(prompt_path, "r", encoding="utf-8") as file:
        return file.read().strip()
    
def load_report(category: str, report_filename: str) -> str:
    report_path = REPORTS_DIR / category / report_filename

    with open(report_path, "r", encoding="utf-8") as file:
        return file.read().strip()

def get_reports_in_category(category: str) -> list[str]:
    category_dir = REPORTS_DIR / category
    report_files = sorted([file.name for file in category_dir.glob("*.md")])
    return report_files


def build_full_prompt(prompt_text: str, report_text: str) -> str:
    return f"{prompt_text}\n\nFinancial report:\n\n{report_text}"

def save_result(model_name: str, category: str, report_filename: str, prompt_name: str, summary_text: str) -> None:
    category_output_dir = OUTPUTS_DIR / category
    category_output_dir.mkdir(parents=True, exist_ok=True)

    report_stem = Path(report_filename).stem
    prompt_stem = Path(prompt_name).stem

    output_filename = f"{report_stem}_{prompt_stem}_{model_name}.txt"
    output_path = category_output_dir / output_filename

    output_path.write_text(summary_text, encoding="utf-8")
    print(f"Saved result to: {output_path}")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def generate_with_gemini(prompt_text: str) -> str:
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt_text,
        config=types.GenerateContentConfig(
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=0,
            top_p=1,
            candidate_count=1,
        )
    )
    return response.text


def generate_with_openai(prompt_text: str) -> str:
    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        input=prompt_text,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return response.output_text


def generate_with_ollama(prompt_text: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt_text,
        "stream": False,
        "options": {
            "num_ctx": 32768,
            "num_predict": MAX_OUTPUT_TOKENS,
            "temperature": 0,
            "top_p": 1
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=1800)
    response.raise_for_status()

    data = response.json()

    print("prompt_eval_count:", data.get("prompt_eval_count"))
    print("eval_count:", data.get("eval_count"))

    return data["response"]

def generate_summary(model_name: str, prompt_text: str) -> str:
    if model_name == "gemini":
        return generate_with_gemini(prompt_text)
    elif model_name == "openai":
        return generate_with_openai(prompt_text)
    elif model_name == "ollama":
        return generate_with_ollama(prompt_text)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# selected_category = "medium"

# prompt_files = ["zero_shot.txt", "few_shot.txt", "chain_of_event.txt"]
# models = ["gemini", "openai", "ollama"]


# prompt_files = ["chain_of_event.txt"]
# models = ["gemini"]

# report_files = get_reports_in_category(selected_category)

# for prompt_name in prompt_files:
#     prompt_text = load_prompt(prompt_name)

#     for report_filename in report_files:
#         report_text = load_report(selected_category, report_filename)
#         full_prompt = build_full_prompt(prompt_text, report_text)

#         print(f"\nCategory: {selected_category} | Report: {report_filename} | Prompt: {prompt_name}")

#         for model_name in models:
#             print(f"Running model: {model_name}")

#             try:
#                 result = generate_summary(model_name, full_prompt)
#                 save_result(model_name, selected_category, report_filename, prompt_name, result)

#             except Exception as error:
#                 print(f"Error for {model_name} | {report_filename} | {prompt_name}: {error}")

#             if model_name == "gemini":
#                 time.sleep(12)

selected_category = "direct"

# prompt_files = ["zero_shot.txt", "few_shot.txt", "chain_of_event.txt"]
prompt_files = ["zero_shot.txt"]
models = ["ollama"]
report_files = ["report_1.md"]
# report_files = get_reports_in_category(selected_category)
for prompt_name in prompt_files:
    prompt_text = load_prompt(prompt_name)

    for report_filename in report_files:
        report_text = load_report(selected_category, report_filename)
        full_prompt = build_full_prompt(prompt_text, report_text)

        print(f"\nCategory: {selected_category} | Report: {report_filename} | Prompt: {prompt_name}")

        for model_name in models:
            print(f"Running model: {model_name}")

            try:
                result = generate_summary(model_name, full_prompt)
                save_result(model_name, selected_category, report_filename, prompt_name, result)

            except Exception as error:
                print(f"Error for {model_name} | {report_filename} | {prompt_name}: {error}")

            if model_name == "gemini":
                time.sleep(12)