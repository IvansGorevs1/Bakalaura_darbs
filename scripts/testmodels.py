from google import genai
from openai import OpenAI
import requests
from pathlib import Path
import time


GEMINI_MODEL = "gemini-2.5-flash"
OPENAI_MODEL = "gpt-5.4-nano"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROMPTS_DIR = DATA_DIR / "prompts"
REPORTS_DIR = DATA_DIR / "reports"
OUTPUTS_DIR = DATA_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

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
    report_files = sorted([file.name for file in category_dir.glob("*.txt")])
    return report_files


def build_full_prompt(prompt_text: str, report_text: str) -> str:
    return f"{prompt_text}\n\nFinancial report:\n\n{report_text}"

def save_result(model_name: str, category: str, report_filename: str, prompt_name: str, summary_text: str) -> None:
    category_output_dir = OUTPUTS_DIR / category
    category_output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{report_filename[:-4]}_{prompt_name[:-4]}_{model_name}.txt"
    output_path = category_output_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(summary_text)

    print(f"Saved result to: {output_path}")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def generate_with_gemini(prompt_text: str) -> str:
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt_text
    )
    return response.text


def generate_with_openai(prompt_text: str) -> str:
    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        input=prompt_text
    )
    return response.output_text


def generate_with_ollama(prompt_text: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt_text,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    data = response.json()
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


selected_category = "short"

prompt_files = ["zero_shot.txt", "few_shot.txt", "chain_of_event.txt"]
models = ["gemini", "openai", "ollama"]

report_files = get_reports_in_category(selected_category)

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