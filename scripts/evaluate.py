from pathlib import Path
import pandas as pd
from rouge_score import rouge_scorer
from bert_score import score as bertscore_score

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

selected_category = "direct"

REFERENCES_DIR = DATA_DIR / "references" / selected_category
OUTPUTS_DIR = DATA_DIR / "outputs" / selected_category
RESULTS_DIR = DATA_DIR / "evaluation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# REPORTS = ["report_1", "report_2", "report_3", "report_4"]
PROMPTS = ["zero_shot", "few_shot", "chain_of_event"]
MODELS = ["gemini", "openai", "ollama"]


def get_reports_from_outputs() -> list:
    reports = set()

    for file_path in OUTPUTS_DIR.glob("report_*.txt"):
        file_name = file_path.stem

        for prompt_name in PROMPTS:
            for model_name in MODELS:
                suffix = f"_{prompt_name}_{model_name}"

                if file_name.endswith(suffix):
                    report_name = file_name.replace(suffix, "")
                    reports.add(report_name)

    return sorted(reports)


REPORTS = get_reports_from_outputs()


def load_text(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()


def compute_rouge(reference_text: str, generated_text: str) -> dict:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )
    scores = scorer.score(reference_text, generated_text)

    return {
        "rouge1_f1": scores["rouge1"].fmeasure,
        "rouge2_f1": scores["rouge2"].fmeasure,
        "rougeL_f1": scores["rougeL"].fmeasure,
    }


def compute_bertscore(reference_text: str, generated_text: str) -> dict:
    precision, recall, f1 = bertscore_score(
        [generated_text],
        [reference_text],
        lang="en",
        verbose=False
    )

    return {
        "bertscore_precision": float(precision[0]),
        "bertscore_recall": float(recall[0]),
        "bertscore_f1": float(f1[0]),
    }


results = []

for report_name in REPORTS:
    reference_number = report_name.replace("report_", "")
    reference_path = REFERENCES_DIR / f"reference_{reference_number}.txt"

    if not reference_path.exists():
        print(f"Missing reference: {reference_path}")
        continue

    reference_text = load_text(reference_path)

    for prompt_name in PROMPTS:
        for model_name in MODELS:
            output_path = OUTPUTS_DIR / f"{report_name}_{prompt_name}_{model_name}.txt"

            if not output_path.exists():
                print(f"Missing output: {output_path}")
                continue

            generated_text = load_text(output_path)

            rouge_scores = compute_rouge(reference_text, generated_text)
            bert_scores = compute_bertscore(reference_text, generated_text)

            row = {
                "report": report_name,
                "prompt": prompt_name,
                "model": model_name,
            }
            row.update(rouge_scores)
            row.update(bert_scores)

            results.append(row)

df = pd.DataFrame(results)

csv_path = RESULTS_DIR / f"{selected_category}_metrics.csv"
xlsx_path = RESULTS_DIR / f"{selected_category}_metrics.xlsx"

df.to_csv(csv_path, index=False, encoding="utf-8-sig")
df.to_excel(xlsx_path, index=False)

print("\nDetailed results:")
print(df)

mean_df = df.groupby(["model", "prompt"], as_index=False)[
    ["rouge1_f1", "rouge2_f1", "rougeL_f1", "bertscore_f1"]
].mean()

mean_csv_path = RESULTS_DIR / f"{selected_category}_metrics_mean.csv"
mean_xlsx_path = RESULTS_DIR / f"{selected_category}_metrics_mean.xlsx"

mean_df.to_csv(mean_csv_path, index=False, encoding="utf-8-sig")
mean_df.to_excel(mean_xlsx_path, index=False)

print("\nMean results:")
print(mean_df)

print(f"\nSaved detailed CSV: {csv_path}")
print(f"Saved detailed Excel: {xlsx_path}")
print(f"Saved mean CSV: {mean_csv_path}")
print(f"Saved mean Excel: {mean_xlsx_path}")