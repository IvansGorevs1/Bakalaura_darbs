import csv
from pathlib import Path
from collections import defaultdict

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

selected_category = "direct"

input_csv_path = DATA_DIR / f"{selected_category}_judge_metrics.csv"
output_csv_path = DATA_DIR / f"{selected_category}_judge_metrics_mean.csv"


def mean(values):
    return round(sum(values) / len(values), 4) if values else 0.0


rows = []

with open(input_csv_path, "r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

model_criterion_groups = defaultdict(list)
prompt_criterion_groups = defaultdict(list)
model_average_groups = defaultdict(list)
prompt_average_groups = defaultdict(list)

for row in rows:
    model = row["model"]
    prompt = row["prompt"]
    criterion = row["criterion"]
    score = float(row["score"])
    average_score = float(row["average_score"])

    model_criterion_groups[(model, criterion)].append(score)
    prompt_criterion_groups[(prompt, criterion)].append(score)

    # average_score repeats 4 times per summary in long format
    if criterion == "faithfulness":
        model_average_groups[model].append(average_score)
        prompt_average_groups[prompt].append(average_score)

mean_rows = []

for model_name in sorted(set(k[0] for k in model_criterion_groups.keys())):
    mean_rows.append({
        "category": selected_category,
        "group_type": "model",
        "group_name": model_name,
        "faithfulness_score_mean": mean(model_criterion_groups[(model_name, "faithfulness")]),
        "numerical_accuracy_score_mean": mean(model_criterion_groups[(model_name, "numerical_accuracy")]),
        "coverage_score_mean": mean(model_criterion_groups[(model_name, "coverage")]),
        "coherence_score_mean": mean(model_criterion_groups[(model_name, "coherence")]),
        "average_score_mean": mean(model_average_groups[model_name]),
    })

for prompt_name in sorted(set(k[0] for k in prompt_criterion_groups.keys())):
    mean_rows.append({
        "category": selected_category,
        "group_type": "prompt",
        "group_name": prompt_name,
        "faithfulness_score_mean": mean(prompt_criterion_groups[(prompt_name, "faithfulness")]),
        "numerical_accuracy_score_mean": mean(prompt_criterion_groups[(prompt_name, "numerical_accuracy")]),
        "coverage_score_mean": mean(prompt_criterion_groups[(prompt_name, "coverage")]),
        "coherence_score_mean": mean(prompt_criterion_groups[(prompt_name, "coherence")]),
        "average_score_mean": mean(prompt_average_groups[prompt_name]),
    })

fieldnames = [
    "category",
    "group_type",
    "group_name",
    "faithfulness_score_mean",
    "numerical_accuracy_score_mean",
    "coverage_score_mean",
    "coherence_score_mean",
    "average_score_mean",
]

with open(output_csv_path, "w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(mean_rows)

print(f"Saved mean judge metrics to: {output_csv_path}")
print(f"Total rows: {len(mean_rows)}")