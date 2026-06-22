import json
import csv
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

JUDGE_RESULTS_DIR = DATA_DIR / "judge_results"
JUDGE_MAPPINGS_DIR = DATA_DIR / "judge_mappings"

selected_category = "direct"

results_category_dir = JUDGE_RESULTS_DIR / selected_category
mappings_category_dir = JUDGE_MAPPINGS_DIR / selected_category

output_csv_path = DATA_DIR / f"{selected_category}_judge_metrics.csv"


def load_json(file_path: Path) -> dict:
    return json.loads(file_path.read_text(encoding="utf-8"))


def load_mapping(file_path: Path) -> dict[str, dict]:
    mapping = {}
    with open(file_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            mapping[row["label"]] = row
    return mapping


rows = []

judge_files = sorted(results_category_dir.glob("*_judge.json"))

print(f"Found {len(judge_files)} judge result files")

for judge_path in judge_files:
    stem = judge_path.stem
    base_name = stem.removesuffix("_judge")

    mapping_path = mappings_category_dir / f"{base_name}_mapping.csv"

    if not mapping_path.exists():
        print(f"Skipping {judge_path.name} because mapping file is missing")
        continue

    scores = load_json(judge_path)
    mapping = load_mapping(mapping_path)

    parts = base_name.split("_")
    report_name = "_".join(parts[:2])   # report_1
    prompt_name = "_".join(parts[2:])   # zero_shot / few_shot / chain_of_event

    for label in ["A", "B", "C"]:
        if label not in scores or label not in mapping:
            continue

        faithfulness_score = scores[label]["faithfulness"]["score"]
        numerical_accuracy_score = scores[label]["numerical_accuracy"]["score"]
        coverage_score = scores[label]["coverage"]["score"]
        coherence_score = scores[label]["coherence"]["score"]

        average_score = round(
            (
                faithfulness_score +
                numerical_accuracy_score +
                coverage_score +
                coherence_score
            ) / 4,
            4
        )

        criteria = [
            ("faithfulness", faithfulness_score),
            ("numerical_accuracy", numerical_accuracy_score),
            ("coverage", coverage_score),
            ("coherence", coherence_score),
        ]

        for criterion_name, criterion_score in criteria:
            rows.append({
                "category": selected_category,
                "report": report_name,
                "prompt": prompt_name,
                "label": label,
                "model": mapping[label]["model"],
                "filename": mapping[label]["filename"],
                "criterion": criterion_name,
                "score": criterion_score,
                "average_score": average_score,
            })

fieldnames = [
    "category",
    "report",
    "prompt",
    "label",
    "model",
    "filename",
    "criterion",
    "score",
    "average_score",
]

with open(output_csv_path, "w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved detailed judge metrics to: {output_csv_path}")
print(f"Total rows: {len(rows)}")