from pathlib import Path
import pandas as pd
import re

from rouge_score import rouge_scorer
from bert_score import BERTScorer


# "annual_report"
# "quarterly_report"

SELECTED_CATEGORY = "annual_report"



# ["openai"]
# ["gemini"]
# ["openai", "gemini"]

SELECTED_MODELS = [
    "gemini",
    "openai",
    "qwen36"
]


PROMPTS = [
    "zero_shot",
    "few_shot",
    "chain_of_event",
]




PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"

REFERENCES_ROOT = DATA_DIR / "references_txt"

# OUTPUTS_ROOT = DATA_DIR / "outputs"
OUTPUTS_ROOT = DATA_DIR / "outputs_reasoning_high"
RESULTS_DIR = DATA_DIR / "evaluation_results_reasoning"

# RESULTS_DIR = DATA_DIR / "evaluation_results"
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)



CATEGORY_CONFIG = {

    "annual_report": {
        "reference_folder":
            "annual_report_reference",
    },

    "quarterly_report": {
        "reference_folder":
            "quarterly_report_reference",
    },
}


if SELECTED_CATEGORY not in CATEGORY_CONFIG:

    raise ValueError(
        "SELECTED_CATEGORY must be "
        "'annual_report' or 'quarterly_report'"
    )


REFERENCES_DIR = (
    REFERENCES_ROOT
    / CATEGORY_CONFIG[
        SELECTED_CATEGORY
    ]["reference_folder"]
)

OUTPUTS_DIR = (
    OUTPUTS_ROOT
    / SELECTED_CATEGORY
)


VALID_MODELS = {
    "nova",
    "gemini",
    "qwen36",
    "openai",
}

for model in SELECTED_MODELS:

    if model not in VALID_MODELS:

        raise ValueError(
            f"Unknown model: {model}"
        )



def natural_sort_key(value: str):

    return [
        int(part)
        if part.isdigit()
        else part.lower()

        for part in re.split(
            r"(\d+)",
            value
        )
    ]


def load_text(
    file_path: Path
) -> str:

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read().strip()



def clean_text_for_metrics(
    text: str
) -> str:

    text = text.replace(
        "**",
        ""
    )

    text = text.replace(
        "__",
        ""
    )

    text = re.sub(
        r"(?m)^\s*#{1,6}\s*",
        "",
        text
    )


    text = re.sub(
        r"(?m)^\s*[-*+]\s+",
        "",
        text
    )

    # Markdown code markers
    text = text.replace(
        "```",
        ""
    )

    text = text.replace(
        "`",
        ""
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def get_reports_from_outputs() -> list[str]:

    reports = set()

    for file_path in OUTPUTS_DIR.glob(
        "*.txt"
    ):

        file_name = file_path.stem

        for prompt_name in PROMPTS:

            for model_name in SELECTED_MODELS:

                suffix = (
                    f"_{prompt_name}_"
                    f"{model_name}"
                )

                if file_name.endswith(
                    suffix
                ):

                    report_name = (
                        file_name[
                            :-len(suffix)
                        ]
                    )

                    reports.add(
                        report_name
                    )

    return sorted(
        reports,
        key=natural_sort_key
    )


def get_reference_path(
    report_name: str
) -> Path:



    reference_name = (
        report_name.replace(
            "report_",
            "reference_",
            1
        )
    )

    return (
        REFERENCES_DIR
        / f"{reference_name}_original.txt"
    )

# ROUGE

rouge_scorer_object = (
    rouge_scorer.RougeScorer(
        [
            "rouge1",
            "rouge2",
            "rougeL",
        ],
        use_stemmer=True
    )
)


def compute_rouge(
    reference_text: str,
    generated_text: str
) -> dict:

    scores = (
        rouge_scorer_object.score(
            reference_text,
            generated_text
        )
    )

    return {

        "rouge1_precision":
            scores["rouge1"].precision,

        "rouge1_recall":
            scores["rouge1"].recall,

        "rouge1_f1":
            scores["rouge1"].fmeasure,

        "rouge2_precision":
            scores["rouge2"].precision,

        "rouge2_recall":
            scores["rouge2"].recall,

        "rouge2_f1":
            scores["rouge2"].fmeasure,

        "rougeL_precision":
            scores["rougeL"].precision,

        "rougeL_recall":
            scores["rougeL"].recall,

        "rougeL_f1":
            scores["rougeL"].fmeasure,
    }


# BERTSCORE

print(
    "\nLoading BERTScore model..."
)

bert_scorer = BERTScorer(
    lang="en",
    rescale_with_baseline=False
)

print(
    "BERTScore model loaded."
)


def compute_bertscore(
    reference_text: str,
    generated_text: str
) -> dict:

    precision, recall, f1 = (
        bert_scorer.score(
            [generated_text],
            [reference_text]
        )
    )

    return {

        "bertscore_precision":
            float(precision[0]),

        "bertscore_recall":
            float(recall[0]),

        "bertscore_f1":
            float(f1[0]),
    }



REPORTS = get_reports_from_outputs()


print("\n" + "=" * 70)
print("EVALUATION SETTINGS")
print("=" * 70)

print(
    f"Category: "
    f"{SELECTED_CATEGORY}"
)

print(
    f"Models: "
    f"{SELECTED_MODELS}"
)

print(
    f"Prompts: "
    f"{PROMPTS}"
)

print(
    f"Reports found: "
    f"{len(REPORTS)}"
)

print(
    f"References directory: "
    f"{REFERENCES_DIR}"
)

print(
    f"Outputs directory: "
    f"{OUTPUTS_DIR}"
)

print("=" * 70)



results = []


for report_index, report_name in enumerate(
    REPORTS,
    start=1
):

    print(
        f"\n[{report_index}/{len(REPORTS)}] "
        f"Report: {report_name}"
    )

    reference_path = (
        get_reference_path(
            report_name
        )
    )

    if not reference_path.exists():

        print(
            f"Missing reference: "
            f"{reference_path}"
        )

        continue


    reference_original = load_text(
        reference_path
    )

    reference_text = (
        clean_text_for_metrics(
            reference_original
        )
    )


    for prompt_name in PROMPTS:

        for model_name in SELECTED_MODELS:

            output_path = (

                OUTPUTS_DIR

                / (
                    f"{report_name}_"
                    f"{prompt_name}_"
                    f"{model_name}.txt"
                )
            )


            if not output_path.exists():

                print(
                    f"Missing output: "
                    f"{output_path.name}"
                )

                continue


            print(
                f"  Evaluating | "
                f"{model_name} | "
                f"{prompt_name}"
            )


            generated_original = (
                load_text(
                    output_path
                )
            )

            generated_text = (
                clean_text_for_metrics(
                    generated_original
                )
            )


            rouge_scores = (
                compute_rouge(
                    reference_text,
                    generated_text
                )
            )



            bert_scores = (
                compute_bertscore(
                    reference_text,
                    generated_text
                )
            )



            reference_words = len(
                reference_text.split()
            )

            generated_words = len(
                generated_text.split()
            )


            if reference_words > 0:

                length_ratio = (
                    generated_words
                    / reference_words
                )

            else:

                length_ratio = 0


            row = {

                "report":
                    report_name,

                "prompt":
                    prompt_name,

                "model":
                    model_name,

                "reference_words":
                    reference_words,

                "generated_words":
                    generated_words,

                "length_ratio":
                    length_ratio,
            }


            row.update(
                rouge_scores
            )

            row.update(
                bert_scores
            )

            results.append(
                row
            )



if not results:

    raise RuntimeError(
        "No evaluation results were generated. "
        "Check file paths and filenames."
    )


df = pd.DataFrame(
    results
)



model_tag = "_".join(
    SELECTED_MODELS
)


csv_path = (

    RESULTS_DIR

    / (
        f"{SELECTED_CATEGORY}_"
        f"{model_tag}_metrics.csv"
    )
)


xlsx_path = (

    RESULTS_DIR

    / (
        f"{SELECTED_CATEGORY}_"
        f"{model_tag}_metrics.xlsx"
    )
)


df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig"
)


df.to_excel(
    xlsx_path,
    index=False
)



mean_columns = [

    "rouge1_precision",
    "rouge1_recall",
    "rouge1_f1",

    "rouge2_precision",
    "rouge2_recall",
    "rouge2_f1",

    "rougeL_precision",
    "rougeL_recall",
    "rougeL_f1",

    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",

    "generated_words",
    "length_ratio",
]


mean_df = (

    df.groupby(
        [
            "model",
            "prompt"
        ],
        as_index=False
    )[mean_columns]

    .mean()
)


mean_csv_path = (

    RESULTS_DIR

    / (
        f"{SELECTED_CATEGORY}_"
        f"{model_tag}_metrics_mean.csv"
    )
)


mean_xlsx_path = (

    RESULTS_DIR

    / (
        f"{SELECTED_CATEGORY}_"
        f"{model_tag}_metrics_mean.xlsx"
    )
)


mean_df.to_csv(
    mean_csv_path,
    index=False,
    encoding="utf-8-sig"
)


mean_df.to_excel(
    mean_xlsx_path,
    index=False
)


print("\n" + "=" * 70)
print("DETAILED RESULTS")
print("=" * 70)

print(
    df[
        [
            "report",
            "model",
            "prompt",
            "rouge1_f1",
            "rouge2_f1",
            "rougeL_f1",
            "bertscore_f1",
            "generated_words",
            "length_ratio",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("MEAN RESULTS")
print("=" * 70)

print(
    mean_df[
        [
            "model",
            "prompt",
            "rouge1_f1",
            "rouge2_f1",
            "rougeL_f1",
            "bertscore_f1",
            "generated_words",
            "length_ratio",
        ]
    ].to_string(
        index=False
    )
)


print("\nSaved files:")

print(
    f"Detailed CSV: "
    f"{csv_path}"
)

print(
    f"Detailed Excel: "
    f"{xlsx_path}"
)

print(
    f"Mean CSV: "
    f"{mean_csv_path}"
)

print(
    f"Mean Excel: "
    f"{mean_xlsx_path}"
)