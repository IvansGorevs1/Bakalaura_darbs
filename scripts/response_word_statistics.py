from pathlib import Path
import re
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

OUTPUTS_DIR = DATA_DIR / "outputs"

CATEGORIES = {
    "annual": OUTPUTS_DIR / "annual_report",
    "quarterly": OUTPUTS_DIR / "quarterly_report",
}

RESULTS_DIR = DATA_DIR / "report_statistics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


TARGET_MIN_WORDS = 350
TARGET_MAX_WORDS = 450


PROMPTS = [
    "zero_shot",
    "few_shot",
    "chain_of_event",
]



def count_words(text: str) -> int:
    """
    Counts words in the model response.

    Numbers are also counted as words.
    Hyphenated words are counted as one word.
    Markdown symbols are ignored.
    """

    words = re.findall(r"\b[\w’'-]+\b", text, flags=re.UNICODE)
    return len(words)


def parse_filename(filename: str):
    """
    Example filenames:

    report_annual_1_zero_shot_gemini.txt
    report_annual_1_few_shot_nova.txt
    report_annual_1_chain_of_event_gemini.txt

    Returns:
        report
        prompt
        model
    """

    stem = Path(filename).stem

    for prompt in PROMPTS:
        marker = f"_{prompt}_"

        if marker in stem:
            report_part, model = stem.split(marker, 1)

            return report_part, prompt, model

    return stem, "unknown", "unknown"



def get_status(word_count: int) -> str:
    if word_count < TARGET_MIN_WORDS:
        return "below"
    elif word_count > TARGET_MAX_WORDS:
        return "above"
    else:
        return "within_range"



rows = []

for category, folder in CATEGORIES.items():

    if not folder.exists():
        print(f"WARNING: Folder not found: {folder}")
        continue

    print(f"\nProcessing: {category}")
    print(f"Folder: {folder}")

    files = sorted(folder.glob("*.txt"))

    print(f"Files found: {len(files)}")

    for file_path in files:

        try:
            text = file_path.read_text(
                encoding="utf-8",
                errors="replace"
            )

            word_count = count_words(text)

            report, prompt, model = parse_filename(file_path.name)

            rows.append({
                "category": category,
                "report": report,
                "prompt": prompt,
                "model": model,
                "file": file_path.name,
                "word_count": word_count,
                "target_min": TARGET_MIN_WORDS,
                "target_max": TARGET_MAX_WORDS,
                "difference_from_min": word_count - TARGET_MIN_WORDS,
                "difference_from_max": word_count - TARGET_MAX_WORDS,
                "status": get_status(word_count),
            })

        except Exception as e:
            print(f"ERROR reading {file_path.name}: {e}")



if not rows:
    print("\nNo response files found.")
    raise SystemExit


df = pd.DataFrame(rows)

df = df.sort_values(
    by=["category", "report", "model", "prompt"]
).reset_index(drop=True)



detailed_file = RESULTS_DIR / "model_response_word_statistics.csv"

df.to_csv(
    detailed_file,
    index=False,
    encoding="utf-8-sig"
)



summary = (
    df.groupby(
        ["category", "model", "prompt"],
        dropna=False
    )
    .agg(
        responses=("word_count", "count"),
        mean_words=("word_count", "mean"),
        median_words=("word_count", "median"),
        min_words=("word_count", "min"),
        max_words=("word_count", "max"),
        std_words=("word_count", "std"),
        within_range=(
            "status",
            lambda x: (x == "within_range").sum()
        ),
        below_range=(
            "status",
            lambda x: (x == "below").sum()
        ),
        above_range=(
            "status",
            lambda x: (x == "above").sum()
        ),
    )
    .reset_index()
)


summary["within_range_percent"] = (
    summary["within_range"]
    / summary["responses"]
    * 100
)

summary["below_range_percent"] = (
    summary["below_range"]
    / summary["responses"]
    * 100
)

summary["above_range_percent"] = (
    summary["above_range"]
    / summary["responses"]
    * 100
)


for column in [
    "mean_words",
    "median_words",
    "std_words",
    "within_range_percent",
    "below_range_percent",
    "above_range_percent",
]:
    summary[column] = summary[column].round(2)


summary_file = RESULTS_DIR / "model_response_word_statistics_summary.csv"

summary.to_csv(
    summary_file,
    index=False,
    encoding="utf-8-sig"
)



model_summary = (
    df.groupby("model")
    .agg(
        responses=("word_count", "count"),
        mean_words=("word_count", "mean"),
        median_words=("word_count", "median"),
        min_words=("word_count", "min"),
        max_words=("word_count", "max"),
        std_words=("word_count", "std"),
        within_range=(
            "status",
            lambda x: (x == "within_range").sum()
        ),
        below_range=(
            "status",
            lambda x: (x == "below").sum()
        ),
        above_range=(
            "status",
            lambda x: (x == "above").sum()
        ),
    )
    .reset_index()
)


model_summary["within_range_percent"] = (
    model_summary["within_range"]
    / model_summary["responses"]
    * 100
)

for column in [
    "mean_words",
    "median_words",
    "std_words",
    "within_range_percent",
]:
    model_summary[column] = model_summary[column].round(2)


model_summary_file = (
    RESULTS_DIR / "model_response_word_statistics_by_model.csv"
)

model_summary.to_csv(
    model_summary_file,
    index=False,
    encoding="utf-8-sig"
)



print("\n" + "=" * 80)
print("WORD COUNT ANALYSIS")
print("=" * 80)

print(
    f"\nTarget range: "
    f"{TARGET_MIN_WORDS}-{TARGET_MAX_WORDS} words"
)

print(f"Total responses: {len(df)}")

print(
    f"Average response length: "
    f"{df['word_count'].mean():.2f} words"
)

print(
    f"Median response length: "
    f"{df['word_count'].median():.2f} words"
)

print(
    f"Minimum response length: "
    f"{df['word_count'].min()} words"
)

print(
    f"Maximum response length: "
    f"{df['word_count'].max()} words"
)


within = (df["status"] == "within_range").sum()
below = (df["status"] == "below").sum()
above = (df["status"] == "above").sum()

print("\nRange compliance:")

print(
    f"Within {TARGET_MIN_WORDS}-{TARGET_MAX_WORDS}: "
    f"{within}/{len(df)} "
    f"({within / len(df) * 100:.2f}%)"
)

print(
    f"Below {TARGET_MIN_WORDS}: "
    f"{below}/{len(df)} "
    f"({below / len(df) * 100:.2f}%)"
)

print(
    f"Above {TARGET_MAX_WORDS}: "
    f"{above}/{len(df)} "
    f"({above / len(df) * 100:.2f}%)"
)


print("\n" + "=" * 80)
print("SUMMARY BY MODEL")
print("=" * 80)

print(
    model_summary.to_string(index=False)
)


print("\n" + "=" * 80)
print("FILES CREATED")
print("=" * 80)

print(f"Detailed: {detailed_file}")
print(f"Summary:  {summary_file}")
print(f"Models:   {model_summary_file}")