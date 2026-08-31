from pathlib import Path
import pandas as pd
import tiktoken


PROJECT_ROOT = Path(__file__).resolve().parent.parent


REFERENCES_DIR = PROJECT_ROOT / "data" / "references_txt"

# Annual references
ANNUAL_DIR = REFERENCES_DIR / "annual_report_reference"


QUARTERLY_DIR_VARIANTS = [
    REFERENCES_DIR / "quarterly_report_reference",
]


OUTPUT_DIR = PROJECT_ROOT / "data" / "report_statistics"

DETAILED_OUTPUT = OUTPUT_DIR / "reference_token_statistics.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "reference_token_summary.csv"


encoding = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(encoding.encode(text))



def find_quarterly_dir() -> Path:


    for folder in QUARTERLY_DIR_VARIANTS:
        if folder.exists():
            return folder

    raise FileNotFoundError(
        "Quarterly references folder not found.\n"
        "Expected one of:\n"
        + "\n".join(str(folder) for folder in QUARTERLY_DIR_VARIANTS)
    )



def analyze_file(file_path: Path, reference_type: str) -> dict:

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    characters = len(text)

    characters_without_spaces = len(
        "".join(text.split())
    )

    words = len(text.split())

    tokens = count_tokens(text)

    return {
        "file": file_path.name,
        "type": reference_type,
        "characters": characters,
        "characters_without_spaces": characters_without_spaces,
        "words": words,
        "tokens": tokens,
        "tokens_per_word": round(
            tokens / words,
            3
        ) if words else 0,
        "file_size_kb": round(
            file_path.stat().st_size / 1024,
            2
        )
    }


def analyze_folder(folder: Path, reference_type: str) -> list:

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder}"
        )

    files = sorted(folder.glob("*.txt"))

    print(
        f"{reference_type}: found {len(files)} TXT references"
    )

    results = []

    for i, file_path in enumerate(files, start=1):

        data = analyze_file(
            file_path,
            reference_type
        )

        results.append(data)

        print(
            f"[{i:02d}/{len(files):02d}] "
            f"{file_path.name}: "
            f"{data['words']:,} words, "
            f"{data['tokens']:,} tokens"
        )

    return results


def create_summary(df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for reference_type in ["Annual", "Quarterly"]:

        subset = df[
            df["type"] == reference_type
        ]

        if subset.empty:
            continue

        total_words = subset["words"].sum()
        total_tokens = subset["tokens"].sum()

        rows.append({
            "type": reference_type,

            "references":
                len(subset),

            "total_words":
                int(total_words),

            "mean_words":
                round(subset["words"].mean(), 2),

            "median_words":
                round(subset["words"].median(), 2),

            "min_words":
                int(subset["words"].min()),

            "max_words":
                int(subset["words"].max()),

            "total_tokens":
                int(total_tokens),

            "mean_tokens":
                round(subset["tokens"].mean(), 2),

            "median_tokens":
                round(subset["tokens"].median(), 2),

            "min_tokens":
                int(subset["tokens"].min()),

            "max_tokens":
                int(subset["tokens"].max()),

            "std_tokens":
                round(subset["tokens"].std(), 2),

            "mean_tokens_per_word":
                round(
                    total_tokens / total_words,
                    3
                ) if total_words else 0
        })



    total_words = df["words"].sum()
    total_tokens = df["tokens"].sum()

    rows.append({
        "type": "All",

        "references":
            len(df),

        "total_words":
            int(total_words),

        "mean_words":
            round(df["words"].mean(), 2),

        "median_words":
            round(df["words"].median(), 2),

        "min_words":
            int(df["words"].min()),

        "max_words":
            int(df["words"].max()),

        "total_tokens":
            int(total_tokens),

        "mean_tokens":
            round(df["tokens"].mean(), 2),

        "median_tokens":
            round(df["tokens"].median(), 2),

        "min_tokens":
            int(df["tokens"].min()),

        "max_tokens":
            int(df["tokens"].max()),

        "std_tokens":
            round(df["tokens"].std(), 2),

        "mean_tokens_per_word":
            round(
                total_tokens / total_words,
                3
            ) if total_words else 0
    })

    return pd.DataFrame(rows)


def main():

    print("=" * 70)
    print("REFERENCE TOKEN STATISTICS")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nReferences directory:")
    print(REFERENCES_DIR)


    if not REFERENCES_DIR.exists():
        raise FileNotFoundError(
            f"References directory not found:\n{REFERENCES_DIR}"
        )

    if not ANNUAL_DIR.exists():
        raise FileNotFoundError(
            f"Annual references folder not found:\n{ANNUAL_DIR}"
        )

    quarterly_dir = find_quarterly_dir()


    print("\nAnnual references:")
    print(ANNUAL_DIR)

    print("\nQuarterly references:")
    print(quarterly_dir)



    print("\n" + "=" * 70)
    print("ANALYZING REFERENCES")
    print("=" * 70 + "\n")

    results = []


    # Annual
    results.extend(
        analyze_folder(
            ANNUAL_DIR,
            "Annual"
        )
    )

    print()


    # Quarterly
    results.extend(
        analyze_folder(
            quarterly_dir,
            "Quarterly"
        )
    )


    if not results:
        print("No TXT references found.")
        return


    df = pd.DataFrame(results)

    df = df.sort_values(
        by=["type", "file"]
    ).reset_index(drop=True)



    annual_count = len(
        df[df["type"] == "Annual"]
    )

    quarterly_count = len(
        df[df["type"] == "Quarterly"]
    )

    print("\n" + "=" * 70)
    print("DATASET CHECK")
    print("=" * 70)

    print(
        f"Annual references:    {annual_count}"
    )

    print(
        f"Quarterly references: {quarterly_count}"
    )

    print(
        f"Total references:     {len(df)}"
    )


    if annual_count != 25:
        print(
            f"WARNING: expected 25 annual references, "
            f"but found {annual_count}."
        )


    if quarterly_count != 25:
        print(
            f"WARNING: expected 25 quarterly references, "
            f"but found {quarterly_count}."
        )


    summary = create_summary(df)


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    df.to_csv(
        DETAILED_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )


    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )


    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)


    for _, row in summary.iterrows():

        print(f"\n{row['type']}")
        print("-" * 40)

        print(
            f"References:            "
            f"{int(row['references']):,}"
        )

        print(
            f"Total words:           "
            f"{int(row['total_words']):,}"
        )

        print(
            f"Mean words/reference:  "
            f"{row['mean_words']:,.0f}"
        )

        print(
            f"Median words:          "
            f"{row['median_words']:,.0f}"
        )

        print(
            f"Minimum words:         "
            f"{int(row['min_words']):,}"
        )

        print(
            f"Maximum words:         "
            f"{int(row['max_words']):,}"
        )

        print(
            f"Total tokens:          "
            f"{int(row['total_tokens']):,}"
        )

        print(
            f"Mean tokens/reference: "
            f"{row['mean_tokens']:,.0f}"
        )

        print(
            f"Median tokens:         "
            f"{row['median_tokens']:,.0f}"
        )

        print(
            f"Minimum tokens:        "
            f"{int(row['min_tokens']):,}"
        )

        print(
            f"Maximum tokens:        "
            f"{int(row['max_tokens']):,}"
        )

        print(
            f"Standard deviation:    "
            f"{row['std_tokens']:,.0f}"
        )

        print(
            f"Tokens per word:       "
            f"{row['mean_tokens_per_word']:.3f}"
        )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"\nDetailed statistics:\n"
        f"{DETAILED_OUTPUT}"
    )

    print(
        f"\nSummary statistics:\n"
        f"{SUMMARY_OUTPUT}"
    )


if __name__ == "__main__":
    main()