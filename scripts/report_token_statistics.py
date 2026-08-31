from pathlib import Path
import pandas as pd
import tiktoken



PROJECT_ROOT = Path(__file__).resolve().parent.parent


REPORTS_DIR = PROJECT_ROOT / "data" / "reports"


ANNUAL_DIR = REPORTS_DIR / "annual_report"
QUARTERLY_DIR = REPORTS_DIR / "quartely_report"

OUTPUT_DIR = PROJECT_ROOT / "data" / "report_statistics"

DETAILED_OUTPUT = OUTPUT_DIR / "report_token_statistics.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "report_token_summary.csv"



encoding = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    """Count tokens in text."""
    return len(encoding.encode(text))


def analyze_file(file_path: Path, report_type: str) -> dict:
    """Calculate statistics for one Markdown report."""

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
        "type": report_type,
        "characters": characters,
        "characters_without_spaces": characters_without_spaces,
        "words": words,
        "tokens": tokens,
        "tokens_per_word": round(tokens / words, 3) if words else 0,
        "file_size_kb": round(file_path.stat().st_size / 1024, 2)
    }


def analyze_folder(folder: Path, report_type: str) -> list:
    """Analyze all Markdown files in a folder."""

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder}"
        )

    files = sorted(folder.glob("*.md"))

    print(
        f"{report_type}: found {len(files)} Markdown reports"
    )

    results = []

    for i, file_path in enumerate(files, start=1):

        data = analyze_file(
            file_path,
            report_type
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
    """Create summary statistics."""

    rows = []

    # Annual + Quarterly
    for report_type in ["Annual", "Quarterly"]:

        subset = df[
            df["type"] == report_type
        ]

        rows.append({
            "type": report_type,
            "reports": len(subset),

            "total_words":
                int(subset["words"].sum()),

            "mean_words":
                round(subset["words"].mean(), 2),

            "median_words":
                round(subset["words"].median(), 2),

            "total_tokens":
                int(subset["tokens"].sum()),

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
                round(subset["tokens"].sum()
                      / subset["words"].sum(), 3)
        })


    rows.append({
        "type": "All",
        "reports": len(df),

        "total_words":
            int(df["words"].sum()),

        "mean_words":
            round(df["words"].mean(), 2),

        "median_words":
            round(df["words"].median(), 2),

        "total_tokens":
            int(df["tokens"].sum()),

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
                df["tokens"].sum()
                / df["words"].sum(),
                3
            )
    })

    return pd.DataFrame(rows)



def main():

    print("=" * 70)
    print("FINANCIAL REPORT TOKEN STATISTICS")
    print("=" * 70)

    print(f"\nProject root:")
    print(PROJECT_ROOT)

    print(f"\nReports directory:")
    print(REPORTS_DIR)

    print("\n" + "=" * 70)
    print("ANALYZING REPORTS")
    print("=" * 70 + "\n")

    results = []

    results.extend(
        analyze_folder(
            ANNUAL_DIR,
            "Annual"
        )
    )

    print()

    results.extend(
        analyze_folder(
            QUARTERLY_DIR,
            "Quarterly"
        )
    )

    if not results:
        print("No Markdown reports found.")
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
        f"Annual reports:    {annual_count}"
    )

    print(
        f"Quarterly reports: {quarterly_count}"
    )

    print(
        f"Total reports:     {len(df)}"
    )

    if annual_count != 25:
        print(
            f"WARNING: expected 25 annual reports, "
            f"but found {annual_count}."
        )

    if quarterly_count != 25:
        print(
            f"WARNING: expected 25 quarterly reports, "
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
            f"Reports:              "
            f"{int(row['reports']):,}"
        )

        print(
            f"Total words:           "
            f"{int(row['total_words']):,}"
        )

        print(
            f"Mean words/report:     "
            f"{row['mean_words']:,.0f}"
        )

        print(
            f"Median words/report:   "
            f"{row['median_words']:,.0f}"
        )

        print(
            f"Total tokens:          "
            f"{int(row['total_tokens']):,}"
        )

        print(
            f"Mean tokens/report:    "
            f"{row['mean_tokens']:,.0f}"
        )

        print(
            f"Median tokens/report:  "
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