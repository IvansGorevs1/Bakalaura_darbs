from pathlib import Path
import pymupdf4llm


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
PDFS_DIR = DATA_DIR / "pdfs_reports"
REPORTS_DIR = DATA_DIR / "reports"

CATEGORIES = ["quartely_report"]


def pdf_to_markdown(pdf_path: Path, md_path: Path) -> None:
    md_text = pymupdf4llm.to_markdown(str(pdf_path))

    with open(md_path, "w", encoding="utf-8") as file:
        file.write(md_text)


for category in CATEGORIES:
    pdf_category_dir = PDFS_DIR / category
    md_category_dir = REPORTS_DIR / category

    md_category_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(pdf_category_dir.glob("*.pdf"))

    print(f"\nCategory: {category}")
    print(f"Found {len(pdf_files)} PDF files")

    for pdf_path in pdf_files:
        md_filename = pdf_path.stem + ".md"
        md_path = md_category_dir / md_filename

        print(f"Converting: {pdf_path.name} -> {md_filename}")
        pdf_to_markdown(pdf_path, md_path)

