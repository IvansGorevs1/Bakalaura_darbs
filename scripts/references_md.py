from pathlib import Path
import pymupdf
import re
import unicodedata


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

PDFS_REFERENCE_DIR = DATA_DIR / "pdfs_reference"
REFERENCES_TXT_DIR = DATA_DIR / "references_txt"

CATEGORIES = [
    "annual_report_reference",
    # "annual_report_reference",
]


def clean_text(text: str) -> str:
    """
    Minimal technical cleanup.
    Does not remove or rewrite document content.
    """


    text = unicodedata.normalize("NFC", text)


    text = text.replace("\u00a0", " ")

    text = text.replace("\u00ad", "")


    text = re.sub(r"[ \t]+\n", "\n", text)


    text = re.sub(r"[ \t]{2,}", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def pdf_to_text(pdf_path: Path, txt_path: Path) -> None:
    document = pymupdf.open(pdf_path)

    pages = []

    for page in document:
        page_text = page.get_text("text", sort=True)
        pages.append(page_text)

    document.close()

    full_text = "\n\n".join(pages)
    full_text = clean_text(full_text)

    txt_path.write_text(full_text, encoding="utf-8")


def build_output_filename(pdf_filename: str) -> str:
    pdf_stem = Path(pdf_filename).stem
    return f"{pdf_stem}_original.txt"


for category in CATEGORIES:
    pdf_category_dir = PDFS_REFERENCE_DIR / category
    txt_category_dir = REFERENCES_TXT_DIR / category

    txt_category_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_category_dir.glob("*.pdf"))

    print(f"\nCategory: {category}")
    print(f"Found {len(pdf_files)} PDF files")

    for pdf_path in pdf_files:
        txt_filename = build_output_filename(pdf_path.name)
        txt_path = txt_category_dir / txt_filename

        print(f"Converting: {pdf_path.name} -> {txt_filename}")

        try:
            pdf_to_text(pdf_path, txt_path)
        except Exception as e:
            print(f"ERROR: {pdf_path.name}: {e}")
