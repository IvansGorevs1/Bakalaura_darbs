from pathlib import Path
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDFS_DIR = DATA_DIR / "pdfs"
REPORTS_DIR = DATA_DIR / "reports"

CATEGORIES = ["short", "medium", "long"]


def pdf_to_text(pdf_path: Path, txt_path: Path) -> None:
    reader = PdfReader(pdf_path)
    full_text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"

    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(full_text)


for category in CATEGORIES:
    pdf_category_dir = PDFS_DIR / category
    txt_category_dir = REPORTS_DIR / category

    txt_category_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(pdf_category_dir.glob("*.pdf"))

    print(f"\nCategory: {category}")
    print(f"Found {len(pdf_files)} PDF files")

    for pdf_path in pdf_files:
        txt_filename = pdf_path.stem + ".txt"
        txt_path = txt_category_dir / txt_filename

        print(f"Converting: {pdf_path.name} -> {txt_filename}")
        pdf_to_text(pdf_path, txt_path)

