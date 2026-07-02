import json
from pathlib import Path

import fitz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "raw" / "Database Management Systems.pdf"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PAGES_PATH = PROCESSED_DIR / "pages.json"


def extract_pages(pdf_path: str | Path = PDF_PATH) -> list[dict]:
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)

    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages.append({"page": page_num, "text": text})

    doc.close()
    return pages


def save_pages(pages: list[dict], output_path: str | Path = PAGES_PATH) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(pages, handle, indent=2, ensure_ascii=False)
    return output_path


def load_pages(input_path: str | Path = PAGES_PATH) -> list[dict]:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"No extracted pages found at {input_path}. Run ingest.py first."
        )

    with input_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    pages = extract_pages(PDF_PATH)
    save_pages(pages, PAGES_PATH)
    print(f"Pages extracted: {len(pages)}")
    print(f"Saved pages to: {PAGES_PATH}")
    print("\nFirst page preview:\n")
    print(pages[0]["text"][:500])