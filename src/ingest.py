import json
import re
from collections import Counter
from pathlib import Path

import fitz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "raw" / "Database Management Systems.pdf"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PAGES_PATH = PROCESSED_DIR / "pages.json"

# Matches headings like:
# 1
# 1.2
# 2.3.4
# 3.2 Functional Dependencies
HEADING_PATTERN = re.compile(r"^\d+(\.\d+)*\s+")


def get_body_font_size(page_dict: dict) -> float:
    """
    Estimate the normal paragraph font size by taking
    the most frequently occurring font size on the page.
    """
    sizes = []

    for block in page_dict["blocks"]:
        if "lines" not in block:
            continue

        for line in block["lines"]:
            for span in line["spans"]:
                if span["text"].strip():
                    sizes.append(round(span["size"]))

    if not sizes:
        return 12

    return Counter(sizes).most_common(1)[0][0]


def heading_level(text: str) -> int:
    """
    Infer heading hierarchy.

    1 Introduction      -> level 1
    2.3 Normal Forms    -> level 2
    3.2.1 BCNF          -> level 3
    """

    match = re.match(r"^(\d+(?:\.\d+)*)", text)

    if not match:
        return 1

    return match.group(1).count(".") + 1


def is_heading(span: dict, body_font_size: float) -> bool:
    """
    Decide whether a span is a heading.
    """

    text = span["text"].strip()

    if not text:
        return False

    font = span["font"].lower()
    size = span["size"]

    # Larger than normal text
    if size >= body_font_size + 2:
        return True

    # Bold text
    if "bold" in font:
        return True

    # Numbered heading
    if HEADING_PATTERN.match(text):
        return True

    return False


def extract_pages(pdf_path: str | Path = PDF_PATH) -> list[dict]:
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)

    pages = []

    for page_num, page in enumerate(doc, start=1):

        page_dict = page.get_text("dict")
        body_font_size = get_body_font_size(page_dict)

        page_blocks = []

        for block in page_dict["blocks"]:

            if "lines" not in block:
                continue

            text_parts = []
            largest_span = None

            for line in block["lines"]:
                for span in line["spans"]:

                    text = span["text"]

                    if text.strip():
                        text_parts.append(text)

                        if (
                            largest_span is None
                            or span["size"] > largest_span["size"]
                        ):
                            largest_span = span

            text = "".join(text_parts).strip()

            if not text:
                continue

            heading = is_heading(largest_span, body_font_size)

            block_data = {
                "type": "heading" if heading else "paragraph",
                "text": text,
                "font": largest_span["font"],
                "font_size": largest_span["size"],
            }

            if heading:
                block_data["level"] = heading_level(text)

            page_blocks.append(block_data)

        pages.append(
            {
                "page": page_num,
                "source_doc": pdf_path.name,
                "blocks": page_blocks,
            }
        )

    doc.close()
    return pages


def save_pages(
    pages: list[dict],
    output_path: str | Path = PAGES_PATH,
) -> Path:

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            pages,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


def load_pages(
    input_path: str | Path = PAGES_PATH,
) -> list[dict]:

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

    print("\nFirst page:\n")
    print(json.dumps(pages[0], indent=2, ensure_ascii=False))