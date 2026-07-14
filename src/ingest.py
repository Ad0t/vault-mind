import json
import re
from collections import Counter
from pathlib import Path

import fitz
import pdfplumber

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


def table_to_markdown(table: list[list[str | None]]) -> str:
    """
    Convert a pdfplumber-extracted table (list of rows, each a list of
    cell strings/None) into a GitHub-flavored Markdown table.
    """

    def clean_cell(cell: str | None) -> str:
        if cell is None:
            return ""
        # Markdown cells can't contain raw newlines or pipes
        return cell.replace("\n", " ").replace("|", "\\|").strip()

    rows = [[clean_cell(cell) for cell in row] for row in table if row]

    if not rows:
        return ""

    header, *body_rows = rows
    col_count = len(header)

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * col_count) + " |",
    ]

    for row in body_rows:
        # Pad/truncate rows that don't match the header column count
        row = (row + [""] * col_count)[:col_count]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def bbox_overlap_ratio(block_bbox: tuple, table_bbox: tuple) -> float:
    """
    Return the fraction of block_bbox's area that overlaps table_bbox.
    Both bboxes are (x0, top, x1, bottom) in PDF point coordinates,
    which is the shared coordinate system used by both PyMuPDF's
    page.get_text("dict") blocks and pdfplumber's table.bbox.
    """

    bx0, btop, bx1, bbottom = block_bbox
    tx0, ttop, tx1, tbottom = table_bbox

    ix0 = max(bx0, tx0)
    itop = max(btop, ttop)
    ix1 = min(bx1, tx1)
    ibottom = min(bbottom, tbottom)

    if ix1 <= ix0 or ibottom <= itop:
        return 0.0

    intersection_area = (ix1 - ix0) * (ibottom - itop)
    block_area = (bx1 - bx0) * (bbottom - btop)

    if block_area <= 0:
        return 0.0

    return intersection_area / block_area


# A single bold *span* is not enough to tell a real heading (its own,
# isolated line -- e.g. "The features of database normalization are as
# follows:") apart from an inline bold lead-in phrase inside a paragraph
# (e.g. "Elimination of Data Redundancy: " followed on the very same
# line by plain continuation text "One of the main features of ...").
# Both patterns are bold and can end in a colon, so span-level checks
# alone can't distinguish them -- checking the *whole line* can: a real
# standalone heading has nothing but heading-styled spans on its line;
# a lead-in phrase shares its line with plain body text.
def is_heading(span: dict, body_font_size: float) -> bool:
    """
    Decide whether a single span, in isolation, looks heading-styled
    (large font, numbered prefix, or bold). This is a per-span signal
    only -- see `line_is_heading` for the whole-line decision that
    actually classifies a line as a heading or not.
    """

    text = span["text"].strip()

    if not text:
        return False

    font = span["font"].lower()
    size = span["size"]

    # Larger than normal text -- the strongest, least ambiguous signal.
    if size >= body_font_size + 2:
        return True

    # Numbered heading (e.g. "3.2 Functional Dependencies") -- reliable
    # regardless of font, since running body text doesn't start this way.
    if HEADING_PATTERN.match(text):
        return True

    # Bold on its own is ambiguous (see module note above) -- handled by
    # line_is_heading, not decided here.
    if "bold" in font:
        return True

    return False


def line_is_heading(spans: list[dict], body_font_size: float) -> bool:
    """
    A line is a heading only if *every* non-empty span on it is
    heading-styled. A line that mixes a bold lead-in phrase with plain
    continuation text (a span that fails the check) is running body
    prose, not a section heading, even though part of it is bold.
    """

    signals = [
        is_heading(span, body_font_size)
        for span in spans
        if span["text"].strip()
    ]

    if not signals:
        return False

    return all(signals)




# Minimum fraction of a text block's area that must fall inside a
# table's bounding box before that block is treated as "part of the
# table" and dropped from normal paragraph/heading extraction.
TABLE_OVERLAP_THRESHOLD = 0.5

# A candidate caption must sit within this many PDF points of the
# table's top or bottom edge to be considered "immediately adjacent".
CAPTION_MAX_GAP = 15

# A candidate caption must horizontally overlap this fraction of its
# own width with the table's horizontal span (guards against merging
# unrelated text in an adjacent column).
CAPTION_MIN_HORIZONTAL_OVERLAP = 0.3

# Captions are short by nature ("Table 3.1: Employee records"). This
# caps how much text can be swept up as a caption, so a full paragraph
# that merely happens to sit close to a table isn't merged into it.
CAPTION_MAX_LENGTH = 200


def horizontal_overlap_ratio(bbox_a: tuple, bbox_b: tuple) -> float:
    """
    Fraction of bbox_a's horizontal span that overlaps bbox_b's.
    """

    ax0, _, ax1, _ = bbox_a
    bx0, _, bx1, _ = bbox_b

    ix0 = max(ax0, bx0)
    ix1 = min(ax1, bx1)

    if ix1 <= ix0:
        return 0.0

    width_a = ax1 - ax0

    if width_a <= 0:
        return 0.0

    return (ix1 - ix0) / width_a


def find_caption_index(
    table_bbox: tuple,
    text_candidates: list[dict],
    consumed_indices: set,
) -> int | None:
    """
    Look for a short text block immediately above or below the table
    (within CAPTION_MAX_GAP points, with meaningful horizontal overlap)
    and return its index in text_candidates, or None if no candidate
    qualifies. Picks the closest qualifying candidate when more than
    one is found.
    """

    tx0, ttop, tx1, tbottom = table_bbox

    best_index = None
    best_gap = None

    for index, candidate in enumerate(text_candidates):

        if index in consumed_indices:
            continue

        bbox = candidate["bbox"]

        if not bbox:
            continue

        if len(candidate["text"]) > CAPTION_MAX_LENGTH:
            continue

        if horizontal_overlap_ratio(bbox, table_bbox) < CAPTION_MIN_HORIZONTAL_OVERLAP:
            continue

        _, ctop, _, cbottom = bbox

        gap_above = ttop - cbottom  # candidate sits above the table
        gap_below = ctop - tbottom  # candidate sits below the table

        if 0 <= gap_above <= CAPTION_MAX_GAP:
            gap = gap_above
        elif 0 <= gap_below <= CAPTION_MAX_GAP:
            gap = gap_below
        else:
            continue

        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_index = index

    return best_index


def extract_tables_for_page(plumber_page, page_num: int) -> list[dict]:
    """
    Detect tables on a single pdfplumber page and convert each one to
    a Markdown block, keeping the table's bbox around so overlapping
    text blocks from PyMuPDF can be filtered out later. Each table
    also carries its bbox, page number, and table_index as metadata --
    not used downstream yet, but preserved for future features like
    citation highlighting.
    """

    tables = []

    for table_index, table in enumerate(plumber_page.find_tables()):
        extracted = table.extract()
        markdown = table_to_markdown(extracted)

        if not markdown:
            continue

        tables.append(
            {
                "bbox": table.bbox,  # (x0, top, x1, bottom)
                "block": {
                    "type": "table",
                    "text": markdown,
                    "bbox": list(table.bbox),
                    "page": page_num,
                    "table_index": table_index,
                },
            }
        )

    return tables


def extract_pages(pdf_path: str | Path = PDF_PATH) -> list[dict]:
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    plumber_doc = pdfplumber.open(pdf_path)

    pages = []

    for page_num, (page, plumber_page) in enumerate(
        zip(doc, plumber_doc.pages), start=1
    ):

        page_dict = page.get_text("dict")
        body_font_size = get_body_font_size(page_dict)

        page_tables = extract_tables_for_page(plumber_page, page_num)
        table_bboxes = [t["bbox"] for t in page_tables]

        # First pass: turn each PyMuPDF block into one or more lightweight
        # candidates (bbox + merged text + dominant span). We classify at
        # the *line* level and only merge consecutive lines that share the
        # same heading/paragraph classification -- a PyMuPDF block can
        # contain a heading line immediately followed by body text (no
        # blank line between them), and concatenating the whole block's
        # text unconditionally used to swallow that body text into the
        # heading (or vice versa), losing it entirely as a paragraph.
        text_candidates = []

        for block in page_dict["blocks"]:

            if "lines" not in block:
                continue

            line_records = []

            for line in block["lines"]:

                line_text_parts = []
                line_largest_span = None

                for span in line["spans"]:
                    span_text = span["text"]

                    if span_text.strip():
                        line_text_parts.append(span_text)

                        if (
                            line_largest_span is None
                            or span["size"] > line_largest_span["size"]
                        ):
                            line_largest_span = span

                line_text = "".join(line_text_parts).strip()

                if not line_text or line_largest_span is None:
                    continue

                line_records.append(
                    {
                        "bbox": line.get("bbox"),
                        "text": line_text,
                        "largest_span": line_largest_span,
                        "is_heading": line_is_heading(line["spans"], body_font_size),
                    }
                )

            def flush_group(group: dict | None) -> None:
                if group is None:
                    return

                text_candidates.append(
                    {
                        "bbox": group["bbox"],
                        "text": " ".join(group["text_parts"]).strip(),
                        "largest_span": group["largest_span"],
                        "is_heading": group["is_heading"],
                    }
                )

            current_group = None

            for record in line_records:

                if (
                    current_group is not None
                    and current_group["is_heading"] == record["is_heading"]
                ):
                    current_group["text_parts"].append(record["text"])

                    if record["largest_span"]["size"] > current_group["largest_span"]["size"]:
                        current_group["largest_span"] = record["largest_span"]

                    bbox_a = current_group["bbox"]
                    bbox_b = record["bbox"]

                    if bbox_a and bbox_b:
                        current_group["bbox"] = (
                            min(bbox_a[0], bbox_b[0]),
                            min(bbox_a[1], bbox_b[1]),
                            max(bbox_a[2], bbox_b[2]),
                            max(bbox_a[3], bbox_b[3]),
                        )
                    elif bbox_b:
                        current_group["bbox"] = bbox_b

                else:
                    flush_group(current_group)

                    current_group = {
                        "is_heading": record["is_heading"],
                        "text_parts": [record["text"]],
                        "largest_span": record["largest_span"],
                        "bbox": record["bbox"],
                    }

            flush_group(current_group)

        consumed_indices = set()

        # Mark candidates that mostly fall inside a table's bbox as
        # consumed -- they're already captured by the table's own
        # Markdown block, so keeping them as paragraphs would
        # duplicate the table's contents.
        for index, candidate in enumerate(text_candidates):
            bbox = candidate["bbox"]

            if bbox and any(
                bbox_overlap_ratio(bbox, table_bbox) >= TABLE_OVERLAP_THRESHOLD
                for table_bbox in table_bboxes
            ):
                consumed_indices.add(index)

        # Merge captions that sit immediately above/below their table
        # into that table's block, and mark them consumed so they
        # don't also show up as a separate paragraph.
        for table in page_tables:
            caption_index = find_caption_index(
                table["bbox"], text_candidates, consumed_indices
            )

            if caption_index is None:
                continue

            caption_text = text_candidates[caption_index]["text"]
            table["block"]["caption"] = caption_text
            table["block"]["text"] = caption_text + "\n\n" + table["block"]["text"]
            consumed_indices.add(caption_index)

        # Second pass: build the final heading/paragraph blocks from
        # whatever candidates are left, then merge them with the
        # table blocks and sort everything by vertical position so
        # the page's original reading order is preserved -- tables no
        # longer get pushed to the end of the page.
        positioned_blocks = []

        for index, candidate in enumerate(text_candidates):

            if index in consumed_indices:
                continue

            largest_span = candidate["largest_span"]
            text = candidate["text"]
            heading = candidate["is_heading"]

            block_data = {
                "type": "heading" if heading else "paragraph",
                "text": text,
                "font": largest_span["font"],
                "font_size": largest_span["size"],
            }

            if heading:
                block_data["level"] = heading_level(text)

            bbox = candidate["bbox"]
            sort_key = bbox[1] if bbox else 0
            positioned_blocks.append((sort_key, block_data))

        for table in page_tables:
            sort_key = table["bbox"][1]
            positioned_blocks.append((sort_key, table["block"]))

        positioned_blocks.sort(key=lambda item: item[0])

        page_blocks = [block for _, block in positioned_blocks]

        pages.append(
            {
                "page": page_num,
                "source_doc": pdf_path.name,
                "blocks": page_blocks,
            }
        )

    doc.close()
    plumber_doc.close()
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