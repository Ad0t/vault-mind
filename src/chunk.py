import hashlib
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest import PAGES_PATH, load_pages

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"


def _section_title(heading: str, found_heading: bool, start_page: int, end_page: int) -> str:
    if found_heading:
        return heading

    if start_page == end_page:
        return f"Page {start_page}"

    return f"Pages {start_page}-{end_page}"


def build_sections(pages: list[dict]) -> list[dict]:
    """
    First pass:
    Structurally split the document into sections based on headings.

    Unlike a per-page split, a section now spans pages: once a heading
    is encountered, everything that follows (paragraphs, across as many
    pages as needed) belongs to that section until the *next* heading
    is encountered -- not until the page ends. A page with no heading
    on it simply continues the previous section.

    Tables are never merged into surrounding paragraph text. Each table
    becomes its own standalone section element, positioned between
    whatever text came before and after it.
    """

    sections = []

    current_heading = "Untitled"
    current_level = 0
    current_text: list[str] = []
    current_start_page = None
    current_end_page = None
    found_heading = False

    def flush_text_section(source_doc: str) -> None:
        if not current_text:
            return

        sections.append(
            {
                "type": "text",
                "start_page": current_start_page,
                "end_page": current_end_page,
                "source_doc": source_doc,
                "section_title": _section_title(
                    current_heading, found_heading, current_start_page, current_end_page
                ),
                "heading_level": current_level,
                "text": "\n".join(current_text).strip(),
                # Table-only metadata -- kept as None here so every
                # section shares the same schema, whether or not it's
                # a table.
                "bbox": None,
                "table_index": None,
                "caption": None,
            }
        )

    source_doc_last = None

    for page in pages:

        page_num = page["page"]
        source_doc = page["source_doc"]
        source_doc_last = source_doc

        for block in page["blocks"]:

            if block["type"] == "heading":

                # Save whatever paragraph text was accumulated under
                # the previous heading before starting the new one.
                flush_text_section(source_doc)
                current_text = []
                current_start_page = None
                current_end_page = None

                found_heading = True
                current_heading = block["text"]
                current_level = block.get("level", 1)

            elif block["type"] == "table":

                # Flush accumulated paragraph text first so the table
                # lands in the correct position in the section order,
                # then emit the table as its own section element.
                flush_text_section(source_doc)
                current_text = []
                current_start_page = None
                current_end_page = None

                sections.append(
                    {
                        "type": "table",
                        "start_page": page_num,
                        "end_page": page_num,
                        "source_doc": source_doc,
                        "section_title": current_heading if found_heading else f"Page {page_num}",
                        "heading_level": current_level,
                        "text": block["text"],
                        # Not used downstream yet, but preserved for
                        # future features (e.g. citation highlighting
                        # back to the original table on the page).
                        "bbox": block.get("bbox"),
                        "table_index": block.get("table_index"),
                        "caption": block.get("caption"),
                    }
                )

            else:  # paragraph

                if current_start_page is None:
                    current_start_page = page_num

                current_end_page = page_num
                current_text.append(block["text"])

    # Save trailing section, if any text is still pending after the
    # last page. Guarded so an empty `pages` list is a no-op.
    if source_doc_last is not None:
        flush_text_section(source_doc_last)

    return sections


def chunk_sections(
    sections: list[dict],
    chunk_size: int = 400,
    chunk_overlap: int = 75,
) -> list[dict]:
    """
    Second pass:
    Semantically split each section into overlapping chunks.

    Table sections are the exception: splitting a Markdown table would
    break its row/column structure, so a table section is always kept
    as a single, unsplit chunk. Text sections go through the normal
    recursive splitter as before.
    """

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list = []
    # Per-source sequential counters so that chunk IDs are globally unique
    # even when different PDFs are processed in separate chunk_sections() calls
    # (as in the Gradio UI).  A 6-char MD5 prefix derived from the source
    # filename makes IDs deterministic: re-ingesting the same PDF always
    # produces the same IDs, so ChromaDB upsert overwrites correctly.
    source_counters: dict[str, int] = {}

    for section in sections:

        if section["type"] == "table":
            split_chunks = [section["text"]]
        else:
            split_chunks = splitter.split_text(section["text"])

        for index, chunk_text in enumerate(split_chunks):
            source = section.get("source_doc", "unknown")
            source_counters[source] = source_counters.get(source, 0) + 1
            seq      = source_counters[source]
            src_hash = hashlib.md5(source.encode()).hexdigest()[:6]

            chunks.append(
                {
                    "chunk_id": f"chunk_{src_hash}_{seq:05d}",
                    "type": section["type"],
                    "start_page": section["start_page"],
                    "end_page": section["end_page"],
                    "source_doc": section["source_doc"],
                    "section_title": section["section_title"],
                    "heading_level": section["heading_level"],
                    "chunk_index": index,
                    "text": chunk_text,
                    "bbox": section.get("bbox"),
                    "table_index": section.get("table_index"),
                    "caption": section.get("caption"),
                }
            )

    return chunks


def save_chunks(
    chunks: list[dict],
    output_path: str | Path = CHUNKS_PATH,
) -> Path:

    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            chunks,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


def load_chunks(
    input_path: str | Path = CHUNKS_PATH,
) -> list[dict]:

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"No chunks found at {input_path}. Run chunk.py first."
        )

    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":

    pages = load_pages(PAGES_PATH)

    sections = build_sections(pages)

    chunks = chunk_sections(sections)

    save_chunks(chunks)

    print(f"Pages loaded      : {len(pages)}")
    print(f"Sections created  : {len(sections)}")
    print(f"Chunks created    : {len(chunks)}")
    print(f"Saved chunks to   : {CHUNKS_PATH}")

    if chunks:
        print("\nFirst chunk:\n")
        print(json.dumps(chunks[0], indent=2, ensure_ascii=False))