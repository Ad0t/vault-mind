import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest import PAGES_PATH, load_pages

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"


def build_sections(pages: list[dict]) -> list[dict]:
    """
    First pass:
    Structurally split pages into sections based on headings.

    If a page has no headings, the entire page becomes one section.
    """

    sections = []

    for page in pages:

        current_heading = "Untitled"
        current_level = 0
        current_text = []

        found_heading = False

        for block in page["blocks"]:

            if block["type"] == "heading":

                found_heading = True

                # Save previous section
                if current_text:
                    sections.append(
                        {
                            "page_number": page["page"],
                            "source_doc": page["source_doc"],
                            "section_title": current_heading,
                            "heading_level": current_level,
                            "text": "\n".join(current_text).strip(),
                        }
                    )

                current_heading = block["text"]
                current_level = block.get("level", 1)
                current_text = []

            else:
                current_text.append(block["text"])

        # Save last section
        if current_text:
            sections.append(
                {
                    "page_number": page["page"],
                    "source_doc": page["source_doc"],
                    "section_title": current_heading if found_heading else f"Page {page['page']}",
                    "heading_level": current_level,
                    "text": "\n".join(current_text).strip(),
                }
            )

    return sections


def chunk_sections(
    sections: list[dict],
    chunk_size: int = 400,
    chunk_overlap: int = 75,
) -> list[dict]:
    """
    Second pass:
    Semantically split each section into overlapping chunks.
    """

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []
    chunk_id = 1

    for section in sections:

        split_chunks = splitter.split_text(section["text"])

        for index, chunk_text in enumerate(split_chunks):

            chunks.append(
                {
                    "chunk_id": f"chunk_{chunk_id:05d}",
                    "page_number": section["page_number"],
                    "source_doc": section["source_doc"],
                    "section_title": section["section_title"],
                    "heading_level": section["heading_level"],
                    "chunk_index": index,
                    "text": chunk_text,
                }
            )

            chunk_id += 1

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