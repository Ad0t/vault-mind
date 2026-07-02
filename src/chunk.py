import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest import PAGES_PATH, load_pages

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"


def chunk_pages(pages: list[dict], chunk_size: int = 500, chunk_overlap: int = 100) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []
    for page in pages:
        for chunk_text in splitter.split_text(page["text"]):
            chunks.append({"page": page["page"], "text": chunk_text})

    return chunks


def save_chunks(chunks: list[dict], output_path: str | Path = CHUNKS_PATH) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(chunks, handle, indent=2, ensure_ascii=False)
    return output_path


def load_chunks(input_path: str | Path = CHUNKS_PATH) -> list[dict]:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"No chunks found at {input_path}. Run chunk.py after ingest.py."
        )

    with input_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    pages = load_pages(PAGES_PATH)
    chunks = chunk_pages(pages)
    save_chunks(chunks, CHUNKS_PATH)
    print(f"Chunks created: {len(chunks)}")
    print(f"Saved chunks to: {CHUNKS_PATH}")