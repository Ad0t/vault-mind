from pathlib import Path

import chromadb

from chunk import CHUNKS_PATH, load_chunks
from embed import EMBEDDINGS_PATH, load_embeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_PATH = PROJECT_ROOT / "vector_store"
COLLECTION_NAME = "dbms"


def store_chunks(
    chunks: list[dict],
    embeddings,
    chroma_path: str | Path = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
):
    """
    Store chunk text, embeddings, and metadata in a persistent ChromaDB collection.
    """

    client = chromadb.PersistentClient(path=str(chroma_path))

    # Recreate collection for a clean ingestion
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(collection_name)

    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "chunk_id": chunk["chunk_id"],
                "start_page": chunk["start_page"],
                "end_page": chunk["end_page"],
                "source_doc": chunk["source_doc"],
                "section_title": chunk["section_title"],
                "heading_level": chunk["heading_level"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in chunks
        ],
    )

    return collection



def delete_source(
    source_doc: str,
    chroma_path: str | Path = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """
    Delete all ChromaDB entries whose source_doc metadata matches source_doc.

    Called before upserting a re-uploaded PDF so that old chunks with
    now-stale IDs don't linger alongside the freshly generated ones.
    Safe to call even when the collection doesn't exist yet.
    """
    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        collection = client.get_collection(collection_name)
        collection.delete(where={"source_doc": source_doc})
    except Exception:
        pass  # collection doesn't exist yet — nothing to delete


def upsert_chunks(
    chunks: list[dict],
    embeddings,
    chroma_path: str | Path = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
):
    """
    Upsert chunks into an existing ChromaDB collection.

    Unlike store_chunks() which deletes and recreates the entire collection,
    this function uses ChromaDB's upsert() so:
      - Existing chunks from other sources are preserved.
      - Re-processing the same PDF overwrites its old chunks (same IDs).
      - New PDFs are simply appended.
    """
    client     = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(collection_name)

    collection.upsert(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "chunk_id":      chunk["chunk_id"],
                "start_page":    chunk["start_page"],
                "end_page":      chunk["end_page"],
                "source_doc":    chunk["source_doc"],
                "section_title": chunk["section_title"],
                "heading_level": chunk["heading_level"],
                "chunk_index":   chunk["chunk_index"],
            }
            for chunk in chunks
        ],
    )

    return collection


if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_PATH)
    embeddings = load_embeddings(EMBEDDINGS_PATH)

    collection = store_chunks(chunks, embeddings)
    print("Collection count:", collection.count())
    print(len(chunks))
    print(f"Stored {collection.count()} chunks in ChromaDB.")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Database location: {CHROMA_PATH}")