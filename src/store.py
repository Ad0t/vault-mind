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


if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_PATH)
    embeddings = load_embeddings(EMBEDDINGS_PATH)

    collection = store_chunks(chunks, embeddings)
    print("Collection count:", collection.count())
    print(len(chunks))
    print(f"Stored {collection.count()} chunks in ChromaDB.")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Database location: {CHROMA_PATH}")