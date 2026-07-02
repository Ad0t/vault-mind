from pathlib import Path

import chromadb

from chunk import CHUNKS_PATH, load_chunks
from embed import EMBEDDINGS_PATH, load_embeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / "vector_store"
COLLECTION_NAME = "dbms"


def store_chunks(chunks: list[dict], embeddings, chroma_path: str | Path = CHROMA_PATH, collection_name: str = COLLECTION_NAME):
    client = chromadb.PersistentClient(path=str(chroma_path))

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(collection_name)
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[{"page": chunk["page"]} for chunk in chunks],
    )
    return collection


if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_PATH)
    embeddings = load_embeddings(EMBEDDINGS_PATH)
    collection = store_chunks(chunks, embeddings)
    print(f"Stored {collection.count()} chunks in ChromaDB.")