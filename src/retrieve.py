import argparse
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / "vector_store"
COLLECTION_NAME = "dbms"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def retrieve(query: str, n_results: int = 3, chroma_path: str | Path = CHROMA_PATH, collection_name: str = COLLECTION_NAME):
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(collection_name)

    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode(query).tolist()

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )


def print_results(results: dict, query: str):
    print(f"\nQuery: {query}")
    print("Top results:\n")

    for index, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0]),
        start=1,
    ):
        print(f"Result {index}")
        print(f"Page: {meta['page']}")
        print(doc[:300])
        print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve the most relevant chunks from the vector store")
    parser.add_argument("--query", default="What is normalization?", help="Search query")
    parser.add_argument("--n-results", type=int, default=3, help="Number of results to return")
    args = parser.parse_args()

    results = retrieve(args.query, args.n_results)
    print_results(results, args.query)