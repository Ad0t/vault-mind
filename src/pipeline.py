from ingest import extract_pages, save_pages
from chunk import chunk_pages, save_chunks
from embed import generate_embeddings, save_embeddings
from store import store_chunks
from retrieve import retrieve, print_results


def run_pipeline(query: str = "What is normalization?"):
    print("Step 1: Ingesting PDF...")
    pages = extract_pages()
    save_pages(pages)

    print("\nStep 2: Chunking text...")
    chunks = chunk_pages(pages)
    save_chunks(chunks)

    print("\nStep 3: Generating embeddings...")
    embeddings = generate_embeddings(chunks)
    save_embeddings(embeddings)

    print("\nStep 4: Storing vectors in ChromaDB...")
    collection = store_chunks(chunks, embeddings)
    print(f"Stored {collection.count()} chunks.")

    print("\nStep 5: Retrieving results...")
    results = retrieve(query)
    print_results(results, query)


if __name__ == "__main__":
    run_pipeline()