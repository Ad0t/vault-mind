from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from chunk import CHUNKS_PATH, load_chunks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def generate_embeddings(chunks: list[dict], model_name: str = MODEL_NAME) -> np.ndarray:
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return embeddings


def save_embeddings(embeddings: np.ndarray, output_path: str | Path = EMBEDDINGS_PATH) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    return output_path


def load_embeddings(input_path: str | Path = EMBEDDINGS_PATH) -> np.ndarray:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"No embeddings found at {input_path}. Run embed.py after chunk.py."
        )
    return np.load(input_path)


if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_PATH)
    embeddings = generate_embeddings(chunks)
    save_embeddings(embeddings, EMBEDDINGS_PATH)
    print(f"Generated {len(embeddings)} embeddings.")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"Saved embeddings to: {EMBEDDINGS_PATH}")