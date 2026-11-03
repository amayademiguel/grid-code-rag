"""Embed every chunk with a local embedding model (Ollama) and store the vectors.

With a few hundred chunks a NumPy matrix is enough; a vector database
(Chroma, FAISS, pgvector) only becomes necessary at a much larger scale.
"""
import json
import shutil

import numpy as np
import ollama

from .config import CHUNKS_FILE, EMBED_MODEL, EMBEDDINGS_FILE, INDEX_CHUNKS_FILE, INDEX_DIR


def embed(texts, batch_size=16):
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        vectors.extend(ollama.embed(model=EMBED_MODEL, input=batch)["embeddings"])
        print(f"  embedded {min(start + batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    matrix = np.array(vectors, dtype=np.float32)
    # Normalise once so cosine similarity becomes a dot product
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def main():
    chunks = [json.loads(line) for line in open(CHUNKS_FILE, encoding="utf-8")]
    print(f"Embedding {len(chunks)} chunks with {EMBED_MODEL} ...")
    matrix = embed([c["text"] for c in chunks])
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_FILE, matrix)
    shutil.copy(CHUNKS_FILE, INDEX_CHUNKS_FILE)  # keep vectors and chunks in sync
    print(f"Saved index: {matrix.shape[0]} vectors of dimension {matrix.shape[1]}")


if __name__ == "__main__":
    main()
