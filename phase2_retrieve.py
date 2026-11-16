"""
PHASE 2: Embed + Retrieve
Goal: turn chunks into vectors and find the chunks that best match a question.
Still no LLM — we only check that the RIGHT chunks are found.
Run:  python phase2_retrieve.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from phase1_chunk import load_text, chunk_text, DATA_FILE, CHUNK_SIZE, CHUNK_OVERLAP

EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3


def embed(model, texts):
    # normalize -> dot product = cosine similarity
    return model.encode(texts, normalize_embeddings=True)


def retrieve(model, question, chunks, chunk_vectors, k):
    q_vec = embed(model, [question])[0]
    scores = chunk_vectors @ q_vec               # similarity with every chunk
    best = np.argsort(scores)[::-1][:k]          # highest scores first
    return [(chunks[i], float(scores[i])) for i in best]


if __name__ == "__main__":
    chunks = chunk_text(load_text(DATA_FILE), CHUNK_SIZE, CHUNK_OVERLAP)

    print("Loading embedding model (first run downloads ~90 MB)...")
    model = SentenceTransformer(EMBED_MODEL)
    chunk_vectors = embed(model, chunks)
    print(f"Vector matrix shape: {chunk_vectors.shape}  (chunks x numbers)\n")

    while True:
        question = input("Question (or 'exit'): ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        for i, (chunk, score) in enumerate(retrieve(model, question, chunks, chunk_vectors, TOP_K), 1):
            print(f"\n--- Match {i} | score = {score:.3f} ---")
            print(chunk[:300] + ("..." if len(chunk) > 300 else ""))
        print()
