"""
PHASE 1: Load + Chunk
Goal: read data/notes.txt and split it into small pieces. No AI yet.
Run:  python phase1_chunk.py
"""

from pathlib import Path

DATA_FILE = Path(__file__).parent / "data" / "notes.txt"
CHUNK_SIZE = 500      # characters per chunk
CHUNK_OVERLAP = 100   # characters shared between neighbouring chunks


def load_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def chunk_text(text, size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        start += size - overlap   # move forward, but keep some overlap
    return chunks


if __name__ == "__main__":
    text = load_text(DATA_FILE)
    print(f"Document length: {len(text)} characters")

    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Number of chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"===== Chunk {i} ({len(chunk)} chars) =====")
        print(chunk)
        print()