"""
PHASE 3: Generate (full RAG)
Goal: send the retrieved chunks + question to the LLM and get a real answer.
Needs Ollama running with:  ollama pull llama3.2
Run:  python phase3_generate.py
"""

import ollama
from sentence_transformers import SentenceTransformer
from phase1_chunk import load_text, chunk_text, DATA_FILE, CHUNK_SIZE, CHUNK_OVERLAP
from phase2_retrieve import embed, retrieve, EMBED_MODEL, TOP_K

LLM_MODEL = "llama3.2"


def generate(question, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question: {question}
Answer:"""
    response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


if __name__ == "__main__":
    chunks = chunk_text(load_text(DATA_FILE), CHUNK_SIZE, CHUNK_OVERLAP)
    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)
    chunk_vectors = embed(model, chunks)
    print(f"Ready! {len(chunks)} chunks indexed. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        results = retrieve(model, question, chunks, chunk_vectors, TOP_K)
        print("\n[Retrieved chunks]")
        for i, (chunk, score) in enumerate(results, 1):
            print(f"  {i}. score={score:.3f} | {chunk[:70].replace(chr(10), ' ')}...")

        try:
            answer = generate(question, [c for c, _ in results])
        except Exception as e:
            print(f"\n[LLM error] {e}\nIs Ollama running? Did you run: ollama pull llama3.2 ?\n")
            continue
        print(f"\nAssistant: {answer}\n")