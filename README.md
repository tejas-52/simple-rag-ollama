# Simple RAG Pipeline (Local, from Scratch)

A minimal **Retrieval-Augmented Generation (RAG)** project built step by step in three phases, so you can see how each part works. Everything runs locally: embeddings come from [sentence-transformers](https://www.sbert.net/) and answers from a local LLM through [Ollama](https://ollama.com/).

The sample knowledge base is `data/notes.txt`, a made-up staff handbook for a café.

## How it works

```
notes.txt ──► chunk ──► embed ──► retrieve top-k ──► LLM answers from the retrieved context
           (phase 1)   (phase 2)    (phase 2)            (phase 3)
```

| Phase | File | What it does |
|-------|------|--------------|
| 1 | `phase1_chunk.py` | Loads the document and splits it into overlapping chunks (500 chars, 100 overlap). No AI. |
| 2 | `phase2_retrieve.py` | Embeds the chunks with `all-MiniLM-L6-v2` and finds the chunks closest to your question (cosine similarity). |
| 3 | `phase3_generate.py` | Sends the top chunks and your question to `llama3.2` through Ollama, which answers **only** from that context. |

## Project structure

```
.
├── data/
│   └── notes.txt          # sample document (fictional)
├── phase1_chunk.py        # load + chunk
├── phase2_retrieve.py     # embed + retrieve
├── phase3_generate.py     # full RAG with an LLM
├── requirements.txt
└── README.md
```

## Setup

**Requirements:** Python 3.9+ and [Ollama](https://ollama.com/download) (needed only for phase 3).

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

# for phase 3
ollama pull llama3.2
```

## Usage

```bash
python phase1_chunk.py      # print the chunks
python phase2_retrieve.py   # ask questions, see which chunks match
python phase3_generate.py   # ask questions, get answers from the LLM
```

The first time you run phase 2 or phase 3, the embedding model (about 90 MB) is downloaded.

### Example (your scores and wording will differ)

```
You: When does the café open on Saturday?

[Retrieved chunks]
  1. score=0.612 | 2. Opening Hours From Monday to Friday the café opens at 7:30 ...

Assistant: On Saturday the café opens at 9:00 and closes at 20:00.
```

## Using your own data

Replace `data/notes.txt` with your own text, or change `DATA_FILE` in `phase1_chunk.py`. You can tune `CHUNK_SIZE`, `CHUNK_OVERLAP` (phase 1), `TOP_K`, `EMBED_MODEL` (phase 2) and `LLM_MODEL` (phase 3).
