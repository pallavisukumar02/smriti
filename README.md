# Smriti · स्मृति

*smṛti (Sanskrit): "that which is remembered"*

**Chat with your documents.** Upload PDFs, Word files or notes, ask questions in plain language, and get answers written only from your files, with every claim cited back to the exact passage and page.

Built with **Python · Streamlit · sentence-transformers · FAISS · BM25 · Groq (Llama 3.3)**.

<!-- Add a screenshot: save it as docs/screenshot.png and uncomment the next line -->
<!-- ![Smriti screenshot](docs/screenshot.png) -->

## Features

- **Hybrid retrieval:** dense semantic search (MiniLM embeddings in a FAISS index) combined with BM25 keyword search through **Reciprocal Rank Fusion**. Semantic search catches paraphrases; BM25 catches exact names, numbers and jargon.
- **Grounded, cited answers:** the LLM answers only from retrieved passages and cites each claim like `[2]`. If the documents don't cover a question, it says so.
- **Source inspector:** every answer shows the passages it used, the page number, whether each was found by semantic or keyword search (with its rank), and your query terms highlighted.
- **Follow-up aware:** the previous question nudges retrieval for follow-ups like "what about the second one?".
- **Many formats:** PDF (with page numbers), Word (.docx, including tables), TXT, Markdown, CSV, HTML and JSON.
- **Tunable:** switch between hybrid, semantic-only and keyword-only search, and change top-k, chunk size and overlap from the sidebar to compare their effect.
- **Tested:** unit tests cover chunking, BM25, RRF, the retriever and the pipeline, and run without downloading a model.

## Architecture

```
                ┌──────────── ingest ────────────┐
 upload ──► loaders.py ──► chunking.py ──┬──► embeddings.py ──► vector_store.py (FAISS)
 (pdf/docx/…)   (text + pages)  (overlapping  │
                                 passages)    └──► bm25.py (sparse index)

                ┌──────────── query ─────────────┐
 question ──► retriever.py ── dense top-25 ─┐
                           └─ BM25 top-25 ──┴─► fusion.py (RRF) ──► top-k passages
                                                                         │
                              prompts.py ──► llm.py (Groq, streamed) ◄───┘
                                                   │
                                        answer with [n] citations
```

| Module | What it does |
|---|---|
| `smriti/loaders.py` | Extracts text from each file type; keeps page numbers for PDFs |
| `smriti/chunking.py` | Packs paragraphs into ~1,000-char chunks with 150-char overlap, splitting long paragraphs by sentence |
| `smriti/embeddings.py` | `all-MiniLM-L6-v2` sentence embeddings, L2-normalised |
| `smriti/vector_store.py` | FAISS `IndexFlatIP` (cosine similarity), with a NumPy fallback |
| `smriti/bm25.py` | Okapi BM25 implemented from scratch |
| `smriti/fusion.py` | Reciprocal Rank Fusion, `score = Σ 1 / (60 + rank)` |
| `smriti/retriever.py` | Hybrid search, per-document diversity cap, follow-up handling |
| `smriti/pipeline.py` | Ties it together: library management, retrieval, prompt building |
| `smriti/llm.py` | Streams answers from Groq with friendly error messages |
| `app.py` | Streamlit interface |

### Design choices

- **Why hybrid search?** Embeddings are good at meaning but weak on rare exact tokens (IDs, names, figures); BM25 is the opposite. RRF merges the two rankings using ranks only, so no score calibration between them is needed.
- **Why exact FAISS search?** A personal library has thousands of chunks, not millions. `IndexFlatIP` is exact and fast at this scale; `IndexHNSWFlat` would be the swap for larger corpora.
- **Why Groq?** It has a free tier and very fast inference, which makes streamed answers feel instant.

## Run locally

```bash
git clone https://github.com/pallavisukumar02/smriti.git
cd smriti
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys) and paste it into the sidebar, or save it in `.streamlit/secrets.toml` (see `secrets.toml.example`). The first run downloads the embedding model (about 90 MB).

## Run the tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Deploy (free) on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click **Create app**.
3. Pick this repo, branch `main`, main file `app.py`.
4. Under **Advanced settings → Secrets**, add `GROQ_API_KEY = "gsk_..."`.
5. Click **Deploy**.

## Limitations and next steps

- Scanned PDFs need OCR (e.g. Tesseract) before they can be indexed.
- The library lives in the session and resets on refresh; persisting the FAISS index to disk is a natural next step.
- Possible extensions: a cross-encoder re-ranker, retrieval evaluation (recall@k, MRR) on a labelled question set, and query rewriting for follow-ups.

## License

MIT
