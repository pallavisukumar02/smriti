"""Smriti · स्मृति: chat with your documents.

Run:  streamlit run app.py
"""

from __future__ import annotations

import html
import os
import re

import streamlit as st

from smriti.embeddings import DEFAULT_MODEL, Embedder
from smriti.llm import GROQ_MODELS, GroqLLM, LLMError
from smriti.loaders import SUPPORTED, LoaderError
from smriti.pipeline import Smriti
from smriti.text import stem, tokenize
from smriti.vector_store import HAS_FAISS

st.set_page_config(page_title="Smriti · Ask your documents", page_icon="📚", layout="wide")

st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Tiro+Devanagari+Sanskrit&display=swap');
  .block-container {max-width: 860px; padding-top: 2rem;}
  .brand {font-family: 'Source Serif 4', Georgia, serif; font-size: 1.6rem; font-weight: 600; margin: 0;}
  .brand span {font-family: 'Tiro Devanagari Sanskrit', serif; color: #1C7C86; font-weight: 400; margin-left: .4rem;}
  .hero {text-align: center; margin: 12vh 0 1rem;}
  .hero .deva {font-family: 'Tiro Devanagari Sanskrit', serif; color: #1C7C86; font-size: 1.4rem;}
  .hero h1 {font-family: 'Source Serif 4', Georgia, serif; font-weight: 500; font-size: 2.4rem; margin: .2rem 0 .4rem;}
  .hero p {color: #6E6B64; margin: 0;}
  [data-testid="stChatMessage"] .stMarkdown p {font-family: 'Source Serif 4', Georgia, serif; font-size: 1.05rem; line-height: 1.7;}
  .src {border: 1px solid rgba(128,128,128,.25); border-radius: 10px; padding: .6rem .8rem; margin: .4rem 0; font-size: .88rem;}
  .src .h {display: flex; gap: .5rem; align-items: baseline; flex-wrap: wrap; margin-bottom: .25rem;}
  .src .n {background: #1C7C86; color: #fff; border-radius: 999px; padding: 0 .45rem; font-size: .72rem; font-weight: 600;}
  .src .m {color: #6E6B64; font-size: .76rem;}
  .src .t {font-family: 'Source Serif 4', Georgia, serif; opacity: .85; line-height: 1.55; max-height: 11em; overflow: auto; white-space: pre-wrap;}
  .src mark {background: #FCEFB4; color: inherit; border-radius: 2px; padding: 0 1px;}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading the embedding model (first run only)…")
def get_embedder(name: str) -> Embedder:
    return Embedder(name)


def secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:  # no secrets.toml present
        value = ""
    return value or os.environ.get(name, "")


def state():
    s = st.session_state
    if "rag" not in s:
        s.rag = Smriti(get_embedder(DEFAULT_MODEL))
        s.messages = []  # {role, content, hits}
        s.uploader_key = 0
        s.errors = []
    return s


S = state()
rag: Smriti = S.rag


# --------------------------------------------------------------------------- helpers
def highlight(text: str, terms: set[str]) -> str:
    out = html.escape(text)
    if not terms:
        return out
    return re.sub(r"[^\W_]+", lambda m: f"<mark>{m.group(0)}</mark>" if stem(m.group(0).lower()) in terms else m.group(0), out)


def render_sources(hits, question: str) -> None:
    if not hits:
        return
    terms = set(tokenize(question))
    with st.expander(f"Sources · {len(hits)} passages"):
        for i, h in enumerate(hits, 1):
            page = f" · page {h.chunk.page}" if h.chunk.page else ""
            st.markdown(
                f"""<div class="src"><div class="h"><span class="n">{i}</span>
                <strong>{html.escape(h.chunk.doc_name)}</strong><span class="m">{page} · {h.matched_by}</span></div>
                <div class="t">{highlight(h.chunk.text, terms)}</div></div>""",
                unsafe_allow_html=True,
            )


def format_citations(text: str) -> str:
    """Turn [1] or [1, 2] into bold superscript-style markers."""
    return re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", lambda m: f"**<sup>[{m.group(1)}]</sup>**", text)


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<p class="brand">Smriti<span>स्मृति</span></p>', unsafe_allow_html=True)
    st.caption("Chat with your documents. Every answer cites its source.")

    if st.button("＋ New chat"):
        S.messages = []
        st.rerun()

    st.subheader("Library")
    uploads = st.file_uploader(
        "Add documents",
        type=list(SUPPORTED),
        accept_multiple_files=True,
        key=f"uploader_{S.uploader_key}",
        label_visibility="collapsed",
    )
    if uploads:
        with st.status("Indexing documents…", expanded=True) as status:
            for f in uploads:
                st.write(f"Reading **{f.name}**")
                try:
                    doc = rag.add_file(f.name, f.getvalue())
                    st.write(f"→ {doc.n_chunks} passages")
                except (LoaderError, ValueError) as e:
                    S.errors.append(f"{f.name}: {e}")
                except Exception as e:  # noqa: BLE001
                    S.errors.append(f"{f.name}: couldn't read this file ({type(e).__name__}).")
            status.update(label="Library updated", state="complete")
        S.uploader_key += 1  # clears the uploader so files aren't re-processed
        st.rerun()

    for err in S.errors:
        st.error(err, icon="⚠️")
    S.errors = []

    if not rag.documents:
        st.caption("No documents yet. Supports PDF, Word, TXT, Markdown, CSV, HTML and JSON.")
    for doc in list(rag.documents.values()):
        c1, c2 = st.columns([5, 1])
        pages = f"{doc.n_pages} pages · " if doc.n_pages else ""
        c1.markdown(f"**{doc.name}**  \n<small>{pages}{doc.words:,} words · {doc.n_chunks} passages</small>",
                    unsafe_allow_html=True)
        if c2.button("✕", key=f"rm_{doc.doc_id}", help=f"Remove {doc.name}"):
            rag.remove(doc.doc_id)
            st.rerun()

    st.subheader("Model")
    api_key = secret("GROQ_API_KEY")
    if not api_key:
        api_key = st.text_input("Groq API key", type="password", placeholder="gsk_…",
                                help="Free at console.groq.com/keys. Kept only for this session.")
    else:
        st.caption("✓ Groq key loaded from secrets")
    model = st.selectbox("Groq model", GROQ_MODELS, index=0)

    with st.expander("Retrieval settings"):
        mode = st.radio("Search mode", ["hybrid", "semantic", "keyword"], horizontal=True,
                        help="Hybrid fuses embeddings (FAISS) and BM25 with Reciprocal Rank Fusion.")
        top_k = st.slider("Passages per answer", 2, 12, 6)
        size = st.slider("Chunk size (characters)", 400, 2000, rag.chunk_size, step=100)
        overlap = st.slider("Chunk overlap", 0, 400, rag.overlap, step=25)
        if (size, overlap) != (rag.chunk_size, rag.overlap):
            if st.button("Re-index library with these settings"):
                with st.spinner("Re-indexing…"):
                    rag.reindex(size, overlap)
                st.rerun()
        st.caption(f"Embeddings: `{DEFAULT_MODEL.split('/')[-1]}` · Index: {'FAISS' if HAS_FAISS else 'NumPy'}"
                   f" · {len(rag.retriever)} passages")


# --------------------------------------------------------------------------- main
if not S.messages:
    st.markdown(
        """<div class="hero"><div class="deva">स्मृति</div><h1>What do you want to know?</h1>
        <p>Add documents in the sidebar, then ask. Answers come only from your files, with citations.</p></div>""",
        unsafe_allow_html=True,
    )
    if rag.documents:
        cols = st.columns(3)
        for col, (label, q) in zip(cols, [
            ("Summarize", "Summarize the main points of these documents."),
            ("Key facts", "What are the key dates, numbers, or facts mentioned?"),
            ("Action items", "List any risks, open questions, or action items."),
        ]):
            if col.button(label):
                S.pending = q
                st.rerun()

for m in S.messages:
    with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "📚"):
        st.markdown(format_citations(m["content"]) if m["role"] == "assistant" else m["content"],
                    unsafe_allow_html=True)
        if m["role"] == "assistant":
            render_sources(m.get("hits"), m.get("question", ""))

placeholder = "Ask anything about your documents" if rag.documents else "Add a document in the sidebar to start"
question = st.chat_input(placeholder, disabled=not rag.documents) or S.pop("pending", None)

if question:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    previous = next((m["content"] for m in reversed(S.messages) if m["role"] == "user"), "")
    hits = rag.retrieve(question, k=top_k, mode=mode, previous=previous)
    history = [{"role": m["role"], "content": m["content"]} for m in S.messages]

    with st.chat_message("assistant", avatar="📚"):
        if not api_key:
            answer = ("Add a Groq API key in the sidebar to get written answers. "
                      "Meanwhile, here are the passages that best match your question.")
            st.markdown(answer)
        else:
            try:
                llm = GroqLLM(api_key, model)
                answer = st.write_stream(rag.answer(llm, question, hits, history))
            except LLMError as e:
                answer = ""
                st.error(str(e), icon="⚠️")
        render_sources(hits, question)

    if answer:
        S.messages.append({"role": "user", "content": question})
        S.messages.append({"role": "assistant", "content": answer, "hits": hits, "question": question})
        st.rerun()
