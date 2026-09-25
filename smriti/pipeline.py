"""The RAG pipeline: ingest -> chunk -> index -> retrieve -> prompt -> generate."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from .chunking import chunk_pages
from .loaders import Page, load_file
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .retriever import Hit, HybridRetriever

BROAD = re.compile(r"\b(summar|overview|main points|key points|tl;?dr|outline|what is this)", re.I)


@dataclass
class Document:
    doc_id: str
    name: str
    pages: list[Page] = field(repr=False)
    n_chunks: int = 0

    @property
    def words(self) -> int:
        return sum(len(p.text.split()) for p in self.pages)

    @property
    def n_pages(self) -> int:
        return sum(1 for p in self.pages if p.page)


class Smriti:
    def __init__(self, embedder, chunk_size: int = 1000, overlap: int = 150):
        self.embedder = embedder
        self.chunk_size, self.overlap = chunk_size, overlap
        self.retriever = HybridRetriever(embedder)
        self.documents: dict[str, Document] = {}

    # ---------------------------------------------------------------- library
    def add_file(self, name: str, data: bytes) -> Document:
        doc_id = hashlib.sha1(name.encode() + data).hexdigest()[:12]
        if doc_id in self.documents:
            return self.documents[doc_id]
        for existing in [d for d in self.documents.values() if d.name == name]:
            self.remove(existing.doc_id)  # re-upload of a changed file replaces it
        pages = load_file(name, data)
        return self._index(Document(doc_id, name, pages))

    def _index(self, doc: Document) -> Document:
        chunks = chunk_pages(doc.pages, doc.doc_id, doc.name, self.chunk_size, self.overlap)
        if not chunks:
            raise ValueError(f"No text found in {doc.name}.")
        self.retriever.add(chunks)
        doc.n_chunks = len(chunks)
        self.documents[doc.doc_id] = doc
        return doc

    def remove(self, doc_id: str) -> None:
        self.documents.pop(doc_id, None)
        self.retriever.remove_doc(doc_id)

    def reindex(self, chunk_size: int, overlap: int) -> None:
        self.chunk_size, self.overlap = chunk_size, overlap
        docs = list(self.documents.values())
        self.retriever = HybridRetriever(self.embedder)
        self.documents = {}
        for d in docs:
            self._index(d)

    # ------------------------------------------------------------------ query
    def retrieve(self, question: str, k: int = 6, mode: str = "hybrid", previous: str = "",
                 max_chars: int = 14000) -> list[Hit]:
        hits = self.retriever.search(question, k=k, mode=mode, previous=previous)
        if len(hits) < 3 or (BROAD.search(question) and len(hits) < k):
            seen = {(h.chunk.doc_id, h.chunk.index) for h in hits}
            for h in self.retriever.spread(k + 2):
                if (h.chunk.doc_id, h.chunk.index) not in seen and len(hits) < k + 2:
                    hits.append(h)
        # keep the prompt within the model's (and Groq free tier's) token budget
        out, total = [], 0
        for h in hits:
            total += len(h.chunk.text)
            if total > max_chars and out:
                break
            out.append(h)
        return out

    def build_messages(self, question: str, hits: list[Hit], history: list[dict]) -> list[dict]:
        past = [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
        user = build_user_prompt(question, hits, [d.name for d in self.documents.values()])
        return [{"role": "system", "content": SYSTEM_PROMPT}, *past, {"role": "user", "content": user}]

    def answer(self, llm, question: str, hits: list[Hit], history: list[dict]) -> Iterator[str]:
        return llm.stream(self.build_messages(question, hits, history))
