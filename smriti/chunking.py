"""Split documents into overlapping passages that respect paragraph and sentence boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .loaders import Page

_SENTENCE = re.compile(r"[^.!?\n]+[.!?]*\s*")


@dataclass
class Chunk:
    doc_id: str
    doc_name: str
    text: str
    page: int | None
    index: int = 0

    @property
    def label(self) -> str:
        return f"{self.doc_name}, page {self.page}" if self.page else self.doc_name


def _clean(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(
    pages: list[Page], doc_id: str, doc_name: str, size: int = 1000, overlap: int = 150
) -> list[Chunk]:
    """Greedy packer: add paragraphs until `size` chars, carry `overlap` chars into the next chunk.

    Paragraphs longer than `size` are split by sentence.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    chunks: list[Chunk] = []

    for page in pages:
        text = _clean(page.text)
        if not text:
            continue
        buf = ""

        def flush(b: str) -> None:
            if b.strip():
                chunks.append(Chunk(doc_id, doc_name, b.strip(), page.page, len(chunks)))

        for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
            if not para:
                continue
            if len(para) > size:
                for sent in _SENTENCE.findall(para) or [para]:
                    if len(buf) + len(sent) > size and buf:
                        flush(buf)
                        buf = buf[-overlap:] + sent
                    else:
                        buf += sent
                buf += "\n\n"
            elif len(buf) + len(para) > size and buf:
                flush(buf)
                buf = buf[-overlap:] + para + "\n\n"
            else:
                buf += para + "\n\n"
        flush(buf)

    return chunks
