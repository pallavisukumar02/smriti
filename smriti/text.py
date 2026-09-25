"""Tokenization helpers shared by BM25 and highlighting."""

from __future__ import annotations

import re
import unicodedata

STOPWORDS = set(
    """a an and are as at be been but by can could did do does for from had has have he her his how i if in
    into is it its me my no not of on or our she so than that the their them then there these they this those
    to too us was we were what when where which while who whom why will with would you your about also any all
    just more most other some such only own same very s t don should now""".split()
)

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def stem(word: str) -> str:
    """A tiny suffix-stripping stemmer. Crude, but fast and dependency-free."""
    w = word
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and (w.endswith("ied") or w.endswith("ies")):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 4 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words, drop stopwords, stem."""
    text = unicodedata.normalize("NFKD", text.lower())
    return [stem(w) for w in _WORD.findall(text) if len(w) > 1 and w not in STOPWORDS]
