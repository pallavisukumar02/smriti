import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smriti.text import tokenize  # noqa: E402


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder, so tests run without downloading a model."""

    dim = 64

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, t in enumerate(texts):
            for tok in tokenize(t):
                out[row, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-12, None)


@pytest.fixture
def embedder():
    return FakeEmbedder()
