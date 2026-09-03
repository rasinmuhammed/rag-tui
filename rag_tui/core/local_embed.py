"""Built-in embedder that works the moment the package is installed.

There is no model file to download, no server to start, and no key to set. The
whole thing is the hashing trick over word and character n-grams, which means
it runs on any machine that can run numpy, including one with no network at
all.

What you get is lexical similarity. Two chunks that share vocabulary land near
each other; two chunks that say the same thing in different words do not. That
is a real limitation and it is worth being blunt about it: this is here so that
`rag-tui doctor` and `rag-tui search` do something useful thirty seconds after
`pip install`, not because hashed bag-of-words competes with a trained encoder.

Point RAG-TUI at Ollama or OpenAI when you want paraphrase to register. For
finding duplicate passages, boilerplate, stray headings and structural damage
in a corpus, lexical overlap is most of what you need anyway.
"""

from __future__ import annotations

import math
import re
import zlib
from typing import Dict, Iterable, List, Sequence

import numpy as np

DEFAULT_DIM = 512

# Char n-gram width. Four is wide enough to carry a morpheme and narrow enough
# that "retrieval" and "retrieved" still share most of their grams.
CHAR_NGRAM = 4

# Words this common carry no signal about what a chunk is about, and because
# they appear in nearly every chunk they drag everything toward a single point.
# Dropping them is what keeps the space from collapsing.
STOPWORDS = frozenset("""
a an and are as at be been but by for from had has have he her his i if in into is it
its of on or she that the their them then there these they this to was were what when
which who will with would you your our us we do does did not no nor so than too very
can could should may might must shall about above after again against all also am any
because before being below between both during each few further here how more most
other own same some such only other over under until up down out off why while
""".split())

_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens with stopwords removed."""
    return [w for w in _WORD_RE.findall(text.lower()) if w not in STOPWORDS]


def _features(text: str) -> Iterable[str]:
    """Yield every feature for a chunk: unigrams, bigrams, and char n-grams.

    Bigrams let a phrase like "rate limit" mean something the two words alone
    do not. Char n-grams keep singular and plural forms close together and stop
    a typo from moving a chunk across the space.
    """
    words = _tokenize(text)

    for word in words:
        yield word

    for left, right in zip(words, words[1:]):
        yield f"{left}_{right}"

    for word in words:
        # Short words are already their own n-gram.
        if len(word) <= CHAR_NGRAM:
            continue
        padded = f"^{word}$"
        for i in range(len(padded) - CHAR_NGRAM + 1):
            yield f"#{padded[i:i + CHAR_NGRAM]}"


class HashingEmbedder:
    """Deterministic lexical embedder built on the signed hashing trick.

    Every feature is hashed to a column and to a sign. The sign is what keeps
    collisions honest: when two unrelated features share a column they cancel
    out on average instead of always reinforcing each other.

    Vectors are L2 normalised, so a dot product between any two of them is
    their cosine similarity, which is exactly what the vector store and the
    doctor's geometry checks expect.
    """

    def __init__(self, dim: int = DEFAULT_DIM):
        if dim < 16:
            raise ValueError(f"dim must be at least 16, got {dim}")
        self.dim = dim
        # Hashing the same token thousands of times across a corpus is pure
        # waste, so results are memoised for the life of the embedder.
        self._hash_cache: Dict[str, tuple] = {}

    def _hash(self, feature: str) -> tuple:
        """Map a feature to (column, sign), memoised.

        crc32 and adler32 are both in the standard library, both implemented in
        C, and different enough from each other to serve as two independent
        hashes. Neither is randomised per process, so a vector computed today
        matches one computed tomorrow, which the on-disk embedding cache relies
        on.
        """
        cached = self._hash_cache.get(feature)
        if cached is not None:
            return cached

        data = feature.encode("utf-8")
        column = zlib.crc32(data) % self.dim
        sign = 1.0 if zlib.adler32(data) & 1 else -1.0
        result = (column, sign)
        self._hash_cache[feature] = result
        return result

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string into a unit-length vector."""
        vector = np.zeros(self.dim, dtype=np.float32)

        counts: Dict[str, int] = {}
        for feature in _features(text):
            counts[feature] = counts.get(feature, 0) + 1

        for feature, count in counts.items():
            column, sign = self._hash(feature)
            # Sublinear term frequency. A word used nine times matters more
            # than one used once, but nowhere near nine times more.
            vector[column] += sign * (1.0 + math.log(count))

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a sequence of strings into a [n_texts, dim] matrix."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed_one(t) for t in texts])
