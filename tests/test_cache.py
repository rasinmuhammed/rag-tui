"""Unit tests for the SQLite embedding cache."""

import numpy as np
import pytest

from rag_tui.core.cache import EmbeddingCache


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr("rag_tui.core.cache.CACHE_DIR", tmp_path)
    c = EmbeddingCache("test_provider", "test_model")
    yield c
    c.close()


def test_cache_miss_returns_none(cache):
    result = cache.get("hello world")
    assert result is None


def test_cache_put_and_get(cache):
    text = "The quick brown fox"
    embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    cache.put(text, embedding)
    retrieved = cache.get(text)
    assert retrieved is not None
    np.testing.assert_array_almost_equal(retrieved, embedding)


def test_cache_different_texts(cache):
    emb1 = np.array([1.0, 0.0], dtype=np.float32)
    emb2 = np.array([0.0, 1.0], dtype=np.float32)
    cache.put("text one", emb1)
    cache.put("text two", emb2)
    np.testing.assert_array_almost_equal(cache.get("text one"), emb1)
    np.testing.assert_array_almost_equal(cache.get("text two"), emb2)


def test_cache_overwrite(cache):
    text = "same text"
    emb1 = np.array([1.0, 2.0], dtype=np.float32)
    emb2 = np.array([3.0, 4.0], dtype=np.float32)
    cache.put(text, emb1)
    cache.put(text, emb2)
    np.testing.assert_array_almost_equal(cache.get(text), emb2)


def test_cache_batch(cache):
    texts = ["alpha", "beta", "gamma"]
    embeddings = [np.array([float(i)], dtype=np.float32) for i in range(3)]
    cache.put_batch(texts, embeddings)
    hits = cache.get_batch(texts)
    assert len(hits) == 3
    for text, expected in zip(texts, embeddings):
        np.testing.assert_array_almost_equal(hits[text], expected)


def test_cache_size(cache):
    assert cache.size == 0
    cache.put("a", np.array([1.0], dtype=np.float32))
    assert cache.size == 1
    cache.put("b", np.array([2.0], dtype=np.float32))
    assert cache.size == 2


def test_cache_clear(cache):
    cache.put("a", np.array([1.0], dtype=np.float32))
    cache.clear()
    assert cache.size == 0
    assert cache.get("a") is None


def test_cache_stats(cache):
    cache.put("x", np.array([1.0, 2.0], dtype=np.float32))
    stats = cache.stats()
    assert stats["entries"] == 1
    assert "size_mb" in stats
    assert "db_path" in stats


def test_cache_empty_get_batch(cache):
    result = cache.get_batch(["not here"])
    assert result == {}
