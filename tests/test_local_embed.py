"""Unit tests for the built-in hashing embedder and the provider wrapping it."""

import asyncio
import math
import subprocess
import sys

import numpy as np
import pytest

from rag_tui.core.local_embed import (
    DEFAULT_DIM,
    STOPWORDS,
    HashingEmbedder,
    _features,
    _tokenize,
)
from rag_tui.core.providers import (
    PROVIDER_CONFIGS,
    LocalProvider,
    ProviderType,
    get_best_provider,
    get_provider,
)


def _sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


# ---------------------------------------------------------------------------
# Tokenising
# ---------------------------------------------------------------------------

def test_tokenize_lowercases_and_splits():
    assert _tokenize("Rate Limits, Explained!") == ["rate", "limits", "explained"]


def test_tokenize_drops_stopwords():
    tokens = _tokenize("the refund is in the console and it was approved")
    assert "the" not in tokens
    assert "refund" in tokens and "console" in tokens and "approved" in tokens


def test_tokenize_keeps_apostrophes_and_digits():
    assert _tokenize("don't ship v2 builds") == ["don't", "ship", "v2", "builds"]


def test_tokenize_empty_string():
    assert _tokenize("") == []


def test_stopwords_are_lowercase_and_nonempty():
    assert STOPWORDS
    assert all(w == w.lower() and w for w in STOPWORDS)


def test_features_include_unigrams_bigrams_and_char_ngrams():
    features = list(_features("refund policy"))

    assert "refund" in features
    assert "refund_policy" in features
    assert any(f.startswith("#") for f in features)


def test_short_words_produce_no_char_ngrams():
    # "api" is shorter than the n-gram width, so it contributes only itself.
    assert not [f for f in _features("api") if f.startswith("#")]


# ---------------------------------------------------------------------------
# Vector properties
# ---------------------------------------------------------------------------

def test_vectors_are_unit_length():
    embedder = HashingEmbedder()
    vector = embedder.embed_one("throttling kicks in above one thousand requests")

    assert vector.shape == (DEFAULT_DIM,)
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)


def test_empty_text_gives_a_zero_vector_not_nan():
    embedder = HashingEmbedder()
    vector = embedder.embed_one("")

    assert not np.isnan(vector).any()
    assert float(np.linalg.norm(vector)) == 0.0


def test_stopword_only_text_gives_a_zero_vector():
    embedder = HashingEmbedder()
    assert float(np.linalg.norm(embedder.embed_one("the and of it was"))) == 0.0


def test_identical_text_is_perfectly_similar():
    embedder = HashingEmbedder()
    text = "refunds are available within thirty days of the original charge"

    assert _sim(embedder.embed_one(text), embedder.embed_one(text)) == pytest.approx(1.0, abs=1e-5)


def test_related_text_beats_unrelated_text():
    embedder = HashingEmbedder()
    anchor = embedder.embed_one("the api enforces rate limits per project")
    related = embedder.embed_one("rate limit ceilings apply to every project")
    unrelated = embedder.embed_one("coral reefs support a quarter of marine species")

    assert _sim(anchor, related) > _sim(anchor, unrelated)
    assert _sim(anchor, unrelated) < 0.2


def test_char_ngrams_bring_morphological_variants_together():
    embedder = HashingEmbedder()
    singular = embedder.embed_one("the retrieval index stores one document")
    plural = embedder.embed_one("retrieved indexes store many documents")

    # No shared word survives stopword removal in exact form, so any similarity
    # here comes from the character n-grams.
    assert _sim(singular, plural) > 0.1


def test_word_order_changes_the_vector_via_bigrams():
    embedder = HashingEmbedder()
    forward = embedder.embed_one("limit rate")
    backward = embedder.embed_one("rate limit")

    assert _sim(forward, backward) < 1.0


def test_repetition_is_damped_not_linear():
    """A tenfold difference in word count barely moves the vector.

    Under linear term frequency the fifty-repeat document would weight
    "refund" ten times heavier than the five-repeat one and drag the whole
    vector away from every other chunk in the corpus.
    """
    embedder = HashingEmbedder()
    context = "policy details apply"
    few = embedder.embed_one(" ".join(["refund"] * 5) + " " + context)
    many = embedder.embed_one(" ".join(["refund"] * 50) + " " + context)

    assert _sim(few, many) > 0.95


def test_term_frequency_weight_follows_one_plus_log():
    """Check the weighting formula itself, away from hashing collisions.

    A wide vector plus two nonsense tokens means each unigram owns its column
    outright, so the ratio between the two columns is the ratio of the weights.
    """
    embedder = HashingEmbedder(dim=8192)
    count = 40
    text = "zqxjv " + " ".join(["wkbfp"] * count)
    vector = embedder.embed_one(text)

    rare_col, _ = embedder._hash("zqxjv")
    common_col, _ = embedder._hash("wkbfp")
    assert rare_col != common_col

    ratio = abs(vector[common_col]) / abs(vector[rare_col])
    assert ratio == pytest.approx(1.0 + math.log(count), rel=0.01)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_two_instances_agree():
    text = "sustained overage triggers automatic throttling"
    assert np.array_equal(
        HashingEmbedder().embed_one(text), HashingEmbedder().embed_one(text)
    )


def test_vectors_are_stable_across_processes():
    """The on-disk embedding cache is only valid if hashing is not randomised.

    Python's built-in hash() is salted per process, so this guards against
    anyone swapping the stdlib checksums for it later.
    """
    script = (
        "from rag_tui.core.local_embed import HashingEmbedder;"
        "v = HashingEmbedder().embed_one('rate limits and throttling behaviour');"
        "print(','.join(f'{x:.6f}' for x in v[:16]))"
    )
    runs = [
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        for _ in range(2)
    ]

    assert runs[0] == runs[1]
    local = HashingEmbedder().embed_one("rate limits and throttling behaviour")
    assert runs[0] == ",".join(f"{x:.6f}" for x in local[:16])


def test_collisions_use_both_signs():
    """Signed hashing is what stops collisions from always reinforcing."""
    embedder = HashingEmbedder()
    signs = {embedder._hash(f"token{i}")[1] for i in range(200)}

    assert signs == {1.0, -1.0}


def test_hash_cache_does_not_change_results():
    embedder = HashingEmbedder()
    first = embedder.embed_one("cached tokens must hash identically")
    second = embedder.embed_one("cached tokens must hash identically")

    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# Batching and configuration
# ---------------------------------------------------------------------------

def test_embed_many_shape():
    embedder = HashingEmbedder()
    matrix = embedder.embed_many(["one document here", "another document", "a third"])

    assert matrix.shape == (3, DEFAULT_DIM)


def test_embed_many_matches_embed_one():
    embedder = HashingEmbedder()
    texts = ["first passage of text", "second passage of text"]
    matrix = embedder.embed_many(texts)

    for row, text in zip(matrix, texts):
        assert np.array_equal(row, embedder.embed_one(text))


def test_embed_many_on_empty_list():
    matrix = HashingEmbedder().embed_many([])

    assert matrix.shape == (0, DEFAULT_DIM)


def test_custom_dimension():
    embedder = HashingEmbedder(dim=128)
    assert embedder.embed_one("some text").shape == (128,)


def test_rejects_absurd_dimension():
    with pytest.raises(ValueError, match="at least 16"):
        HashingEmbedder(dim=4)


# ---------------------------------------------------------------------------
# Provider integration
# ---------------------------------------------------------------------------

def test_local_provider_is_always_reachable():
    provider = get_provider(ProviderType.LOCAL)

    assert isinstance(provider, LocalProvider)
    assert asyncio.run(provider.check_connection()) is True


def test_local_provider_embeds_to_plain_lists():
    provider = get_provider(ProviderType.LOCAL)
    vectors = asyncio.run(provider.embed_batch(["alpha text", "beta text"]))

    assert len(vectors) == 2
    assert all(isinstance(v, list) for v in vectors)
    assert all(isinstance(x, float) for x in vectors[0])
    assert len(vectors[0]) == PROVIDER_CONFIGS[ProviderType.LOCAL].embedding_dim


def test_local_provider_single_embed_matches_batch():
    provider = get_provider(ProviderType.LOCAL)
    single = asyncio.run(provider.embed("shared text"))
    batched = asyncio.run(provider.embed_batch(["shared text"]))

    assert single == batched[0]


def test_local_provider_refuses_to_fake_generation():
    provider = get_provider(ProviderType.LOCAL)

    with pytest.raises(NotImplementedError, match="embeddings only"):
        asyncio.run(provider.generate("what is the refund policy?"))


def test_local_provider_stream_also_refuses():
    provider = get_provider(ProviderType.LOCAL)

    async def drain():
        async for _ in provider.stream_generate("hello"):
            pass

    with pytest.raises(NotImplementedError, match="embeddings only"):
        asyncio.run(drain())


def test_local_config_declares_no_llm_support():
    config = PROVIDER_CONFIGS[ProviderType.LOCAL]

    assert config.supports_embedding is True
    assert config.supports_llm is False


def test_fallback_is_used_when_nothing_else_is_reachable(monkeypatch):
    """With every network provider unreachable, embeddings still work."""
    async def unreachable(self):
        return False

    for cls_name in ("OllamaProvider", "OpenAIProvider", "GroqProvider", "GoogleProvider"):
        monkeypatch.setattr(
            f"rag_tui.core.providers.{cls_name}.check_connection", unreachable
        )

    embedding_provider, llm_provider = asyncio.run(get_best_provider())

    assert isinstance(embedding_provider, LocalProvider)
    # It must not pose as a language model, or judge mode would silently
    # produce meaningless scores.
    assert llm_provider is None


def test_real_provider_wins_over_the_fallback(monkeypatch):
    async def reachable(self):
        return True

    async def unreachable(self):
        return False

    monkeypatch.setattr("rag_tui.core.providers.OllamaProvider.check_connection", reachable)
    for cls_name in ("OpenAIProvider", "GroqProvider", "GoogleProvider"):
        monkeypatch.setattr(
            f"rag_tui.core.providers.{cls_name}.check_connection", unreachable
        )

    embedding_provider, _ = asyncio.run(get_best_provider())

    assert not isinstance(embedding_provider, LocalProvider)
