"""Unit tests for the ChunkOptimizer (killer feature)."""

import asyncio
from typing import List
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from rag_tui.core.optimizer import (
    ChunkOptimizer,
    OptimizationReport,
    OptimizationResult,
)


# ---------------------------------------------------------------------------
# Mock embedding provider
# ---------------------------------------------------------------------------

class MockEmbeddingProvider:
    """Returns deterministic random-ish embeddings for testing."""

    embedding_dim = 8

    async def embed(self, text: str) -> List[float]:
        # Deterministic: hash-based so same text → same vector
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.random(self.embedding_dim).tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [await self.embed(t) for t in texts]


SAMPLE_TEXT = """
Retrieval-Augmented Generation (RAG) is a technique that combines retrieval
with generation. The process involves chunking documents, embedding them,
and searching for relevant chunks at query time. Chunk size is critical.
Large chunks preserve more context but reduce precision. Small chunks enable
more precise retrieval but may miss broader context.
"""

SAMPLE_QUERIES = [
    "What is RAG?",
    "How does chunking work?",
    "What affects retrieval precision?",
]


# ---------------------------------------------------------------------------
# OptimizationResult helpers
# ---------------------------------------------------------------------------

def test_optimization_result_config():
    """OptimizationResult.config should produce correct ChunkConfig."""
    from rag_tui.core.metrics import calculate_batch_metrics

    batch = calculate_batch_metrics([])
    r = OptimizationResult(
        chunk_size=256, overlap_percent=10, strategy="token",
        num_chunks=5, metrics=batch,
    )
    cfg = r.config
    assert cfg.chunk_size == 256
    assert cfg.overlap_percent == 10
    assert cfg.strategy == "token"
    assert cfg.overlap_tokens == 26  # round(256 * 0.10)


def test_optimization_result_to_dict():
    from rag_tui.core.metrics import calculate_batch_metrics

    batch = calculate_batch_metrics([])
    r = OptimizationResult(
        chunk_size=200, overlap_percent=15, strategy="sentence",
        num_chunks=8, metrics=batch, rank=1, score=0.75,
    )
    d = r.to_dict()
    assert d["rank"] == 1
    assert d["chunk_size"] == 200
    assert d["strategy"] == "sentence"
    assert "metrics" in d


# ---------------------------------------------------------------------------
# Full optimizer run (using mock provider)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_optimizer_returns_report():
    optimizer = ChunkOptimizer(
        sizes=[100, 200],
        overlap_percents=[10],
        strategies=["token"],
        top_k=3,
        threshold=0.5,
        max_concurrent=2,
    )
    provider = MockEmbeddingProvider()
    report = await optimizer.optimize(SAMPLE_TEXT, SAMPLE_QUERIES, provider)

    assert isinstance(report, OptimizationReport)
    assert report.total_configs_tested == 2  # 2 sizes × 1 overlap × 1 strategy
    assert len(report.results) == 2
    assert report.best is not None
    assert report.best.rank == 1


@pytest.mark.asyncio
async def test_optimizer_results_ranked():
    optimizer = ChunkOptimizer(
        sizes=[64, 128, 256],
        overlap_percents=[5, 15],
        strategies=["token"],
        max_concurrent=3,
    )
    provider = MockEmbeddingProvider()
    report = await optimizer.optimize(SAMPLE_TEXT, SAMPLE_QUERIES, provider)

    # 3 sizes × 2 overlaps × 1 strategy = 6 configs
    assert report.total_configs_tested == 6
    ranks = [r.rank for r in report.results]
    assert sorted(ranks) == list(range(1, 7))


@pytest.mark.asyncio
async def test_optimizer_scores_between_0_and_1():
    optimizer = ChunkOptimizer(
        sizes=[200],
        overlap_percents=[10],
        strategies=["token"],
    )
    provider = MockEmbeddingProvider()
    report = await optimizer.optimize(SAMPLE_TEXT, SAMPLE_QUERIES, provider)
    for r in report.results:
        assert 0.0 <= r.score <= 1.0


@pytest.mark.asyncio
async def test_optimizer_progress_callback():
    messages = []
    optimizer = ChunkOptimizer(
        sizes=[100, 200],
        overlap_percents=[10],
        strategies=["token"],
    )
    provider = MockEmbeddingProvider()
    await optimizer.optimize(
        SAMPLE_TEXT, SAMPLE_QUERIES, provider,
        progress_cb=messages.append,
    )
    assert len(messages) == 2


@pytest.mark.asyncio
async def test_optimizer_ab_test():
    optimizer = ChunkOptimizer(top_k=3)
    provider = MockEmbeddingProvider()
    result = await optimizer.ab_test(
        text=SAMPLE_TEXT,
        queries=SAMPLE_QUERIES,
        embedding_provider=provider,
        config_a={"chunk_size": 100, "overlap_percent": 10, "strategy": "token"},
        config_b={"chunk_size": 300, "overlap_percent": 15, "strategy": "token"},
    )
    assert result["winner"] in ("A", "B")
    assert "config_a" in result
    assert "config_b" in result
    assert "score_delta" in result


@pytest.mark.asyncio
async def test_optimizer_multiple_strategies():
    optimizer = ChunkOptimizer(
        sizes=[200],
        overlap_percents=[10],
        strategies=["token", "sentence", "recursive"],
        max_concurrent=3,
    )
    provider = MockEmbeddingProvider()
    report = await optimizer.optimize(SAMPLE_TEXT, SAMPLE_QUERIES, provider)
    assert report.total_configs_tested == 3
    strategies_tested = {r.strategy for r in report.results}
    assert strategies_tested == {"token", "sentence", "recursive"}


def test_recommend():
    from rag_tui.core.metrics import calculate_batch_metrics

    optimizer = ChunkOptimizer()
    batch = calculate_batch_metrics([])
    results = [
        OptimizationResult(100, 10, "token", 5, batch, rank=2, score=0.5),
        OptimizationResult(200, 10, "token", 8, batch, rank=1, score=0.8),
        OptimizationResult(300, 15, "token", 6, batch, rank=3, score=0.3),
    ]
    best = optimizer.recommend(results)
    assert best.rank == 1
    assert best.chunk_size == 200


def test_recommend_empty_raises():
    optimizer = ChunkOptimizer()
    with pytest.raises(ValueError):
        optimizer.recommend([])


# ---------------------------------------------------------------------------
# Report serialization
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_report_to_dict():
    optimizer = ChunkOptimizer(
        sizes=[200], overlap_percents=[10], strategies=["token"]
    )
    provider = MockEmbeddingProvider()
    report = await optimizer.optimize(SAMPLE_TEXT, SAMPLE_QUERIES, provider)
    d = report.to_dict()
    assert "best" in d
    assert "all_results" in d
    assert "total_configs_tested" in d
    assert d["total_configs_tested"] == 1
