"""Headless Python API for RAG-TUI."""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

import numpy as np

from rag_tui.core.engine import ChunkingEngine
from rag_tui.core.metrics import ChunkConfig, QueryResult, calculate_batch_metrics, export_config
from rag_tui.core.providers import ProviderType, get_best_provider, get_provider
from rag_tui.core.strategies import StrategyType
from rag_tui.core.vector import VectorStore


def chunk(text: str, strategy: str = "token", chunk_size: int = 200, overlap_percent: int = 10) -> Dict:
    """Chunk text and return JSON-serializable output.

    Returns a dict with stats + chunks, designed for headless usage.
    """
    engine = ChunkingEngine()
    overlap_tokens = max(0, int(round(chunk_size * (overlap_percent / 100.0))))
    strategy_type = StrategyType(strategy)

    chunks = engine.chunk_text(
        text=text,
        chunk_size=chunk_size,
        overlap=overlap_tokens,
        strategy_type=strategy_type,
    )
    stats = engine.get_chunk_stats(chunks)

    return {
        "strategy": strategy_type.value,
        "chunk_size": chunk_size,
        "overlap_percent": overlap_percent,
        "overlap_tokens": overlap_tokens,
        "stats": stats,
        "chunks": [
            {"index": i, "start": start, "end": end, "text": chunk}
            for i, (chunk, start, end) in enumerate(chunks)
        ],
    }


async def eval_async(
    queries: List[str],
    docs: str,
    strategy: str = "token",
    chunk_size: int = 200,
    overlap_percent: int = 10,
    top_k: int = 3,
    threshold: float = 0.5,
    provider: Optional[str] = None,
) -> Dict:
    """Evaluate retrieval quality for a list of queries against docs."""
    engine = ChunkingEngine()
    overlap_tokens = max(0, int(round(chunk_size * (overlap_percent / 100.0))))
    strategy_type = StrategyType(strategy)

    chunk_results = engine.chunk_text(
        text=docs,
        chunk_size=chunk_size,
        overlap=overlap_tokens,
        strategy_type=strategy_type,
    )
    chunks = [c[0] for c in chunk_results]

    if provider is None:
        embedding_provider, _ = await get_best_provider()
        if embedding_provider is None:
            raise RuntimeError("No embedding provider available.")
    else:
        provider_type = ProviderType(provider)
        embedding_provider = get_provider(provider_type)
        if not await embedding_provider.check_connection():
            raise RuntimeError(f"Provider '{provider}' is not available.")

    chunk_embeddings = await embedding_provider.embed_batch(chunks)
    embedding_dim = len(chunk_embeddings[0]) if chunk_embeddings else 0

    store = VectorStore(embedding_dim=embedding_dim)
    store.add_chunks(chunks, np.array(chunk_embeddings))

    query_results: List[QueryResult] = []
    for query in queries:
        query_embedding = np.array(await embedding_provider.embed(query))
        matches = store.search(query_embedding, top_k=top_k)
        retrieved = [(m[0], m[1]) for m in matches]
        top_score = retrieved[0][1] if retrieved else 0.0
        avg_score = sum(r[1] for r in retrieved) / len(retrieved) if retrieved else 0.0
        query_results.append(
            QueryResult(
                query=query,
                chunks_retrieved=retrieved,
                top_score=top_score,
                avg_score=avg_score,
            )
        )

    batch = calculate_batch_metrics(query_results, threshold=threshold)
    output = batch.to_dict()
    output["config"] = {
        "strategy": strategy_type.value,
        "chunk_size": chunk_size,
        "overlap_percent": overlap_percent,
        "overlap_tokens": overlap_tokens,
        "top_k": top_k,
        "threshold": threshold,
        "provider": provider or "auto",
    }
    return output


def eval(
    queries: List[str],
    docs: str,
    strategy: str = "token",
    chunk_size: int = 200,
    overlap_percent: int = 10,
    top_k: int = 3,
    threshold: float = 0.5,
    provider: Optional[str] = None,
) -> Dict:
    """Sync wrapper for eval_async.

    If running inside an event loop, call eval_async instead.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        raise RuntimeError("An event loop is running. Use eval_async instead.")

    return asyncio.run(
        eval_async(
            queries=queries,
            docs=docs,
            strategy=strategy,
            chunk_size=chunk_size,
            overlap_percent=overlap_percent,
            top_k=top_k,
            threshold=threshold,
            provider=provider,
        )
    )


def export(format: str = "json", strategy: str = "token", chunk_size: int = 200, overlap_percent: int = 10) -> str:
    """Export chunk config for external frameworks."""
    overlap_tokens = max(0, int(round(chunk_size * (overlap_percent / 100.0))))
    config = ChunkConfig(
        strategy=strategy,
        chunk_size=chunk_size,
        overlap_percent=overlap_percent,
        overlap_tokens=overlap_tokens,
    )
    return export_config(config, format=format)


__all__ = ["chunk", "eval", "eval_async", "export"]
