"""Headless Python API for RAG-TUI.

Public surface:
    chunk()         - chunk text, return stats + chunk list
    eval()          - evaluate retrieval quality for a list of queries
    eval_async()    - async version of eval()
    eval_dataset()  - eval() but takes a CSV/JSONL dataset file
    optimize()      - auto-recommend the best chunking config
    optimize_async()- async version of optimize()
    compare()       - compare a new run against a saved baseline
    export()        - export config for LangChain / LlamaIndex
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from rag_tui.core.engine import ChunkingEngine
from rag_tui.core.metrics import (
    BatchTestResult,
    ChunkConfig,
    QueryResult,
    calculate_batch_metrics,
    compare_results,
    export_config,
    load_dataset,
)
from rag_tui.core.optimizer import ChunkOptimizer, OptimizationReport
from rag_tui.core.providers import ProviderType, get_best_provider, get_provider
from rag_tui.core.strategies import StrategyType
from rag_tui.core.vector import VectorStore


# ---------------------------------------------------------------------------
# chunk()
# ---------------------------------------------------------------------------

def chunk(
    text: str,
    strategy: str = "token",
    chunk_size: int = 200,
    overlap_percent: int = 10,
) -> Dict:
    """Chunk text and return JSON-serialisable output.

    Returns a dict with stats + chunks, designed for headless usage.

    Example::

        import rag_tui.api as rag
        result = rag.chunk(open("doc.txt").read(), chunk_size=256)
        print(result["stats"])
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
            {"index": i, "start": start, "end": end, "text": text_chunk}
            for i, (text_chunk, start, end) in enumerate(chunks)
        ],
    }


# ---------------------------------------------------------------------------
# eval() / eval_async()
# ---------------------------------------------------------------------------

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
    """Evaluate retrieval quality for a list of queries against docs.

    Returns a dict with IR metrics (MRR, nDCG@k, Recall@k, Precision@k,
    hit rate) plus per-query details.
    """
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

    embedding_provider = await _resolve_provider(provider)

    chunk_embeddings = await embedding_provider.embed_batch(chunks)
    store = VectorStore(embedding_dim=len(chunk_embeddings[0]))
    store.add_chunks(chunks, np.array(chunk_embeddings))

    query_results: List[QueryResult] = []
    for query in queries:
        q_emb = np.array(await embedding_provider.embed(query))
        matches = store.search(q_emb, top_k=top_k)
        retrieved = [(m[0], m[1]) for m in matches]
        top_score = retrieved[0][1] if retrieved else 0.0
        avg_score = sum(s for _, s in retrieved) / len(retrieved) if retrieved else 0.0
        query_results.append(QueryResult(
            query=query,
            chunks_retrieved=retrieved,
            top_score=top_score,
            avg_score=avg_score,
        ))

    batch = calculate_batch_metrics(query_results, threshold=threshold, top_k=top_k)
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
    """Synchronous wrapper for eval_async().

    If already inside a running event loop, call eval_async() instead.
    """
    _ensure_no_running_loop()
    return asyncio.run(eval_async(
        queries=queries,
        docs=docs,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap_percent=overlap_percent,
        top_k=top_k,
        threshold=threshold,
        provider=provider,
    ))


# ---------------------------------------------------------------------------
# eval_dataset() - CSV / JSONL evaluation
# ---------------------------------------------------------------------------

async def eval_dataset_async(
    dataset_path: str,
    docs: str,
    strategy: str = "token",
    chunk_size: int = 200,
    overlap_percent: int = 10,
    top_k: int = 3,
    threshold: float = 0.5,
    provider: Optional[str] = None,
) -> Dict:
    """Evaluate retrieval quality using a CSV or JSONL query dataset.

    Dataset format:
        CSV:   query,relevant_chunk,answer   (only ``query`` is required)
        JSONL: {"query": "...", "relevant_chunk": "...", "answer": "..."}

    The optional ``relevant_chunk`` field enables exact ground-truth matching
    when present; otherwise score-based metrics are used.
    """
    rows = load_dataset(dataset_path)
    if not rows:
        raise ValueError(f"Dataset is empty: {dataset_path}")
    queries = [row["query"] for row in rows if row.get("query")]
    if not queries:
        raise ValueError("Dataset has no 'query' column/key.")

    result = await eval_async(
        queries=queries,
        docs=docs,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap_percent=overlap_percent,
        top_k=top_k,
        threshold=threshold,
        provider=provider,
    )
    result["dataset"] = {"path": dataset_path, "rows": len(rows)}
    return result


def eval_dataset(
    dataset_path: str,
    docs: str,
    strategy: str = "token",
    chunk_size: int = 200,
    overlap_percent: int = 10,
    top_k: int = 3,
    threshold: float = 0.5,
    provider: Optional[str] = None,
) -> Dict:
    """Synchronous wrapper for eval_dataset_async()."""
    _ensure_no_running_loop()
    return asyncio.run(eval_dataset_async(
        dataset_path=dataset_path,
        docs=docs,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap_percent=overlap_percent,
        top_k=top_k,
        threshold=threshold,
        provider=provider,
    ))


# ---------------------------------------------------------------------------
# optimize() - killer feature
# ---------------------------------------------------------------------------

async def optimize_async(
    docs: str,
    queries: List[str],
    strategies: Optional[List[str]] = None,
    sizes: Optional[List[int]] = None,
    overlap_percents: Optional[List[int]] = None,
    top_k: int = 3,
    threshold: float = 0.5,
    provider: Optional[str] = None,
    max_concurrent: int = 3,
) -> Dict:
    """Auto-recommend the best chunking configuration.

    Tests all combinations of (chunk_size, overlap_percent, strategy) in
    parallel and returns them ranked by a composite IR metric score:
        0.35 * MRR + 0.35 * nDCG@k + 0.20 * Recall@k + 0.10 * Precision@k

    Example::

        result = await rag.optimize_async(
            docs=open("doc.txt").read(),
            queries=["What is RAG?", "How does chunking work?"],
            strategies=["token", "sentence"],
        )
        print(result["best"])
    """
    embedding_provider = await _resolve_provider(provider)

    optimizer = ChunkOptimizer(
        sizes=sizes,
        overlap_percents=overlap_percents,
        strategies=strategies or ["token"],
        top_k=top_k,
        threshold=threshold,
        max_concurrent=max_concurrent,
    )
    report = await optimizer.optimize(docs, queries, embedding_provider)
    return report.to_dict()


def optimize(
    docs: str,
    queries: List[str],
    strategies: Optional[List[str]] = None,
    sizes: Optional[List[int]] = None,
    overlap_percents: Optional[List[int]] = None,
    top_k: int = 3,
    threshold: float = 0.5,
    provider: Optional[str] = None,
    max_concurrent: int = 3,
) -> Dict:
    """Synchronous wrapper for optimize_async().

    Returns a dict with ``best`` config and ``all_results`` ranked list.
    """
    _ensure_no_running_loop()
    return asyncio.run(optimize_async(
        docs=docs,
        queries=queries,
        strategies=strategies,
        sizes=sizes,
        overlap_percents=overlap_percents,
        top_k=top_k,
        threshold=threshold,
        provider=provider,
        max_concurrent=max_concurrent,
    ))


# ---------------------------------------------------------------------------
# compare() - baseline regression detection
# ---------------------------------------------------------------------------

def compare(
    baseline_path: str,
    current: Dict,
    current_config: Optional[Dict] = None,
) -> Dict:
    """Compare a current eval result against a saved baseline JSON file.

    The baseline should be the JSON output previously returned by eval()
    (saved with json.dump).  Returns a dict with per-metric deltas and a
    boolean ``overall_improved`` flag.

    Example::

        baseline = rag.eval(queries, docs, chunk_size=200)
        json.dump(baseline, open("baseline.json", "w"))

        # After tuning:
        current = rag.eval(queries, docs, chunk_size=300)
        report = rag.compare("baseline.json", current)
        print(report["overall_improved"])
    """
    with open(baseline_path, encoding="utf-8") as f:
        baseline_data = json.load(f)

    def _make_result(data: Dict) -> BatchTestResult:
        m = data.get("metrics", {})
        return BatchTestResult(
            queries=[],
            timestamp=data.get("timestamp", ""),
            total_queries=data.get("total_queries", 0),
            avg_top_score=m.get("avg_top_score", 0.0),
            avg_retrieval_score=m.get("avg_retrieval_score", 0.0),
            hit_rate=m.get("hit_rate", 0.0),
            mrr=m.get("mrr", 0.0),
            ndcg_at_k=m.get("ndcg_at_k", 0.0),
            recall_at_k=m.get("recall_at_k", 0.0),
            precision_at_k=m.get("precision_at_k", 0.0),
        )

    def _make_config(data: Dict) -> ChunkConfig:
        c = data.get("config", {})
        return ChunkConfig(
            strategy=c.get("strategy", "token"),
            chunk_size=c.get("chunk_size", 200),
            overlap_percent=c.get("overlap_percent", 10),
            overlap_tokens=c.get("overlap_tokens", 20),
        )

    baseline_result = _make_result(baseline_data)
    current_result = _make_result(current)
    baseline_cfg = _make_config(baseline_data)
    current_cfg = _make_config(current) if current_config is None else ChunkConfig(
        strategy=current_config.get("strategy", "token"),
        chunk_size=current_config.get("chunk_size", 200),
        overlap_percent=current_config.get("overlap_percent", 10),
        overlap_tokens=current_config.get("overlap_tokens", 20),
    )

    comparison = compare_results(baseline_result, current_result, baseline_cfg, current_cfg)
    return comparison.to_dict()


# ---------------------------------------------------------------------------
# export()
# ---------------------------------------------------------------------------

def export(
    format: str = "json",
    strategy: str = "token",
    chunk_size: int = 200,
    overlap_percent: int = 10,
) -> str:
    """Export chunk config for external frameworks.

    format: "json" | "langchain" | "llamaindex"
    """
    overlap_tokens = max(0, int(round(chunk_size * (overlap_percent / 100.0))))
    config = ChunkConfig(
        strategy=strategy,
        chunk_size=chunk_size,
        overlap_percent=overlap_percent,
        overlap_tokens=overlap_tokens,
    )
    return export_config(config, format=format)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _resolve_provider(name: Optional[str]):
    if name is None:
        embedding_provider, _ = await get_best_provider()
        if embedding_provider is None:
            raise RuntimeError(
                "No embedding provider available. "
                "Run Ollama locally or set OPENAI_API_KEY / GOOGLE_API_KEY."
            )
        return embedding_provider
    provider_type = ProviderType(name)
    p = get_provider(provider_type)
    if not await p.check_connection():
        raise RuntimeError(f"Provider '{name}' is not reachable. Check credentials or service.")
    return p


def _ensure_no_running_loop():
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            raise RuntimeError(
                "An event loop is already running. "
                "Call the *_async() variant instead."
            )
    except RuntimeError as e:
        if "no running" in str(e).lower() or "no current" in str(e).lower():
            return
        raise


__all__ = [
    "chunk",
    "eval",
    "eval_async",
    "eval_dataset",
    "eval_dataset_async",
    "optimize",
    "optimize_async",
    "compare",
    "export",
]
