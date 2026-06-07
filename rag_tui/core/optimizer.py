"""Automated chunk size recommender for RAG-TUI.

Tests multiple (chunk_size, overlap, strategy) configurations in parallel
against a set of queries and ranks them by retrieval quality (MRR, nDCG@k).
This is the "killer feature" that turns RAG-TUI from a debugger into a
recommendation engine.

Usage (async):
    optimizer = ChunkOptimizer()
    results = await optimizer.optimize(text, queries, provider)
    best = optimizer.recommend(results)

Usage (sync, Python API):
    results = ChunkOptimizer.optimize_sync(text, queries, provider)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from rag_tui.core.engine import ChunkingEngine
from rag_tui.core.metrics import (
    BatchTestResult,
    ChunkConfig,
    QueryResult,
    calculate_batch_metrics,
)
from rag_tui.core.strategies import StrategyType
from rag_tui.core.vector import VectorStore


# Sensible search space defaults
DEFAULT_SIZES = [64, 128, 200, 256, 320, 400, 512]
DEFAULT_OVERLAPS = [5, 10, 15, 20]


@dataclass
class OptimizationResult:
    """Result for a single (chunk_size, overlap, strategy) configuration."""
    chunk_size: int
    overlap_percent: int
    strategy: str
    num_chunks: int
    metrics: BatchTestResult
    rank: int = 0
    score: float = 0.0  # composite ranking score

    @property
    def config(self) -> ChunkConfig:
        overlap_tokens = max(0, int(round(self.chunk_size * self.overlap_percent / 100)))
        return ChunkConfig(
            strategy=self.strategy,
            chunk_size=self.chunk_size,
            overlap_percent=self.overlap_percent,
            overlap_tokens=overlap_tokens,
        )

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "chunk_size": self.chunk_size,
            "overlap_percent": self.overlap_percent,
            "strategy": self.strategy,
            "num_chunks": self.num_chunks,
            "score": round(self.score, 4),
            "metrics": self.metrics.to_dict()["metrics"],
        }

    def summary(self) -> str:
        m = self.metrics
        return (
            f"#{self.rank}  size={self.chunk_size}  overlap={self.overlap_percent}%  "
            f"strategy={self.strategy}  chunks={self.num_chunks}  "
            f"MRR={m.mrr:.3f}  nDCG={m.ndcg_at_k:.3f}  "
            f"Recall={m.recall_at_k:.3f}  Hit={m.hit_rate:.1%}"
        )


@dataclass
class OptimizationReport:
    """Full optimization report with all tested configs ranked."""
    results: List[OptimizationResult]
    best: OptimizationResult
    queries_used: int
    total_configs_tested: int

    def to_dict(self) -> dict:
        return {
            "best": self.best.to_dict(),
            "queries_used": self.queries_used,
            "total_configs_tested": self.total_configs_tested,
            "all_results": [r.to_dict() for r in self.results],
        }

    def summary(self) -> str:
        lines = [
            f"Tested {self.total_configs_tested} configs against {self.queries_used} queries.",
            f"Best: chunk_size={self.best.chunk_size}, overlap={self.best.overlap_percent}%, "
            f"strategy={self.best.strategy}",
            f"      MRR={self.best.metrics.mrr:.3f}  "
            f"nDCG@{self.best.metrics.top_k}={self.best.metrics.ndcg_at_k:.3f}  "
            f"Hit Rate={self.best.metrics.hit_rate:.1%}",
        ]
        return "\n".join(lines)


class ChunkOptimizer:
    """Tests multiple chunking configurations and recommends the best one.

    Ranking uses a weighted composite score:
        score = 0.35*MRR + 0.35*nDCG@k + 0.20*Recall@k + 0.10*Precision@k
    MRR and nDCG are weighted highest because they reward ranking quality,
    not just binary retrieval success.
    """

    WEIGHTS = {
        "mrr": 0.35,
        "ndcg_at_k": 0.35,
        "recall_at_k": 0.20,
        "precision_at_k": 0.10,
    }

    def __init__(
        self,
        sizes: Optional[List[int]] = None,
        overlap_percents: Optional[List[int]] = None,
        strategies: Optional[List[str]] = None,
        top_k: int = 3,
        threshold: float = 0.5,
        max_concurrent: int = 3,
    ) -> None:
        self.sizes = sizes or DEFAULT_SIZES
        self.overlap_percents = overlap_percents or DEFAULT_OVERLAPS
        self.strategies = strategies or ["token"]
        self.top_k = top_k
        self.threshold = threshold
        self.max_concurrent = max_concurrent

    def _composite_score(self, m: BatchTestResult) -> float:
        return (
            self.WEIGHTS["mrr"] * m.mrr
            + self.WEIGHTS["ndcg_at_k"] * m.ndcg_at_k
            + self.WEIGHTS["recall_at_k"] * m.recall_at_k
            + self.WEIGHTS["precision_at_k"] * m.precision_at_k
        )

    async def _eval_single_config(
        self,
        text: str,
        queries: List[str],
        embedding_provider,
        chunk_size: int,
        overlap_percent: int,
        strategy: str,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> OptimizationResult:
        """Evaluate a single (chunk_size, overlap, strategy) config."""
        engine = ChunkingEngine()
        overlap_tokens = max(0, int(round(chunk_size * overlap_percent / 100)))
        strategy_type = StrategyType(strategy)

        raw_chunks = engine.chunk_text(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap_tokens,
            strategy_type=strategy_type,
        )
        chunk_texts = [c[0] for c in raw_chunks]

        if not chunk_texts:
            empty = calculate_batch_metrics([], self.threshold, self.top_k)
            return OptimizationResult(
                chunk_size=chunk_size,
                overlap_percent=overlap_percent,
                strategy=strategy,
                num_chunks=0,
                metrics=empty,
            )

        # Embed chunks
        chunk_embeddings = await embedding_provider.embed_batch(chunk_texts)
        emb_np = np.array(chunk_embeddings, dtype=np.float32)

        store = VectorStore(embedding_dim=emb_np.shape[1])
        store.add_chunks(chunk_texts, emb_np)

        # Run queries
        query_results: List[QueryResult] = []
        for query in queries:
            q_emb = np.array(
                await embedding_provider.embed(query), dtype=np.float32
            )
            matches = store.search(q_emb, top_k=self.top_k)
            retrieved = [(m[0], m[1]) for m in matches]
            top_score = retrieved[0][1] if retrieved else 0.0
            avg_score = sum(s for _, s in retrieved) / len(retrieved) if retrieved else 0.0
            query_results.append(
                QueryResult(
                    query=query,
                    chunks_retrieved=retrieved,
                    top_score=top_score,
                    avg_score=avg_score,
                )
            )

        batch = calculate_batch_metrics(query_results, self.threshold, self.top_k)

        if progress_cb:
            progress_cb(
                f"chunk_size={chunk_size} overlap={overlap_percent}% "
                f"strategy={strategy}: "
                f"MRR={batch.mrr:.3f} nDCG={batch.ndcg_at_k:.3f} "
                f"Hit={batch.hit_rate:.1%}"
            )

        store.shutdown()

        return OptimizationResult(
            chunk_size=chunk_size,
            overlap_percent=overlap_percent,
            strategy=strategy,
            num_chunks=len(chunk_texts),
            metrics=batch,
        )

    async def optimize(
        self,
        text: str,
        queries: List[str],
        embedding_provider,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> OptimizationReport:
        """Test all configurations and return a ranked OptimizationReport.

        Args:
            text: Document text to chunk and index.
            queries: List of test queries to evaluate retrieval.
            embedding_provider: An LLMProvider instance that supports embed().
            progress_cb: Optional callback(message) called after each config.

        Returns:
            OptimizationReport with all results ranked by composite score.
        """
        configs = [
            (size, overlap, strategy)
            for strategy in self.strategies
            for size in self.sizes
            for overlap in self.overlap_percents
        ]

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def bounded_eval(size, overlap, strategy):
            async with semaphore:
                return await self._eval_single_config(
                    text, queries, embedding_provider,
                    size, overlap, strategy, progress_cb,
                )

        tasks = [bounded_eval(s, o, st) for s, o, st in configs]
        raw_results: List[OptimizationResult] = await asyncio.gather(*tasks)

        # Score and rank
        for r in raw_results:
            r.score = self._composite_score(r.metrics)

        ranked = sorted(raw_results, key=lambda r: r.score, reverse=True)
        for i, r in enumerate(ranked, 1):
            r.rank = i

        return OptimizationReport(
            results=ranked,
            best=ranked[0],
            queries_used=len(queries),
            total_configs_tested=len(configs),
        )

    async def ab_test(
        self,
        text: str,
        queries: List[str],
        embedding_provider,
        config_a: dict,
        config_b: dict,
    ) -> dict:
        """Run A/B test between exactly two configurations.

        Each config dict should have: chunk_size, overlap_percent, strategy.
        Returns a dict with both results and the winner.
        """
        result_a, result_b = await asyncio.gather(
            self._eval_single_config(
                text, queries, embedding_provider,
                config_a["chunk_size"],
                config_a.get("overlap_percent", 10),
                config_a.get("strategy", "token"),
            ),
            self._eval_single_config(
                text, queries, embedding_provider,
                config_b["chunk_size"],
                config_b.get("overlap_percent", 10),
                config_b.get("strategy", "token"),
            ),
        )
        score_a = self._composite_score(result_a.metrics)
        score_b = self._composite_score(result_b.metrics)
        result_a.score = score_a
        result_b.score = score_b

        winner = "A" if score_a >= score_b else "B"
        return {
            "winner": winner,
            "config_a": result_a.to_dict(),
            "config_b": result_b.to_dict(),
            "score_delta": round(abs(score_a - score_b), 4),
        }

    def recommend(self, results: List[OptimizationResult]) -> OptimizationResult:
        """Return the top-ranked result (already ranked by optimize())."""
        if not results:
            raise ValueError("No optimization results to recommend from.")
        return min(results, key=lambda r: r.rank)
