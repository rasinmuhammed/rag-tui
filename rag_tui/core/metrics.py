"""Metrics and batch testing for RAG-TUI.

Provides standard IR metrics: MRR, nDCG@k, Recall@k, Precision@k,
hit rate, and baseline comparison for regression detection.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import json
from datetime import datetime


@dataclass
class QueryResult:
    """Result of a single query."""
    query: str
    chunks_retrieved: List[Tuple[str, float]]  # (chunk_text, score)
    top_score: float
    avg_score: float
    response: Optional[str] = None
    # Optional: index of the ground-truth relevant chunk within chunks_retrieved
    ground_truth_rank: Optional[int] = None


@dataclass
class BatchTestResult:
    """Results of batch testing with full IR metric suite."""
    queries: List[QueryResult]
    timestamp: str
    total_queries: int
    # Score-based
    avg_top_score: float
    avg_retrieval_score: float
    hit_rate: float          # fraction with top_score >= threshold
    # Standard IR metrics
    mrr: float = 0.0         # Mean Reciprocal Rank
    ndcg_at_k: float = 0.0  # nDCG @ top_k
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    top_k: int = 3
    threshold: float = 0.5

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "total_queries": self.total_queries,
            "top_k": self.top_k,
            "threshold": self.threshold,
            "metrics": {
                "hit_rate": round(self.hit_rate, 4),
                "mrr": round(self.mrr, 4),
                "ndcg_at_k": round(self.ndcg_at_k, 4),
                "recall_at_k": round(self.recall_at_k, 4),
                "precision_at_k": round(self.precision_at_k, 4),
                "avg_top_score": round(self.avg_top_score, 4),
                "avg_retrieval_score": round(self.avg_retrieval_score, 4),
            },
            "queries": [
                {
                    "query": q.query,
                    "top_score": round(q.top_score, 4),
                    "avg_score": round(q.avg_score, 4),
                    "chunks_count": len(q.chunks_retrieved),
                }
                for q in self.queries
            ],
        }

    def summary_line(self) -> str:
        return (
            f"Hit Rate: {self.hit_rate:.1%}  "
            f"MRR: {self.mrr:.3f}  "
            f"nDCG@{self.top_k}: {self.ndcg_at_k:.3f}  "
            f"Recall@{self.top_k}: {self.recall_at_k:.3f}  "
            f"P@{self.top_k}: {self.precision_at_k:.3f}"
        )


@dataclass
class ChunkConfig:
    """Chunking configuration for export and comparison."""
    strategy: str
    chunk_size: int
    overlap_percent: int
    overlap_tokens: int
    label: str = ""

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "chunk_size": self.chunk_size,
            "overlap_percent": self.overlap_percent,
            "overlap_tokens": self.overlap_tokens,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_langchain(self) -> str:
        return f'''from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size={self.chunk_size * 4},  # ~{self.chunk_size} tokens
    chunk_overlap={self.overlap_tokens * 4},
    length_function=len,
)
'''

    def to_llamaindex(self) -> str:
        return f'''from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size={self.chunk_size * 4},
    chunk_overlap={self.overlap_tokens * 4},
)
'''

    @classmethod
    def from_dict(cls, data: dict) -> "ChunkConfig":
        return cls(
            strategy=data.get("strategy", "token"),
            chunk_size=data.get("chunk_size", 200),
            overlap_percent=data.get("overlap_percent", 10),
            overlap_tokens=data.get("overlap_tokens", 20),
        )


@dataclass
class MetricDelta:
    """Change in a single metric between baseline and current run."""
    metric: str
    baseline: float
    current: float
    delta: float
    delta_pct: float
    improved: bool

    def to_display(self) -> str:
        sign = "+" if self.delta >= 0 else ""
        trend = "▲" if self.improved else "▼"
        return (
            f"{trend} {self.metric}: "
            f"{self.baseline:.3f} → {self.current:.3f} "
            f"({sign}{self.delta:.3f}, {sign}{self.delta_pct:.1f}%)"
        )


@dataclass
class BaselineComparison:
    """Comparison between a saved baseline and current run."""
    baseline_config: ChunkConfig
    current_config: ChunkConfig
    baseline_result: BatchTestResult
    current_result: BatchTestResult
    deltas: List[MetricDelta] = field(default_factory=list)
    overall_improved: bool = False

    def to_dict(self) -> dict:
        return {
            "baseline_config": self.baseline_config.to_dict(),
            "current_config": self.current_config.to_dict(),
            "baseline_metrics": self.baseline_result.to_dict()["metrics"],
            "current_metrics": self.current_result.to_dict()["metrics"],
            "deltas": [
                {
                    "metric": d.metric,
                    "baseline": d.baseline,
                    "current": d.current,
                    "delta": d.delta,
                    "delta_pct": d.delta_pct,
                    "improved": d.improved,
                }
                for d in self.deltas
            ],
            "overall_improved": self.overall_improved,
        }


# ---------------------------------------------------------------------------
# IR metric computation
# ---------------------------------------------------------------------------

def _dcg_at_k(scores: List[float], k: int) -> float:
    """Discounted Cumulative Gain at rank k using scores as relevance grades."""
    top = scores[:k]
    return sum(score / math.log2(i + 2) for i, score in enumerate(top))


def compute_ndcg_at_k(scores: List[float], k: int) -> float:
    """nDCG@k using similarity scores as graded relevance (no binary labels needed)."""
    if not scores:
        return 0.0
    ideal = sorted(scores, reverse=True)
    idcg = _dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return _dcg_at_k(scores, k) / idcg


def compute_mrr(results: List[QueryResult], threshold: float = 0.5) -> float:
    """Mean Reciprocal Rank.

    For each query, the reciprocal rank is 1/rank of the first retrieved chunk
    whose score >= threshold. If none qualify, RR = 0.
    """
    if not results:
        return 0.0
    rrs = []
    for r in results:
        rr = 0.0
        for rank, (_, score) in enumerate(r.chunks_retrieved, 1):
            if score >= threshold:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    return sum(rrs) / len(rrs)


def compute_recall_at_k(results: List[QueryResult], k: int = 3, threshold: float = 0.5) -> float:
    """Recall@k: fraction of queries where any top-k result exceeds the threshold."""
    if not results:
        return 0.0
    hits = sum(
        1 for r in results
        if any(score >= threshold for _, score in r.chunks_retrieved[:k])
    )
    return hits / len(results)


def compute_precision_at_k(results: List[QueryResult], k: int = 3, threshold: float = 0.5) -> float:
    """Precision@k: mean fraction of top-k results that exceed the threshold."""
    if not results:
        return 0.0
    precisions = []
    for r in results:
        top = r.chunks_retrieved[:k]
        if not top:
            precisions.append(0.0)
        else:
            relevant = sum(1 for _, score in top if score >= threshold)
            precisions.append(relevant / len(top))
    return sum(precisions) / len(precisions)


def compute_mean_ndcg(results: List[QueryResult], k: int = 3) -> float:
    """Mean nDCG@k across all queries."""
    if not results:
        return 0.0
    scores_list = [[s for _, s in r.chunks_retrieved] for r in results]
    ndcg_vals = [compute_ndcg_at_k(scores, k) for scores in scores_list]
    return sum(ndcg_vals) / len(ndcg_vals)


# ---------------------------------------------------------------------------
# Batch metrics
# ---------------------------------------------------------------------------

def calculate_batch_metrics(
    results: List[QueryResult],
    threshold: float = 0.5,
    top_k: int = 3,
) -> BatchTestResult:
    """Calculate the full IR metric suite from batch query results."""
    if not results:
        return BatchTestResult(
            queries=[],
            timestamp=datetime.now().isoformat(),
            total_queries=0,
            avg_top_score=0.0,
            avg_retrieval_score=0.0,
            hit_rate=0.0,
            mrr=0.0,
            ndcg_at_k=0.0,
            recall_at_k=0.0,
            precision_at_k=0.0,
            top_k=top_k,
            threshold=threshold,
        )

    avg_top = sum(r.top_score for r in results) / len(results)
    avg_ret = sum(r.avg_score for r in results) / len(results)
    hits = sum(1 for r in results if r.top_score >= threshold)

    return BatchTestResult(
        queries=results,
        timestamp=datetime.now().isoformat(),
        total_queries=len(results),
        avg_top_score=avg_top,
        avg_retrieval_score=avg_ret,
        hit_rate=hits / len(results),
        mrr=compute_mrr(results, threshold),
        ndcg_at_k=compute_mean_ndcg(results, top_k),
        recall_at_k=compute_recall_at_k(results, top_k, threshold),
        precision_at_k=compute_precision_at_k(results, top_k, threshold),
        top_k=top_k,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def compare_results(
    baseline: BatchTestResult,
    current: BatchTestResult,
    baseline_config: ChunkConfig,
    current_config: ChunkConfig,
) -> BaselineComparison:
    """Compare current run against a saved baseline, flagging regressions."""
    metric_names = ["hit_rate", "mrr", "ndcg_at_k", "recall_at_k", "precision_at_k"]
    baseline_vals = [
        baseline.hit_rate, baseline.mrr, baseline.ndcg_at_k,
        baseline.recall_at_k, baseline.precision_at_k,
    ]
    current_vals = [
        current.hit_rate, current.mrr, current.ndcg_at_k,
        current.recall_at_k, current.precision_at_k,
    ]

    deltas = []
    improved_count = 0
    for name, bv, cv in zip(metric_names, baseline_vals, current_vals):
        delta = cv - bv
        delta_pct = (delta / bv * 100) if bv != 0 else 0.0
        improved = delta >= 0
        if improved:
            improved_count += 1
        deltas.append(MetricDelta(
            metric=name,
            baseline=round(bv, 4),
            current=round(cv, 4),
            delta=round(delta, 4),
            delta_pct=round(delta_pct, 2),
            improved=improved,
        ))

    return BaselineComparison(
        baseline_config=baseline_config,
        current_config=current_config,
        baseline_result=baseline,
        current_result=current,
        deltas=deltas,
        overall_improved=improved_count >= len(metric_names) // 2 + 1,
    )


# ---------------------------------------------------------------------------
# Dataset loading (CSV / JSONL)
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load a query dataset from CSV or JSONL.

    CSV columns: query  (required), relevant_chunk (optional), answer (optional)
    JSONL keys:  query  (required), relevant_chunk (optional), answer (optional)

    Returns a list of dicts with at least a "query" key.
    """
    import csv
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = p.suffix.lower()

    if suffix == ".jsonl":
        rows = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if suffix in (".csv", ".tsv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        with open(p, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            rows = [dict(row) for row in reader]
        return rows

    raise ValueError(f"Unsupported dataset format '{suffix}'. Use .jsonl, .csv, or .tsv")


# ---------------------------------------------------------------------------
# Config export
# ---------------------------------------------------------------------------

def export_config(config: ChunkConfig, format: str = "json") -> str:
    if format == "json":
        return config.to_json()
    elif format == "langchain":
        return config.to_langchain()
    elif format == "llamaindex":
        return config.to_llamaindex()
    else:
        raise ValueError(f"Unknown format: {format}")
