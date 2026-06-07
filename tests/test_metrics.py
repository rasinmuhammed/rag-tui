"""Unit tests for the metrics module: MRR, nDCG, Recall, Precision, baseline comparison."""

import json
import math
import os
import tempfile

import pytest

from rag_tui.core.metrics import (
    BatchTestResult,
    ChunkConfig,
    MetricDelta,
    QueryResult,
    calculate_batch_metrics,
    compare_results,
    compute_mean_ndcg,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    load_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_result(scores, query="test query"):
    chunks = [(f"chunk_{i}", s) for i, s in enumerate(scores)]
    top = max(scores) if scores else 0.0
    avg = sum(scores) / len(scores) if scores else 0.0
    return QueryResult(query=query, chunks_retrieved=chunks, top_score=top, avg_score=avg)


# ---------------------------------------------------------------------------
# nDCG
# ---------------------------------------------------------------------------

def test_ndcg_perfect_ranking():
    # Perfectly ordered high scores → nDCG = 1.0
    scores = [0.9, 0.8, 0.7]
    assert compute_ndcg_at_k(scores, k=3) == pytest.approx(1.0)


def test_ndcg_worst_ranking():
    # Reversed scores → nDCG < 1
    scores = [0.7, 0.8, 0.9]
    ndcg = compute_ndcg_at_k(scores, k=3)
    assert ndcg < 1.0
    assert ndcg > 0.0


def test_ndcg_empty():
    assert compute_ndcg_at_k([], k=3) == 0.0


def test_ndcg_single():
    assert compute_ndcg_at_k([0.8], k=3) == pytest.approx(1.0)


def test_ndcg_all_zero():
    assert compute_ndcg_at_k([0.0, 0.0, 0.0], k=3) == 0.0


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------

def test_mrr_first_hit():
    # Top result is above threshold → RR = 1.0
    r = _make_result([0.9, 0.4, 0.3])
    assert compute_mrr([r], threshold=0.5) == pytest.approx(1.0)


def test_mrr_second_hit():
    r = _make_result([0.4, 0.8, 0.3])
    assert compute_mrr([r], threshold=0.5) == pytest.approx(0.5)


def test_mrr_no_hit():
    r = _make_result([0.3, 0.2, 0.1])
    assert compute_mrr([r], threshold=0.5) == pytest.approx(0.0)


def test_mrr_multiple_queries():
    r1 = _make_result([0.9, 0.1])  # RR = 1.0
    r2 = _make_result([0.3, 0.8])  # RR = 0.5
    mrr = compute_mrr([r1, r2], threshold=0.5)
    assert mrr == pytest.approx(0.75)


def test_mrr_empty():
    assert compute_mrr([]) == 0.0


# ---------------------------------------------------------------------------
# Recall@k
# ---------------------------------------------------------------------------

def test_recall_at_k_all_hit():
    r1 = _make_result([0.8, 0.7])
    r2 = _make_result([0.6, 0.5])
    assert compute_recall_at_k([r1, r2], k=2, threshold=0.5) == pytest.approx(1.0)


def test_recall_at_k_none_hit():
    r1 = _make_result([0.3, 0.2])
    assert compute_recall_at_k([r1], k=2, threshold=0.5) == pytest.approx(0.0)


def test_recall_at_k_partial():
    r1 = _make_result([0.8])
    r2 = _make_result([0.2])
    assert compute_recall_at_k([r1, r2], k=1, threshold=0.5) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Precision@k
# ---------------------------------------------------------------------------

def test_precision_at_k_all_relevant():
    r = _make_result([0.9, 0.8, 0.7])
    assert compute_precision_at_k([r], k=3, threshold=0.5) == pytest.approx(1.0)


def test_precision_at_k_half_relevant():
    r = _make_result([0.9, 0.3])  # 1 of 2 above threshold
    assert compute_precision_at_k([r], k=2, threshold=0.5) == pytest.approx(0.5)


def test_precision_at_k_none_relevant():
    r = _make_result([0.2, 0.1])
    assert compute_precision_at_k([r], k=2, threshold=0.5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# calculate_batch_metrics
# ---------------------------------------------------------------------------

def test_calculate_batch_metrics_empty():
    result = calculate_batch_metrics([], threshold=0.5, top_k=3)
    assert result.total_queries == 0
    assert result.mrr == 0.0
    assert result.ndcg_at_k == 0.0


def test_calculate_batch_metrics_full():
    results = [
        _make_result([0.9, 0.7, 0.5], "q1"),
        _make_result([0.3, 0.2, 0.1], "q2"),
    ]
    batch = calculate_batch_metrics(results, threshold=0.5, top_k=3)
    assert batch.total_queries == 2
    assert batch.hit_rate == 0.5  # only q1 has top_score >= 0.5
    assert batch.mrr > 0
    assert batch.ndcg_at_k > 0
    assert 0.0 <= batch.recall_at_k <= 1.0
    assert 0.0 <= batch.precision_at_k <= 1.0


def test_calculate_batch_metrics_to_dict():
    results = [_make_result([0.8])]
    batch = calculate_batch_metrics(results)
    d = batch.to_dict()
    assert "metrics" in d
    assert "mrr" in d["metrics"]
    assert "ndcg_at_k" in d["metrics"]
    assert "recall_at_k" in d["metrics"]
    assert "precision_at_k" in d["metrics"]


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def _make_batch(hit_rate, mrr, ndcg, recall, precision):
    return BatchTestResult(
        queries=[],
        timestamp="",
        total_queries=10,
        avg_top_score=0.5,
        avg_retrieval_score=0.4,
        hit_rate=hit_rate,
        mrr=mrr,
        ndcg_at_k=ndcg,
        recall_at_k=recall,
        precision_at_k=precision,
    )


def _dummy_config(strategy="token", size=200):
    return ChunkConfig(strategy=strategy, chunk_size=size, overlap_percent=10, overlap_tokens=20)


def test_compare_results_improved():
    baseline = _make_batch(0.5, 0.5, 0.5, 0.5, 0.5)
    current = _make_batch(0.7, 0.7, 0.7, 0.7, 0.7)
    cmp = compare_results(baseline, current, _dummy_config(), _dummy_config(size=300))
    assert cmp.overall_improved is True
    assert all(d.improved for d in cmp.deltas)


def test_compare_results_regression():
    baseline = _make_batch(0.8, 0.8, 0.8, 0.8, 0.8)
    current = _make_batch(0.4, 0.4, 0.4, 0.4, 0.4)
    cmp = compare_results(baseline, current, _dummy_config(), _dummy_config(size=100))
    assert cmp.overall_improved is False


def test_compare_results_deltas():
    baseline = _make_batch(0.5, 0.6, 0.7, 0.8, 0.9)
    current = _make_batch(0.6, 0.6, 0.7, 0.8, 0.9)
    cmp = compare_results(baseline, current, _dummy_config(), _dummy_config())
    hit_delta = next(d for d in cmp.deltas if d.metric == "hit_rate")
    assert hit_delta.delta == pytest.approx(0.1, abs=1e-6)
    assert hit_delta.improved is True


def test_metric_delta_display():
    d = MetricDelta(
        metric="mrr", baseline=0.5, current=0.7, delta=0.2, delta_pct=40.0, improved=True
    )
    line = d.to_display()
    assert "mrr" in line
    assert "0.500" in line
    assert "0.700" in line


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def test_load_dataset_csv(tmp_path):
    csv_file = tmp_path / "queries.csv"
    csv_file.write_text("query,answer\nWhat is RAG?,Retrieval-Augmented Generation\nHow to chunk?,Split text\n")
    rows = load_dataset(str(csv_file))
    assert len(rows) == 2
    assert rows[0]["query"] == "What is RAG?"
    assert rows[1]["query"] == "How to chunk?"


def test_load_dataset_jsonl(tmp_path):
    jsonl_file = tmp_path / "queries.jsonl"
    jsonl_file.write_text(
        '{"query": "What is RAG?", "answer": "RAG is..."}\n'
        '{"query": "How to embed?", "answer": "Use a model"}\n'
    )
    rows = load_dataset(str(jsonl_file))
    assert len(rows) == 2
    assert rows[0]["query"] == "What is RAG?"


def test_load_dataset_missing_file():
    with pytest.raises(FileNotFoundError):
        load_dataset("/nonexistent/path/queries.csv")


def test_load_dataset_bad_extension(tmp_path):
    f = tmp_path / "queries.xml"
    f.write_text("<root/>")
    with pytest.raises(ValueError, match="Unsupported"):
        load_dataset(str(f))


def test_load_dataset_tsv(tmp_path):
    tsv_file = tmp_path / "queries.tsv"
    tsv_file.write_text("query\tanswer\nWhat is RAG?\tRAG is...\n")
    rows = load_dataset(str(tsv_file))
    assert len(rows) == 1
    assert rows[0]["query"] == "What is RAG?"
