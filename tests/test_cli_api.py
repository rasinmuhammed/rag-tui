"""Smoke tests for the headless CLI and Python API.

These tests exercise the public surface without requiring a live embedding
provider (pure chunking + config export paths only).
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_cli(args):
    result = subprocess.run(
        [sys.executable, "-m", "rag_tui", *args],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"CLI failed with args {args}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return result.stdout


# ---------------------------------------------------------------------------
# chunk subcommand
# ---------------------------------------------------------------------------

def test_cli_chunk_json():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Hello world. " * 50)
        tmp = f.name
    try:
        output = _run_cli(["chunk", "--file", tmp, "--strategy", "token",
                           "--chunk-size", "50", "--overlap-percent", "10"])
        payload = json.loads(output)
        assert payload["strategy"] == "token"
        assert payload["chunks"], "Expected at least one chunk"
        assert "stats" in payload
        assert payload["stats"]["total_chunks"] > 0
    finally:
        os.unlink(tmp)


def test_cli_chunk_csv():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Alpha beta gamma. " * 30)
        tmp = f.name
    try:
        output = _run_cli(["chunk", "--file", tmp, "--format", "csv",
                           "--strategy", "sentence", "--chunk-size", "100"])
        lines = [l for l in output.strip().splitlines() if l]
        assert lines[0].startswith("index"), "Expected CSV header"
        assert len(lines) > 1
    finally:
        os.unlink(tmp)


def test_cli_chunk_all_strategies():
    strategies = ["token", "sentence", "paragraph", "recursive", "fixed_chars"]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Test sentence one. Test sentence two. And a third one.\n\n"
                "Second paragraph here. More text follows.\n")
        tmp = f.name
    try:
        for strat in strategies:
            out = _run_cli(["chunk", "--file", tmp, "--strategy", strat, "--chunk-size", "50"])
            payload = json.loads(out)
            assert payload["strategy"] == strat
    finally:
        os.unlink(tmp)


def test_cli_chunk_text_inline():
    out = _run_cli(["chunk", "--text", "Hello world. " * 10, "--chunk-size", "30"])
    payload = json.loads(out)
    assert payload["chunks"]


# ---------------------------------------------------------------------------
# export subcommand
# ---------------------------------------------------------------------------

def test_cli_export_json():
    out = _run_cli(["export", "--format", "json", "--chunk-size", "200"])
    payload = json.loads(out)
    assert payload["chunk_size"] == 200
    assert "overlap_tokens" in payload
    assert "strategy" in payload


def test_cli_export_langchain():
    out = _run_cli(["export", "--format", "langchain", "--chunk-size", "300", "--strategy", "recursive"])
    assert "RecursiveCharacterTextSplitter" in out
    assert "chunk_size=" in out


def test_cli_export_llamaindex():
    out = _run_cli(["export", "--format", "llamaindex", "--chunk-size", "256"])
    assert "SentenceSplitter" in out


# ---------------------------------------------------------------------------
# Python API: chunk
# ---------------------------------------------------------------------------

def test_api_chunk_token():
    from rag_tui import api
    result = api.chunk("Hello world. " * 20, strategy="token", chunk_size=50)
    assert result["chunks"]
    assert result["strategy"] == "token"
    assert result["stats"]["total_chunks"] > 0


def test_api_chunk_all_strategies():
    from rag_tui import api
    text = "Test. " * 30
    for strat in ["token", "sentence", "paragraph", "recursive", "fixed_chars"]:
        result = api.chunk(text, strategy=strat, chunk_size=80)
        assert result["strategy"] == strat


def test_api_chunk_stats_keys():
    from rag_tui import api
    result = api.chunk("Sample text " * 20, chunk_size=50)
    stats = result["stats"]
    assert "total_chunks" in stats
    assert "avg_chunk_size" in stats
    assert "total_characters" in stats
    assert "total_tokens_est" in stats


# ---------------------------------------------------------------------------
# Python API: export
# ---------------------------------------------------------------------------

def test_api_export_json():
    from rag_tui import api
    out = api.export(format="json", chunk_size=200)
    data = json.loads(out)
    assert data["chunk_size"] == 200


def test_api_export_langchain():
    from rag_tui import api
    out = api.export(format="langchain", strategy="recursive", chunk_size=400)
    assert "RecursiveCharacterTextSplitter" in out


def test_api_export_llamaindex():
    from rag_tui import api
    out = api.export(format="llamaindex", chunk_size=256)
    assert "SentenceSplitter" in out


# ---------------------------------------------------------------------------
# Python API: eval_dataset (no provider needed for loading)
# ---------------------------------------------------------------------------

def test_api_load_dataset_csv(tmp_path):
    from rag_tui.core.metrics import load_dataset
    f = tmp_path / "q.csv"
    f.write_text("query,answer\nWhat is RAG?,RAG is...\nHow to chunk?,Split text\n")
    rows = load_dataset(str(f))
    assert len(rows) == 2
    assert rows[0]["query"] == "What is RAG?"


def test_api_load_dataset_jsonl(tmp_path):
    from rag_tui.core.metrics import load_dataset
    f = tmp_path / "q.jsonl"
    f.write_text('{"query":"A"}\n{"query":"B"}\n')
    rows = load_dataset(str(f))
    assert len(rows) == 2
    assert rows[0]["query"] == "A"


# ---------------------------------------------------------------------------
# Python API: compare (no provider needed)
# ---------------------------------------------------------------------------

def test_api_compare(tmp_path):
    from rag_tui import api

    baseline = {
        "timestamp": "2024-01-01",
        "total_queries": 5,
        "metrics": {
            "hit_rate": 0.6,
            "mrr": 0.6,
            "ndcg_at_k": 0.6,
            "recall_at_k": 0.6,
            "precision_at_k": 0.6,
            "avg_top_score": 0.6,
            "avg_retrieval_score": 0.5,
        },
        "config": {
            "strategy": "token",
            "chunk_size": 200,
            "overlap_percent": 10,
            "overlap_tokens": 20,
        },
        "queries": [],
    }
    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text(json.dumps(baseline))

    current = {
        "timestamp": "2024-01-02",
        "total_queries": 5,
        "metrics": {
            "hit_rate": 0.8,
            "mrr": 0.8,
            "ndcg_at_k": 0.8,
            "recall_at_k": 0.8,
            "precision_at_k": 0.8,
            "avg_top_score": 0.8,
            "avg_retrieval_score": 0.7,
        },
        "config": {
            "strategy": "token",
            "chunk_size": 300,
            "overlap_percent": 15,
            "overlap_tokens": 45,
        },
        "queries": [],
    }

    report = api.compare(str(baseline_file), current)
    assert report["overall_improved"] is True
    assert len(report["deltas"]) > 0
    mrr_delta = next(d for d in report["deltas"] if d["metric"] == "mrr")
    assert mrr_delta["improved"] is True
    assert mrr_delta["delta"] == pytest.approx(0.2, abs=1e-6)


# ---------------------------------------------------------------------------
# CLI: compare subcommand
# ---------------------------------------------------------------------------

def test_cli_compare(tmp_path):
    baseline = {
        "timestamp": "2024-01-01",
        "total_queries": 5,
        "metrics": {"hit_rate": 0.5, "mrr": 0.5, "ndcg_at_k": 0.5,
                    "recall_at_k": 0.5, "precision_at_k": 0.5,
                    "avg_top_score": 0.5, "avg_retrieval_score": 0.4},
        "config": {"strategy": "token", "chunk_size": 200,
                   "overlap_percent": 10, "overlap_tokens": 20},
        "queries": [],
    }
    current = {
        "timestamp": "2024-01-02",
        "total_queries": 5,
        "metrics": {"hit_rate": 0.8, "mrr": 0.8, "ndcg_at_k": 0.8,
                    "recall_at_k": 0.8, "precision_at_k": 0.8,
                    "avg_top_score": 0.8, "avg_retrieval_score": 0.7},
        "config": {"strategy": "token", "chunk_size": 300,
                   "overlap_percent": 15, "overlap_tokens": 45},
        "queries": [],
    }
    bl = tmp_path / "baseline.json"
    cu = tmp_path / "current.json"
    bl.write_text(json.dumps(baseline))
    cu.write_text(json.dumps(current))

    out = _run_cli(["compare", "--baseline", str(bl), "--current", str(cu)])
    data = json.loads(out)
    assert data["overall_improved"] is True


import pytest
