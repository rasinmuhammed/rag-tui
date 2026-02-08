"""Smoke tests for the headless CLI and Python API."""

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
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_cli_chunk_json():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as handle:
        handle.write("Hello world. " * 50)
        temp_path = handle.name

    try:
        output = _run_cli(
            [
                "chunk",
                "--file",
                temp_path,
                "--format",
                "json",
                "--strategy",
                "token",
                "--chunk-size",
                "50",
                "--overlap-percent",
                "10",
            ]
        )
        payload = json.loads(output)
        assert payload["strategy"] == "token"
        assert payload["chunks"], "Expected at least one chunk"
    finally:
        os.unlink(temp_path)


def test_cli_export_json():
    output = _run_cli(["export", "--format", "json", "--chunk-size", "200"])
    payload = json.loads(output)
    assert payload["chunk_size"] == 200
    assert "overlap_tokens" in payload


def test_api_chunk_and_export():
    from rag_tui import api

    chunked = api.chunk("Hello world. " * 10, strategy="sentence", chunk_size=100)
    assert chunked["chunks"], "Expected chunks from API"

    export = api.export(format="langchain", strategy="recursive", chunk_size=400)
    assert "RecursiveCharacterTextSplitter" in export
