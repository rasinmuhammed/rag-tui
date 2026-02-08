# RAG-TUI 0.0.4b1 Release Notes

Date: 2026-02-08

## Highlights
- Headless CLI for chunking, evaluation, and export workflows.
- Python API (`rag_tui.api`) for programmatic chunking, eval, and config export.
- `rag-tui` now defaults to the CLI (TUI still launches with no args, or `rag-tui ui`).

## New CLI Commands
- `rag-tui chunk` outputs JSON or CSV chunk data.
- `rag-tui eval` runs batch retrieval evaluation and outputs JSON metrics.
- `rag-tui export` generates framework-ready configs.

## Python API
- `api.chunk(text, strategy, config)` returns JSON-serializable chunk data.
- `api.eval(queries, docs, ...)` runs a headless retrieval evaluation.
- `api.export(format, config)` returns LangChain/LlamaIndex/JSON snippets.

## Notes
- The `custom` chunking strategy remains TUI-only (requires user-provided code).

## Files of Interest
- `rag_tui/cli.py`
- `rag_tui/api.py`
- `pyproject.toml`
- `CHANGELOG.md`

