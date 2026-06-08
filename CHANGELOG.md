# Changelog

All notable changes to RAG-TUI are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-06-08

This is the first stable release. The version jump from 0.0.4-beta reflects the addition of production-grade evaluation, automated optimization, and a complete headless API that together make this a useful tool for real RAG development workflows, not just a visualization aid.

### Added

**Automated chunk optimizer**
- New `Optimize` tab in the TUI with full configuration UI
- `ChunkOptimizer` sweeps all combinations of chunk sizes, overlap percentages, and strategies in parallel using asyncio concurrency with a configurable semaphore
- Default search space: 7 sizes (64, 128, 200, 256, 320, 400, 512) x 4 overlaps (5%, 10%, 15%, 20%) x N strategies
- Results ranked by composite score: `0.35 * MRR + 0.35 * nDCG@k + 0.20 * Recall@k + 0.10 * Precision@k`
- Apply any result to the current session with one click
- `rag-tui optimize` CLI subcommand with `--strategies`, `--sizes`, `--overlaps`, `--max-concurrent` flags
- `api.optimize()` and `api.optimize_async()` for programmatic use

**Full IR metric suite**
- MRR (Mean Reciprocal Rank)
- nDCG@k (Normalized Discounted Cumulative Gain)
- Recall@k
- Precision@k
- Hit Rate (previously the only metric)
- All metrics displayed in the Batch tab alongside per-query details

**Baseline comparison and regression detection**
- Save any batch run as a named baseline from the Batch tab
- Compare any subsequent run against the saved baseline
- Metric-by-metric delta table with regression flagging
- `rag-tui compare --baseline baseline.json --current new.json` CLI command
- `api.compare()` for programmatic comparison

**Persistent embedding cache**
- SQLite-backed cache stored at `~/.rag-tui/cache/`
- Cache key: SHA-256 of input text + provider name + model name
- Thread-safe WAL mode, stores numpy arrays as pickle blobs
- Shared across TUI, CLI, and API -- repeated evaluations during tuning sessions are fast
- Especially impactful during optimizer sweeps where overlapping chunks are re-embedded

**Dataset import**
- `--dataset-file` flag for `eval` and `optimize` CLI commands accepts CSV and JSONL
- Required column: `query`. Optional: `relevant_chunk`, `answer`
- `api.eval_dataset()` and `api.eval_dataset_async()` in the Python API

**Python API expansion**
- `eval_dataset()` and `eval_dataset_async()`
- `optimize()` and `optimize_async()`
- `compare()`
- All existing functions (`chunk`, `eval`, `export`) unchanged

**Docker support**
- `Dockerfile` with python:3.11-slim base
- `docker-compose.yml` with Ollama sidecar
- `OLLAMA_HOST` environment variable respected throughout (TUI, CLI, API)

**Security**
- Custom chunker and cleaner code now runs under RestrictedPython
- Blocks dunder attribute access (`__class__`, `__subclasses__`, etc.) at AST compile time
- `open()`, `eval()`, `exec()`, `__import__()` and other dangerous builtins removed from sandbox
- RestrictedPython added as a hard dependency

### Changed

- Batch tab now shows all five IR metrics (previously only hit rate and average scores)
- `save-baseline-btn` and `compare-baseline-btn` added to Batch tab
- Tab names cleaned up (emoji removed for terminal compatibility)
- Graceful degradation when no embedding provider is available: TUI stays open in view-only mode
- `cli.py`: `from rag_tui.app import main` moved inside the `ui` branch so headless commands do not import textual
- `core/__init__.py`: `VectorStore`, `EmbeddingCache`, `OllamaLLM` now lazy-loaded so importing the package does not require usearch or ollama to be installed
- `pyproject.toml`: added `asyncio_mode = "auto"` for pytest, added keywords and classifiers

### Fixed

- `exec()` in custom chunker/cleaner was called with full `__builtins__`, allowing unrestricted file I/O and shell access. Fixed with RestrictedPython sandbox. (Issue #1)

---

## [0.0.4-beta] - 2026-02-08

### Added
- Headless CLI with `chunk`, `eval`, and `export` subcommands
- Python API (`rag_tui.api`) for chunking, evaluation, and config export
- Headless module entrypoint routes through CLI

---

## [0.0.3-beta] - 2025-12-12

### Added
- Overlap visualization: shared text between adjacent chunks highlighted in gold
- Copy to clipboard button on each chunk card
- Strategy keyboard shortcuts 1-5

### Changed
- Chunk cards show overlap indicator and overlap character count
- Help overlay updated with new shortcuts

---

## [0.0.2-beta] - 2025-12-10

### Fixed
- Ollama 500 errors during rapid parameter changes: added retries, exponential backoff, global async lock, and automatic in-flight cancellation

### Changed
- `OllamaProvider.embed_batch()` processes embeddings sequentially with rate limiting

---

## [0.0.1-beta] - 2025-12-05

### Added
- Initial release
- Interactive TUI for chunking visualization and debugging
- Six chunking strategies: Token, Sentence, Paragraph, Recursive, Fixed Characters, Custom
- Multi-provider support: Ollama, OpenAI, Groq, Google Gemini
- Real-time chunk visualization with statistics
- Semantic search with vector store
- Batch query testing
- Config export for JSON, LangChain, LlamaIndex
- Quick text cleaning utilities
- Preset configurations
- Chat interface with RAG context
