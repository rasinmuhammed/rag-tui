# Changelog

All notable changes to RAG-TUI are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-09-03

Three things in this release. RAG-TUI now works the second `pip install` finishes, with no
provider of any kind; it can find retrieval defects in a corpus without being given a single
labelled query; and it can be handed directly to an agent as an MCP server.

### Added

**MCP server, so an agent can tune retrieval directly**

- `rag_tui.mcp_server` exposes six tools over the Model Context Protocol:
  `diagnose_corpus`, `chunk_document`, `evaluate_retrieval`, `optimize_chunking`,
  `export_chunking_config`, `list_strategies`, matching the CLI one for one
- `rag-tui mcp` runs it over stdio. Point a client's config at
  `{"command": "rag-tui", "args": ["mcp"]}` and it just works
- Every tool takes `text` or `path`, never both, matching the CLI's `--text`/`--file` split
- Output is capped by default (`max_findings`, `max_chunks`, `max_results`, all overridable)
  so a large corpus cannot flood the calling agent's context window in one call; the true
  totals (`findings_omitted`, `chunks_omitted`, `results_omitted`) are always included
- Every result names which embedder actually ran, the same disclosure `doctor` already made,
  so a caller cannot mistake a lexical-fallback score for a semantic one
- Lives behind a `mcp` extra (`pip install "rag-tui[mcp]"`). It pulls in a real dependency
  tree (starlette, uvicorn, pydantic) that most installs never touch, so it stays opt-in;
  `rag-tui mcp` without it prints the install command instead of a stack trace
- 29 unit tests exercise every tool through `call_tool()`, plus one end-to-end test that
  drives a real subprocess over the actual JSON-RPC/stdio wire protocol
- `api.eval_async()` and `api.optimize_async()` now report the embedder that actually ran
  rather than echoing back the requested provider name (or the literal string `"auto"` when
  none was given), matching what `doctor_async()` already did. The CLI's own separate `eval`
  implementation had the identical bug and got the identical fix

**Built-in embedder, so there is nothing to set up**

- New `rag_tui.core.local_embed` module and a `local` provider. Roughly 150 lines of numpy
  implementing the signed hashing trick over word unigrams, word bigrams and character
  4-grams, with sublinear term frequency and L2 normalisation
- No model download, no server, no API key, no network. `pip install rag-tui` followed by
  `rag-tui doctor --file doc.md` produces a full report in well under a second
- Selected automatically only when nothing better is reachable, so Ollama, OpenAI and Gemini
  still win when they are available
- Hashing uses `zlib.crc32` and `zlib.adler32` rather than the salted built-in `hash()`, so
  vectors are identical across processes and the on-disk embedding cache stays valid
- Declares `supports_llm = False` and raises from `generate()`. Chat and `--use-judge` refuse
  rather than dressing up a bag-of-words as a relevance judgement
- Every run says which embedder produced its numbers, and falling back to the built-in one
  prints a warning explaining that it reads vocabulary rather than meaning

**`rag-tui doctor`, retrievability analysis with no query set**

- New `rag_tui.core.doctor` module. Embeds the corpus once and analyses the chunk-to-chunk
  cosine similarity matrix for defects that hurt retrieval regardless of query
- Hub detection using k-occurrence skewness from the hubness literature. Flagged at 2 sigma
  above the mean and 3x the average neighbourhood count, tuned so healthy corpora say nothing
- Orphan detection, split by cause: chunks shadowed by a near-twin that outranks them
  everywhere (the report names the twin), and isolated content islands
- Duplicate clusters, grouped transitively with union-find
- Fracture detection for boundaries that cut a sentence, requiring both a missing terminal
  punctuation mark and a lowercase continuation so headings and lists are left alone
- Boilerplate detection for chunks too short or too repetitive to carry meaning, reported once
  per class rather than once per chunk
- A composite retrievability score out of 100 with a letter grade, weighted by how directly
  each defect degrades retrieval and rate-based so it compares across corpus sizes
- Every finding carries a severity, the chunk indices, the retrieval consequence, and a fix
- `--fail-under SCORE` exits 1, so a knowledge base can be gated in CI with no golden queries
- `--format json` for scripts, `--neighbors` to tune the neighbourhood size k
- `api.doctor()` and `api.doctor_async()`
- 94 new unit tests across the embedder, the diagnostics, and the new strategies

**Two structure-aware chunking strategies**

- `markdown` splits on ATX headings and prefixes every chunk with its heading trail, so a
  chunk carries where it came from. Bare headings are never emitted as their own chunk,
  which is the most common source of unretrievable fragments. Fenced code blocks are skipped
  so a Python comment is not mistaken for an h1. On the demo corpus this alone moves the
  retrievability score from 86 to 93
- `hierarchical` returns small chunks for matching and attaches the larger passage each came
  from in `metadata["parent_text"]`, the small-to-find/large-to-answer pattern
- Both are wired into the TUI dropdown, the optimizer sweep, and keys `6` and `7`
- The TUI's three hardcoded strategy lookup tables are now derived from the enum, so adding a
  strategy no longer means remembering to update three places
- `ChunkingEngine.chunk_detailed()` returns full `ChunkResult` objects with metadata, and
  `rag-tui chunk --format json` and `api.chunk()` now both emit that metadata. Hierarchical is
  unusable headlessly without it; `api.chunk()` was still calling the metadata-discarding
  `chunk_text()` even after the CLI's own copy of this same code was fixed, and the MCP
  server's tests are what caught it

**Demo**

- `assets/demo/doctor.tape` records the README GIF with VHS, so the demo is scripted and
  regenerates as the output changes rather than going stale

### Fixed

- **CLI exit codes were always 0.** `main()` called `sys.exit()` inside a `try` whose handler
  caught `SystemExit` and discarded it, so every non-zero exit code was swallowed. A provider
  outage printed `error: Provider 'openai' is not reachable` and still exited 0, which meant
  CI pipelines running `rag-tui eval` read a total failure as a success. `KeyboardInterrupt`
  now exits 130
- `rag-tui --version` did not exist. It does now
- `test_package_version` asserted a literal version string, breaking on every release. It now
  checks that `__init__.py` and `pyproject.toml` agree, which is the invariant worth guarding

### Changed

- README rewritten. It leads with the zero-setup path and the query-free diagnostics, states
  plainly what the built-in embedder cannot do, and has a section listing what the tool does
  not handle
- The four TUI screenshots in `assets/` are now actually shown in the README

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

**LLM-as-Judge evaluation**
- New "Run with Judge" button in the Batch tab
- `LLMJudge` class in `core/judge.py` scores each retrieved chunk for relevance (0-1) using your LLM provider
- `score_faithfulness`: asks whether the retrieved chunks are sufficient to answer the query
- All five IR metrics (MRR, nDCG@k, Recall@k, Precision@k, Hit Rate) use judge labels instead of cosine similarity when available, so the numbers reflect actual retrieval quality
- Works fully offline with Ollama. No OpenAI key required for judging.
- `--use-judge` flag added to `rag-tui eval` CLI command
- `QueryResult.relevance_labels` and `QueryResult.faithfulness` fields added
- `BatchTestResult.eval_mode` field labels every result as "judge", "ground_truth", or "similarity" so proxy metrics are never confused for real ones
- `ground_truth_labels()` helper for dataset-based evaluation with known relevant chunks

**Security**
- Custom chunker and cleaner code now runs under RestrictedPython
- Blocks dunder attribute access (`__class__`, `__subclasses__`, etc.) at AST compile time
- `open()`, `eval()`, `exec()`, `__import__()` and other dangerous builtins removed from sandbox
- RestrictedPython added as a hard dependency

**UI/UX overhaul**
- Persistent status strip docked below the strategy bar shows: active strategy, chunk size, overlap, provider, and chunk count at all times. No more hunting through tabs to remember what config is active.
- Batch results now display as color-coded metric bars (green >= 0.7, amber >= 0.4, red < 0.4) with Rich markup. Metric health is readable at a glance.
- Baseline comparison redesigned with ▲/▼ arrows, color-coded deltas, and absolute + percentage change per metric.
- Chunk cards reduced from max-height 20 to 13, making more chunks visible at once. Quality indicators replaced with clean ASCII (◉/◎/◌) instead of emoji for compatibility. Border style changed to `tall` for better depth perception. Color rotation preserved.
- Search result cards: rank and similarity bar combined into a single header line. Bar width increased from 30 to 36 characters for better score granularity.
- Full CSS design system in `app.tcss`: consistent border hierarchy (tall=interactive, solid=structural), spacing rhythm, and semantic color usage for metrics.

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
