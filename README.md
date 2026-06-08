# RAG-TUI

**The terminal-native debugger and optimizer for RAG chunking pipelines.**

You built a RAG system. It retrieves the wrong chunks half the time and you have no idea why. You tweak the chunk size, re-run the pipeline, test again, and still can't see what's happening. RAG-TUI exists to fix that workflow. It makes chunking visible, measurable, and tunable, right in your terminal.

[![PyPI version](https://badge.fury.io/py/rag-tui.svg)](https://pypi.org/project/rag-tui/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What it does

RAG-TUI is a full-featured interactive tool for debugging and optimizing RAG chunking pipelines. It runs entirely in your terminal, works with any embedding provider, and ships with both a visual TUI and a headless CLI and Python API for CI pipelines.

The core loop is simple: load your document, tune your chunking strategy, run retrieval tests, and get real IR metrics back immediately. When you find a configuration that works, export it directly to LangChain or LlamaIndex format.

---

## Install

```bash
pip install rag-tui
```

Python 3.10 or higher required.

---

## Quick start

```bash
# Launch the interactive TUI
rag-tui

# Press L to load a sample document and start exploring immediately
```

If you want to use local embeddings with no API keys, install Ollama and run `ollama serve`. RAG-TUI will detect it automatically.

---

## The TUI: seven tabs, one workflow

### Input tab

Paste your document directly or load from a file. Supports `.txt`, `.md`, `.py`, `.js`, `.ts`, `.json`, `.yaml`, `.yml`, `.xml`, `.html`, `.css`, `.sql`, `.sh`, `.rst`, `.tex`, `.csv`, and `.pdf` out of the box.

The Quick Clean button normalizes whitespace, removes page numbers, and strips horizontal rules before you start chunking. You can also write your own Python cleaner function and apply it here.

### Chunks tab

This is where the core debugging happens. You can see every chunk your document produces with live parameter controls:

- Chunk size slider
- Overlap percentage slider
- Strategy selector (Token, Sentence, Paragraph, Recursive, Fixed Characters, Custom)

Adjust any parameter and the chunks re-render immediately. Overlap text between adjacent chunks is highlighted so you can see exactly how much context is shared. Each chunk card shows its character count, token estimate, and position in the document.

**Six chunking strategies:**

| Strategy | Best for |
|---|---|
| Token | Precise token-budget control, models with hard token limits |
| Sentence | Natural language documents, QA systems |
| Paragraph | Articles, documentation, structured prose |
| Recursive | Code, mixed content, nested structure |
| Fixed Characters | Byte-level control, preprocessing pipelines |
| Custom | Your own Python function, any splitting logic you need |

Keyboard shortcuts `1` through `5` switch strategies instantly.

### Search tab

Type any query and see which chunks get retrieved. Results show similarity scores, rank, and the matched text so you can understand what the vector store is actually doing.

### Batch tab

Paste a list of queries and run a full retrieval evaluation in one shot. You get five metrics back:

- **Hit Rate**: the fraction of queries where at least one relevant chunk was retrieved
- **MRR** (Mean Reciprocal Rank): how high the first relevant chunk ranks on average
- **nDCG@k** (Normalized Discounted Cumulative Gain): a graded measure of ranking quality
- **Recall@k**: how much relevant content is captured in the top k results
- **Precision@k**: how much of the top k results is actually relevant

**Baseline comparison**: save any run as your baseline, tune your config, run again, and get a metric-by-metric delta table. Regressions are flagged immediately.

### Optimize tab

This is the killer feature. Give it a list of test queries, select which strategies and parameter ranges to sweep, and it runs every combination in parallel. The results come back ranked by a composite score: `0.35 * MRR + 0.35 * nDCG@k + 0.20 * Recall@k + 0.10 * Precision@k`. The top result is highlighted as the recommendation. You can apply any result to your current session with one click.

Default sweep: 7 chunk sizes (64, 128, 200, 256, 320, 400, 512) x 4 overlaps (5%, 10%, 15%, 20%) x however many strategies you select. That's up to 140 configurations tested automatically with async concurrency control so it does not hammer your embedding provider.

### Settings tab

Presets for common use cases (QA retrieval, summarization, code search, long documents), custom chunker code editor, custom cleaner code editor, and export buttons for JSON, LangChain, and LlamaIndex formats.

The custom code editor is sandboxed using RestrictedPython. Dunder escapes, file I/O, `__import__`, and dangerous builtins are blocked at the AST level.

### Chat tab

Chat with your document using the indexed chunks as retrieval context. Requires an LLM provider.

---

## Embedding providers

RAG-TUI auto-detects the best available provider at startup. You can also select one explicitly.

| Provider | How to enable |
|---|---|
| Ollama (local) | Run `ollama serve` or set `OLLAMA_HOST` |
| OpenAI | Set `OPENAI_API_KEY` |
| Groq | Set `GROQ_API_KEY` |
| Google Gemini | Set `GOOGLE_API_KEY` |

All providers share a persistent SQLite embedding cache stored at `~/.rag-tui/cache/`. Re-embedding the same text twice is free.

---

## Headless CLI

For scripts, CI pipelines, and "just give me JSON".

### Chunk a document

```bash
rag-tui chunk --file doc.txt --strategy sentence --chunk-size 256 --overlap-percent 10 --format json
rag-tui chunk --file doc.txt --format csv
```

Output includes every chunk with its start/end positions, plus aggregate stats.

### Evaluate retrieval quality

```bash
# Queries from a file
rag-tui eval --file doc.txt --queries-file queries.txt --chunk-size 200 --top-k 3

# Queries from a CSV/JSONL dataset
rag-tui eval --file doc.txt --dataset-file queries.csv --strategy sentence

# Save as a baseline for later comparison
rag-tui eval --file doc.txt --queries-file queries.txt --save-baseline baseline.json
```

### Auto-optimize chunk configuration

```bash
# Sweep all default sizes and overlaps
rag-tui optimize --file doc.txt --queries-file queries.txt

# Narrow the search space
rag-tui optimize --file doc.txt --queries-file queries.txt \
  --strategies token,sentence \
  --sizes 128,200,256,320 \
  --overlaps 5,10,15

# Use a specific provider
rag-tui optimize --file doc.txt --queries-file queries.txt --provider openai
```

### Compare two runs

```bash
# Generate a baseline
rag-tui eval --file doc.txt --queries-file queries.txt --chunk-size 200 --save-baseline v1.json

# Run with new config
rag-tui eval --file doc.txt --queries-file queries.txt --chunk-size 300 > v2.json

# Compare them
rag-tui compare --baseline v1.json --current v2.json
```

### Export a config

```bash
rag-tui export --strategy recursive --chunk-size 600 --overlap-percent 15 --format langchain
rag-tui export --strategy sentence --chunk-size 256 --format llamaindex
```

---

## Python API

Use RAG-TUI in notebooks, evaluation scripts, or CI pipelines.

### Chunking

```python
from rag_tui import api

result = api.chunk(
    text="Your document text here.",
    strategy="sentence",
    chunk_size=256,
    overlap_percent=10,
)

for chunk in result["chunks"]:
    print(chunk["text"], chunk["start"], chunk["end"])
```

### Retrieval evaluation

```python
metrics = api.eval(
    queries=["What is RAG?", "How does chunking affect retrieval?"],
    docs="Your document text here.",
    strategy="token",
    chunk_size=200,
    overlap_percent=10,
    top_k=3,
)

print(metrics["metrics"]["mrr"])
print(metrics["metrics"]["ndcg_at_k"])
print(metrics["metrics"]["hit_rate"])
```

### Dataset evaluation

```python
# Load queries from CSV or JSONL
metrics = api.eval_dataset(
    dataset_path="queries.csv",
    docs="Your document text here.",
    strategy="sentence",
    chunk_size=256,
)
```

### Automated optimization

```python
report = api.optimize(
    text="Your document text here.",
    queries=["What is RAG?", "How does chunking work?"],
    strategies=["token", "sentence"],
    sizes=[128, 200, 256, 320],
    overlaps=[5, 10, 15],
)

best = report["ranked_results"][0]
print(best["chunk_size"], best["strategy"], best["score"])
```

### Baseline comparison

```python
baseline = api.eval(queries, docs, chunk_size=200)
current = api.eval(queries, docs, chunk_size=300)

comparison = api.compare(baseline, current)
print(comparison["overall_improved"])
for delta in comparison["deltas"]:
    print(delta["metric"], delta["delta"], delta["direction"])
```

### Async versions

Every function has an async counterpart: `eval_async`, `eval_dataset_async`, `optimize_async`.

```python
import asyncio
from rag_tui import api

async def main():
    metrics = await api.eval_async(queries, docs, chunk_size=256)
    report = await api.optimize_async(text, queries)

asyncio.run(main())
```

### Config export

```python
langchain_code = api.export(format="langchain", strategy="recursive", chunk_size=600)
llamaindex_code = api.export(format="llamaindex", strategy="sentence", chunk_size=256)
```

---

## Embedding cache

Every embedding result is cached in a local SQLite database at `~/.rag-tui/cache/`. The cache key is the SHA-256 hash of the input text plus the provider and model name, so changing providers invalidates the cache correctly.

This makes repeated evaluations during tuning sessions fast. The optimizer benefit is especially large: a 140-config sweep where half the text chunks overlap across configurations can save 30-50% of embedding API calls.

---

## Docker

```bash
# Build and start with Ollama sidecar
docker-compose up

# Set a remote Ollama instance
OLLAMA_HOST=http://your-server:11434 docker-compose up
```

The `OLLAMA_HOST` environment variable is respected everywhere: TUI, CLI, and API.

---

## Dataset format

The `--dataset-file` flag and `eval_dataset` API accept CSV and JSONL.

**CSV:**
```csv
query,relevant_chunk,answer
What is RAG?,RAG is a technique...,
How does chunking work?,Chunking splits...,
```

**JSONL:**
```jsonl
{"query": "What is RAG?", "relevant_chunk": "RAG is a technique..."}
{"query": "How does chunking work?"}
```

The `query` column is required. `relevant_chunk` and `answer` are optional.

---

## CI integration example

```yaml
# .github/workflows/rag-eval.yml
- name: Evaluate RAG chunking
  run: |
    pip install rag-tui
    rag-tui eval \
      --file docs/knowledge-base.txt \
      --dataset-file tests/eval-queries.csv \
      --strategy sentence \
      --chunk-size 256 \
      --save-baseline baseline.json
    rag-tui compare --baseline baseline-main.json --current baseline.json
```

---

## Common workflows

**Debug a failing query in 5 minutes:**
1. Load your document in the Input tab
2. Switch to Search, type the query that's failing
3. See which chunks are being retrieved and their scores
4. Go to Chunks, adjust size and strategy until the right content appears
5. Run a batch test to verify you didn't break other queries

**Find the optimal config for a new document type:**
1. Prepare 10-20 representative queries in a text file
2. Run `rag-tui optimize --file doc.txt --queries-file queries.txt`
3. Check the ranked results, apply the top config
4. Export to LangChain or LlamaIndex

**Catch regressions before deploying a config change:**
1. Run eval on your current config, save as baseline
2. Make your config change
3. Run eval again
4. Compare: the output tells you which metrics improved and which regressed

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| `L` | Load sample document |
| `R` | Re-chunk current document |
| `D` | Toggle dark/light theme |
| `E` | Export current config |
| `1-5` | Switch chunking strategy |
| `F1` / `?` | Open help overlay |
| `Q` | Quit |

---

## Contributing

Open an issue or pull request. If you find a chunking strategy that works better for a specific document type and have the eval numbers to back it up, that is a welcome contribution.

---

## License

MIT.
