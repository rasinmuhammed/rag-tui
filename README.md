# RAG-TUI v0.0.4 Beta

## The terminal sidekick for chunking, retrieval, and “why did my RAG answer do that?”

You know the moment when your LLM confidently answers with something you never said? Yeah. RAG‑TUI exists for that exact moment. It lets you *see* your chunks, tweak them in real time, and test retrieval before you ship.

Now with a headless CLI **and** a Python API — because sometimes you want the debug UI, and sometimes you want a CI‑friendly report.

---

## What Is This?

**RAG‑TUI** is a beautiful terminal‑based debugger for Retrieval‑Augmented Generation pipelines. It helps you tune chunking strategies, inspect retrieval results, run batch tests, and export production‑ready configs.

In short: **fewer hallucinations, more receipts**.

---

## Quick Start (30 seconds)

### Install
```bash
pip install rag-tui
```

### Launch the TUI
```bash
rag-tui
```

Pro tip: Press `L` in the app to load a sample document and start experimenting immediately.

---

## New: Headless CLI

For scripts, CI, and “just give me JSON”.

### Chunk text
```bash
rag-tui chunk --file docs.txt --strategy token --chunk-size 200 --overlap-percent 10 --format json
```

### Evaluate retrieval
```bash
rag-tui eval --file docs.txt --queries-file queries.txt --top-k 3 --threshold 0.5
```

### Export config
```bash
rag-tui export --strategy recursive --chunk-size 600 --overlap-percent 15 --format langchain
```

If you want the UI explicitly:
```bash
rag-tui ui
```

---

## New: Python API (Headless Mode)

Use RAG‑TUI directly inside notebooks or pipelines.

```python
from rag_tui import api

# Chunk text
result = api.chunk(
    text="Hello world. Here's a tiny document.",
    strategy="sentence",
    chunk_size=200,
    overlap_percent=10,
)

# Evaluate retrieval
metrics = api.eval(
    queries=["What is this about?"],
    docs="Hello world. Here's a tiny document.",
    strategy="token",
    chunk_size=200,
    overlap_percent=10,
)

# Export config
langchain_code = api.export(
    format="langchain",
    strategy="recursive",
    chunk_size=600,
    overlap_percent=15,
)
```

---

## What You Can Do in the TUI

### 1) Visualize chunking strategies
Pick from six chunkers and instantly see the results:
- Token
- Sentence
- Paragraph
- Recursive
- Fixed characters
- Custom (your own Python)

### 2) Tune chunk size + overlap live
Slide, watch, repeat. No guessing.

### 3) Search and inspect retrieval
Type a query and see exactly which chunks are returned.

### 4) Batch test
Paste a list of queries and get metrics like hit rate and average scores.

### 5) Export production configs
Generate LangChain or LlamaIndex splitters without hand‑translating numbers.

---

## Providers Supported

Pick your flavor:
- **Ollama** (local, free, private)
- **OpenAI**
- **Groq** (fast)
- **Google Gemini**

Set your API key env vars, or just run Ollama locally and you’re good.

---

## File Types Supported

`.txt`, `.md`, `.py`, `.js`, `.json`, `.yaml`, `.pdf`, `.csv`, and more. If it’s text‑ish, RAG‑TUI probably speaks it.

---

## Why People Use RAG‑TUI

- “My chunks were too big, but I couldn’t *see* it.”
- “Retrieval looked fine… until I inspected the top‑k results.”
- “My RAG answers were drifting, and I needed a clean baseline.”

RAG‑TUI makes chunking and retrieval *visible*.

---

## Common Workflows

### Debug a RAG failure in 5 minutes
1. Load your document.
2. Try a few queries in **Search**.
3. Adjust chunk size + overlap.
4. Repeat until retrieval actually makes sense.

### Generate a production config
1. Tune your chunks in the TUI.
2. Export LangChain or LlamaIndex config.
3. Paste into production. Done.

---

## Tips and Tricks

- Press `L` to load sample text instantly.
- Use smaller chunks for QA, larger ones for summarization.
- If in doubt, start with the **Q&A Retrieval** preset.

---

## Installation Notes

Python 3.10+ required.

```bash
pip install rag-tui
```

---

## Contributing

Got ideas? We want them. Open an issue or send a PR. If you fix a weird chunking bug, we’ll probably name a preset after you.

---

## License

MIT. Do cool things.
