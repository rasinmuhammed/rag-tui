# RAG-TUI

Your retrieval keeps returning the wrong chunk and you cannot see why.

So you change `chunk_size` from 512 to 256. Re-embed. Re-run. You get a different wrong answer. You put it back to 512 and try a different overlap. Somewhere in there you stopped engineering and started turning a dial in the dark.

RAG-TUI puts the dial and the readout on the same screen.

[![PyPI version](https://badge.fury.io/py/rag-tui.svg)](https://pypi.org/project/rag-tui/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![rag-tui doctor finding defects in a corpus, then the same corpus after switching to markdown chunking](assets/doctor-demo.gif)

## Install

```bash
pip install rag-tui
```

Python 3.10 or newer. That is the whole setup. No model to download, no server to start, no API key.

```bash
rag-tui doctor --file your-docs.md
```

That command works on a laptop in airplane mode, thirty seconds after the install finishes. Read on for why that is unusual, and for what it costs you.

## Nobody has a golden query set

Ragas, TruLens and DeepEval all open the same way: bring your labelled queries. It is a fair thing to ask for. It is also why most teams never evaluate their retrieval at all. Writing fifty representative questions with known-correct answers is a solid week of work, and that week competes against shipping features. It loses every time.

`rag-tui doctor` does not ask for queries. It embeds your chunks once and then studies the shape of the space they landed in. Plenty of things that wreck retrieval leave a mark there before any query shows up.

```bash
rag-tui doctor --file knowledge-base.md
```

```
╭───────────────────────────── rag-tui doctor ─────────────────────────────╮
│ Retrievability  86/100  (B)                                              │
│ 17 chunks · paragraph · size 25 · overlap 10% · hubness skew -0.22       │
│ embeddings: Ollama (Local)                                               │
╰──────────────────────────────────────────────────────────────────────────╯

▲ DUPLICATE  4 near-duplicate chunks (#1, #6, #11, #16)
   Peak similarity 1.000. "Copyright Acme Corp. All rights reserved…"
   → Deduplicate before indexing. If the repetition comes from overlap,
     lower --overlap-percent; if it is in the source, strip it in cleaning.

• BOILERPLATE  5 chunks too short to be retrievable (#0, #2, #7, #9, #12)
   Under 80 characters. #0 "# Acme Cloud Knowledge Base"; #2 "## Billing…"
   → This usually means a heading level is being split into its own chunk.

• ORPHAN  Chunk #12 is unreachable from any neighbourhood
   "## Data residency". Never appears in another chunk's neighbourhood.
   → Low reachability. Verify a representative query actually retrieves it.
```

Four findings, each naming the chunk and what to do about it. No queries were involved.

### The five checks

**Hub.** A chunk that sits in far more top-k neighbourhoods than its share. Hubs come back for questions they have nothing to do with, and every slot a hub takes is a slot the right chunk does not get. This is the [hubness phenomenon](https://www.jmlr.org/papers/v11/radovanovic10a.html), a property of high-dimensional space that has been studied since 2010 and that no other RAG tool surfaces. In practice a hub is almost always a generic summary paragraph or a footer that got embedded along with everything else.

**Orphan.** A chunk that appears in nobody's neighbourhood. There are two ways to end up here and they need opposite fixes, so the report tells you which one you have. Either a near-twin is beating it in every ranking they both enter (the report names the twin), or it is a content island with nothing semantically adjacent to it. The first is a dedup problem. The second might be your most valuable unique content, or it might be a mangled table.

**Duplicate.** Near-identical chunks, grouped into clusters. Two copies of a fact do not double your chances of retrieving it. They split the ranking mass and take two slots in a top-3, which means one query can spend its entire context budget restating a single sentence.

**Fracture.** A chunk boundary that cuts a sentence in half. The claim ends up split across two chunks and neither half answers the question. The check only fires when the first chunk ends without terminal punctuation *and* the next one starts lowercase, so headings and bullet lists do not trip it.

**Boilerplate.** Chunks too short or too repetitive to mean anything. Nav bars, page numbers, stray headings. They flatten toward the corpus average and crowd out real content.

All of it comes out of a single similarity matrix over your chunk embeddings, so the cost is one embedding pass and some numpy. Embeddings are cached on disk, so the second run after a config change is nearly free.

### Putting it in CI

The report ends in a score out of 100, and `--fail-under` turns that score into a build gate.

```yaml
# .github/workflows/rag-doctor.yml
- name: Check corpus retrievability
  run: |
    pip install rag-tui
    rag-tui doctor --file docs/knowledge-base.md --chunk-size 256 --fail-under 75
```

Now a pull request that pastes a duplicate section into the knowledge base fails the build, and it does so without anyone maintaining a golden query set. If you do have one, run `rag-tui eval` alongside it and get both.

## Embeddings

RAG-TUI picks the best embedder it can find at startup, in this order.

| Provider | How to enable | Notes |
|---|---|---|
| Ollama | `ollama serve` or set `OLLAMA_HOST` | Local, free, semantic. The one to use |
| OpenAI | `OPENAI_API_KEY` | Best quality, costs money, sends your text to OpenAI |
| Google Gemini | `GOOGLE_API_KEY` | Free tier available |
| Groq | `GROQ_API_KEY` | Text generation only, no embeddings |
| Built-in | Nothing. It is always there | Lexical only. See below |

### About the built-in one

The fallback is roughly 150 lines of numpy: the hashing trick over word unigrams, bigrams and character 4-grams, with sublinear term frequency and L2 normalisation. There is no model file and no network call, which is why `pip install` is the entire setup.

It is genuinely useful and genuinely limited, and the difference matters, so RAG-TUI says which one it used at the top of every report and warns you on stderr when it falls back.

What it catches perfectly well: duplicate passages, boilerplate, stray headings, fractured sentences, structural damage of every kind. On the example corpus above it scores 88 where Ollama scores 86, and it flags the same defects.

What it cannot see: paraphrase. "How do I get my money back" and "refund policy" share no vocabulary, so to a lexical embedder they are unrelated. If your queries reword things, and real user queries always do, start Ollama.

Think of it as the smoke test you can always run, not the encoder you ship on.

## The TUI

`rag-tui` with no arguments opens the full interface. A status strip under the tab bar always shows the active strategy, size, overlap, provider and chunk count, so you never have to go looking for what is currently loaded.

### Input

![Input tab](assets/input-tab.png)

Paste text or load a file. Handles `.txt`, `.md`, `.pdf`, `.py`, `.js`, `.ts`, `.json`, `.yaml`, `.xml`, `.html`, `.css`, `.sql`, `.sh`, `.rst`, `.tex` and `.csv`.

Quick Clean normalises whitespace, strips page numbers and removes horizontal rules. You can also write your own cleaner in Python and run it here.

### Chunks

![Chunks tab, re-rendering live as you drag the size slider](assets/chunks-tab.png)

This is where you actually debug. Chunk size and overlap are sliders, strategy is a dropdown, and the chunk list re-renders as you move them. Overlapping text between adjacent chunks is highlighted so you can see exactly how much context is shared, and each card shows its character count, token estimate and position.

Seven strategies, and keys `1` through `7` switch between them instantly.

| Strategy | Reach for it when |
|---|---|
| Token | Your model has a hard token ceiling you cannot cross |
| Sentence | Prose and QA, where a split mid-sentence costs you the answer |
| Paragraph | Articles and documentation with real paragraph structure |
| Recursive | Code and mixed content with nested structure |
| Fixed characters | You need byte-level predictability in a preprocessing step |
| Markdown | Docs and wikis. Splits on headings and keeps the heading trail |
| Hierarchical | Small chunks to match on, larger parent windows to answer from |
| Custom | Your own Python function, sandboxed |

### Search

![Search tab](assets/search-tab.png)

Type a query, see what comes back and at what score. This is the fastest way to answer "why on earth did it return that."

### Batch

Paste a list of queries and get the full IR suite back as colour-coded bars: hit rate, MRR, nDCG@k, recall@k and precision@k.

Two modes. **Run Batch Test** scores by cosine threshold, takes seconds, and is a proxy. **Run with Judge** has your local LLM score each retrieved chunk for relevance from 0 to 1, then computes the metrics from those scores instead of from vector distance. Judge mode also reports faithfulness, meaning whether the retrieved chunks are actually sufficient to answer. Results are labelled with which mode produced them so you never mistake a proxy for the real thing.

Save any run as a baseline, change your config, run again, and you get a metric-by-metric delta table with regressions in red.

### Optimize

Give it queries, pick which strategies and ranges to sweep, and it runs every combination concurrently and ranks them by `0.35·MRR + 0.35·nDCG@k + 0.20·Recall@k + 0.10·Precision@k`.

The default sweep is 7 sizes by 4 overlaps by however many strategies you tick, so up to 140 configurations. Concurrency is capped so you do not hammer your embedding provider. Apply any result to the session with one click.

### Settings and Chat

Presets for common jobs (QA, summarisation, code search, long documents), the custom chunker and cleaner editors, and export to JSON, LangChain or LlamaIndex.

Custom code runs under a RestrictedPython AST sandbox. Dunder escapes, file I/O, `__import__` and the dangerous builtins are blocked before evaluation.

![Chat tab](assets/chat-tab.png)

Chat queries your document through the indexed chunks. Needs a real LLM, so Ollama or an API key.

## Command line

Everything the TUI does, minus the pictures, for scripts and CI.

```bash
# Diagnose. No queries needed
rag-tui doctor --file doc.md
rag-tui doctor --file doc.md --format json
rag-tui doctor --file doc.md --fail-under 70
rag-tui doctor --file doc.md --strategy sentence --chunk-size 256 --neighbors 10

# Chunk
rag-tui chunk --file doc.txt --strategy sentence --chunk-size 256 --format json
rag-tui chunk --file doc.txt --format csv

# Evaluate
rag-tui eval --file doc.txt --queries-file queries.txt --top-k 3
rag-tui eval --file doc.txt --queries-file queries.txt --use-judge
rag-tui eval --file doc.txt --dataset-file queries.csv --save-baseline v1.json

# Sweep
rag-tui optimize --file doc.txt --queries-file queries.txt
rag-tui optimize --file doc.txt --queries-file queries.txt \
  --strategies token,sentence --sizes 128,200,256,320 --overlaps 5,10,15

# Compare and export
rag-tui compare --baseline v1.json --current v2.json
rag-tui export --strategy recursive --chunk-size 600 --format langchain
```

`--neighbors` sets the neighbourhood size k for hub and orphan detection. Raise it on large corpora where top-5 is not representative of how you actually retrieve.

Every command exits non-zero on failure, so `set -e` behaves the way you expect.

## MCP server

Point Claude Code, Claude Desktop, or Cursor at RAG-TUI and an agent can tune a chunking
pipeline directly, in the same session where it is writing the retrieval code.

```bash
pip install "rag-tui[mcp]"
```

The extra is separate from the base install on purpose. It pulls in a real dependency tree
(starlette, uvicorn, pydantic) that most people never touch, and pulling that in for everyone
would work against the entire point of this project, which is that the base install has
nothing to set up.

Add it to a client's config:

```json
{
  "mcpServers": {
    "rag-tui": { "command": "rag-tui", "args": ["mcp"] }
  }
}
```

Six tools, matching the CLI one for one:

| Tool | What it does |
|---|---|
| `diagnose_corpus` | Structural defects, no query set. The one to reach for first |
| `chunk_document` | Split a document and inspect exactly what came out |
| `evaluate_retrieval` | IR metrics against real queries |
| `optimize_chunking` | Sweep configs, ranked |
| `export_chunking_config` | LangChain or LlamaIndex code for a config worth keeping |
| `list_strategies` | What each chunking strategy is for |

Every tool takes `text` or `path`, never both, the same split as the CLI's `--text`/`--file`.
Output is capped by default (`max_findings`, `max_chunks`, `max_results`, all overridable) so
one call on a large corpus cannot flood the agent's context window, and every response says
which embedder actually ran, so a lexical-fallback score is never mistaken for a semantic one.

## Python API

```python
import rag_tui.api as rag

# Diagnose
report = rag.doctor(open("doc.md").read(), strategy="sentence", chunk_size=256)
print(report["retrievability_score"], report["grade"])
for finding in report["findings"]:
    print(finding["severity"], finding["kind"], finding["message"])
    print("  fix:", finding["suggestion"])

# Chunk
result = rag.chunk(text, strategy="sentence", chunk_size=256, overlap_percent=10)

# Evaluate
metrics = rag.eval(queries=["what is rag?"], docs=text, chunk_size=200, top_k=3)
print(metrics["metrics"]["mrr"])

# Sweep
best = rag.optimize(text, queries, strategies=["token", "sentence"])["ranked_results"][0]

# Regression check
comparison = rag.compare(rag.eval(queries, docs, chunk_size=200),
                         rag.eval(queries, docs, chunk_size=300))
```

Everything that touches the network has an async twin: `doctor_async`, `eval_async`, `eval_dataset_async`, `optimize_async`.

## Datasets

`--dataset-file` and `eval_dataset` take CSV or JSONL. Only `query` is required.

```csv
query,relevant_chunk,answer
What is RAG?,RAG is a technique...,
```

```jsonl
{"query": "What is RAG?", "relevant_chunk": "RAG is a technique..."}
```

## Caching

Every embedding lands in SQLite at `~/.rag-tui/cache/`, keyed by the SHA-256 of the text plus the provider and model name, so switching providers invalidates correctly rather than silently mixing vector spaces.

This matters most during a sweep. A 140-configuration run where half the chunks repeat across configs saves 30 to 50 percent of the embedding calls.

## Docker

```bash
docker-compose up
OLLAMA_HOST=http://your-server:11434 docker-compose up
```

`OLLAMA_HOST` is respected everywhere: TUI, CLI and API.

## What this does not do

Worth knowing before you file an issue.

The doctor finds structural problems, not factual ones. It can tell you a chunk is unreachable, duplicated or cut in half. It cannot tell you the chunk is wrong, out of date, or contradicts the chunk next to it.

Findings are risk signals read off the geometry, not proofs. A chunk flagged as an isolated island might be exactly the unique content you care most about. Read them, do not just gate on them.

There is no reranker support, so what you are measuring is first-stage retrieval only.

There is no semantic chunking, meaning splits placed where the embedding drifts rather than where the punctuation falls. Markdown and hierarchical cover the structural cases; semantic splitting is the next one.

The built-in embedder is lexical, as covered above. It will not see paraphrase and it is not pretending to.

## How it compares

| | RAG-TUI | Ragas | TruLens | DeepEval |
|---|---|---|---|---|
| Works with no query set | Yes | No | No | No |
| Runs with zero setup | Yes | No | No | No |
| Fully offline, including the judge | Yes | No | No | No |
| Visual chunking debugger | Yes | No | No | No |
| Automatic config sweep | Yes | No | No | No |
| CI-ready CLI | Yes | Partial | Partial | Yes |
| Breadth of metrics | Focused on retrieval | Broad | Broad | Broadest |

That last row is the honest one. If you need agent tracing, multi-turn evaluation and fifty metric types, DeepEval is a better tool and this is not trying to replace it. RAG-TUI is for the part of the problem where you are staring at a chunk and wondering why the vector store hates it.

## Keyboard shortcuts

| Key | Action |
|---|---|
| `L` | Load sample document |
| `R` | Re-chunk |
| `D` | Toggle theme |
| `E` | Export config |
| `1`-`5` | Switch strategy |
| `F1` or `?` | Help |
| `Q` | Quit |

## Regenerating the demo

The GIF at the top is scripted, not hand-recorded, so it stays honest as the output changes.

```bash
brew install vhs
vhs assets/demo/doctor.tape
```

## Contributing

Issues and pull requests welcome. If you have found a chunking strategy that beats the defaults for a particular kind of document, bring the eval numbers and it will very likely get merged.

## License

MIT.
