# RAG‑TUI Growth Analysis + 3–6 Month Roadmap to 100k+ Downloads

Date: 2026-02-08

## Objective
Provide a market‑grounded analysis of commonly searched RAG tools, map those tools to user jobs, identify gaps RAG‑TUI can fill, and deliver a prioritized roadmap and GTM plan targeting 100k+ PyPI downloads in 3–6 months.

## 1) Market Landscape: Commonly Referenced RAG Tools by Job

Notes:
- “Commonly referenced” here is grounded in official project pages plus open‑source availability and community usage signals (GitHub, PyPI). This is not an exhaustive list of every tool in the ecosystem.

### A. RAG frameworks and orchestration
Primary job: building RAG pipelines end‑to‑end.
- LangChain: [GitHub](https://github.com/langchain-ai/langchain)
- LlamaIndex: [GitHub](https://github.com/run-llama/llama_index)
- Haystack: [GitHub](https://github.com/deepset-ai/haystack)
- DSPy: [GitHub](https://github.com/stanfordnlp/dspy)
- Semantic Kernel: [GitHub](https://github.com/microsoft/semantic-kernel)

### B. Retrieval evaluation and testing
Primary job: benchmarking retrieval quality and RAG outcomes.
- RAGAS: [GitHub](https://github.com/explodinggradients/ragas)
- TruLens: [GitHub](https://github.com/truera/trulens)
- DeepEval: [GitHub](https://github.com/confident-ai/deepeval)
- Giskard: [GitHub](https://github.com/Giskard-AI/giskard)
- promptfoo: [GitHub](https://github.com/promptfoo/promptfoo)

### C. Vector databases and ANN search
Primary job: store/query embeddings efficiently.
- Pinecone: [Website](https://www.pinecone.io)
- Weaviate: [Website](https://weaviate.io)
- Qdrant: [Website](https://qdrant.tech)
- Milvus: [Website](https://milvus.io)
- Chroma: [Website](https://www.trychroma.com)
- pgvector: [GitHub](https://github.com/pgvector/pgvector)

### D. Document ingestion and parsing
Primary job: convert raw docs into clean text/chunks.
- Unstructured: [GitHub](https://github.com/Unstructured-IO/unstructured)
- LlamaParse: [Docs](https://docs.llamaindex.ai/en/stable/llamaparse/)
- Docling: [Project](https://github.com/docling-project)
- Apache Tika: [Website](https://tika.apache.org)

### E. Reranking and hybrid retrieval
Primary job: improve ranking quality after initial retrieval.
- bge‑reranker (FlagEmbedding): [GitHub](https://github.com/FlagOpen/FlagEmbedding)
- Cohere Rerank: [Docs](https://docs.cohere.com/docs/rerank)
- BM25 / Elastic: [Docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html)

### F. Embeddings and foundation model providers
Primary job: generate embeddings and/or LLM responses.
- OpenAI embeddings: [Docs](https://platform.openai.com/docs/guides/embeddings)
- Cohere embeddings: [Docs](https://docs.cohere.com/docs/embeddings)
- VoyageAI embeddings: [Docs](https://docs.voyageai.com/)
- Nomic embeddings: [Docs](https://docs.nomic.ai/atlas/embeddings/)
- Jina embeddings: [Docs](https://finetuner.jina.ai/)
- Mistral embeddings: [Docs](https://docs.mistral.ai/capabilities/embeddings/)

### G. Observability and monitoring
Primary job: trace, evaluate, and monitor RAG usage.
- Langfuse: [GitHub](https://github.com/langfuse/langfuse)
- Helicone: [Website](https://www.helicone.ai)
- Arize Phoenix: [GitHub](https://github.com/Arize-ai/phoenix)

## 2) Popularity Signals (Sample)

The goal is directional evidence that these tools are searched and adopted, not a full ranking.
- PyPI download signals:
  - LangChain: [pypistats](https://pypistats.org/packages/langchain)
  - LlamaIndex: [pypistats](https://pypistats.org/packages/llama-index)
  - RAGAS: [pypistats](https://pypistats.org/packages/ragas)
  - DeepEval: [pypistats](https://pypistats.org/packages/deepeval)
- GitHub community signals (open‑source visibility):
  - LangChain, LlamaIndex, Haystack, DSPy, Semantic Kernel, RAGAS, TruLens, DeepEval, Giskard, promptfoo, Langfuse, Phoenix (see GitHub links above).

## 3) Map RAG‑TUI to User Jobs (Current State)

Current RAG‑TUI strengths (from README and codebase):
- Chunking visualization across multiple strategies.
- Real‑time parameter tuning (chunk size, overlap).
- Search tab for retrieval inspection.
- Batch testing tab with basic metrics.
- Export configuration snippets for LangChain and LlamaIndex.
- Multi‑provider embeddings and LLMs (Ollama, OpenAI, Groq, Google Gemini).

## 4) Gap Analysis (High‑Intent, Adjacent Opportunities)

High‑impact gaps that align with the market:
1. Headless CLI and Python API
- Market: most teams want CI‑friendly evaluation and reproducible configs.
- Gap: RAG‑TUI is TUI‑only today.

2. Retrieval evaluation metrics parity with specialized tools
- Market: tools like RAGAS/TruLens/DeepEval standardize eval metrics.
- Gap: RAG‑TUI does not yet expose standard IR metrics (MRR, nDCG, recall@k) in a headless workflow.

3. Dataset‑centric eval workflows
- Market: teams benchmark multiple queries/datasets.
- Gap: no dataset import/export formats (CSV/JSONL) for automated runs.

4. Integrations with vector DBs and rerankers
- Market: Qdrant/Chroma and rerankers are common in RAG stacks.
- Gap: no adapters or CLI hooks to test retrieval settings directly against these stores.

5. Observability and regression testing
- Market: Langfuse/Phoenix emphasize traceability and drift monitoring.
- Gap: no baseline comparison or regression reports in RAG‑TUI.

## 5) Prioritized Roadmap (3–6 Months)

### Phase 1 (Weeks 0–2): Packaging + API Foundation
- Add headless CLI surface:
  - `rag-tui chunk` for chunking analysis outputs in JSON/CSV.
  - `rag-tui eval` for batch retrieval evaluation.
  - `rag-tui export` for framework configs.
- Add Python API surface:
  - `rag_tui.api.chunk(text, strategy, config)`
  - `rag_tui.api.eval(queries, docs, embeddings, metrics)`
  - `rag_tui.api.export(format, config)`
- Add headless mode for CI usage with deterministic outputs.

### Phase 2 (Weeks 2–6): Retrieval Evaluation and Diagnostics
- Add standardized metrics: hit rate, MRR, nDCG, recall@k.
- Add dataset import (CSV/JSONL).
- Add regression testing: save baseline config + compare against new runs.

### Phase 3 (Weeks 4–8): High‑Intent Integrations
- Add LangChain and LlamaIndex adapters (export + CLI interop).
- Add vector DB connectors (Qdrant + Chroma first).
- Add reranker hooks (bge‑reranker or Cohere Rerank).

### Phase 4 (Weeks 6–12): Distribution and Developer Experience
- Produce 5‑minute quickstart, 30‑minute walkthrough, CI‑ready examples.
- Publish “RAG tuning checklist” and “best chunking strategy” guides.
- Curated presets aligned with popular frameworks.

## 6) GTM Plan (Download‑Focused)

### Documentation and onboarding
- Short quickstart, long tutorial, CI/headless usage examples.
- “RAG tuning checklist” landing page referencing RAG‑TUI.

### Content and community
- Publish comparisons: “best chunking strategies for {docs, code, chat}.”
- Post benchmark results in RAG community channels.
- Provide a reusable “RAG‑TUI eval template.”

### Partnerships and integrations
- Provide recipes with 2 frameworks (LangChain, LlamaIndex) and 2 vector DBs (Qdrant, Chroma).

### Metrics and targets
- Weekly PyPI download tracking.
- GitHub stars‑to‑PyPI conversion tracking.

## 7) Success Metrics and Verification

- 100k+ monthly downloads by end of month 6.
- 10+ public issues/PRs from external contributors.
- 5+ public examples or blog posts referencing RAG‑TUI.
- CLI/API adoption: 30% of users use headless mode (measured via optional telemetry or self‑reporting surveys).

## 8) Proposed Deliverables Checklist

- Market tool landscape report with sources and signals. (This document)
- Gap analysis matrix and prioritized feature list. (Included above)
- 3–6 month roadmap with milestones. (Included above)
- GTM plan with concrete actions and expected impact. (Included above)

