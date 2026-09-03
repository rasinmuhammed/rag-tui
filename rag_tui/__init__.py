"""RAG-TUI: Interactive chunking debugger and optimizer for RAG pipelines.

Features:
    - Six chunking strategies (Token, Sentence, Paragraph, Recursive, …)
    - Real-time chunk visualization with overlap highlighting
    - Multi-provider LLM/embedding (Ollama, OpenAI, Groq, Gemini)
    - Full IR metric suite: MRR, nDCG@k, Recall@k, Precision@k
    - Query-free corpus diagnostics (hubs, orphans, duplicates, fractures)
    - Automated chunk-size recommender (A/B optimizer)
    - Baseline comparison & regression detection
    - Dataset import (CSV / JSONL)
    - Persistent SQLite embedding cache
    - Config export for LangChain and LlamaIndex
    - Headless CLI and Python API for CI pipelines

Usage:
    rag-tui                          # launch interactive TUI
    rag-tui chunk --file doc.txt     # headless chunking
    rag-tui doctor --file doc.txt    # diagnose retrieval defects, no queries
    rag-tui optimize --file doc.txt  # auto-recommend config
"""

__version__ = "0.2.0"
__author__ = "Muhammed Rasin"

__all__ = ["__version__", "__author__", "api"]
