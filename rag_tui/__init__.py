"""RAG-TUI: Interactive chunking debugger and optimizer for RAG pipelines.

Features:
    - Six chunking strategies (Token, Sentence, Paragraph, Recursive, …)
    - Real-time chunk visualization with overlap highlighting
    - Multi-provider LLM/embedding (Ollama, OpenAI, Groq, Gemini)
    - Full IR metric suite: MRR, nDCG@k, Recall@k, Precision@k
    - Automated chunk-size recommender (A/B optimizer)
    - Baseline comparison & regression detection
    - Dataset import (CSV / JSONL)
    - Persistent SQLite embedding cache
    - Config export for LangChain and LlamaIndex
    - Headless CLI and Python API for CI pipelines

Usage:
    rag-tui                          # launch interactive TUI
    rag-tui chunk --file doc.txt     # headless chunking
    rag-tui optimize --file doc.txt  # auto-recommend config
"""

__version__ = "0.1.0"
__author__ = "Muhammed Rasin"

__all__ = ["__version__", "__author__", "api"]
