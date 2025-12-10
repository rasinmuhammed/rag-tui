"""RAG-TUI: Interactive Chunking Debugger for RAG Pipelines.

A beautiful terminal UI for visualizing, debugging, and tuning 
text chunking in RAG (Retrieval-Augmented Generation) pipelines.

Features:
    - Multiple chunking strategies (Token, Sentence, Paragraph, Recursive)
    - Real-time chunk visualization with quality indicators
    - Multi-provider LLM support (Ollama, OpenAI, Groq, Gemini)
    - Batch testing with hit rate metrics
    - Export to LangChain/LlamaIndex format
    - Interactive Chat with your documents

Usage:
    python -m rag_tui.app
    # or after installation:
    rag-tui
"""

__version__ = "0.0.2-beta"
__author__ = "Muhammed Rasin"

__all__ = ["__version__", "__author__"]
