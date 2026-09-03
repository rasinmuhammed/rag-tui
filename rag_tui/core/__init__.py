"""RAG-TUI core: lazy re-exports so importing submodules stays lightweight."""

from rag_tui.core.strategies import (
    StrategyType,
    ChunkingStrategy,
    TokenStrategy,
    SentenceStrategy,
    ParagraphStrategy,
    RecursiveStrategy,
    FixedCharsStrategy,
    CustomStrategy,
    get_strategy,
    get_strategy_info,
    STRATEGIES,
)
from rag_tui.core.engine import ChunkingEngine
from rag_tui.core.file_handler import read_file, FileInfo, SUPPORTED_EXTENSIONS
from rag_tui.core.metrics import ChunkConfig, QueryResult, BatchTestResult, calculate_batch_metrics
from rag_tui.core.doctor import DoctorReport, Finding, analyze as analyze_corpus


def __getattr__(name):
    """Lazy imports for heavy optional deps (usearch, ollama)."""
    if name in ("VectorStore", "EmbeddingCache"):
        from rag_tui.core.vector import VectorStore, EmbeddingCache
        globals()["VectorStore"] = VectorStore
        globals()["EmbeddingCache"] = EmbeddingCache
        return globals()[name]
    if name == "OllamaLLM":
        from rag_tui.core.llm import OllamaLLM
        globals()["OllamaLLM"] = OllamaLLM
        return OllamaLLM
    raise AttributeError(f"module 'rag_tui.core' has no attribute {name!r}")


__all__ = [
    "ChunkingEngine",
    "StrategyType",
    "ChunkingStrategy",
    "TokenStrategy",
    "SentenceStrategy",
    "ParagraphStrategy",
    "RecursiveStrategy",
    "FixedCharsStrategy",
    "CustomStrategy",
    "get_strategy",
    "get_strategy_info",
    "STRATEGIES",
    "VectorStore",
    "EmbeddingCache",
    "OllamaLLM",
    "read_file",
    "FileInfo",
    "SUPPORTED_EXTENSIONS",
    "ChunkConfig",
    "QueryResult",
    "BatchTestResult",
    "calculate_batch_metrics",
    "DoctorReport",
    "Finding",
    "analyze_corpus",
]
