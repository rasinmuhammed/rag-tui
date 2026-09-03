"""Chunking engine for RAG-TUI.

Provides async text chunking with multiple strategy support.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Callable

from rag_tui.core.strategies import (
    StrategyType,
    ChunkResult,
    ChunkingStrategy, 
    get_strategy,
    get_strategy_info
)


class ChunkingEngine:
    """High-performance async chunking engine with multiple strategies."""
    
    def __init__(self):
        """Initialize the chunking engine."""
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._strategy_cache = {}
        self._current_strategy_type = StrategyType.TOKEN
        self._custom_function: Optional[Callable] = None
    
    def get_strategy(self, strategy_type: StrategyType) -> ChunkingStrategy:
        """Get a strategy instance (cached).
        
        Args:
            strategy_type: The type of strategy
            
        Returns:
            Strategy instance
        """
        if strategy_type not in self._strategy_cache:
            self._strategy_cache[strategy_type] = get_strategy(strategy_type)
        
        strategy = self._strategy_cache[strategy_type]
        
        # Set custom function if this is a custom strategy
        if strategy_type == StrategyType.CUSTOM and self._custom_function:
            strategy.set_function(self._custom_function)
        
        return strategy
    
    def set_strategy(self, strategy_type: StrategyType):
        """Set the current chunking strategy.
        
        Args:
            strategy_type: The strategy to use
        """
        self._current_strategy_type = strategy_type
    
    def set_custom_function(self, func: Callable):
        """Set a custom chunking function.
        
        Args:
            func: Function with signature (text, chunk_size, overlap) -> List[Tuple[str, int, int]]
        """
        self._custom_function = func
        if StrategyType.CUSTOM in self._strategy_cache:
            self._strategy_cache[StrategyType.CUSTOM].set_function(func)
    
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 200,
        overlap: int = 20,
        strategy_type: Optional[StrategyType] = None
    ) -> List[Tuple[str, int, int]]:
        """Synchronously split text into chunks.
        
        Args:
            text: The text to chunk
            chunk_size: Target chunk size in tokens
            overlap: Overlap amount in tokens
            strategy_type: Strategy to use (defaults to current)
            
        Returns:
            List of (text, start_pos, end_pos) tuples
        """
        return [
            (r.text, r.start_pos, r.end_pos)
            for r in self.chunk_detailed(text, chunk_size, overlap, strategy_type)
        ]

    def chunk_detailed(
        self,
        text: str,
        chunk_size: int = 200,
        overlap: int = 20,
        strategy_type: Optional[StrategyType] = None,
    ) -> List["ChunkResult"]:
        """Chunk text and keep each chunk's metadata.

        Same work as chunk_text(), but nothing is thrown away. Strategies like
        hierarchical put their whole point in metadata (the parent window a
        child came from), so anything that needs it should call this.

        Args:
            text: The text to chunk
            chunk_size: Target chunk size in tokens
            overlap: Overlap amount in tokens
            strategy_type: Strategy to use (defaults to current)

        Returns:
            List of ChunkResult objects
        """
        if not text or not text.strip():
            return []

        strategy_type = strategy_type or self._current_strategy_type
        strategy = self.get_strategy(strategy_type)

        return strategy.chunk(text, chunk_size, overlap)
    
    async def chunk_text_async(
        self,
        text: str,
        chunk_size: int = 200,
        overlap: int = 20,
        strategy_type: Optional[StrategyType] = None
    ) -> List[Tuple[str, int, int]]:
        """Asynchronously split text into chunks.
        
        Runs chunking in a thread pool to avoid blocking the UI.
        
        Args:
            text: The text to chunk
            chunk_size: Target chunk size in tokens
            overlap: Overlap amount in tokens
            strategy_type: Strategy to use (defaults to current)
            
        Returns:
            List of (text, start_pos, end_pos) tuples
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.chunk_text,
            text,
            chunk_size,
            overlap,
            strategy_type
        )
    
    def get_chunk_stats(self, chunks: List[Tuple[str, int, int]]) -> dict:
        """Calculate statistics for a list of chunks.
        
        Args:
            chunks: List of (text, start, end) tuples
            
        Returns:
            Dictionary of statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "total_tokens_est": 0
            }
        
        sizes = [len(c[0]) for c in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_characters": sum(sizes),
            "avg_chunk_size": sum(sizes) / len(sizes),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "total_tokens_est": sum(sizes) // 4
        }
    
    @staticmethod
    def get_available_strategies() -> List[dict]:
        """Get information about all available strategies.
        
        Returns:
            List of strategy info dicts
        """
        return get_strategy_info()
    
    def shutdown(self):
        """Shutdown the executor."""
        self._executor.shutdown(wait=False)
