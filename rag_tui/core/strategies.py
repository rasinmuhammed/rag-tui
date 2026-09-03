"""Chunking strategies for RAG-TUI.

Provides multiple built-in chunking strategies and support for custom strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Callable, Optional
from enum import Enum
import re


class StrategyType(Enum):
    """Available chunking strategies."""
    TOKEN = "token"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    RECURSIVE = "recursive"
    FIXED_CHARS = "fixed_chars"
    MARKDOWN = "markdown"
    HIERARCHICAL = "hierarchical"
    CUSTOM = "custom"


@dataclass
class ChunkResult:
    """Result of chunking operation."""
    text: str
    start_pos: int
    end_pos: int
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ChunkingStrategy(ABC):
    """Base class for chunking strategies."""
    
    name: str = "base"
    description: str = "Base chunking strategy"
    
    @abstractmethod
    def chunk(self, text: str, chunk_size: int, overlap: int) -> List[ChunkResult]:
        """Split text into chunks.
        
        Args:
            text: The text to chunk
            chunk_size: Target size for chunks
            overlap: Amount of overlap between chunks
            
        Returns:
            List of ChunkResult objects
        """
        pass


class TokenStrategy(ChunkingStrategy):
    """Token-based chunking using Chonkie."""
    
    name = "token"
    description = "Split by token count (best for general text)"
    
    def __init__(self):
        self._chunker_cache = {}
    
    def _get_chunker(self, chunk_size: int, overlap: int):
        """Get or create a chunker with given params."""
        key = (chunk_size, overlap)
        if key not in self._chunker_cache:
            from chonkie import TokenChunker
            self._chunker_cache[key] = TokenChunker(
                chunk_size=chunk_size,
                chunk_overlap=overlap
            )
        return self._chunker_cache[key]
    
    def chunk(self, text: str, chunk_size: int, overlap: int) -> List[ChunkResult]:
        chunker = self._get_chunker(chunk_size, overlap)
        chunks = chunker.chunk(text)
        
        results = []
        for chunk in chunks:
            results.append(ChunkResult(
                text=chunk.text,
                start_pos=chunk.start_index,
                end_pos=chunk.end_index,
                metadata={"token_count": chunk.token_count}
            ))
        return results


class SentenceStrategy(ChunkingStrategy):
    """Sentence-based chunking - splits at sentence boundaries."""
    
    name = "sentence"
    description = "Split at sentence boundaries (best for natural language)"
    
    # Sentence ending patterns
    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+')
    
    def chunk(self, text: str, chunk_size: int, overlap: int) -> List[ChunkResult]:
        # Split into sentences
        sentences = self.SENTENCE_ENDINGS.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        results = []
        current_chunk = []
        current_length = 0
        chunk_start = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_length + sentence_len > chunk_size * 4 and current_chunk:  # ~4 chars per token
                chunk_text = ' '.join(current_chunk)
                results.append(ChunkResult(
                    text=chunk_text,
                    start_pos=chunk_start,
                    end_pos=chunk_start + len(chunk_text),
                    metadata={"sentence_count": len(current_chunk)}
                ))
                
                # Handle overlap by keeping last sentences
                overlap_chars = overlap * 4
                overlap_text = ""
                overlap_sentences = []
                for s in reversed(current_chunk):
                    if len(overlap_text) + len(s) < overlap_chars:
                        overlap_sentences.insert(0, s)
                        overlap_text = ' '.join(overlap_sentences)
                    else:
                        break
                
                chunk_start += len(chunk_text) - len(overlap_text)
                current_chunk = overlap_sentences
                current_length = len(overlap_text)
            
            current_chunk.append(sentence)
            current_length += sentence_len
        
        # Add remaining
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            results.append(ChunkResult(
                text=chunk_text,
                start_pos=chunk_start,
                end_pos=chunk_start + len(chunk_text),
                metadata={"sentence_count": len(current_chunk)}
            ))
        
        return results


class ParagraphStrategy(ChunkingStrategy):
    """Paragraph-based chunking - splits at double newlines."""
    
    name = "paragraph"
    description = "Split at paragraph breaks (best for structured documents)"
    
    def chunk(self, text: str, chunk_size: int, overlap: int) -> List[ChunkResult]:
        # Split by double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        results = []
        current_chunk = []
        current_length = 0
        chunk_start = 0
        
        for para in paragraphs:
            para_len = len(para)
            
            if current_length + para_len > chunk_size * 4 and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                results.append(ChunkResult(
                    text=chunk_text,
                    start_pos=chunk_start,
                    end_pos=chunk_start + len(chunk_text),
                    metadata={"paragraph_count": len(current_chunk)}
                ))
                
                # Simple overlap: keep last paragraph if within overlap
                if len(current_chunk[-1]) < overlap * 4:
                    chunk_start += len(chunk_text) - len(current_chunk[-1])
                    current_chunk = [current_chunk[-1]]
                    current_length = len(current_chunk[0])
                else:
                    chunk_start += len(chunk_text)
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(para)
            current_length += para_len
        
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            results.append(ChunkResult(
                text=chunk_text,
                start_pos=chunk_start,
                end_pos=chunk_start + len(chunk_text),
                metadata={"paragraph_count": len(current_chunk)}
            ))
        
        return results


class RecursiveStrategy(ChunkingStrategy):
    """Recursive character splitting - tries multiple separators."""
    
    name = "recursive"
    description = "Try multiple separators recursively (best for code/mixed content)"
    
    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
    
    def chunk(self, text: str, chunk_size: int, overlap: int) -> List[ChunkResult]:
        target_size = chunk_size * 4  # chars
        
        def split_recursive(text: str, separators: List[str]) -> List[str]:
            if not separators:
                return [text]
            
            sep = separators[0]
            if not sep:
                # Final fallback: split by character
                return [text[i:i+target_size] for i in range(0, len(text), target_size - overlap * 4)]
            
            parts = text.split(sep)
            
            result = []
            current = []
            current_len = 0
            
            for part in parts:
                if current_len + len(part) > target_size and current:
                    result.append(sep.join(current))
                    current = []
                    current_len = 0
                
                if len(part) > target_size:
                    if current:
                        result.append(sep.join(current))
                        current = []
                        current_len = 0
                    result.extend(split_recursive(part, separators[1:]))
                else:
                    current.append(part)
                    current_len += len(part) + len(sep)
            
            if current:
                result.append(sep.join(current))
            
            return result
        
        chunks = split_recursive(text, self.SEPARATORS)
        
        results = []
        pos = 0
        for chunk_text in chunks:
            start = text.find(chunk_text, pos)
            if start == -1:
                start = pos
            results.append(ChunkResult(
                text=chunk_text,
                start_pos=start,
                end_pos=start + len(chunk_text),
                metadata={"method": "recursive"}
            ))
            pos = start + len(chunk_text)
        
        return results


class FixedCharsStrategy(ChunkingStrategy):
    """Fixed character count chunking."""
    
    name = "fixed_chars"
    description = "Split by fixed character count (simple, fast)"
    
    def chunk(self, text: str, chunk_size: int, overlap: int) -> List[ChunkResult]:
        char_size = chunk_size * 4  # Approximate chars per token
        char_overlap = overlap * 4
        
        results = []
        start = 0
        
        while start < len(text):
            end = min(start + char_size, len(text))
            chunk_text = text[start:end]
            
            results.append(ChunkResult(
                text=chunk_text,
                start_pos=start,
                end_pos=end,
                metadata={"char_count": len(chunk_text)}
            ))
            
            start = end - char_overlap
            if start >= len(text) - char_overlap:
                break
        
        return results


class MarkdownStrategy(ChunkingStrategy):
    """Split on markdown headings and keep the heading trail on every chunk.

    Splitting a structured document on size alone throws away the one piece of
    context the author already gave you. A paragraph under "Billing > Refunds"
    reads very differently from the same words under "Billing > Disputes", and
    once the heading is gone neither the embedder nor the model can tell.

    Every chunk here is prefixed with its heading trail, so a chunk carries
    where it came from. Headings are never emitted alone: a bare "## Rate
    limits" is a useless chunk that no query retrieves, so it is attached to
    the section body underneath it.

    Only ATX headings (the ``#`` kind) are recognised. Fenced code blocks are
    skipped, so a Python comment does not get mistaken for an ``<h1>``.
    """

    name = "markdown"
    description = "Split on headings, keep the heading trail (best for docs and wikis)"

    HEADING = re.compile(r'^(#{1,6})\s+(.*?)\s*#*\s*$')
    FENCE = re.compile(r'^\s*(```|~~~)')

    def chunk(self, text: str, chunk_size: int, overlap: int) -> List[ChunkResult]:
        target = chunk_size * 4
        sections = self._split_sections(text)

        results: List[ChunkResult] = []
        for path, body, body_start in sections:
            if not body.strip():
                continue
            crumb = " > ".join(path)
            for piece, start, end in self._split_body(body, body_start, target, overlap * 4):
                chunk_text = f"{crumb}\n\n{piece}" if crumb else piece
                results.append(ChunkResult(
                    text=chunk_text,
                    start_pos=start,
                    end_pos=end,
                    metadata={
                        "heading_path": list(path),
                        "heading_depth": len(path),
                        # The prefix is context, not source text. Anything that
                        # maps a chunk back to the file needs to skip it.
                        "prefix_chars": len(chunk_text) - len(piece),
                    },
                ))

        # A document with no headings at all still has to produce something.
        if not results and text.strip():
            return ParagraphStrategy().chunk(text, chunk_size, overlap)

        return results

    def _split_sections(self, text: str):
        """Yield (heading_path, body_text, body_start_offset) per section."""
        sections = []
        stack: List[str] = []
        body_lines: List[str] = []
        body_start = 0
        offset = 0
        in_fence = False

        def flush():
            if body_lines:
                sections.append((tuple(stack), "".join(body_lines), body_start))

        for line in text.splitlines(keepends=True):
            if self.FENCE.match(line):
                in_fence = not in_fence

            match = None if in_fence else self.HEADING.match(line)
            if match:
                flush()
                level = len(match.group(1))
                title = match.group(2).strip()
                del stack[level - 1:]
                stack.append(title)
                body_lines = []
                body_start = offset + len(line)
            else:
                if not body_lines and not line.strip():
                    # Skip blank lines so a body starts at real content.
                    body_start = offset + len(line)
                else:
                    body_lines.append(line)
            offset += len(line)

        flush()
        return sections

    def _split_body(self, body: str, body_start: int, target: int, overlap_chars: int):
        """Break one section into size-limited pieces at paragraph breaks."""
        if len(body.strip()) <= target:
            stripped = body.strip()
            lead = len(body) - len(body.lstrip())
            yield stripped, body_start + lead, body_start + lead + len(stripped)
            return

        pos = 0
        buffer: List[str] = []
        buffer_start = 0
        buffer_len = 0

        for para in re.split(r'(\n\s*\n)', body):
            if not para:
                continue
            if para.strip() == "":
                if buffer:
                    buffer.append(para)
                pos += len(para)
                continue

            if buffer_len + len(para) > target and buffer:
                piece = "".join(buffer).strip()
                yield piece, body_start + buffer_start, body_start + buffer_start + len(piece)
                keep = buffer[-1] if len(buffer[-1]) <= overlap_chars else ""
                buffer = [keep] if keep else []
                buffer_len = len(keep)
                buffer_start = pos - len(keep)

            if not buffer:
                buffer_start = pos
            buffer.append(para)
            buffer_len += len(para)
            pos += len(para)

        if buffer:
            piece = "".join(buffer).strip()
            if piece:
                yield piece, body_start + buffer_start, body_start + buffer_start + len(piece)


class HierarchicalStrategy(ChunkingStrategy):
    """Small chunks to find with, large chunks to answer from.

    This is the pattern that resolves the oldest tradeoff in retrieval. Small
    chunks match precisely because they are about one thing, and then hand the
    model a fragment too narrow to reason over. Large chunks give the model
    enough to work with and match badly because their embedding is an average
    of six different topics.

    So do both. The chunks returned here are small and are what you embed and
    rank. Each one carries the larger passage it came from in
    ``metadata["parent_text"]``, and that is what you put in the prompt once
    the child has won the ranking.

    ``chunk_size`` sizes the child. The parent is ``PARENT_MULTIPLIER`` times
    larger.
    """

    name = "hierarchical"
    description = "Small chunks for matching, parent windows for context"

    PARENT_MULTIPLIER = 4
    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+')

    def chunk(self, text: str, chunk_size: int, overlap: int) -> List[ChunkResult]:
        child_target = chunk_size * 4
        parent_target = child_target * self.PARENT_MULTIPLIER

        results: List[ChunkResult] = []
        for parent_index, (parent_text, parent_start) in enumerate(
            self._windows(text, parent_target)
        ):
            children = list(self._windows(parent_text, child_target))
            for child_text, child_offset in children:
                start = parent_start + child_offset
                results.append(ChunkResult(
                    text=child_text,
                    start_pos=start,
                    end_pos=start + len(child_text),
                    metadata={
                        "parent_index": parent_index,
                        "parent_start": parent_start,
                        "parent_end": parent_start + len(parent_text),
                        "parent_text": parent_text,
                        "children_in_parent": len(children),
                    },
                ))

        return results

    def _windows(self, text: str, target: int):
        """Split text into windows of about ``target`` chars on sentence breaks."""
        if not text.strip():
            return

        sentences = [s for s in self.SENTENCE_ENDINGS.split(text) if s.strip()]
        if not sentences:
            return

        buffer: List[str] = []
        buffer_len = 0
        cursor = 0
        start = None

        for sentence in sentences:
            found = text.find(sentence, cursor)
            if found == -1:
                found = cursor
            if start is None:
                start = found

            if buffer_len + len(sentence) > target and buffer:
                window = text[start:cursor].strip()
                if window:
                    lead = len(text[start:cursor]) - len(text[start:cursor].lstrip())
                    yield window, start + lead
                start = found
                buffer = []
                buffer_len = 0

            buffer.append(sentence)
            buffer_len += len(sentence)
            cursor = found + len(sentence)

        if buffer and start is not None:
            window = text[start:].strip()
            if window:
                lead = len(text[start:]) - len(text[start:].lstrip())
                yield window, start + lead


class CustomStrategy(ChunkingStrategy):
    """Custom user-defined chunking strategy."""
    
    name = "custom"
    description = "User-defined Python function"
    
    def __init__(self, chunk_fn: Optional[Callable] = None):
        """Initialize with a custom chunking function.
        
        Args:
            chunk_fn: Function with signature (text, chunk_size, overlap) -> List[Tuple[str, int, int]]
        """
        self._chunk_fn = chunk_fn
    
    def set_function(self, chunk_fn: Callable):
        """Set the custom chunking function."""
        self._chunk_fn = chunk_fn
    
    def chunk(self, text: str, chunk_size: int, overlap: int) -> List[ChunkResult]:
        if not self._chunk_fn:
            raise ValueError("Custom chunking function not set. Use set_function() first.")
        
        raw_chunks = self._chunk_fn(text, chunk_size, overlap)
        
        results = []
        for item in raw_chunks:
            if isinstance(item, tuple) and len(item) >= 3:
                chunk_text, start, end = item[:3]
                metadata = item[3] if len(item) > 3 else {}
            else:
                chunk_text = str(item)
                start = text.find(chunk_text)
                end = start + len(chunk_text)
                metadata = {}
            
            results.append(ChunkResult(
                text=chunk_text,
                start_pos=start,
                end_pos=end,
                metadata=metadata
            ))
        
        return results


# Strategy registry
STRATEGIES = {
    StrategyType.TOKEN: TokenStrategy,
    StrategyType.SENTENCE: SentenceStrategy,
    StrategyType.PARAGRAPH: ParagraphStrategy,
    StrategyType.RECURSIVE: RecursiveStrategy,
    StrategyType.FIXED_CHARS: FixedCharsStrategy,
    StrategyType.MARKDOWN: MarkdownStrategy,
    StrategyType.HIERARCHICAL: HierarchicalStrategy,
    StrategyType.CUSTOM: CustomStrategy,
}


def get_strategy(strategy_type: StrategyType) -> ChunkingStrategy:
    """Get a chunking strategy instance.
    
    Args:
        strategy_type: The type of strategy to get
        
    Returns:
        An instance of the requested strategy
    """
    strategy_class = STRATEGIES.get(strategy_type)
    if not strategy_class:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    return strategy_class()


def get_strategy_info() -> List[dict]:
    """Get information about all available strategies.
    
    Returns:
        List of dicts with name, description for each strategy
    """
    info = []
    for strategy_type, strategy_class in STRATEGIES.items():
        info.append({
            "type": strategy_type,
            "name": strategy_class.name,
            "description": strategy_class.description
        })
    return info
