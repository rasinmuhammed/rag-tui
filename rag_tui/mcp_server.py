"""MCP server exposing RAG-TUI as tools for Claude Code, Claude Desktop, or Cursor.

Point an agent at this and it can tune a retrieval pipeline directly: diagnose
a corpus for structural defects, sweep chunking configs, evaluate retrieval
quality, all without shelling out to `rag-tui` and parsing terminal output.

Run it:

    rag-tui mcp

Or point a client's config at it directly:

    {
      "mcpServers": {
        "rag-tui": { "command": "rag-tui", "args": ["mcp"] }
      }
    }

Needs the ``mcp`` extra, which pulls in a real dependency tree (starlette,
uvicorn, pydantic) that most installs never touch, so it stays opt-in::

    pip install "rag-tui[mcp]"

Every tool below takes exactly one of ``text`` (paste it inline) or ``path``
(a file already on disk), never both, matching the CLI's ``--text``/``--file``
split. All of the actual work is delegated to :mod:`rag_tui.api`; this module
picks defaults suited to an agent instead of a human at a terminal: bounded
output so one call cannot flood the context window, and a plain
``{"error": ...}`` dict instead of a stack trace when something goes wrong.

A note on docstrings in this file: every ``@server.tool()`` description below
is read straight from the wrapped function's docstring, so it has to be a
plain string literal, not an f-string. An f-string as the first statement of
a function body compiles to a JoinedStr expression, not a Constant, so Python
does not treat it as a docstring at all and ``__doc__`` silently comes back
as None. The strategy list mentioned in a few of them is therefore spelled
out literally rather than interpolated from ``_SELECTABLE_STRATEGIES``; call
``list_strategies`` for the live version.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from mcp.server.mcpserver import MCPServer
except ImportError as exc:  # pragma: no cover - exercised via the CLI, not directly
    raise ImportError(
        "The MCP server needs the 'mcp' package. Install it with:\n\n"
        "    pip install \"rag-tui[mcp]\"\n"
    ) from exc

from rag_tui import __version__
from rag_tui.core.file_handler import read_file
from rag_tui.core.strategies import StrategyType, get_strategy_info

# Custom needs a Python function passed in, which has no shape an MCP client
# can send, so it stays TUI-only here too.
_SELECTABLE_STRATEGIES = [s.value for s in StrategyType if s is not StrategyType.CUSTOM]

# Caps that keep one tool call from dumping an entire corpus, or an entire
# 140-config sweep, into the calling agent's context window.
DEFAULT_MAX_CHUNKS = 40
DEFAULT_CHUNK_CHARS = 500
DEFAULT_MAX_FINDINGS = 40
DEFAULT_MAX_RESULTS = 10


def _load_text(text: Optional[str], path: Optional[str]) -> str:
    """Resolve the shared text/path pair every tool below takes.

    Raises ValueError on a bad combination, and whatever read_file() raises
    (FileNotFoundError, ValueError on an unsupported extension) on a bad path.
    Both are caught by the tool wrappers and turned into an error dict, never
    a stack trace across the MCP transport.
    """
    if (text is None) == (path is None):
        raise ValueError("Pass exactly one of 'text' or 'path', not both and not neither.")
    if path is not None:
        content, _info = read_file(path)
        return content
    return text  # type: ignore[return-value]


def create_server() -> "MCPServer":
    """Build the MCPServer instance with every tool registered."""

    server = MCPServer(
        name="rag-tui",
        version=__version__,
        instructions=(
            "Tools for debugging and tuning a RAG chunking pipeline. Start with "
            "diagnose_corpus: it finds structural retrieval defects (duplicate "
            "passages, unreachable chunks, sentence fractures) from the chunk "
            "embeddings alone, with no query set required. Reach for "
            "evaluate_retrieval or optimize_chunking once you have example "
            "queries to test against. Every tool takes 'text' or 'path', never "
            "both. When no embedding provider is configured, calls fall back to "
            "a built-in lexical embedder: it catches structural damage but not "
            "paraphrase, and every result names which embedder actually ran."
        ),
    )

    # -----------------------------------------------------------------
    # list_strategies
    # -----------------------------------------------------------------

    @server.tool(structured_output=False)
    def list_strategies() -> Dict[str, Any]:
        """List the chunking strategies available to every other tool here.

        Returns each strategy's name and a one-line description of what it is
        best suited for. Call this before choosing a strategy argument
        elsewhere if you are not already sure which one fits the document.
        """
        info = [
            {"name": entry["type"].value, "description": entry["description"]}
            for entry in get_strategy_info()
            if entry["type"] is not StrategyType.CUSTOM
        ]
        return {"strategies": info}

    # -----------------------------------------------------------------
    # diagnose_corpus
    # -----------------------------------------------------------------

    @server.tool(structured_output=False)
    async def diagnose_corpus(
        text: Optional[str] = None,
        path: Optional[str] = None,
        strategy: str = "token",
        chunk_size: int = 200,
        overlap_percent: int = 10,
        neighbors: int = 5,
        provider: Optional[str] = None,
        fail_under: Optional[float] = None,
        max_findings: int = DEFAULT_MAX_FINDINGS,
    ) -> Dict[str, Any]:
        """Find structural retrieval defects in a document. No query set needed.

        Chunks the document, embeds it once, and reads the geometry of the
        resulting vector space for five kinds of problem: hub chunks that get
        retrieved for unrelated queries, orphan chunks nothing else retrieves,
        near-duplicate chunks splitting each other's ranking, sentence
        fractures at chunk boundaries, and boilerplate too short or repetitive
        to mean anything.

        Use this first, before writing any test queries. It is the fastest way
        to tell whether a chunking choice is structurally sound.

        Args:
            text: The document text, inline. Exactly one of text/path.
            path: Path to a document file on disk. Exactly one of text/path.
            strategy: Chunking strategy. One of: token, sentence, paragraph,
                recursive, fixed_chars, markdown, hierarchical. Call
                list_strategies for what each one is for.
            chunk_size: Target chunk size in tokens.
            overlap_percent: Overlap between adjacent chunks, as a percent.
            neighbors: Neighbourhood size k used for hub/orphan detection.
                Raise it for a large corpus where top-5 is not representative.
            provider: Embedding provider name (ollama, openai, google). Left
                unset, the best available one is auto-detected, falling back
                to a built-in lexical embedder if nothing else is reachable.
            fail_under: If given, adds a `passed` boolean: whether the score
                met this threshold. Use it to gate a change without needing
                the caller to compare the score itself.
            max_findings: Cap on findings returned, most severe first, so one
                badly-chunked document cannot flood the response.

        Returns:
            retrievability_score, grade, hubness_skew, counts by finding kind,
            and the findings themselves, each with a severity, the affected
            chunk indices, why it hurts retrieval, and a concrete fix.
        """
        try:
            docs = _load_text(text, path)
            from rag_tui.api import doctor_async

            report = await doctor_async(
                docs=docs,
                strategy=strategy,
                chunk_size=chunk_size,
                overlap_percent=overlap_percent,
                neighbors=neighbors,
                provider=provider,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller, not raised
            return {"error": str(exc)}

        total = len(report["findings"])
        if total > max_findings:
            report["findings"] = report["findings"][:max_findings]
            report["findings_omitted"] = total - max_findings

        if fail_under is not None:
            report["passed"] = report["retrievability_score"] >= fail_under

        return report

    # -----------------------------------------------------------------
    # chunk_document
    # -----------------------------------------------------------------

    @server.tool(structured_output=False)
    def chunk_document(
        text: Optional[str] = None,
        path: Optional[str] = None,
        strategy: str = "token",
        chunk_size: int = 200,
        overlap_percent: int = 10,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
        max_chars_per_chunk: int = DEFAULT_CHUNK_CHARS,
    ) -> Dict[str, Any]:
        """Split a document into chunks and show what came out.

        Needs no embedding provider, so it always works, including with
        nothing configured. Use it to inspect exactly what a strategy and
        size produce before spending an embedding call on them, or to see the
        parent windows the hierarchical strategy attaches to each chunk.

        Args:
            text: The document text, inline. Exactly one of text/path.
            path: Path to a document file on disk. Exactly one of text/path.
            strategy: Chunking strategy. One of: token, sentence, paragraph,
                recursive, fixed_chars, markdown, hierarchical. Call
                list_strategies for what each one is for.
            chunk_size: Target chunk size in tokens.
            overlap_percent: Overlap between adjacent chunks, as a percent.
            max_chunks: Cap on chunks returned in full. `stats` still reflects
                the whole document even when chunks are capped.
            max_chars_per_chunk: Truncate each returned chunk's text beyond
                this length (the hierarchical strategy's parent_text is
                truncated the same way). Does not affect chunking itself.

        Returns:
            stats for the whole document (total chunks, size distribution)
            plus up to max_chunks chunks, each with its position, text, and
            any strategy-specific metadata (heading path, parent window).
        """
        try:
            docs = _load_text(text, path)
            from rag_tui.api import chunk as chunk_fn

            result = chunk_fn(
                text=docs,
                strategy=strategy,
                chunk_size=chunk_size,
                overlap_percent=overlap_percent,
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

        def _truncate(value: str) -> str:
            if len(value) <= max_chars_per_chunk:
                return value
            return value[:max_chars_per_chunk] + f"... [{len(value) - max_chars_per_chunk} more chars]"

        all_chunks = result["chunks"]
        shown = []
        for entry in all_chunks[:max_chunks]:
            entry = dict(entry)
            entry["text"] = _truncate(entry["text"])
            if "metadata" in entry and isinstance(entry["metadata"], dict):
                meta = dict(entry["metadata"])
                if isinstance(meta.get("parent_text"), str):
                    meta["parent_text"] = _truncate(meta["parent_text"])
                entry["metadata"] = meta
            shown.append(entry)

        result["chunks"] = shown
        if len(all_chunks) > max_chunks:
            result["chunks_omitted"] = len(all_chunks) - max_chunks
        return result

    # -----------------------------------------------------------------
    # evaluate_retrieval
    # -----------------------------------------------------------------

    @server.tool(structured_output=False)
    async def evaluate_retrieval(
        queries: List[str],
        text: Optional[str] = None,
        path: Optional[str] = None,
        strategy: str = "token",
        chunk_size: int = 200,
        overlap_percent: int = 10,
        top_k: int = 3,
        threshold: float = 0.5,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run retrieval for real queries and score it with standard IR metrics.

        Chunks and embeds the document, retrieves the top_k chunks for every
        query, and reports hit rate, MRR, nDCG@k, recall@k and precision@k.
        Unlike diagnose_corpus, this needs example queries, and the scores
        are only as representative as the queries are.

        Args:
            queries: The test queries to retrieve against.
            text: The document text, inline. Exactly one of text/path.
            path: Path to a document file on disk. Exactly one of text/path.
            strategy: Chunking strategy. One of: token, sentence, paragraph,
                recursive, fixed_chars, markdown, hierarchical.
            chunk_size: Target chunk size in tokens.
            overlap_percent: Overlap between adjacent chunks, as a percent.
            top_k: Number of chunks retrieved per query.
            threshold: Cosine similarity above which a retrieved chunk counts
                as relevant, for the score-based metrics.
            provider: Embedding provider name. Left unset, auto-detected with
                a lexical fallback, same as diagnose_corpus.

        Returns:
            The five IR metrics, per-query scores, and the config actually
            used, including which embedder produced the numbers.
        """
        if not queries:
            return {"error": "queries must be a non-empty list."}
        try:
            docs = _load_text(text, path)
            from rag_tui.api import eval_async

            return await eval_async(
                queries=queries,
                docs=docs,
                strategy=strategy,
                chunk_size=chunk_size,
                overlap_percent=overlap_percent,
                top_k=top_k,
                threshold=threshold,
                provider=provider,
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    # -----------------------------------------------------------------
    # optimize_chunking
    # -----------------------------------------------------------------

    @server.tool(structured_output=False)
    async def optimize_chunking(
        queries: List[str],
        text: Optional[str] = None,
        path: Optional[str] = None,
        strategies: Optional[List[str]] = None,
        sizes: Optional[List[int]] = None,
        overlap_percents: Optional[List[int]] = None,
        top_k: int = 3,
        threshold: float = 0.5,
        provider: Optional[str] = None,
        max_concurrent: int = 3,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> Dict[str, Any]:
        """Sweep chunking configs and rank them for a set of test queries.

        Tests every combination of strategy, chunk size, and overlap in
        parallel and ranks them by a composite score. This is the expensive
        tool: strategies times sizes times overlaps configs, each needing an
        embedding pass. Narrow the ranges for a quick check, or run the
        defaults for a thorough sweep (up to 140 configs).

        Args:
            queries: The test queries every config is scored against.
            text: The document text, inline. Exactly one of text/path.
            path: Path to a document file on disk. Exactly one of text/path.
            strategies: Strategies to try. Defaults to ["token"] if omitted.
                One of: token, sentence, paragraph, recursive, fixed_chars,
                markdown, hierarchical.
            sizes: Chunk sizes to try. Defaults to a standard sweep.
            overlap_percents: Overlap percentages to try. Defaults to a
                standard sweep.
            top_k: Number of chunks retrieved per query for scoring.
            threshold: Cosine similarity above which a chunk counts relevant.
            provider: Embedding provider name. Left unset, auto-detected with
                a lexical fallback.
            max_concurrent: Cap on concurrent embedding calls, so this does
                not hammer a rate-limited provider.
            max_results: Cap on ranked configs returned. `best` is always
                included and `total_configs_tested` always reflects the full
                sweep even when the list is capped.

        Returns:
            The best config, the full ranked list (capped at max_results),
            how many configs were tested, and which embedder ran.
        """
        if not queries:
            return {"error": "queries must be a non-empty list."}
        try:
            docs = _load_text(text, path)
            from rag_tui.api import optimize_async

            result = await optimize_async(
                docs=docs,
                queries=queries,
                strategies=strategies,
                sizes=sizes,
                overlap_percents=overlap_percents,
                top_k=top_k,
                threshold=threshold,
                provider=provider,
                max_concurrent=max_concurrent,
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

        all_results = result["all_results"]
        if len(all_results) > max_results:
            result["all_results"] = all_results[:max_results]
            result["results_omitted"] = len(all_results) - max_results
        return result

    # -----------------------------------------------------------------
    # export_chunking_config
    # -----------------------------------------------------------------

    @server.tool(structured_output=False)
    def export_chunking_config(
        format: str = "langchain",
        strategy: str = "token",
        chunk_size: int = 200,
        overlap_percent: int = 10,
    ) -> Dict[str, Any]:
        """Generate the code to reproduce a chunking config in another framework.

        Use this once diagnose_corpus or optimize_chunking has pointed at a
        config worth keeping, to hand it off as ready-to-paste code rather
        than a description of numbers.

        Args:
            format: One of: json, langchain, llamaindex.
            strategy: Chunking strategy name, for the config's own record.
            chunk_size: Target chunk size in tokens.
            overlap_percent: Overlap between adjacent chunks, as a percent.

        Returns:
            The generated code (or JSON config) as a string, and the format
            it was generated for.
        """
        try:
            from rag_tui.api import export

            code = export(
                format=format,
                strategy=strategy,
                chunk_size=chunk_size,
                overlap_percent=overlap_percent,
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        return {"format": format, "code": code}

    return server


def main() -> None:
    """Entry point for `rag-tui mcp`. Runs over stdio."""
    server = create_server()
    server.run()


if __name__ == "__main__":
    main()
