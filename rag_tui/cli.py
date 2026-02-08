"""Command-line interface for RAG-TUI.

Defaults to launching the TUI when no subcommand is provided.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from dataclasses import asdict
from typing import List, Optional

import numpy as np

from rag_tui.app import main as run_tui
from rag_tui.core.engine import ChunkingEngine
from rag_tui.core.file_handler import read_file
from rag_tui.core.metrics import ChunkConfig, QueryResult, calculate_batch_metrics, export_config
from rag_tui.core.providers import ProviderType, get_best_provider, get_provider
from rag_tui.core.strategies import StrategyType
from rag_tui.core.vector import VectorStore


def _parse_strategy(value: str) -> StrategyType:
    try:
        strategy = StrategyType(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Unknown strategy '{value}'. Choices: {[s.value for s in StrategyType]}"
        ) from exc
    if strategy == StrategyType.CUSTOM:
        raise argparse.ArgumentTypeError(
            "Strategy 'custom' is only available in the TUI where you can provide code."
        )
    return strategy


def _read_text(file_path: Optional[str], text: Optional[str]) -> str:
    if file_path:
        content, _ = read_file(file_path)
        return content
    if text is not None:
        return text
    # Fall back to stdin
    return sys.stdin.read()


def _overlap_tokens(chunk_size: int, overlap_percent: int) -> int:
    return max(0, int(round(chunk_size * (overlap_percent / 100.0))))


async def _resolve_embedding_provider(name: Optional[str]):
    if name is None:
        provider, _ = await get_best_provider()
        if provider is None:
            raise RuntimeError("No embedding provider available. Set API keys or run Ollama.")
        return provider

    provider_type = ProviderType(name)
    provider = get_provider(provider_type)
    if not await provider.check_connection():
        raise RuntimeError(f"Provider '{name}' is not available. Check credentials or service.")
    return provider


async def _run_chunk(args: argparse.Namespace) -> int:
    text = _read_text(args.file, args.text)

    engine = ChunkingEngine()
    overlap = _overlap_tokens(args.chunk_size, args.overlap_percent)
    chunks = engine.chunk_text(
        text=text,
        chunk_size=args.chunk_size,
        overlap=overlap,
        strategy_type=args.strategy,
    )
    stats = engine.get_chunk_stats(chunks)

    if args.format == "json":
        payload = {
            "strategy": args.strategy.value,
            "chunk_size": args.chunk_size,
            "overlap_percent": args.overlap_percent,
            "overlap_tokens": overlap,
            "stats": stats,
            "chunks": [
                {
                    "index": i,
                    "start": start,
                    "end": end,
                    "text": chunk,
                }
                for i, (chunk, start, end) in enumerate(chunks)
            ],
        }
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    writer = csv.writer(sys.stdout)
    writer.writerow(["index", "start", "end", "text"])
    for i, (chunk, start, end) in enumerate(chunks):
        writer.writerow([i, start, end, chunk])
    return 0


async def _run_eval(args: argparse.Namespace) -> int:
    text = _read_text(args.file, None)
    queries_text = _read_text(args.queries_file, args.queries_text)
    queries = [q.strip() for q in queries_text.splitlines() if q.strip()]
    if not queries:
        raise RuntimeError("No queries provided. Use --queries-file or --queries-text.")

    engine = ChunkingEngine()
    overlap = _overlap_tokens(args.chunk_size, args.overlap_percent)
    chunk_results = engine.chunk_text(
        text=text,
        chunk_size=args.chunk_size,
        overlap=overlap,
        strategy_type=args.strategy,
    )
    chunks = [c[0] for c in chunk_results]

    provider = await _resolve_embedding_provider(args.provider)
    chunk_embeddings = await provider.embed_batch(chunks)
    embedding_dim = len(chunk_embeddings[0]) if chunk_embeddings else 0

    store = VectorStore(embedding_dim=embedding_dim)
    store.add_chunks(chunks, np.array(chunk_embeddings))

    query_results: List[QueryResult] = []
    for query in queries:
        query_embedding = np.array(await provider.embed(query))
        matches = store.search(query_embedding, top_k=args.top_k)
        retrieved = [(m[0], m[1]) for m in matches]
        top_score = retrieved[0][1] if retrieved else 0.0
        avg_score = sum(r[1] for r in retrieved) / len(retrieved) if retrieved else 0.0
        query_results.append(
            QueryResult(
                query=query,
                chunks_retrieved=retrieved,
                top_score=top_score,
                avg_score=avg_score,
            )
        )

    batch = calculate_batch_metrics(query_results, threshold=args.threshold)
    output = batch.to_dict()
    output["config"] = {
        "strategy": args.strategy.value,
        "chunk_size": args.chunk_size,
        "overlap_percent": args.overlap_percent,
        "overlap_tokens": overlap,
        "top_k": args.top_k,
        "threshold": args.threshold,
        "provider": args.provider or "auto",
    }

    json.dump(output, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


async def _run_export(args: argparse.Namespace) -> int:
    overlap = _overlap_tokens(args.chunk_size, args.overlap_percent)
    config = ChunkConfig(
        strategy=args.strategy.value,
        chunk_size=args.chunk_size,
        overlap_percent=args.overlap_percent,
        overlap_tokens=overlap,
    )
    sys.stdout.write(export_config(config, format=args.format))
    if not sys.stdout.isatty():
        sys.stdout.write("\n")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-tui",
        description="RAG-TUI: chunking debugger with headless CLI and TUI.",
    )

    subparsers = parser.add_subparsers(dest="command")

    ui_parser = subparsers.add_parser("ui", help="Launch the interactive TUI")
    ui_parser.set_defaults(func=lambda _: run_tui())

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text and output JSON/CSV")
    chunk_parser.add_argument("--file", help="Path to input file")
    chunk_parser.add_argument("--text", help="Inline text input")
    chunk_parser.add_argument(
        "--strategy",
        type=_parse_strategy,
        default=StrategyType.TOKEN,
        help="Chunking strategy",
    )
    chunk_parser.add_argument("--chunk-size", type=int, default=200)
    chunk_parser.add_argument("--overlap-percent", type=int, default=10)
    chunk_parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
    )
    chunk_parser.set_defaults(func=_run_chunk)

    eval_parser = subparsers.add_parser("eval", help="Run batch retrieval evaluation")
    eval_parser.add_argument("--file", required=True, help="Document file to index")
    eval_parser.add_argument("--queries-file", help="File with queries (one per line)")
    eval_parser.add_argument("--queries-text", help="Inline queries (one per line)")
    eval_parser.add_argument(
        "--strategy",
        type=_parse_strategy,
        default=StrategyType.TOKEN,
        help="Chunking strategy",
    )
    eval_parser.add_argument("--chunk-size", type=int, default=200)
    eval_parser.add_argument("--overlap-percent", type=int, default=10)
    eval_parser.add_argument("--top-k", type=int, default=3)
    eval_parser.add_argument("--threshold", type=float, default=0.5)
    eval_parser.add_argument(
        "--provider",
        choices=[p.value for p in ProviderType],
        help="Embedding provider (default: auto-detect)",
    )
    eval_parser.set_defaults(func=_run_eval)

    export_parser = subparsers.add_parser("export", help="Export chunk config for frameworks")
    export_parser.add_argument(
        "--strategy",
        type=_parse_strategy,
        default=StrategyType.TOKEN,
        help="Chunking strategy",
    )
    export_parser.add_argument("--chunk-size", type=int, default=200)
    export_parser.add_argument("--overlap-percent", type=int, default=10)
    export_parser.add_argument(
        "--format",
        choices=["json", "langchain", "llamaindex"],
        default="json",
    )
    export_parser.set_defaults(func=_run_export)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        run_tui()
        return

    result = args.func(args)
    if asyncio.iscoroutine(result):
        asyncio.run(result)


if __name__ == "__main__":
    main()
