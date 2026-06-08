"""Command-line interface for RAG-TUI.

Subcommands:
    (none)      Launch the interactive TUI
    ui          Launch the interactive TUI
    chunk       Chunk a file / stdin and print JSON or CSV
    eval        Batch retrieval evaluation (queries from file, text, or dataset)
    optimize    Auto-recommend the best chunking config
    compare     Compare a new eval result against a saved baseline
    export      Print LangChain / LlamaIndex boilerplate for a config
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from rag_tui.core.engine import ChunkingEngine
from rag_tui.core.file_handler import read_file
from rag_tui.core.metrics import (
    ChunkConfig,
    QueryResult,
    calculate_batch_metrics,
    compare_results,
    export_config,
    load_dataset,
)
from rag_tui.core.optimizer import ChunkOptimizer
from rag_tui.core.providers import ProviderType, get_best_provider, get_provider
from rag_tui.core.strategies import StrategyType
from rag_tui.core.vector import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_strategy(value: str) -> StrategyType:
    try:
        strategy = StrategyType(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Unknown strategy '{value}'. Choices: {[s.value for s in StrategyType]}"
        ) from exc
    if strategy == StrategyType.CUSTOM:
        raise argparse.ArgumentTypeError(
            "Strategy 'custom' is only available in the TUI."
        )
    return strategy


def _read_text(file_path: Optional[str], text: Optional[str]) -> str:
    if file_path:
        content, _ = read_file(file_path)
        return content
    if text is not None:
        return text
    return sys.stdin.read()


def _overlap_tokens(chunk_size: int, overlap_percent: int) -> int:
    return max(0, int(round(chunk_size * (overlap_percent / 100.0))))


async def _resolve_embedding_provider(name: Optional[str]):
    if name is None:
        provider, _ = await get_best_provider()
        if provider is None:
            _die(
                "No embedding provider available. "
                "Run Ollama (ollama serve) or set OPENAI_API_KEY / GOOGLE_API_KEY."
            )
        return provider
    provider_type = ProviderType(name)
    provider = get_provider(provider_type)
    if not await provider.check_connection():
        _die(f"Provider '{name}' is not reachable. Check credentials or service.")
    return provider


def _die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def _emit_json(obj) -> None:
    json.dump(obj, sys.stdout, indent=2)
    sys.stdout.write("\n")


# ---------------------------------------------------------------------------
# chunk
# ---------------------------------------------------------------------------

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
        _emit_json({
            "strategy": args.strategy.value,
            "chunk_size": args.chunk_size,
            "overlap_percent": args.overlap_percent,
            "overlap_tokens": overlap,
            "stats": stats,
            "chunks": [
                {"index": i, "start": start, "end": end, "text": chunk}
                for i, (chunk, start, end) in enumerate(chunks)
            ],
        })
    else:
        writer = csv.writer(sys.stdout)
        writer.writerow(["index", "start", "end", "text"])
        for i, (chunk, start, end) in enumerate(chunks):
            writer.writerow([i, start, end, chunk])
    return 0


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

async def _build_eval_output(
    text: str,
    queries: List[str],
    strategy: StrategyType,
    chunk_size: int,
    overlap_percent: int,
    top_k: int,
    threshold: float,
    provider_name: Optional[str],
    use_judge: bool = False,
) -> dict:
    engine = ChunkingEngine()
    overlap = _overlap_tokens(chunk_size, overlap_percent)
    chunk_results = engine.chunk_text(
        text=text,
        chunk_size=chunk_size,
        overlap=overlap,
        strategy_type=strategy,
    )
    chunks = [c[0] for c in chunk_results]

    provider = await _resolve_embedding_provider(provider_name)
    chunk_embeddings = await provider.embed_batch(chunks)
    store = VectorStore(embedding_dim=len(chunk_embeddings[0]))
    store.add_chunks(chunks, np.array(chunk_embeddings))

    judge = None
    if use_judge:
        from rag_tui.core.judge import LLMJudge
        if not hasattr(provider, "generate"):
            _die("--use-judge requires a provider with LLM generation support (Ollama, OpenAI, Groq).")
        judge = LLMJudge(provider)
        print("Judge mode: scoring each retrieved chunk with LLM...", file=sys.stderr)

    query_results: List[QueryResult] = []
    for query in queries:
        q_emb = np.array(await provider.embed(query))
        matches = store.search(q_emb, top_k=top_k)
        retrieved = [(m[0], m[1]) for m in matches]
        top_score = retrieved[0][1] if retrieved else 0.0
        avg_score = sum(s for _, s in retrieved) / len(retrieved) if retrieved else 0.0
        relevance_labels = None
        faithfulness = None
        if judge and retrieved:
            chunks_text = [c for c, _ in retrieved]
            relevance_labels, faithfulness = await judge.evaluate_query(query, chunks_text)
        query_results.append(QueryResult(
            query=query,
            chunks_retrieved=retrieved,
            top_score=top_score,
            avg_score=avg_score,
            relevance_labels=relevance_labels,
            faithfulness=faithfulness,
        ))

    batch = calculate_batch_metrics(query_results, threshold=threshold, top_k=top_k)
    output = batch.to_dict()
    output["config"] = {
        "strategy": strategy.value,
        "chunk_size": chunk_size,
        "overlap_percent": overlap_percent,
        "overlap_tokens": overlap,
        "top_k": top_k,
        "threshold": threshold,
        "provider": provider_name or "auto",
    }
    return output


async def _run_eval(args: argparse.Namespace) -> int:
    text = _read_text(args.file, None)

    # Query source: --dataset-file > --queries-file > --queries-text
    if args.dataset_file:
        rows = load_dataset(args.dataset_file)
        if not rows:
            _die(f"Dataset is empty: {args.dataset_file}")
        queries = [r["query"] for r in rows if r.get("query")]
        if not queries:
            _die("Dataset has no 'query' column/key.")
    else:
        queries_text = _read_text(args.queries_file, args.queries_text)
        queries = [q.strip() for q in queries_text.splitlines() if q.strip()]
        if not queries:
            _die("No queries provided. Use --queries-file, --queries-text, or --dataset-file.")

    output = await _build_eval_output(
        text=text,
        queries=queries,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap_percent=args.overlap_percent,
        top_k=args.top_k,
        threshold=args.threshold,
        provider_name=args.provider,
        use_judge=getattr(args, "use_judge", False),
    )

    if args.save_baseline:
        path = Path(args.save_baseline)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Baseline saved → {path}", file=sys.stderr)

    _emit_json(output)
    return 0


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------

async def _run_optimize(args: argparse.Namespace) -> int:
    text = _read_text(args.file, None)

    # Queries
    if args.dataset_file:
        rows = load_dataset(args.dataset_file)
        queries = [r["query"] for r in rows if r.get("query")]
    else:
        queries_text = _read_text(args.queries_file, args.queries_text)
        queries = [q.strip() for q in queries_text.splitlines() if q.strip()]

    if not queries:
        _die("No queries provided for optimization.")

    provider = await _resolve_embedding_provider(args.provider)

    strategies = [s.strip() for s in args.strategies.split(",")]
    sizes = [int(s) for s in args.sizes.split(",")]
    overlaps = [int(s) for s in args.overlaps.split(",")]

    printed = [0]

    def progress_cb(msg: str) -> None:
        print(f"  {msg}", file=sys.stderr)
        printed[0] += 1

    print(
        f"Optimizing: {len(sizes)} sizes × {len(overlaps)} overlaps × "
        f"{len(strategies)} strategies = {len(sizes)*len(overlaps)*len(strategies)} configs",
        file=sys.stderr,
    )

    optimizer = ChunkOptimizer(
        sizes=sizes,
        overlap_percents=overlaps,
        strategies=strategies,
        top_k=args.top_k,
        threshold=args.threshold,
        max_concurrent=args.max_concurrent,
    )
    report = await optimizer.optimize(text, queries, provider, progress_cb=progress_cb)

    _emit_json(report.to_dict())

    print(f"\n{report.summary()}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

async def _run_compare(args: argparse.Namespace) -> int:
    with open(args.baseline, encoding="utf-8") as f:
        baseline_data = json.load(f)

    with open(args.current, encoding="utf-8") as f:
        current_data = json.load(f)

    from rag_tui.core.metrics import BatchTestResult, ChunkConfig

    def _make_result(data):
        m = data.get("metrics", {})
        return BatchTestResult(
            queries=[],
            timestamp=data.get("timestamp", ""),
            total_queries=data.get("total_queries", 0),
            avg_top_score=m.get("avg_top_score", 0.0),
            avg_retrieval_score=m.get("avg_retrieval_score", 0.0),
            hit_rate=m.get("hit_rate", 0.0),
            mrr=m.get("mrr", 0.0),
            ndcg_at_k=m.get("ndcg_at_k", 0.0),
            recall_at_k=m.get("recall_at_k", 0.0),
            precision_at_k=m.get("precision_at_k", 0.0),
        )

    def _make_config(data):
        c = data.get("config", {})
        return ChunkConfig(
            strategy=c.get("strategy", "token"),
            chunk_size=c.get("chunk_size", 200),
            overlap_percent=c.get("overlap_percent", 10),
            overlap_tokens=c.get("overlap_tokens", 20),
        )

    comparison = compare_results(
        _make_result(baseline_data),
        _make_result(current_data),
        _make_config(baseline_data),
        _make_config(current_data),
    )

    _emit_json(comparison.to_dict())

    print("\n── Metric deltas ──", file=sys.stderr)
    for d in comparison.deltas:
        print(f"  {d.to_display()}", file=sys.stderr)

    status = "IMPROVED" if comparison.overall_improved else "REGRESSION"
    print(f"\nOverall: {status}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-tui",
        description="RAG-TUI: interactive chunking debugger with headless CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  rag-tui                                      # launch TUI\n"
            "  rag-tui chunk --file doc.txt --chunk-size 256\n"
            "  rag-tui eval --file doc.txt --queries-file q.txt --chunk-size 256\n"
            "  rag-tui eval --file doc.txt --dataset-file queries.csv --save-baseline bl.json\n"
            "  rag-tui optimize --file doc.txt --queries-file q.txt --strategies token,sentence\n"
            "  rag-tui compare --baseline bl.json --current new.json\n"
            "  rag-tui export --format langchain --chunk-size 256\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    # ui
    subparsers.add_parser("ui", help="Launch the interactive TUI")

    # chunk
    cp = subparsers.add_parser("chunk", help="Chunk text and output JSON/CSV")
    cp.add_argument("--file", help="Path to input file")
    cp.add_argument("--text", help="Inline text input")
    cp.add_argument("--strategy", type=_parse_strategy, default=StrategyType.TOKEN)
    cp.add_argument("--chunk-size", type=int, default=200)
    cp.add_argument("--overlap-percent", type=int, default=10)
    cp.add_argument("--format", choices=["json", "csv"], default="json")
    cp.set_defaults(func=_run_chunk)

    # eval
    ep = subparsers.add_parser("eval", help="Run batch retrieval evaluation")
    ep.add_argument("--file", required=True, help="Document file to index")
    ep.add_argument("--queries-file", help="File with queries (one per line)")
    ep.add_argument("--queries-text", help="Inline queries (one per line)")
    ep.add_argument("--dataset-file", help="CSV/JSONL dataset with a 'query' column")
    ep.add_argument("--strategy", type=_parse_strategy, default=StrategyType.TOKEN)
    ep.add_argument("--chunk-size", type=int, default=200)
    ep.add_argument("--overlap-percent", type=int, default=10)
    ep.add_argument("--top-k", type=int, default=3)
    ep.add_argument("--threshold", type=float, default=0.5)
    ep.add_argument(
        "--provider", choices=[p.value for p in ProviderType],
        help="Embedding provider (default: auto-detect)",
    )
    ep.add_argument(
        "--save-baseline", metavar="FILE",
        help="Save this run's metrics to FILE for later comparison",
    )
    ep.add_argument(
        "--use-judge", action="store_true", default=False,
        help="Score retrieved chunks with LLM judge for real relevance metrics (requires LLM provider)",
    )
    ep.set_defaults(func=_run_eval)

    # optimize
    op = subparsers.add_parser("optimize", help="Auto-recommend the best chunking config")
    op.add_argument("--file", required=True, help="Document file to index")
    op.add_argument("--queries-file", help="File with queries (one per line)")
    op.add_argument("--queries-text", help="Inline queries (one per line)")
    op.add_argument("--dataset-file", help="CSV/JSONL dataset with a 'query' column")
    op.add_argument(
        "--strategies", default="token",
        help="Comma-separated strategies to test (default: token)",
    )
    op.add_argument(
        "--sizes", default="64,128,200,256,320,400,512",
        help="Comma-separated chunk sizes to test",
    )
    op.add_argument(
        "--overlaps", default="5,10,15,20",
        help="Comma-separated overlap percentages to test",
    )
    op.add_argument("--top-k", type=int, default=3)
    op.add_argument("--threshold", type=float, default=0.5)
    op.add_argument(
        "--provider", choices=[p.value for p in ProviderType],
        help="Embedding provider (default: auto-detect)",
    )
    op.add_argument("--max-concurrent", type=int, default=3)
    op.set_defaults(func=_run_optimize)

    # compare
    comp = subparsers.add_parser("compare", help="Compare eval result against a saved baseline")
    comp.add_argument("--baseline", required=True, help="Baseline JSON file (from eval --save-baseline)")
    comp.add_argument("--current", required=True, help="Current eval JSON file to compare")
    comp.set_defaults(func=_run_compare)

    # export
    xp = subparsers.add_parser("export", help="Export chunk config for frameworks")
    xp.add_argument("--strategy", type=_parse_strategy, default=StrategyType.TOKEN)
    xp.add_argument("--chunk-size", type=int, default=200)
    xp.add_argument("--overlap-percent", type=int, default=10)
    xp.add_argument("--format", choices=["json", "langchain", "llamaindex"], default="json")
    xp.set_defaults(func=_run_export)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None or args.command == "ui":
        from rag_tui.app import main as run_tui
        run_tui()
        return

    if not hasattr(args, "func"):
        parser.print_help()
        return

    result = args.func(args)
    if asyncio.iscoroutine(result):
        try:
            sys.exit(asyncio.run(result))
        except (KeyboardInterrupt, SystemExit):
            pass
        except Exception as exc:
            _die(str(exc))


if __name__ == "__main__":
    main()
