"""RAG-TUI v0.1.0: Interactive Chunking Debugger.

A terminal UI for visualizing, debugging, tuning, and auto-optimizing
RAG chunking pipelines. Features:
  - Six chunking strategies with live parameter tuning
  - Multi-provider embedding/LLM (Ollama, OpenAI, Groq, Google)
  - Semantic search with similarity visualization
  - Batch evaluation: MRR, nDCG@k, Recall@k, Precision@k, Hit Rate
  - Baseline comparison & regression detection
  - Automated chunk-size recommender (A/B test optimizer)
  - Config export for LangChain and LlamaIndex
  - Persistent embedding cache (SQLite)
"""

import asyncio
import json
from pathlib import Path
from typing import List, Optional, Tuple

from RestrictedPython import compile_restricted_exec, safe_builtins, safe_globals

import numpy as np

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import (
    Button, Footer, Header, Input, Label, RichLog,
    Select, Static, TabbedContent, TabPane, TextArea,
)

from rag_tui.core.engine import ChunkingEngine, StrategyType
from rag_tui.core.file_handler import format_file_size, read_file
from rag_tui.core.metrics import (
    BatchTestResult, ChunkConfig, QueryResult,
    calculate_batch_metrics, compare_results, export_config,
)
from rag_tui.core.optimizer import ChunkOptimizer
from rag_tui.core.presets import list_presets, load_preset
from rag_tui.core.providers import LLMProvider, get_best_provider
from rag_tui.core.vector import VectorStore
from rag_tui.widgets.chunk_card import ChunkList
from rag_tui.widgets.help_overlay import HelpOverlay
from rag_tui.widgets.optimize_panel import OptimizePanel
from rag_tui.widgets.parameter_panel import ParameterPanel
from rag_tui.widgets.search_panel import SearchPanel


SAMPLE_TEXT = """Retrieval-Augmented Generation (RAG) is a powerful technique for enhancing large language models with external knowledge. The RAG pipeline consists of several key components: document ingestion, text chunking, embedding generation, vector storage, and retrieval.

During the ingestion phase, documents are split into smaller, manageable chunks. The chunk size and overlap are critical parameters that affect retrieval quality. Smaller chunks provide more precise retrieval but may lose context, while larger chunks maintain context but reduce precision.

Each chunk is then converted into a dense vector embedding using a specialized embedding model. These embeddings capture the semantic meaning of the text, allowing for similarity-based retrieval. The embeddings are stored in a vector database that enables efficient similarity search.

When a user submits a query, it is also embedded using the same model. The system then retrieves the most semantically similar chunks from the vector store. These chunks serve as context for the language model, which generates a response grounded in the retrieved information.

This approach significantly reduces hallucinations and allows LLMs to work with up-to-date, proprietary, or domain-specific information that wasn't part of their original training data."""

STRATEGY_OPTIONS = [
    ("Token-based (Chonkie)", "token"),
    ("Sentence boundaries", "sentence"),
    ("Paragraph breaks", "paragraph"),
    ("Recursive splitting", "recursive"),
    ("Fixed characters", "fixed_chars"),
]


class RAGTUIApp(App):
    """RAG-TUI v0.1.0 — Interactive chunking debugger and optimizer."""

    CSS_PATH = Path(__file__).parent / "styles" / "app.tcss"
    TITLE = "RAG-TUI v0.1.0"
    SUB_TITLE = "Interactive Chunking Debugger"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("l", "load_sample", "Load Sample"),
        Binding("r", "rechunk", "Rechunk"),
        Binding("d", "toggle_dark", "Toggle Theme"),
        Binding("e", "export_config", "Export Config"),
        Binding("f1", "show_help", "Help"),
        Binding("question_mark", "show_help", "Help", show=False),
        Binding("1", "strategy_token", "Token", show=False),
        Binding("2", "strategy_sentence", "Sentence", show=False),
        Binding("3", "strategy_paragraph", "Paragraph", show=False),
        Binding("4", "strategy_recursive", "Recursive", show=False),
        Binding("5", "strategy_fixed", "Fixed", show=False),
    ]

    def __init__(self):
        super().__init__()
        self.chunking_engine = ChunkingEngine()
        self.vector_store = VectorStore(embedding_dim=768)
        self.embedding_provider: Optional[LLMProvider] = None
        self.llm_provider: Optional[LLMProvider] = None

        self._current_text = ""
        self._current_chunks: List[Tuple] = []
        self._chunk_size = 200
        self._overlap_percent = 10
        self._current_strategy = StrategyType.TOKEN
        self._debounce_task: Optional[asyncio.Task] = None
        self._embedding_task: Optional[asyncio.Task] = None
        self._batch_results: List[QueryResult] = []
        self._file_info = None
        self._custom_cleaner = None

        # Baseline comparison state
        self._baseline_result: Optional[BatchTestResult] = None
        self._baseline_config: Optional[ChunkConfig] = None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal(id="strategy-bar"):
            yield Label("Strategy:")
            yield Select(STRATEGY_OPTIONS, value="token", id="strategy-select")
            yield Label("  │  File:")
            yield Input(placeholder="Path to file...", id="file-path-input")
            yield Button("Load", id="load-file-btn", variant="primary")

        yield Static("", id="status-strip", markup=True)

        with TabbedContent(id="main-tabs"):
            # 1 — Input
            with TabPane("Input", id="input-tab"):
                yield Static("Paste or load your document:", classes="tab-intro")
                yield TextArea(id="text-input", language="markdown")
                with Horizontal(classes="button-row"):
                    yield Button("Load Sample", id="load-sample-btn", variant="default")
                    yield Button("Quick Clean", id="quick-clean-btn", variant="primary")
                    yield Button("Clear", id="clear-btn", variant="warning")

            # 2 — Chunks
            with TabPane("Chunks", id="chunks-tab"):
                yield ParameterPanel(
                    chunk_size=self._chunk_size,
                    overlap_percent=self._overlap_percent,
                    id="parameter-panel",
                )
                yield Static("", id="chunk-summary", classes="tab-intro")
                yield ChunkList(id="chunk-list")

            # 3 — Search
            with TabPane("Search", id="search-tab"):
                yield SearchPanel(id="search-panel")

            # 4 — Batch
            with TabPane("Batch", id="batch-tab"):
                with VerticalScroll():
                    yield Static("Enter multiple queries (one per line):", classes="tab-intro")
                    yield TextArea(id="batch-queries")
                    with Horizontal(classes="button-row"):
                        yield Button("Run Batch Test", id="run-batch-btn", variant="success")
                        yield Button("Run with Judge", id="run-judge-btn", variant="primary")
                        yield Button("Save as Baseline", id="save-baseline-btn", variant="default")
                        yield Button("Compare to Baseline", id="compare-baseline-btn", variant="default")
                        yield Button("Clear Results", id="clear-batch-btn", variant="default")
                    yield Static(
                        "Tip: 'Run with Judge' scores each retrieved chunk using your LLM for real relevance metrics.",
                        classes="tab-intro",
                        id="judge-tip",
                    )
                    yield Static("", id="batch-results", classes="batch-results")
                    yield Static("", id="baseline-comparison", classes="batch-results")

            # 5 — Optimize  (killer feature)
            with TabPane("Optimize", id="optimize-tab"):
                yield OptimizePanel(id="optimize-panel")

            # 6 — Settings & Export
            with TabPane("Settings", id="settings-tab"):
                with VerticalScroll():
                    yield Static("Recommendations", classes="section-title")
                    yield Static("", id="recommendations-panel", classes="recommendations-panel")

                    yield Static("Quick Presets", classes="section-title")
                    preset_options = [(p.name, p.name) for p in list_presets()]
                    yield Select(preset_options, id="preset-select", prompt="Select a preset...")
                    yield Button("Apply Preset", id="apply-preset-btn", variant="primary")

                    yield Static("Export Configuration", classes="section-title")
                    with Horizontal(classes="button-row"):
                        yield Button("JSON", id="export-json-btn", variant="primary")
                        yield Button("LangChain", id="export-langchain-btn", variant="default")
                        yield Button("LlamaIndex", id="export-llamaindex-btn", variant="default")
                    yield Static("", id="export-preview", classes="export-preview")

                    yield Static("Custom Chunker", classes="section-title")
                    yield Static(
                        "def chunk_name(text, chunk_size, overlap) -> [(text, start, end), ...]",
                        classes="tab-intro",
                    )
                    yield TextArea(id="custom-code", language="python")
                    yield Button("Apply Custom Chunker", id="apply-custom-btn", variant="success")

                    yield Static("Custom Cleaner (Optional)", classes="section-title")
                    yield Static("def clean_name(text) -> cleaned_text", classes="tab-intro")
                    yield TextArea(id="custom-cleaner-code", language="python")
                    yield Button("Apply Custom Cleaner", id="apply-cleaner-btn", variant="primary")

            # 7 — Chat
            with TabPane("Chat", id="chat-tab"):
                yield RichLog(id="chat-log", markup=True, wrap=True)
                yield Static("", id="chat-live-response", classes="live-response")
                with Horizontal(id="chat-input-row"):
                    yield Input(placeholder="Ask a question...", id="chat-input")
                    yield Button("Send", id="chat-send-btn", variant="primary")

        yield Footer()

    # ------------------------------------------------------------------
    # Mount
    # ------------------------------------------------------------------

    async def on_mount(self) -> None:
        self.embedding_provider, self.llm_provider = await get_best_provider()

        if self.embedding_provider:
            self.notify(
                f"Embeddings: {self.embedding_provider.config.name}",
                severity="information", timeout=4,
            )
        else:
            self.notify(
                "No embedding provider — Search & Optimize disabled. "
                "Run Ollama or set OPENAI_API_KEY.",
                severity="warning", timeout=8,
            )

        if self.llm_provider:
            self.notify(
                f"LLM: {self.llm_provider.config.name}",
                severity="information", timeout=4,
            )
            self.sub_title = f"Provider: {self.llm_provider.config.name}"
        else:
            self.notify(
                "No LLM provider — Chat disabled. Chunking & Export still work.",
                severity="warning", timeout=8,
            )
            self.sub_title = "View-Only Mode (no provider)"

        await self.action_load_sample()
        self._update_export_preview()
        self._update_status_strip()

    def _update_status_strip(self) -> None:
        """Refresh the always-visible status bar with current config context."""
        from rich.markup import escape
        try:
            strategy = self._current_strategy.value if self._current_strategy else "token"
            chunks = len(self._current_chunks)
            chunks_str = f"[cyan]{chunks}[/cyan] chunk{'s' if chunks != 1 else ''}"
            provider_str = (
                f"[green]{escape(self.embedding_provider.config.name)}[/green]"
                if self.embedding_provider
                else "[dim]no provider[/dim]"
            )
            strip = self.query_one("#status-strip", Static)
            strip.update(
                f" [bold cyan]{escape(strategy)}[/bold cyan]"
                f"  [dim]·[/dim]  [cyan]{self._chunk_size}[/cyan] tok"
                f"  [dim]·[/dim]  [cyan]{self._overlap_percent}%[/cyan] ovlp"
                f"  [dim]│[/dim]  {provider_str}"
                f"  [dim]│[/dim]  {chunks_str}"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Event handlers — top-level widgets
    # ------------------------------------------------------------------

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "strategy-select":
            strategy_map = {
                "token": StrategyType.TOKEN,
                "sentence": StrategyType.SENTENCE,
                "paragraph": StrategyType.PARAGRAPH,
                "recursive": StrategyType.RECURSIVE,
                "fixed_chars": StrategyType.FIXED_CHARS,
            }
            self._current_strategy = strategy_map.get(str(event.value), StrategyType.TOKEN)
            self.chunking_engine.set_strategy(self._current_strategy)
            self._update_status_strip()
            asyncio.create_task(self._rechunk())
            self.notify(f"Strategy: {event.value}", timeout=1)

    async def on_parameter_panel_parameters_changed(
        self, event: ParameterPanel.ParametersChanged
    ) -> None:
        self._chunk_size = event.chunk_size
        self._overlap_percent = event.overlap_percent
        self._update_export_preview()
        self._update_status_strip()
        await self._debounced_rechunk()

    async def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id == "text-input":
            self._current_text = event.text_area.text
            self._update_recommendations()
            await self._debounced_rechunk()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "file-path-input":
            await self.action_load_file()
        elif event.input.id == "chat-input":
            await self._submit_chat_message()

    # ------------------------------------------------------------------
    # Optimize panel messages
    # ------------------------------------------------------------------

    async def on_optimize_panel_optimize_requested(
        self, event: OptimizePanel.OptimizeRequested
    ) -> None:
        """Run the optimizer when user hits Run Optimization."""
        if not self.embedding_provider:
            self.notify(
                "Embedding provider not available. "
                "Run Ollama or set OPENAI_API_KEY.",
                severity="error",
            )
            return

        if not self._current_text.strip():
            self.notify("Load a document first (Input tab).", severity="warning")
            return

        panel = self.query_one("#optimize-panel", OptimizePanel)
        panel.set_running(True)
        panel.log_progress(
            f"Starting: {len(event.sizes)} sizes × "
            f"{len(event.overlaps)} overlaps × "
            f"{len(event.strategies)} strategies = "
            f"{len(event.sizes)*len(event.overlaps)*len(event.strategies)} configs"
        )

        async def _run():
            try:
                optimizer = ChunkOptimizer(
                    sizes=event.sizes,
                    overlap_percents=event.overlaps,
                    strategies=event.strategies,
                    max_concurrent=3,
                )
                report = await optimizer.optimize(
                    text=self._current_text,
                    queries=event.queries,
                    embedding_provider=self.embedding_provider,
                    progress_cb=lambda msg: self.call_from_thread(
                        panel.log_progress, msg
                    ) if False else panel.log_progress(msg),
                )
                panel.show_results(report.results)
                self.notify(
                    f"Optimization complete! Best: {report.best.chunk_size} tokens, "
                    f"{report.best.overlap_percent}% overlap, {report.best.strategy}",
                    timeout=6,
                )
            except Exception as exc:
                self.notify(f"Optimization error: {exc}", severity="error")
            finally:
                panel.set_running(False)

        asyncio.create_task(_run())

    def on_optimize_panel_config_applied(self, event: OptimizePanel.ConfigApplied) -> None:
        """Apply the selected optimization result to current chunking params."""
        self._chunk_size = event.chunk_size
        self._overlap_percent = event.overlap_percent
        strategy_map = {
            "token": StrategyType.TOKEN,
            "sentence": StrategyType.SENTENCE,
            "paragraph": StrategyType.PARAGRAPH,
            "recursive": StrategyType.RECURSIVE,
            "fixed_chars": StrategyType.FIXED_CHARS,
        }
        self._current_strategy = strategy_map.get(event.strategy, StrategyType.TOKEN)
        self.chunking_engine.set_strategy(self._current_strategy)

        try:
            param_panel = self.query_one("#parameter-panel", ParameterPanel)
            param_panel.chunk_size = event.chunk_size
            param_panel.overlap_percent = event.overlap_percent
        except Exception:
            pass

        try:
            selector = self.query_one("#strategy-select", Select)
            selector.value = event.strategy
        except Exception:
            pass

        self._update_export_preview()
        self._update_status_strip()
        asyncio.create_task(self._rechunk())
        self.notify(
            f"Applied: {event.chunk_size} tokens, {event.overlap_percent}% overlap, {event.strategy}",
            timeout=3,
        )

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    async def _debounced_rechunk(self) -> None:
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        self._debounce_task = asyncio.create_task(self._delayed_rechunk())

    async def _delayed_rechunk(self) -> None:
        await asyncio.sleep(0.3)
        await self._rechunk()

    async def _rechunk(self) -> None:
        if not self._current_text.strip():
            self._current_chunks = []
            self._update_chunk_display()
            return
        try:
            overlap_tokens = int(self._chunk_size * (self._overlap_percent / 100))
            chunks = await self.chunking_engine.chunk_text_async(
                self._current_text,
                chunk_size=self._chunk_size,
                overlap=overlap_tokens,
                strategy_type=self._current_strategy,
            )
            self._current_chunks = chunks
            self._update_chunk_display()
            if self.embedding_provider and chunks:
                await self._update_embeddings()
        except Exception as exc:
            self.notify(f"Chunking error: {exc}", severity="error")

    def _update_chunk_display(self) -> None:
        try:
            self.query_one("#chunk-list", ChunkList).update_chunks(self._current_chunks)
            summary = self.query_one("#chunk-summary", Static)
            if self._current_chunks:
                stats = self.chunking_engine.get_chunk_stats(self._current_chunks)
                summary.update(
                    f"{stats['total_chunks']} chunks  ·  "
                    f"avg {stats['avg_chunk_size']:.0f} chars  ·  "
                    f"total {stats['total_characters']:,} chars  ·  "
                    f"~{stats['total_tokens_est']} tokens"
                )
                try:
                    self.query_one("#parameter-panel", ParameterPanel).update_chunk_count(
                        stats["total_chunks"]
                    )
                except Exception:
                    pass
            else:
                summary.update("No chunks yet")
            self._update_status_strip()
        except Exception:
            pass

    async def _update_embeddings(self) -> None:
        if not self.embedding_provider:
            return

        if self._embedding_task and not self._embedding_task.done():
            self._embedding_task.cancel()
            try:
                await self._embedding_task
            except asyncio.CancelledError:
                pass

        chunk_texts = [c[0] for c in self._current_chunks]
        if not chunk_texts:
            return

        async def do_embedding():
            try:
                embeddings = await self.embedding_provider.embed_batch(chunk_texts)
                emb_np = np.array(embeddings, dtype=np.float32)
                if emb_np.shape[1] != self.vector_store.embedding_dim:
                    self.vector_store = VectorStore(embedding_dim=emb_np.shape[1])
                self.vector_store.clear()
                self.vector_store.add_chunks(chunk_texts, emb_np)
                self.notify(f"Embeddings ready ({len(chunk_texts)} chunks)", timeout=2)
            except asyncio.CancelledError:
                pass
            except Exception:
                self.notify("Embeddings pending (retrying)...", severity="information", timeout=3)

        self._embedding_task = asyncio.create_task(do_embedding())

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def on_search_panel_query_submitted(
        self, event: SearchPanel.QuerySubmitted
    ) -> None:
        if not self.embedding_provider:
            self.notify(
                "Embedding provider not available — Search disabled.",
                severity="warning",
            )
            return
        if not self._current_chunks:
            self.notify("Load a document first.", severity="warning")
            return

        if event.action == "search":
            await self._search(event.query)
        else:
            await self._generate(event.query)

    async def _search(self, query: str) -> List[Tuple]:
        try:
            q_emb = await self.embedding_provider.embed(query)
            q_np = np.array(q_emb, dtype=np.float32)
            results = await self.vector_store.search_async(q_np, top_k=5)
            self.query_one("#search-panel", SearchPanel).update_results(results)
            return results
        except Exception as exc:
            self.notify(f"Search error: {exc}", severity="error")
            return []

    async def _generate(self, query: str) -> None:
        if not self.llm_provider:
            self.notify("LLM provider not available.", severity="warning")
            return
        results = await self._search(query)
        if not results:
            return
        try:
            prompt = self.llm_provider.build_rag_prompt(query, [r[0] for r in results[:3]])
            chat_log = self.query_one("#chat-log", RichLog)
            live = self.query_one("#chat-live-response", Static)
            chat_log.write(f"[bold cyan]User:[/] {query}")
            tabs = self.query_one("#main-tabs", TabbedContent)
            tabs.active = "chat-tab"
            live.styles.display = "block"
            live.update("Thinking...")
            buf = ""
            try:
                async for token in self.llm_provider.stream_generate(prompt):
                    buf += token
                    live.update(buf)
            except Exception as exc:
                buf += f"\n[Error: {exc}]"
            live.update("")
            live.styles.display = "none"
            chat_log.write(f"[bold green]AI:[/] {buf}\n")
            self.notify("Response complete", timeout=2)
        except Exception as exc:
            self.notify(f"Generation error: {exc}", severity="error")

    # ------------------------------------------------------------------
    # Batch testing
    # ------------------------------------------------------------------

    async def _run_batch_test(self, use_judge: bool = False) -> Optional[BatchTestResult]:
        if not self.embedding_provider:
            self.notify("Embedding provider not available.", severity="error")
            return None

        queries_input = self.query_one("#batch-queries", TextArea)
        queries = [q.strip() for q in queries_input.text.split("\n") if q.strip()]
        if not queries:
            self.notify("Enter queries first.", severity="warning")
            return None

        if use_judge and not self.llm_provider:
            self.notify("Judge mode requires an LLM provider (run Ollama or set an API key).", severity="warning")
            return None

        judge = None
        if use_judge:
            from rag_tui.core.judge import LLMJudge
            judge = LLMJudge(self.llm_provider)

        msg = f"Running {len(queries)} queries" + (" with LLM judge..." if judge else "...")
        self.notify(msg, timeout=3)

        results = []
        for query in queries:
            try:
                search_results = await self._search(query)
                if search_results:
                    scores = [r[1] for r in search_results]
                    chunks_text = [r[0] for r in search_results]
                    relevance_labels = None
                    faithfulness = None
                    if judge:
                        relevance_labels, faithfulness = await judge.evaluate_query(
                            query, chunks_text
                        )
                    results.append(QueryResult(
                        query=query,
                        chunks_retrieved=[(r[0][:100], r[1]) for r in search_results],
                        top_score=max(scores),
                        avg_score=sum(scores) / len(scores),
                        relevance_labels=relevance_labels,
                        faithfulness=faithfulness,
                    ))
                else:
                    results.append(QueryResult(
                        query=query, chunks_retrieved=[],
                        top_score=0.0, avg_score=0.0,
                    ))
            except Exception as exc:
                self.notify(f"Query failed: {query[:30]} ({exc})", severity="warning")

        batch = calculate_batch_metrics(results)
        self._batch_results = results
        self._render_batch_results(batch)
        done_msg = "Batch test complete" + (" (Judge mode)" if judge else "")
        self.notify(done_msg, timeout=3)
        return batch

    @staticmethod
    def _metric_bar(label: str, value: float, bar_width: int = 24) -> str:
        """Render a labeled metric bar with Rich color coding."""
        filled = int(bar_width * min(1.0, max(0.0, value)))
        empty = bar_width - filled
        if value >= 0.7:
            bar_color, val_color = "green", "bright_green"
        elif value >= 0.4:
            bar_color, val_color = "yellow", "bright_yellow"
        else:
            bar_color, val_color = "red", "bright_red"
        bar = f"[{bar_color}]{'█' * filled}[/{bar_color}][dim]{'░' * empty}[/dim]"
        val = f"[{val_color}]{value:.3f}[/{val_color}]"
        return f"  {label:<14} {bar}  {val}"

    def _render_batch_results(self, batch: BatchTestResult) -> None:
        from rich.markup import escape
        results_widget = self.query_one("#batch-results", Static)

        mode_styles = {
            "judge": "[bold green]LLM Judge[/bold green] [dim]· real relevance via LLM scoring[/dim]",
            "ground_truth": "[bold cyan]Ground Truth[/bold cyan] [dim]· labeled dataset[/dim]",
            "similarity": "[yellow]Cosine Similarity[/yellow] [dim]· proxy metric (no ground truth)[/dim]",
        }
        mode_line = mode_styles.get(batch.eval_mode, escape(batch.eval_mode))
        divider = f"[dim]{'─' * 46}[/dim]"

        lines = [
            f"[bold]Batch Results[/bold]  [dim]{batch.total_queries} queries[/dim]",
            f"  Eval: {mode_line}",
            divider,
            self._metric_bar("Hit Rate", batch.hit_rate),
            self._metric_bar("MRR", batch.mrr),
            self._metric_bar(f"nDCG@{batch.top_k}", batch.ndcg_at_k),
            self._metric_bar(f"Recall@{batch.top_k}", batch.recall_at_k),
            self._metric_bar(f"Precision@{batch.top_k}", batch.precision_at_k),
        ]
        if batch.avg_faithfulness is not None:
            lines.append(self._metric_bar("Faithfulness", batch.avg_faithfulness))
        lines.extend([
            divider,
            f"[bold]Query Details[/bold]  [dim]top {min(10, batch.total_queries)}[/dim]",
        ])

        for i, r in enumerate(batch.queries[:10], 1):
            if r.relevance_labels:
                relevant = any(lbl >= 0.5 for lbl in r.relevance_labels)
            else:
                relevant = r.top_score >= batch.threshold
            status = "[green]✓[/green]" if relevant else "[red]✗[/red]"
            score_color = "green" if r.top_score >= 0.7 else ("yellow" if r.top_score >= 0.4 else "red")
            faith_str = ""
            if r.faithfulness is not None:
                fc = "green" if r.faithfulness >= 0.7 else ("yellow" if r.faithfulness >= 0.4 else "red")
                faith_str = f"  faith=[{fc}]{r.faithfulness:.2f}[/{fc}]"
            lines.append(
                f"  {status} [dim]{i:>2}.[/dim] [italic]{escape(r.query[:52])}[/italic]"
            )
            lines.append(
                f"       score=[{score_color}]{r.top_score:.3f}[/{score_color}]{faith_str}"
            )
            if r.chunks_retrieved:
                preview = escape(r.chunks_retrieved[0][0][:64].replace("\n", " "))
                lines.append(f"       [dim]↳ {preview}[/dim]")

        results_widget.update("\n".join(lines))

    def _save_baseline(self, batch: BatchTestResult) -> None:
        self._baseline_result = batch
        self._baseline_config = ChunkConfig(
            strategy=self._current_strategy.value,
            chunk_size=self._chunk_size,
            overlap_percent=self._overlap_percent,
            overlap_tokens=int(self._chunk_size * self._overlap_percent / 100),
        )
        self.notify(
            f"Baseline saved: {self._current_strategy.value}, "
            f"{self._chunk_size} tokens, {self._overlap_percent}% overlap",
            timeout=4,
        )

    async def _compare_to_baseline(self) -> None:
        if not self._baseline_result:
            self.notify("No baseline saved. Run a batch test first, then Save as Baseline.", severity="warning")
            return

        current_batch = await self._run_batch_test()
        if not current_batch:
            return

        current_cfg = ChunkConfig(
            strategy=self._current_strategy.value,
            chunk_size=self._chunk_size,
            overlap_percent=self._overlap_percent,
            overlap_tokens=int(self._chunk_size * self._overlap_percent / 100),
        )

        comparison = compare_results(
            self._baseline_result, current_batch,
            self._baseline_config, current_cfg,
        )

        cmp_widget = self.query_one("#baseline-comparison", Static)
        divider = f"[dim]{'─' * 46}[/dim]"
        lines = [
            "[bold]Baseline Comparison[/bold]",
            divider,
            f"  Baseline: [cyan]{self._baseline_config.strategy}[/cyan]"
            f" / [cyan]{self._baseline_config.chunk_size}[/cyan] tok"
            f" / [cyan]{self._baseline_config.overlap_percent}%[/cyan] ovlp",
            f"  Current:  [cyan]{current_cfg.strategy}[/cyan]"
            f" / [cyan]{current_cfg.chunk_size}[/cyan] tok"
            f" / [cyan]{current_cfg.overlap_percent}%[/cyan] ovlp",
            divider,
            "[bold]Metric Deltas[/bold]",
        ]
        for d in comparison.deltas:
            sign = "+" if d.delta >= 0 else ""
            arrow = "[green]▲[/green]" if d.improved else "[red]▼[/red]"
            delta_color = "green" if d.improved else "red"
            lines.append(
                f"  {arrow} {d.metric:<16}"
                f" [dim]{d.baseline:.3f}[/dim] → [bold]{d.current:.3f}[/bold]"
                f"  [{delta_color}]{sign}{d.delta:.3f} ({sign}{d.delta_pct:.1f}%)[/{delta_color}]"
            )

        if comparison.overall_improved:
            verdict = "[bold green]IMPROVED[/bold green]"
        else:
            verdict = "[bold red]REGRESSION[/bold red]"
        lines.extend([divider, f"  Verdict: {verdict}"])
        cmp_widget.update("\n".join(lines))

        severity = "information" if comparison.overall_improved else "warning"
        self.notify(f"Comparison: {verdict}", severity=severity, timeout=4)

    # ------------------------------------------------------------------
    # Button dispatcher
    # ------------------------------------------------------------------

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""

        match btn_id:
            case "load-file-btn":
                await self.action_load_file()
            case "load-sample-btn":
                await self.action_load_sample()
            case "quick-clean-btn":
                self._quick_clean()
            case "clear-btn":
                self._clear_text_widget()
            case "run-batch-btn":
                await self._run_batch_test()
            case "run-judge-btn":
                await self._run_batch_test(use_judge=True)
            case "save-baseline-btn":
                if self._batch_results:
                    batch = calculate_batch_metrics(self._batch_results)
                    self._save_baseline(batch)
                else:
                    self.notify("Run a batch test first.", severity="warning")
            case "compare-baseline-btn":
                await self._compare_to_baseline()
            case "clear-batch-btn":
                self._batch_results = []
                self.query_one("#batch-results", Static).update("")
                self.query_one("#baseline-comparison", Static).update("")
            case "export-json-btn":
                self._copy_config("json")
            case "export-langchain-btn":
                self._copy_config("langchain")
            case "export-llamaindex-btn":
                self._copy_config("llamaindex")
            case "apply-custom-btn":
                self._apply_custom_chunker()
            case "apply-cleaner-btn":
                self._apply_custom_cleaner()
            case "apply-preset-btn":
                await self._apply_selected_preset()
            case "chat-send-btn":
                await self._submit_chat_message()

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    async def _load_file(self, file_path: str) -> None:
        try:
            content, info = read_file(file_path)
            self._file_info = info
            text_area = self.query_one("#text-input", TextArea)
            text_area.text = content
            self._current_text = content
            await self._rechunk()
            self.notify(
                f"Loaded: {info.name} "
                f"({format_file_size(info.size_bytes)}, {info.line_count} lines)",
                timeout=3,
            )
            self.query_one("#main-tabs", TabbedContent).active = "input-tab"
        except Exception as exc:
            self.notify(f"Error loading file: {exc}", severity="error")

    # ------------------------------------------------------------------
    # Export preview
    # ------------------------------------------------------------------

    def _update_export_preview(self) -> None:
        config = ChunkConfig(
            strategy=self._current_strategy.value,
            chunk_size=self._chunk_size,
            overlap_percent=self._overlap_percent,
            overlap_tokens=int(self._chunk_size * self._overlap_percent / 100),
        )
        try:
            self.query_one("#export-preview", Static).update(
                f"```json\n{config.to_json()}\n```"
            )
        except Exception:
            pass

    def _copy_config(self, format: str) -> None:
        config = ChunkConfig(
            strategy=self._current_strategy.value,
            chunk_size=self._chunk_size,
            overlap_percent=self._overlap_percent,
            overlap_tokens=int(self._chunk_size * self._overlap_percent / 100),
        )
        output = export_config(config, format)
        try:
            self.query_one("#export-preview", Static).update(f"```\n{output}\n```")
        except Exception:
            pass
        self.notify(f"{format.upper()} config generated — copy from Settings tab", timeout=3)

    # ------------------------------------------------------------------
    # Custom chunker / cleaner
    # ------------------------------------------------------------------

    def _exec_restricted(self, code: str) -> dict:
        """Compile and run user code under RestrictedPython's AST sandbox.

        Blocks dunder attribute access (__class__, __subclasses__, etc.),
        open(), exec(), eval(), __import__(), and all dangerous builtins
        at the AST transform level, not just via a runtime allowlist.
        """
        result = compile_restricted_exec(code)
        if result.errors:
            raise SyntaxError("\n".join(result.errors))
        globs = {**safe_globals, "__builtins__": safe_builtins}
        local_vars: dict = {}
        exec(result.code, globs, local_vars)  # noqa: S102
        return local_vars

    def _apply_custom_chunker(self) -> None:
        code = self.query_one("#custom-code", TextArea).text
        if not code.strip():
            self.notify("Enter Python code first.", severity="warning")
            return
        try:
            local_vars = self._exec_restricted(code)
            chunk_fn = next(
                (obj for name, obj in local_vars.items()
                 if callable(obj) and name.startswith("chunk")),
                None,
            )
            if not chunk_fn:
                self.notify("Define a function starting with 'chunk'.", severity="warning")
                return
            self.chunking_engine.set_custom_function(chunk_fn)
            self._current_strategy = StrategyType.CUSTOM
            self.chunking_engine.set_strategy(StrategyType.CUSTOM)
            asyncio.create_task(self._rechunk())
            self.notify("Custom chunker applied!", timeout=2)
        except SyntaxError as exc:
            self.notify(f"Syntax error: {exc}", severity="error")
        except Exception as exc:
            self.notify(f"Code error: {exc}", severity="error")

    def _apply_custom_cleaner(self) -> None:
        try:
            code = self.query_one("#custom-cleaner-code", TextArea).text.strip()
            if not code:
                self._custom_cleaner = None
                self.notify("Custom cleaner cleared.", timeout=2)
                return
            local_vars = self._exec_restricted(code)
            clean_fn = next(
                (obj for name, obj in local_vars.items()
                 if callable(obj) and name.startswith("clean")),
                None,
            )
            if not clean_fn:
                self.notify("Define a function starting with 'clean'.", severity="warning")
                return
            self._custom_cleaner = clean_fn
            self.notify("Custom cleaner applied! Use Quick Clean to run it.", timeout=3)
        except SyntaxError as exc:
            self.notify(f"Syntax error: {exc}", severity="error")
        except Exception as exc:
            self.notify(f"Code error: {exc}", severity="error")

    def _quick_clean(self) -> None:
        import re

        text_area = self.query_one("#text-input", TextArea)
        text = text_area.text
        if not text.strip():
            self.notify("No text to clean.", severity="warning")
            return
        original_len = len(text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"^\s*Page \d+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*-{3,}\s*$", "", text, flags=re.MULTILINE)
        if self._custom_cleaner:
            try:
                text = self._custom_cleaner(text)
            except Exception as exc:
                self.notify(f"Custom cleaner error: {exc}", severity="warning")
        text_area.load_text(text.strip())
        self._current_text = text.strip()
        self.notify(f"Cleaned! Removed {original_len - len(text.strip())} chars", timeout=2)
        asyncio.create_task(self._debounced_rechunk())

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    async def _apply_selected_preset(self) -> None:
        try:
            preset_select = self.query_one("#preset-select", Select)
            if preset_select.value == Select.BLANK:
                self.notify("Select a preset first.", severity="warning")
                return
            preset = load_preset(str(preset_select.value))
            if not preset:
                self.notify("Preset not found.", severity="error")
                return
            self._chunk_size = preset.chunk_size
            self._overlap_percent = preset.overlap_percent
            strategy_map = {
                "token": StrategyType.TOKEN,
                "sentence": StrategyType.SENTENCE,
                "paragraph": StrategyType.PARAGRAPH,
                "recursive": StrategyType.RECURSIVE,
                "fixed": StrategyType.FIXED_CHARS,
                "fixed_chars": StrategyType.FIXED_CHARS,
            }
            if preset.strategy in strategy_map:
                self._current_strategy = strategy_map[preset.strategy]
                self.chunking_engine.set_strategy(self._current_strategy)
            param_panel = self.query_one("#parameter-panel", ParameterPanel)
            param_panel.chunk_size = preset.chunk_size
            param_panel.overlap_percent = preset.overlap_percent
            self._update_export_preview()
            self.notify(f"Applied preset: {preset.name}")
            await self._debounced_rechunk()
        except Exception as exc:
            self.notify(f"Error applying preset: {exc}", severity="error")

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _update_recommendations(self) -> None:
        try:
            panel = self.query_one("#recommendations-panel", Static)
            text_len = len(self._current_text)
            token_estimate = text_len // 4
            if text_len == 0:
                panel.update("Load text to see recommendations.")
                return
            if token_estimate < 500:
                rec_size, rec_overlap, exp_chunks, tip = (
                    "100-150", "15-20%", "3-5", "Small document — use precise chunks"
                )
            elif token_estimate < 2000:
                rec_size, rec_overlap, exp_chunks, tip = (
                    "200-300", "10-15%", "8-15", "Medium document — balanced approach"
                )
            else:
                rec_size, rec_overlap, exp_chunks, tip = (
                    "300-500", "10-15%", "15+", "Large document — consider larger chunks"
                )
            panel.update(
                f"Text: {text_len:,} chars (~{token_estimate:,} tokens)\n\n"
                f"Suggested chunk size: {rec_size} tokens\n"
                f"Suggested overlap:    {rec_overlap}\n"
                f"Expected chunks:      {exp_chunks}\n\n"
                f"Tip: {tip}"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    async def _submit_chat_message(self) -> None:
        chat_input = self.query_one("#chat-input", Input)
        query = chat_input.value.strip()
        if not query:
            return
        chat_input.value = ""
        if not self.llm_provider:
            self.notify("LLM provider not available.", severity="error")
            return
        await self._generate(query)

    # ------------------------------------------------------------------
    # Clear helpers
    # ------------------------------------------------------------------

    def _clear_text_widget(self) -> None:
        text_area = self.query_one("#text-input", TextArea)
        text_area.load_text("")
        self._current_text = ""
        self._current_chunks = []
        self._update_chunk_display()
        self.notify("Cleared text.")

    # ------------------------------------------------------------------
    # Actions (key bindings)
    # ------------------------------------------------------------------

    async def action_load_sample(self) -> None:
        text_area = self.query_one("#text-input", TextArea)
        text_area.text = SAMPLE_TEXT
        self._current_text = SAMPLE_TEXT
        await self._rechunk()
        self.notify("Sample text loaded.", timeout=2)

    async def action_load_file(self) -> None:
        path = self.query_one("#file-path-input", Input).value.strip()
        if path:
            await self._load_file(path)
        else:
            self.notify("Enter a file path first.", severity="warning")

    async def action_rechunk(self) -> None:
        await self._rechunk()

    def action_toggle_dark(self) -> None:
        self.dark = not self.dark

    def action_export_config(self) -> None:
        self._copy_config("json")

    def action_show_help(self) -> None:
        self.push_screen(HelpOverlay())

    def _switch_strategy(self, strategy: StrategyType, name: str) -> None:
        self._current_strategy = strategy
        self.chunking_engine.set_strategy(strategy)
        try:
            self.query_one("#strategy-select", Select).value = strategy.value
        except Exception:
            pass
        asyncio.create_task(self._rechunk())
        self.notify(f"Strategy: {name}", timeout=1)

    def action_strategy_token(self) -> None:
        self._switch_strategy(StrategyType.TOKEN, "Token-based")

    def action_strategy_sentence(self) -> None:
        self._switch_strategy(StrategyType.SENTENCE, "Sentence")

    def action_strategy_paragraph(self) -> None:
        self._switch_strategy(StrategyType.PARAGRAPH, "Paragraph")

    def action_strategy_recursive(self) -> None:
        self._switch_strategy(StrategyType.RECURSIVE, "Recursive")

    def action_strategy_fixed(self) -> None:
        self._switch_strategy(StrategyType.FIXED_CHARS, "Fixed chars")


def main():
    RAGTUIApp().run()


if __name__ == "__main__":
    main()
