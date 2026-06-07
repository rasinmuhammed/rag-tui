"""Optimize panel widget for RAG-TUI.

Provides the full A/B testing and chunk-size recommendation UI:
  - Query input
  - Strategy / size / overlap configuration
  - Async progress log
  - Ranked results table
  - "Apply Best" and per-row "Apply" buttons
"""

from __future__ import annotations

from typing import List

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Input, Label, RichLog, Static, TextArea

from rag_tui.core.optimizer import OptimizationResult


_STRATEGY_CHOICES = [
    ("Token", "token"),
    ("Sentence", "sentence"),
    ("Paragraph", "paragraph"),
    ("Recursive", "recursive"),
    ("Fixed-chars", "fixed_chars"),
]


class OptimizePanel(Vertical):
    """Full optimization / A-B-test UI panel."""

    DEFAULT_CSS = """
    OptimizePanel {
        height: 1fr;
        padding: 1;
    }
    OptimizePanel .opt-section {
        text-style: bold;
        color: $accent;
        padding: 0 1;
        margin-top: 1;
    }
    OptimizePanel .opt-row {
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
    }
    OptimizePanel .opt-label {
        width: 16;
        color: $text;
    }
    OptimizePanel .opt-input {
        width: 1fr;
    }
    OptimizePanel .opt-btn-row {
        height: auto;
        padding: 0 1;
        margin: 1 0;
    }
    OptimizePanel .opt-log {
        height: 9;
        border: solid $primary-darken-2;
        margin: 0 1;
    }
    OptimizePanel .opt-recommendation {
        color: $success;
        text-style: bold;
        padding: 1;
        margin: 0 1;
        border: solid $success;
        height: auto;
    }
    OptimizePanel .opt-results {
        height: 1fr;
        border: solid $primary;
        margin: 0 1;
        padding: 0;
    }
    OptimizePanel .opt-result-row {
        height: 3;
        padding: 0 1;
        border-bottom: solid $primary-darken-2;
    }
    OptimizePanel .opt-result-text {
        width: 1fr;
        content-align: left middle;
    }
    OptimizePanel .opt-apply-btn {
        width: 10;
        min-width: 10;
    }
    OptimizePanel .opt-empty {
        color: $text-muted;
        text-align: center;
        padding: 4;
    }
    """

    class ConfigApplied(Message):
        """User clicked Apply on an optimization result."""
        def __init__(self, chunk_size: int, overlap_percent: int, strategy: str) -> None:
            self.chunk_size = chunk_size
            self.overlap_percent = overlap_percent
            self.strategy = strategy
            super().__init__()

    class OptimizeRequested(Message):
        """User clicked Run Optimization."""
        def __init__(
            self,
            queries: List[str],
            strategies: List[str],
            sizes: List[int],
            overlaps: List[int],
        ) -> None:
            self.queries = queries
            self.strategies = strategies
            self.sizes = sizes
            self.overlaps = overlaps
            super().__init__()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._results: List[OptimizationResult] = []
        self._running = False
        self._active_strategies: set[str] = {"token"}

    def compose(self) -> ComposeResult:
        yield Static("Test Queries (one per line):", classes="opt-section")
        yield TextArea(
            "What is the main topic?\nHow does this work?\nWhat are the key findings?",
            id="opt-queries",
        )

        yield Static("Strategies to test:", classes="opt-section")
        with Horizontal(classes="opt-row", id="opt-strat-row"):
            for label, value in _STRATEGY_CHOICES:
                yield Button(label, id=f"opt-strat-{value}")

        with Horizontal(classes="opt-row"):
            yield Label("Chunk sizes:", classes="opt-label")
            yield Input("64,128,200,256,320,400,512", id="opt-sizes", classes="opt-input")

        with Horizontal(classes="opt-row"):
            yield Label("Overlaps (%):", classes="opt-label")
            yield Input("5,10,15,20", id="opt-overlaps", classes="opt-input")

        with Horizontal(classes="opt-btn-row"):
            yield Button("Run Optimization", id="opt-run-btn", variant="success")
            yield Button("Clear", id="opt-clear-btn", variant="default")

        yield Static("Progress:", classes="opt-section")
        yield RichLog(id="opt-log", classes="opt-log", markup=True, wrap=True)

        yield Static("", id="opt-rec", classes="opt-recommendation")

        yield Static("Results (ranked by MRR + nDCG):", classes="opt-section")
        with VerticalScroll(classes="opt-results", id="opt-results-scroll"):
            yield Static("Run optimization to see results.", classes="opt-empty", id="opt-empty")

    # ------------------------------------------------------------------
    # Mount: apply initial button styles
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        self._refresh_strategy_buttons()

    def _refresh_strategy_buttons(self) -> None:
        for label, value in _STRATEGY_CHOICES:
            try:
                btn = self.query_one(f"#opt-strat-{value}", Button)
                active = value in self._active_strategies
                btn.variant = "primary" if active else "default"
                prefix = "[x] " if active else "    "
                btn.label = f"{prefix}{label}"
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Button handling
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""

        if btn_id.startswith("opt-strat-"):
            strategy = btn_id.removeprefix("opt-strat-")
            if strategy in self._active_strategies:
                self._active_strategies.discard(strategy)
            else:
                self._active_strategies.add(strategy)
            self._refresh_strategy_buttons()

        elif btn_id == "opt-run-btn":
            if not self._running:
                self._fire_optimize()

        elif btn_id == "opt-clear-btn":
            self._clear()

        elif btn_id.startswith("opt-apply-"):
            idx = int(btn_id.removeprefix("opt-apply-"))
            if idx < len(self._results):
                r = self._results[idx]
                self.post_message(
                    self.ConfigApplied(r.chunk_size, r.overlap_percent, r.strategy)
                )

    def _fire_optimize(self) -> None:
        queries_text = self.query_one("#opt-queries", TextArea).text
        queries = [q.strip() for q in queries_text.splitlines() if q.strip()]
        if not queries:
            self.app.notify("Enter at least one query.", severity="warning")
            return

        strategies = list(self._active_strategies)
        if not strategies:
            self.app.notify("Select at least one strategy.", severity="warning")
            return

        sizes = self._parse_int_list("#opt-sizes")
        overlaps = self._parse_int_list("#opt-overlaps")

        self.post_message(
            self.OptimizeRequested(
                queries=queries,
                strategies=strategies,
                sizes=sizes,
                overlaps=overlaps,
            )
        )

    def _parse_int_list(self, widget_id: str) -> List[int]:
        raw = self.query_one(widget_id, Input).value
        result = []
        for p in raw.split(","):
            p = p.strip()
            if p:
                try:
                    result.append(int(p))
                except ValueError:
                    pass
        return result or [200]

    def _clear(self) -> None:
        self._results = []
        try:
            self.query_one("#opt-log", RichLog).clear()
            self.query_one("#opt-rec", Static).update("")
            scroll = self.query_one("#opt-results-scroll", VerticalScroll)
            scroll.remove_children()
            scroll.mount(Static("Run optimization to see results.", classes="opt-empty", id="opt-empty"))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API (called from app.py)
    # ------------------------------------------------------------------

    def set_running(self, running: bool) -> None:
        self._running = running
        try:
            btn = self.query_one("#opt-run-btn", Button)
            if running:
                btn.label = "Running..."
                btn.variant = "warning"
            else:
                btn.label = "Run Optimization"
                btn.variant = "success"
        except Exception:
            pass

    def log_progress(self, message: str) -> None:
        try:
            self.query_one("#opt-log", RichLog).write(message)
        except Exception:
            pass

    def show_results(self, results: List[OptimizationResult]) -> None:
        """Render ranked results table (called synchronously from app)."""
        self._results = results
        scroll = self.query_one("#opt-results-scroll", VerticalScroll)
        scroll.remove_children()

        if not results:
            scroll.mount(Static("No results.", classes="opt-empty"))
            return

        best = results[0]
        rec = self.query_one("#opt-rec", Static)
        rec.update(
            f"Recommended: {best.chunk_size} tokens | "
            f"{best.overlap_percent}% overlap | "
            f"{best.strategy} strategy  "
            f"[MRR: {best.metrics.mrr:.3f}  "
            f"nDCG: {best.metrics.ndcg_at_k:.3f}  "
            f"Hit Rate: {best.metrics.hit_rate:.1%}  "
            f"Score: {best.score:.3f}]"
        )

        for i, r in enumerate(results[:25]):
            medals = {1: "BEST", 2: " #2 ", 3: " #3 "}
            badge = medals.get(r.rank, f"#{r.rank:<3}")
            m = r.metrics
            text = (
                f"[bold]{badge}[/bold]  "
                f"size=[cyan]{r.chunk_size:>4}[/cyan]  "
                f"ovlp=[cyan]{r.overlap_percent:>2}%[/cyan]  "
                f"strat=[cyan]{r.strategy:<10}[/cyan]  "
                f"chunks={r.num_chunks:>3}  "
                f"MRR=[green]{m.mrr:.3f}[/green]  "
                f"nDCG=[green]{m.ndcg_at_k:.3f}[/green]  "
                f"Hit=[green]{m.hit_rate:.0%}[/green]  "
                f"Score=[yellow]{r.score:.3f}[/yellow]"
            )
            row = Horizontal(classes="opt-result-row")
            scroll.mount(row)
            row.mount(Static(text, classes="opt-result-text", markup=True))
            row.mount(Button("Apply", id=f"opt-apply-{i}", classes="opt-apply-btn", variant="primary"))
