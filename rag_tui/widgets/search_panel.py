"""Search panel with query input and similarity results."""

from textual.widgets import Static, Input, Button
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.app import ComposeResult
from textual.message import Message
from rich.text import Text
from typing import List, Tuple


class SimilarityBar(Static):
    """Single-line rank + score bar: `#1  ████████████████░░░░  0.847`"""

    DEFAULT_CSS = """
    SimilarityBar {
        height: 1;
        width: 100%;
        padding: 0 1;
    }
    """

    def __init__(self, rank: int, score: float, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.score = score

    def on_mount(self) -> None:
        self._render_bar()

    def _render_bar(self) -> None:
        width = 36
        filled = int(width * min(1.0, max(0.0, self.score)))
        empty = width - filled

        if self.score >= 0.7:
            color = "green"
        elif self.score >= 0.4:
            color = "yellow"
        else:
            color = "red"

        bar = Text()
        bar.append(f"#{self.rank}", style="bold yellow")
        bar.append("  ")
        bar.append("█" * filled, style=color)
        bar.append("░" * empty, style="dim")
        bar.append(f"  {self.score:.3f}", style=f"bold {color}")
        self.update(bar)


class ResultCard(Static):
    """A card displaying a single search result."""

    DEFAULT_CSS = """
    ResultCard {
        height: auto;
        margin: 0 0 1 0;
        border: tall $primary-darken-1;
        background: $surface;
    }

    ResultCard .result-header {
        height: 1;
        background: $surface-darken-1;
    }

    ResultCard .result-content {
        padding: 1 1 0 1;
        color: $text;
        height: auto;
    }
    """

    def __init__(self, rank: int, chunk_text: str, score: float, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.chunk_text = chunk_text
        self.score = score

    def compose(self) -> ComposeResult:
        yield SimilarityBar(self.rank, self.score, classes="result-header")
        preview = self.chunk_text[:280]
        if len(self.chunk_text) > 280:
            preview += "…"
        yield Static(preview, classes="result-content")


class SearchPanel(Vertical):
    """Panel for query input and search results."""

    DEFAULT_CSS = """
    SearchPanel {
        height: 1fr;
        padding: 1;
    }
    """

    class QuerySubmitted(Message):
        """Message when query is submitted."""

        def __init__(self, query: str, action: str):
            self.query = query
            self.action = action
            super().__init__()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._results = []

    def compose(self) -> ComposeResult:
        with Vertical(classes="query-section"):
            with Horizontal(classes="query-row"):
                yield Input(
                    placeholder="Ask a question about the text...",
                    id="query-input",
                )
                yield Button("Search", variant="primary", id="search-btn")
                yield Button("Generate", variant="success", id="generate-btn")

        with VerticalScroll(classes="results-section", id="results-container"):
            yield Static(
                "Enter a query and press Search to find relevant chunks.",
                classes="empty-results",
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        query_input = self.query_one("#query-input", Input)
        query = query_input.value.strip()
        if query:
            action = "search" if event.button.id == "search-btn" else "generate"
            self.post_message(self.QuerySubmitted(query, action))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "query-input" and event.value.strip():
            self.post_message(self.QuerySubmitted(event.value.strip(), "search"))

    def update_results(self, results: List[Tuple[str, float, dict]]) -> None:
        self._results = results
        container = self.query_one("#results-container", VerticalScroll)
        container.remove_children()

        if not results:
            container.mount(Static("No matching chunks found.", classes="empty-results"))
            return

        for i, (chunk_text, score, _) in enumerate(results, 1):
            container.mount(ResultCard(rank=i, chunk_text=chunk_text, score=score))

    @property
    def current_query(self) -> str:
        return self.query_one("#query-input", Input).value
