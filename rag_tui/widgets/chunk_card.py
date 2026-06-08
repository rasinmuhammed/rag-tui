"""Chunk card widget for displaying individual chunks with colors."""

from textual.widgets import Static, Button
from textual.containers import Vertical, Container, VerticalScroll, Horizontal
from textual.app import ComposeResult


CHUNK_COLORS = [
    ("#3b82f6", "#1e3a5f"),  # Blue
    ("#22c55e", "#166534"),  # Green
    ("#a855f7", "#581c87"),  # Purple
    ("#f97316", "#9a3412"),  # Orange
    ("#06b6d4", "#155e75"),  # Cyan
    ("#ec4899", "#831843"),  # Pink
]


class ChunkCard(Container):
    """A styled card displaying a single chunk with colored border."""

    DEFAULT_CSS = """
    ChunkCard {
        height: auto;
        max-height: 13;
        margin: 0 0 1 0;
        padding: 0;
        background: $surface;
    }

    ChunkCard .chunk-header-row {
        height: 1;
        width: 100%;
    }

    ChunkCard .chunk-header {
        width: 1fr;
        height: 1;
        padding: 0 1;
        text-style: bold;
        background: $surface-darken-1;
    }

    ChunkCard .copy-btn {
        width: 6;
        min-width: 6;
        height: 1;
        background: $surface-darken-1;
        border: none;
        color: $text-muted;
    }

    ChunkCard .chunk-scroll {
        height: auto;
        max-height: 8;
        overflow-y: auto;
        padding: 0 1;
    }

    ChunkCard .chunk-content {
        height: auto;
    }

    ChunkCard .chunk-meta {
        height: 1;
        padding: 0 1;
        color: $text-muted;
        background: $surface-darken-1;
    }
    """

    def __init__(
        self,
        chunk_text: str,
        chunk_index: int,
        start_pos: int,
        end_pos: int,
        token_count: int = 0,
        overlap_text: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.chunk_text = chunk_text
        self.chunk_index = chunk_index
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.token_count = token_count
        self.overlap_text = overlap_text

        color_idx = chunk_index % len(CHUNK_COLORS)
        self.border_color, self.bg_color = CHUNK_COLORS[color_idx]

    def _quality_tag(self) -> str:
        text = self.chunk_text.strip()
        parts = []
        if text and text[-1] in ".!?":
            parts.append("[green]◉[/green]")
        elif text and text[-1] in ",:;":
            parts.append("[yellow]◎[/yellow]")
        else:
            parts.append("[red]◌[/red]")
        if self.token_count < 50:
            parts.append("[yellow]SHORT[/yellow]")
        elif self.token_count > 600:
            parts.append("[yellow]LONG[/yellow]")
        if text and text[0].islower():
            parts.append("[dim]↩CUT[/dim]")
        return " ".join(parts)

    def _format_content_with_overlap(self) -> str:
        if not self.overlap_text or self.overlap_text not in self.chunk_text:
            return self.chunk_text
        overlap_start = self.chunk_text.rfind(self.overlap_text)
        if overlap_start == -1:
            return self.chunk_text
        before = self.chunk_text[:overlap_start]
        overlap = self.chunk_text[overlap_start:]
        return f"{before}[bold yellow on #4a3f00]{overlap}[/]"

    def on_mount(self) -> None:
        self.styles.border = ("tall", self.border_color)
        self.styles.background = self.bg_color

    async def on_button_pressed(self, event) -> None:
        if hasattr(event.button, "id") and event.button.id and event.button.id.startswith("copy-"):
            event.stop()
            try:
                import pyperclip
                pyperclip.copy(self.chunk_text)
                self.app.notify("Copied to clipboard", timeout=2)
            except ImportError:
                try:
                    import subprocess
                    subprocess.run(["pbcopy"], input=self.chunk_text.encode(), check=True)
                    self.app.notify("Copied to clipboard", timeout=2)
                except Exception:
                    self.app.notify("Install pyperclip for clipboard support", severity="warning")
            except Exception as exc:
                self.app.notify(f"Copy failed: {exc}", severity="warning")

    def compose(self) -> ComposeResult:
        quality = self._quality_tag()
        overlap_tag = "  [dim]⇌[/dim]" if self.overlap_text else ""
        header = (
            f"[bold]#{self.chunk_index + 1}[/bold]"
            f"  [dim]│[/dim]  {len(self.chunk_text)} ch"
            f"  [dim]·[/dim]  ~{self.token_count} tok"
            f"  [dim]│[/dim]  {quality}{overlap_tag}"
        )
        with Horizontal(classes="chunk-header-row"):
            yield Static(header, classes="chunk-header", markup=True)
            yield Button("Copy", id=f"copy-{self.chunk_index}", classes="copy-btn")

        with VerticalScroll(classes="chunk-scroll"):
            content = self._format_content_with_overlap()
            yield Static(content, classes="chunk-content", markup=True)

        meta_parts = [f"pos {self.start_pos}–{self.end_pos}"]
        if self.overlap_text:
            meta_parts.append(f"overlap {len(self.overlap_text)} ch")
        yield Static("  ".join(meta_parts), classes="chunk-meta")


class ChunkList(Vertical):
    """Container for displaying multiple chunk cards."""

    DEFAULT_CSS = """
    ChunkList {
        height: 1fr;
        padding: 0 1;
        overflow-y: auto;
    }

    ChunkList .empty-state {
        color: $text-muted;
        text-align: center;
        padding: 4;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._chunks = []

    def update_chunks(self, chunks: list) -> None:
        self._chunks = chunks
        self._rebuild_cards()

    def _rebuild_cards(self) -> None:
        self.remove_children()

        if not self._chunks:
            self.mount(Static("No chunks yet. Load text to begin.", classes="empty-state"))
            return

        for i, (text, start, end) in enumerate(self._chunks):
            token_estimate = len(text) // 4
            overlap_text = ""
            if i < len(self._chunks) - 1:
                _, next_start, next_end = self._chunks[i + 1]
                if next_start < end:
                    overlap_len = end - next_start
                    overlap_text = text[-overlap_len:] if overlap_len <= len(text) else ""
            card = ChunkCard(
                chunk_text=text,
                chunk_index=i,
                start_pos=start,
                end_pos=end,
                token_count=token_estimate,
                overlap_text=overlap_text,
                id=f"chunk-{i}",
            )
            self.mount(card)
