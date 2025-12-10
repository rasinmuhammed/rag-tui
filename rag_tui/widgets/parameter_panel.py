"""Parameter controls with visual sliders for RAG-TUI."""

from textual.widgets import Static, Label, Button, Input, ProgressBar
from textual.containers import Horizontal, Container
from textual.app import ComposeResult
from textual.message import Message


class ParameterControl(Container):
    """A labeled parameter input with +/- buttons and visual bar."""
    
    DEFAULT_CSS = """
    ParameterControl {
        height: auto;
        width: 100%;
        padding: 0 1;
        margin: 0;
    }
    
    ParameterControl .param-row {
        height: auto;
        align: center middle;
    }
    
    ParameterControl .param-label {
        width: 12;
        color: $text;
    }
    
    ParameterControl .param-btn {
        min-width: 3;
        margin: 0;
    }
    
    ParameterControl .param-input {
        width: 10;
        text-align: center;
        margin: 0 1;
    }
    
    ParameterControl .param-bar {
        width: 1fr;
        margin: 0 1;
    }
    
    ParameterControl .param-suffix {
        width: 8;
        color: $text-muted;
    }
    """
    
    class Changed(Message):
        """Value changed."""
        def __init__(self, param_control: "ParameterControl", value: int):
            self.param_control = param_control
            self.value = value
            super().__init__()
    
    def __init__(
        self, 
        label: str,
        min_value: int = 0,
        max_value: int = 100,
        value: int = 50,
        step: int = 10,
        suffix: str = "",
        param_id: str = "",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.label_text = label
        self.min_value = min_value
        self.max_value = max_value
        self._value = value
        self.step = step
        self.suffix = suffix
        self.param_id = param_id
    
    def compose(self) -> ComposeResult:
        with Horizontal(classes="param-row"):
            yield Label(self.label_text, classes="param-label")
            yield Button("◀", id=f"{self.param_id}-dec", classes="param-btn", variant="default")
            yield Input(str(self._value), id=f"{self.param_id}-input", classes="param-input", type="integer")
            yield Button("▶", id=f"{self.param_id}-inc", classes="param-btn", variant="default")
            yield ProgressBar(total=self.max_value, show_eta=False, show_percentage=False, classes="param-bar", id=f"{self.param_id}-bar")
            yield Static(self.suffix, classes="param-suffix")
    
    def on_mount(self) -> None:
        """Set initial progress bar value."""
        self._update_bar()
    
    def _update_bar(self) -> None:
        """Update the progress bar."""
        try:
            bar = self.query_one(f"#{self.param_id}-bar", ProgressBar)
            bar.update(progress=self._value)
        except Exception:
            pass
    
    def _update_input(self) -> None:
        """Update the input field."""
        try:
            inp = self.query_one(f"#{self.param_id}-input", Input)
            inp.value = str(self._value)
        except Exception:
            pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle +/- button presses."""
        if event.button.id == f"{self.param_id}-dec":
            self._value = max(self.min_value, self._value - self.step)
            self._update_input()
            self._update_bar()
            self.post_message(self.Changed(self, self._value))
        elif event.button.id == f"{self.param_id}-inc":
            self._value = min(self.max_value, self._value + self.step)
            self._update_input()
            self._update_bar()
            self.post_message(self.Changed(self, self._value))
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle direct input."""
        if event.input.id == f"{self.param_id}-input":
            try:
                new_val = int(event.value)
                self._value = max(self.min_value, min(self.max_value, new_val))
                self._update_input()
                self._update_bar()
                self.post_message(self.Changed(self, self._value))
            except ValueError:
                self._update_input()
    
    @property
    def value(self) -> int:
        return self._value


class ParameterPanel(Container):
    """Panel with chunking parameter controls."""
    
    DEFAULT_CSS = """
    ParameterPanel {
        height: auto;
        width: 100%;
        background: $surface-darken-1;
        border: solid $primary;
        padding: 0 1;
        margin: 0;
    }
    
    ParameterPanel .panel-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    
    ParameterPanel .status-row {
        height: auto;
        margin-top: 1;
        padding-top: 1;
        border-top: solid $primary-darken-2;
    }
    
    ParameterPanel .chunk-status {
        color: $success;
        text-style: bold;
        width: 1fr;
    }
    """
    
    class ParametersChanged(Message):
        """Parameters have been applied."""
        def __init__(self, chunk_size: int, overlap_percent: int):
            self.chunk_size = chunk_size
            self.overlap_percent = overlap_percent
            super().__init__()
    
    def __init__(
        self,
        chunk_size: int = 200,
        overlap_percent: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._chunk_size = chunk_size
        self._overlap_percent = overlap_percent
        self._chunk_count = 0
    
    def compose(self) -> ComposeResult:
        yield Static("📐 Chunking Parameters", classes="panel-title")
        
        yield ParameterControl(
            label="Chunk Size:",
            min_value=50,
            max_value=1000,
            value=self._chunk_size,
            step=50,
            suffix="tokens",
            param_id="chunk-size",
            id="chunk-size-control"
        )
        
        yield ParameterControl(
            label="Overlap:",
            min_value=0,
            max_value=50,
            value=self._overlap_percent,
            step=5,
            suffix="%",
            param_id="overlap",
            id="overlap-control"
        )
        
        with Horizontal(classes="status-row"):
            yield Static(f"⚡ {self._chunk_count} chunks", classes="chunk-status", id="chunk-status")
    
    def on_parameter_control_changed(self, event: ParameterControl.Changed) -> None:
        """Handle control value changes."""
        if "chunk-size" in (event.param_control.param_id or ""):
            self._chunk_size = event.value
        elif "overlap" in (event.param_control.param_id or ""):
            self._overlap_percent = event.value
        
        self.post_message(self.ParametersChanged(self._chunk_size, self._overlap_percent))
    
    def update_chunk_count(self, count: int) -> None:
        """Update the displayed chunk count."""
        self._chunk_count = count
        try:
            status = self.query_one("#chunk-status", Static)
            status.update(f"⚡ {count} chunks")
        except Exception:
            pass
    
    @property
    def chunk_size(self) -> int:
        return self._chunk_size
    
    @chunk_size.setter
    def chunk_size(self, value: int) -> None:
        """Set chunk size and update UI control."""
        self._chunk_size = value
        try:
            control = self.query_one("#chunk-size-control", ParameterControl)
            control._value = value
            control._update_input()
            control._update_bar()
        except Exception:
            pass
    
    @property
    def overlap_percent(self) -> int:
        return self._overlap_percent
    
    @overlap_percent.setter
    def overlap_percent(self, value: int) -> None:
        """Set overlap percent and update UI control."""
        self._overlap_percent = value
        try:
            control = self.query_one("#overlap-control", ParameterControl)
            control._value = value
            control._update_input()
            control._update_bar()
        except Exception:
            pass
