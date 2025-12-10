"""Help overlay widget for RAG-TUI."""

from textual.widgets import Button, Markdown
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.app import ComposeResult
from textual.binding import Binding


HELP_MARKDOWN = """
# 🚀 RAG-TUI v0.0.2 Beta

## ⌨️ Keyboard Shortcuts
* **Q**        : Quit application
* **L**        : Load sample text
* **R**        : Rechunk current text
* **D**        : Toggle dark/light theme
* **E**        : Export current configuration
* **F1 / ?**   : Show this help
* **Tab**      : Switch tabs
* **Esc**      : Close modal / Cancel

## 📝 Tabs Guide
1. **Input**: Paste text or load files (.txt, .md, .py, .pdf).
2. **Chunks**: View chunking results visually. Adjust **Size** & **Overlap** sliders.
3. **Search**: Test semantic search (requires embedding provider).
4. **Batch**: Run multiple queries to test hit rate.
5. **Settings**: Export to LangChain/LlamaIndex, Custom Python strategies.
6. **Chat**: Chat with your document (requires LLM).


## 🧩 Chunking Strategies
* **Token** (Chonkie): Best for general text.
* **Sentence**: Keeps sentences intact.
* **Paragraph**: Good for structured docs.
* **Recursive**: Adaptive (good for code/mixed).
* **Fixed**: Simple char count (fast).
* **Custom**: Define your own Python function.

## 🚦 Quality Indicators (Chunks Tab)
* 🟢 Complete sentence (ends with . ! ?)
* 🟡 Mid-phrase (ends with , : ;)
* 🔴 Cut off (ends with other char)
* ⚠️SHORT Token count < 50
* ⚠️LONG Token count > 600
* ↪️CUT Starts mid-sentence (lowercase)

## ✏️ Custom Chunking Guide

Your function MUST have this signature:
```python
def chunk_name(text, chunk_size, overlap):
    # text: str - The full document text
    # chunk_size: int - Target chunk size
    # overlap: int - Overlap between chunks
    # Returns: List of tuples (chunk_text, start_pos, end_pos)
    return [(chunk, start, end), ...]
```

**Example 1: Split by sentences**
```python
def chunk_sentences(text, chunk_size, overlap):
    import re
    sentences = re.split(r'(?<=[.!?])\\s+', text)
    results = []
    pos = 0
    for s in sentences:
        start = text.find(s, pos)
        results.append((s, start, start + len(s)))
        pos = start + len(s)
    return results
```

**Example 2: Split by headers**
```python
def chunk_headers(text, chunk_size, overlap):
    import re
    parts = re.split(r'(^#{1,3} .+$)', text, flags=re.M)
    # ... process parts
    return results
```

## 📊 Batch Testing Guide

**Purpose**: Test if your chunking config retrieves the right content.

**How to use**:
1. Go to **Batch** tab
2. Enter test queries (one per line):
   ```
   What is chunking?
   How does embedding work?
   What chunk size is best?
   ```
3. Click **Run Batch Test**

**Understanding Results**:
* **Hit Rate (>0.5)**: % of queries with good matches
  - >80% = 🟢 Excellent
  - 60-80% = 🟡 Needs tuning
  - <60% = 🔴 Change strategy/size
* **Avg Top Score**: Best match quality (higher = better)
* **Avg Retrieval Score**: Overall retrieval quality

**Best Practice**: Create 10-20 realistic queries. Compare hit rates across different chunk sizes (150 vs 300 vs 500).

## 🔌 LLM Providers
Auto-detected on startup:
1. **Ollama** (Local): Run `ollama serve` (Free).
2. **OpenAI**: Set `OPENAI_API_KEY`.
3. **Groq**: Set `GROQ_API_KEY` (Fast).
4. **Google**: Set `GOOGLE_API_KEY`.

## 💡 Tips
* **Smaller chunks** = More precision, less context.
* **Larger chunks** = More context, "fuzzier" retrieval.
* Use **Batch** tab to find the sweet spot.
"""


class HelpOverlay(ModalScreen):
    """Full-screen help modal."""
    
    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close"),
        Binding("question_mark", "close", "Close"),
    ]
    
    DEFAULT_CSS = """
    HelpOverlay {
        align: center middle;
        background: $background 50%;
    }
    
    #help-container {
        width: 85;
        height: 90%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        overflow-y: auto;
    }
    
    #help-content {
        height: 1fr;
        color: $text;
        scrollbar-gutter: stable;
    }
    
    #close-btn {
        width: 100%;
        margin-top: 1;
    }
    
    Markdown {
        background: $surface;
        padding: 1;
    }
    
    Markdown H1 {
        text-align: center;
        background: $primary;
        color: $text;
        padding: 1;
    }
    
    Markdown H2 {
        border-bottom: solid $secondary;
        color: $accent;
        margin-top: 1;
        padding-bottom: 0;
    }
    """
    
    def compose(self) -> ComposeResult:
        with Vertical(id="help-container"):
            with Vertical(id="help-content"):
                yield Markdown(HELP_MARKDOWN)
            yield Button("Close (Esc)", variant="primary", id="close-btn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-btn":
            self.dismiss()
            
    def action_close(self) -> None:
        self.dismiss()
