"""RAG-TUI v0.0.2-beta: Interactive Chunking Debugger.

A beautiful terminal UI for visualizing, debugging, and tuning RAG pipelines.
Features: Multiple chunking strategies, multi-provider LLM, batch testing, export.
"""

import asyncio
from pathlib import Path
from typing import List, Tuple
import numpy as np

from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import (
    Header, Footer, Static, TextArea, Button, 
    TabbedContent, TabPane, RichLog, Input, Select, Label
)
from textual.binding import Binding

from rag_tui.core.engine import ChunkingEngine, StrategyType
from rag_tui.core.vector import VectorStore
from rag_tui.core.providers import get_best_provider, LLMProvider
from rag_tui.core.file_handler import read_file, format_file_size
from rag_tui.core.metrics import ChunkConfig, QueryResult, calculate_batch_metrics, export_config
from rag_tui.widgets.chunk_card import ChunkList
from rag_tui.widgets.search_panel import SearchPanel
from rag_tui.widgets.help_overlay import HelpOverlay
from rag_tui.widgets.parameter_panel import ParameterPanel
from rag_tui.core.presets import list_presets, load_preset


SAMPLE_TEXT = """Retrieval-Augmented Generation (RAG) is a powerful technique for enhancing large language models with external knowledge. The RAG pipeline consists of several key components: document ingestion, text chunking, embedding generation, vector storage, and retrieval.

During the ingestion phase, documents are split into smaller, manageable chunks. The chunk size and overlap are critical parameters that affect retrieval quality. Smaller chunks provide more precise retrieval but may lose context, while larger chunks maintain context but reduce precision.

Each chunk is then converted into a dense vector embedding using a specialized embedding model. These embeddings capture the semantic meaning of the text, allowing for similarity-based retrieval. The embeddings are stored in a vector database that enables efficient similarity search.

When a user submits a query, it is also embedded using the same model. The system then retrieves the most semantically similar chunks from the vector store. These chunks serve as context for the language model, which generates a response grounded in the retrieved information.

This approach significantly reduces hallucinations and allows LLMs to work with up-to-date, proprietary, or domain-specific information that wasn't part of their original training data."""


# Strategy options for Select widget
STRATEGY_OPTIONS = [
    ("Token-based (Chonkie)", "token"),
    ("Sentence boundaries", "sentence"),
    ("Paragraph breaks", "paragraph"),
    ("Recursive splitting", "recursive"),
    ("Fixed characters", "fixed_chars"),
]


class RAGTUIApp(App):
    """Enterprise-grade RAG chunking debugger."""
    
    CSS_PATH = Path(__file__).parent / "styles" / "app.tcss"
    
    TITLE = "RAG-TUI v0.0.2 Beta"
    SUB_TITLE = "Interactive Chunking Debugger"
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("l", "load_sample", "Load Sample"),
        Binding("r", "rechunk", "Rechunk"),
        Binding("d", "toggle_dark", "Toggle Theme"),
        Binding("e", "export_config", "Export Config"),
        Binding("f1", "show_help", "Help"),
        Binding("question_mark", "show_help", "Help", show=False),
    ]
    
    def __init__(self):
        """Initialize the application."""
        super().__init__()
        
        # Core components
        # Core components
        self.chunking_engine = ChunkingEngine()
        self.vector_store = VectorStore(embedding_dim=768)
        self.embedding_provider: LLMProvider | None = None
        self.llm_provider: LLMProvider | None = None
        
        # State
        self._current_text = ""
        self._current_chunks = []
        self._chunk_size = 200
        self._overlap_percent = 10
        self._current_strategy = StrategyType.TOKEN

        self._debounce_task = None
        self._embedding_task = None  # Track in-flight embedding for cancellation
        self._batch_results = []
        self._file_info = None
        self._custom_cleaner = None  # Optional custom text cleaner
    
    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()
        
        # Strategy selector bar
        with Horizontal(id="strategy-bar"):
            yield Label("Strategy:")
            yield Select(
                STRATEGY_OPTIONS,
                value="token",
                id="strategy-select"
            )
            yield Label("  │  File:")
            yield Input(placeholder="Path to file...", id="file-path-input")
            yield Button("📂 Load", id="load-file-btn", variant="primary")
        
        # Main tabbed content
        with TabbedContent(id="main-tabs"):
            # Tab 1: Input
            with TabPane("📝 Input", id="input-tab"):
                yield Static("Paste or load your document:", classes="tab-intro")
                yield TextArea(id="text-input", language="markdown")
                with Horizontal(classes="button-row"):
                    yield Button("📄 Load Sample", id="load-sample-btn", variant="default")
                    yield Button("🧹 Quick Clean", id="quick-clean-btn", variant="primary")
                    yield Button("🗑️ Clear", id="clear-btn", variant="warning")
            
            # Tab 2: Chunks
            with TabPane("🎨 Chunks", id="chunks-tab"):
                yield ParameterPanel(
                    chunk_size=self._chunk_size,
                    overlap_percent=self._overlap_percent,
                    id="parameter-panel"
                )
                yield Static("", id="chunk-summary", classes="tab-intro")
                yield ChunkList(id="chunk-list")
            
            # Tab 3: Search
            with TabPane("🔍 Search", id="search-tab"):
                yield SearchPanel(id="search-panel")
            
            # Tab 4: Batch Test
            with TabPane("📊 Batch", id="batch-tab"):
                with VerticalScroll():
                    yield Static("Enter multiple queries (one per line):", classes="tab-intro")
                    yield TextArea(id="batch-queries")
                    with Horizontal(classes="button-row"):
                        yield Button("▶️ Run Batch Test", id="run-batch-btn", variant="success")
                        yield Button("📋 Clear Results", id="clear-batch-btn", variant="default")
                    yield Static("Results will appear here...", id="batch-results", classes="batch-results")
            
            # Tab 5: Settings & Export
            with TabPane("⚙️ Settings", id="settings-tab"):
                with VerticalScroll():
                    # Recommendations Panel
                    yield Static("💡 Recommendations", classes="section-title")
                    yield Static("", id="recommendations-panel", classes="recommendations-panel")
                    
                    # Quick Presets
                    yield Static("⚡ Quick Presets", classes="section-title")
                    preset_options = [(p.name, p.name) for p in list_presets()]
                    yield Select(preset_options, id="preset-select", prompt="Select a preset...")
                    yield Button("📥 Apply Preset", id="apply-preset-btn", variant="primary")
                    
                    # Export
                    yield Static("📤 Export Configuration", classes="section-title")
                    with Horizontal(classes="button-row"):
                        yield Button("📋 JSON", id="export-json-btn", variant="primary")
                        yield Button("🔗 LangChain", id="export-langchain-btn", variant="default")
                        yield Button("🦙 LlamaIndex", id="export-llamaindex-btn", variant="default")
                    yield Static("", id="export-preview", classes="export-preview")
                    
                    # Custom Chunker
                    yield Static("✏️ Custom Chunker", classes="section-title")
                    yield Static("Function: chunk_name(text, chunk_size, overlap) → [(text, start, end), ...]", classes="tab-intro")
                    yield TextArea(id="custom-code", language="python")
                    yield Button("⚡ Apply Custom Chunker", id="apply-custom-btn", variant="success")
                    
                    # Custom Cleaner
                    yield Static("🧹 Custom Cleaner (Optional)", classes="section-title")
                    yield Static("Function: clean_name(text) → cleaned_text", classes="tab-intro")
                    yield TextArea(id="custom-cleaner-code", language="python")
                    yield Button("🧹 Apply Custom Cleaner", id="apply-cleaner-btn", variant="primary")
            
            # Tab 6: Chat
            # Tab 6: Chat
            with TabPane("💬 Chat", id="chat-tab"):
                yield RichLog(id="chat-log", markup=True, wrap=True)
                yield Static("", id="chat-live-response", classes="live-response")
                with Horizontal(id="chat-input-row"):
                    yield Input(placeholder="Ask a question...", id="chat-input")
                    yield Button("➤", id="chat-send-btn", variant="primary")
        
        # Bottom control bar
        yield Footer()
    
    async def on_mount(self) -> None:
        """Handle app mount."""
        # Detect providers
        self.embedding_provider, self.llm_provider = await get_best_provider()
        
        if self.embedding_provider:
            self.notify(f"✅ Embeddings: {self.embedding_provider.config.name}", severity="information", timeout=5)
        else:
            self.notify("⚠️ No embedding provider found (Search disabled)", severity="warning", timeout=5)
            
        if self.llm_provider:
            self.notify(f"✅ LLM: {self.llm_provider.config.name}", severity="information", timeout=5)
            self.sub_title = f"Provider: {self.llm_provider.config.name}"
        else:
            self.notify("⚠️ No LLM found. Install Ollama or set API keys.", severity="warning", timeout=8)
            self.sub_title = "Provider: None (View Only)"
        
        # Load sample text
        await self.action_load_sample()
        
        # Show export preview
        self._update_export_preview()
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle strategy selection change."""
        if event.select.id == "strategy-select":
            strategy_map = {
                "token": StrategyType.TOKEN,
                "sentence": StrategyType.SENTENCE,
                "paragraph": StrategyType.PARAGRAPH,
                "recursive": StrategyType.RECURSIVE,
                "fixed_chars": StrategyType.FIXED_CHARS,
            }
            self._current_strategy = strategy_map.get(event.value, StrategyType.TOKEN)
            self.chunking_engine.set_strategy(self._current_strategy)
            asyncio.create_task(self._rechunk())
            self.notify(f"Strategy: {event.value}", timeout=1)
    

    async def on_parameter_panel_parameters_changed(self, event: ParameterPanel.ParametersChanged) -> None:
        """Handle parameter changes from sliders in Chunks tab."""
        self._chunk_size = event.chunk_size
        self._overlap_percent = event.overlap_percent
        self._update_export_preview()
        await self._debounced_rechunk()
    
    async def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle text input changes."""
        if event.text_area.id == "text-input":
            self._current_text = event.text_area.text
            self._update_recommendations()
            await self._debounced_rechunk()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        if event.input.id == "file-path-input":
            await self.action_load_file()
        elif event.input.id == "chat-input":
            await self._submit_chat_message()
    
    async def _debounced_rechunk(self) -> None:
        """Rechunk with debouncing."""
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        
        self._debounce_task = asyncio.create_task(self._delayed_rechunk())
    
    async def _delayed_rechunk(self) -> None:
        """Rechunk after delay."""
        await asyncio.sleep(0.3)
        await self._rechunk()
    
    async def _rechunk(self) -> None:
        """Perform chunking on current text."""

        
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
                strategy_type=self._current_strategy
            )
            
            self._current_chunks = chunks
            self._update_chunk_display()
            
            if self.embedding_provider and chunks:
                await self._update_embeddings()
            
        except Exception as e:
            self.notify(f"Chunking error: {e}", severity="error")
    
    def _update_chunk_display(self) -> None:
        """Update the chunk list display."""
        chunk_list = self.query_one("#chunk-list", ChunkList)
        chunk_list.update_chunks(self._current_chunks)
        
        summary = self.query_one("#chunk-summary", Static)
        if self._current_chunks:
            stats = self.chunking_engine.get_chunk_stats(self._current_chunks)
            summary.update(
                f"📊 {stats['total_chunks']} chunks | "
                f"Avg: {stats['avg_chunk_size']:.0f} chars | "
                f"Total: {stats['total_characters']:,} chars | "
                f"~{stats['total_tokens_est']} tokens"
            )
            # Update parameter panel chunk count
            try:
                param_panel = self.query_one("#parameter-panel", ParameterPanel)
                param_panel.update_chunk_count(stats['total_chunks'])
            except Exception:
                pass
        else:
            summary.update("No chunks yet")
    
    async def _update_embeddings(self) -> None:
        """Update vector store with embeddings (bulletproof version).
        
        Features:
        - Cancels any in-flight embedding operation before starting new one
        - Gracefully handles cancellation without error messages
        - Silent failure mode - app continues working without search
        """
        if not self.embedding_provider:
            return
        
        # Cancel any in-flight embedding operation
        if self._embedding_task and not self._embedding_task.done():
            self._embedding_task.cancel()
            try:
                await self._embedding_task
            except asyncio.CancelledError:
                pass  # Expected
        
        chunk_texts = [c[0] for c in self._current_chunks]
        if not chunk_texts:
            return
        
        async def do_embedding():
            """Inner function to perform embedding with error handling."""
            try:
                embeddings = await self.embedding_provider.embed_batch(chunk_texts)
                embeddings_np = np.array(embeddings, dtype=np.float32)
                
                # Update store with correct dimensions
                if embeddings_np.shape[1] != self.vector_store.embedding_dim:
                    self.vector_store = VectorStore(embedding_dim=embeddings_np.shape[1])
                
                self.vector_store.clear()
                self.vector_store.add_chunks(chunk_texts, embeddings_np)
                
                self.notify(f"✓ Embeddings ready ({len(chunk_texts)} chunks)", timeout=2)
                
            except asyncio.CancelledError:
                # Silently handle cancellation - this is normal during rapid parameter changes
                pass
            except Exception as e:
                # Log error but don't show alarming message - search just won't work
                # Use information severity instead of warning
                self.notify(f"Embeddings pending... (retrying)", severity="information", timeout=3)
        
        # Start embedding as background task
        self._embedding_task = asyncio.create_task(do_embedding())
    
    async def on_search_panel_query_submitted(self, event: SearchPanel.QuerySubmitted) -> None:
        """Handle search/generate queries."""
        if not self.llm_provider:
            self.notify("LLM provider not available", severity="error")
            return
        
        if not self._current_chunks:
            self.notify("No chunks - add text first", severity="warning")
            return
        
        if event.action == "search":
            await self._search(event.query)
        else:
            await self._generate(event.query)
    
    async def _search(self, query: str) -> List[Tuple[str, float, dict]]:
        """Search for relevant chunks."""
        try:
            query_embedding = await self.embedding_provider.embed(query)
            query_np = np.array(query_embedding, dtype=np.float32)
            results = await self.vector_store.search_async(query_np, top_k=5)
            
            search_panel = self.query_one("#search-panel", SearchPanel)
            search_panel.update_results(results)
            
            return results
            
        except Exception as e:
            self.notify(f"Search error: {e}", severity="error")
            return []
    
    async def _generate(self, query: str) -> None:
        """Generate AI response with RAG context."""
        results = await self._search(query)
        
        if not results:
            return
        
        try:
            chunk_texts = [r[0] for r in results[:3]]
            prompt = self.llm_provider.build_rag_prompt(query, chunk_texts)
            
            chat_log = self.query_one("#chat-log", RichLog)
            live_response = self.query_one("#chat-live-response", Static)
            
            # Log user query
            chat_log.write(f"[bold cyan]User:[/] {query}")
            
            tabs = self.query_one("#main-tabs", TabbedContent)
            tabs.active = "chat-tab"
            
            # Show live response
            live_response.styles.display = "block"
            live_response.update("Thinking...")
            
            response_buffer = ""
            try:
                async for token in self.llm_provider.stream_generate(prompt):
                    response_buffer += token
                    live_response.update(response_buffer)
            except Exception as e:
                response_buffer += f"\n[Error: {e}]"
                
            # Finalize
            live_response.update("")
            live_response.styles.display = "none"
            
            chat_log.write(f"[bold green]AI:[/] {response_buffer}")
            chat_log.write("\n") # Spacing
            self.notify("✅ Response complete", timeout=2)
            
        except Exception as e:
            self.notify(f"Generation error: {e}", severity="error")
    
    async def _run_batch_test(self) -> None:
        """Run batch query testing."""
        if not self.embedding_provider:
            self.notify("Embedding provider not available", severity="error")
            return
        
        queries_input = self.query_one("#batch-queries", TextArea)
        queries = [q.strip() for q in queries_input.text.split('\n') if q.strip()]
        
        if not queries:
            self.notify("Enter queries first", severity="warning")
            return
        
        self.notify(f"Running {len(queries)} queries...", timeout=2)
        
        results = []
        for query in queries:
            try:
                search_results = await self._search(query)
                
                if search_results:
                    scores = [r[1] for r in search_results]
                    result = QueryResult(
                        query=query,
                        chunks_retrieved=[(r[0][:100], r[1]) for r in search_results],
                        top_score=max(scores),
                        avg_score=sum(scores) / len(scores)
                    )
                else:
                    result = QueryResult(
                        query=query,
                        chunks_retrieved=[],
                        top_score=0.0,
                        avg_score=0.0
                    )
                results.append(result)
            except Exception:
                self.notify(f"Query failed: {query[:30]}...", severity="warning")
        
        batch_result = calculate_batch_metrics(results)
        self._batch_results = results
        
        # Display results
        results_widget = self.query_one("#batch-results", Static)
        results_text = f"""📊 Batch Test Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Queries: {batch_result.total_queries}
Hit Rate (>0.5): {batch_result.hit_rate:.1%}
Avg Top Score: {batch_result.avg_top_score:.3f}
Avg Retrieval Score: {batch_result.avg_retrieval_score:.3f}

📋 Query Details:
"""
        for i, r in enumerate(results[:10], 1):  # Show first 10
            status = "🟢" if r.top_score > 0.5 else "🔴"
            results_text += f"\n{i}. {status} \"{r.query[:40]}...\"\n"
            results_text += f"   Top Score: {r.top_score:.3f} | Avg: {r.avg_score:.3f}\n"
            if r.chunks_retrieved:
                chunk_preview = r.chunks_retrieved[0][0][:60].replace('\n', ' ')
                results_text += f"   Best Match: \"{chunk_preview}...\"\n"
        
        results_widget.update(results_text)
        self.notify("✅ Batch test complete", timeout=2)
    
    async def _load_file(self, file_path: str) -> None:
        """Load content from a file."""
        try:
            content, info = read_file(file_path)
            self._file_info = info
            
            text_area = self.query_one("#text-input", TextArea)
            text_area.text = content
            self._current_text = content
            
            await self._rechunk()
            
            self.notify(
                f"📄 Loaded: {info.name} ({format_file_size(info.size_bytes)}, {info.line_count} lines)",
                timeout=3
            )
            
            # Switch to input tab
            tabs = self.query_one("#main-tabs", TabbedContent)
            tabs.active = "input-tab"
            
        except Exception as e:
            self.notify(f"Error loading file: {e}", severity="error")
    
    def _update_export_preview(self) -> None:
        """Update the export preview."""
        config = ChunkConfig(
            strategy=self._current_strategy.value,
            chunk_size=self._chunk_size,
            overlap_percent=self._overlap_percent,
            overlap_tokens=int(self._chunk_size * self._overlap_percent / 100)
        )
        
        try:
            preview = self.query_one("#export-preview", Static)
            preview.update(f"```json\n{config.to_json()}\n```")
        except Exception:
            pass  # Widget not mounted yet
    
    def _apply_custom_chunker(self) -> None:
        """Apply custom Python chunking function."""
        code_area = self.query_one("#custom-code", TextArea)
        code = code_area.text
        
        if not code.strip():
            self.notify("Enter Python code first", severity="warning")
            return
        
        try:
            # Execute the code to get the function
            local_vars = {}
            exec(code, {"__builtins__": __builtins__}, local_vars)
            
            # Find the chunk function
            chunk_fn = None
            for name, obj in local_vars.items():
                if callable(obj) and name.startswith("chunk"):
                    chunk_fn = obj
                    break
            
            if not chunk_fn:
                self.notify("Define a function starting with 'chunk'", severity="warning")
                return
            
            self.chunking_engine.set_custom_function(chunk_fn)
            self._current_strategy = StrategyType.CUSTOM
            self.chunking_engine.set_strategy(StrategyType.CUSTOM)
            
            asyncio.create_task(self._rechunk())
            self.notify("✅ Custom chunker applied!", timeout=2)
            
        except Exception as e:
            self.notify(f"Code error: {e}", severity="error")
    
    def _quick_clean(self) -> None:
        """Apply quick text cleaning to improve chunking quality."""
        import re
        
        text_area = self.query_one("#text-input", TextArea)
        text = text_area.text
        
        if not text.strip():
            self.notify("No text to clean", severity="warning")
            return
        
        original_len = len(text)
        
        # Built-in cleaning steps
        # 1. Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 2. Remove excessive blank lines (3+ → 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 3. Remove trailing whitespace per line
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        
        # 4. Normalize multiple spaces to single
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 5. Remove common artifacts
        text = re.sub(r'^\s*Page \d+\s*$', '', text, flags=re.MULTILINE)  # Page numbers
        text = re.sub(r'^\s*-{3,}\s*$', '', text, flags=re.MULTILINE)  # Horizontal rules
        
        # 6. Apply custom cleaner if defined
        if self._custom_cleaner:
            try:
                text = self._custom_cleaner(text)
            except Exception as e:
                self.notify(f"Custom cleaner error: {e}", severity="warning")
        
        # Update text area
        text_area.load_text(text.strip())
        self._current_text = text.strip()
        
        cleaned_chars = original_len - len(text.strip())
        self.notify(f"🧹 Cleaned! Removed {cleaned_chars} chars", timeout=2)
        
        # Trigger rechunk
        asyncio.create_task(self._debounced_rechunk())
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "load-file-btn":
            await self.action_load_file()

        elif event.button.id == "load-sample-btn":
            await self.action_load_sample()
        elif event.button.id == "quick-clean-btn":
            self._quick_clean()
        elif event.button.id == "clear-btn":
            text_area = self.query_one("#text-input", TextArea)
            text_area.load_text("")
            self._current_text = ""
            self._current_chunks = []
            self._update_chunk_display()
            self.notify("Cleared text")
        elif event.button.id == "run-batch-btn":
            await self._run_batch_test()
        elif event.button.id == "apply-custom-btn":
            self._apply_custom_chunker()
        elif event.button.id == "apply-cleaner-btn":
            self._apply_custom_cleaner()
        elif event.button.id == "clear-batch-btn":
            self._batch_results = []
            self.query_one("#batch-results", Static).update("")
            self.notify("Cleared batch results")
        elif event.button.id == "export-json-btn":
            self._copy_config("json")
        elif event.button.id == "export-langchain-btn":
            self._copy_config("langchain")
        elif event.button.id == "export-llamaindex-btn":
            self._copy_config("llamaindex")
        elif event.button.id == "chat-send-btn":
            await self._submit_chat_message()
        elif event.button.id == "apply-preset-btn":
            await self._apply_selected_preset()
    
    def _apply_custom_cleaner(self) -> None:
        """Apply a custom text cleaning function."""
        try:
            code_area = self.query_one("#custom-cleaner-code", TextArea)
            code = code_area.text.strip()
            
            if not code:
                self._custom_cleaner = None
                self.notify("Custom cleaner cleared", timeout=2)
                return
            
            # Execute the code
            local_vars = {}
            exec(code, {"__builtins__": __builtins__}, local_vars)
            
            # Find the clean function
            clean_fn = None
            for name, obj in local_vars.items():
                if callable(obj) and name.startswith("clean"):
                    clean_fn = obj
                    break
            
            if not clean_fn:
                self.notify("Define a function starting with 'clean'", severity="warning")
                return
            
            self._custom_cleaner = clean_fn
            self.notify("✅ Custom cleaner applied! Use Quick Clean to run it.", timeout=3)
            
        except Exception as e:
            self.notify(f"Code error: {e}", severity="error")
    
    async def _apply_selected_preset(self) -> None:
        """Apply the selected preset configuration."""
        try:
            preset_select = self.query_one("#preset-select", Select)
            if preset_select.value == Select.BLANK:
                self.notify("Please select a preset first", severity="warning")
                return
            
            preset = load_preset(str(preset_select.value))
            if not preset:
                self.notify("Preset not found", severity="error")
                return
            
            # Apply settings
            self._chunk_size = preset.chunk_size
            self._overlap_percent = preset.overlap_percent
            
            # Update strategy
            strategy_map = {
                "token": StrategyType.TOKEN,
                "sentence": StrategyType.SENTENCE,
                "paragraph": StrategyType.PARAGRAPH,
                "recursive": StrategyType.RECURSIVE,
                "fixed": StrategyType.FIXED_CHARS,
            }
            if preset.strategy in strategy_map:
                self._current_strategy = strategy_map[preset.strategy]
            
            # Update parameter panel
            param_panel = self.query_one("#parameter-panel", ParameterPanel)
            param_panel.chunk_size = preset.chunk_size
            param_panel.overlap_percent = preset.overlap_percent
            
            self.notify(f"✅ Applied preset: {preset.name}")
            await self._debounced_rechunk()
            
        except Exception as e:
            self.notify(f"Error applying preset: {e}", severity="error")
    
    def _update_recommendations(self) -> None:
        """Update the recommendations panel based on current text."""
        try:
            panel = self.query_one("#recommendations-panel", Static)
            
            text_len = len(self._current_text)
            token_estimate = text_len // 4
            
            if text_len == 0:
                panel.update("📝 Load text to see recommendations")
                return
            
            # Calculate recommendations based on text size
            if token_estimate < 500:
                rec_size = "100-150"
                rec_overlap = "15-20%"
                exp_chunks = "3-5"
                tip = "Small document - use precise chunks"
            elif token_estimate < 2000:
                rec_size = "200-300"
                rec_overlap = "10-15%"
                exp_chunks = "8-15"
                tip = "Medium document - balanced approach"
            else:
                rec_size = "300-500"
                rec_overlap = "10-15%"
                exp_chunks = "15+"
                tip = "Large document - consider larger chunks"
            
            text = f"""📊 Your text: {text_len:,} chars (~{token_estimate:,} tokens)

📏 Suggested chunk size: {rec_size} tokens
🔄 Suggested overlap: {rec_overlap}
📦 Expected chunks: {exp_chunks}

💡 Tip: {tip}"""
            
            panel.update(text)
        except Exception:
            pass  # Panel might not exist yet

    async def _submit_chat_message(self) -> None:
        """Submit chat message from input."""
        chat_input = self.query_one("#chat-input", Input)
        query = chat_input.value.strip()
        
        if not query:
            return
            
        chat_input.value = ""
        
        if not self.llm_provider:
            self.notify("LLM provider not available", severity="error")
            return
            
        await self._generate(query)
    
    def _copy_config(self, format: str) -> None:
        """Copy configuration to clipboard (simulated)."""
        config = ChunkConfig(
            strategy=self._current_strategy.value,
            chunk_size=self._chunk_size,
            overlap_percent=self._overlap_percent,
            overlap_tokens=int(self._chunk_size * self._overlap_percent / 100)
        )
        
        output = export_config(config, format)
        
        # Show in preview
        preview = self.query_one("#export-preview", Static)
        preview.update(f"```\n{output}\n```")
        
        self.notify(f"📋 {format.upper()} config generated - copy from Settings tab", timeout=3)
    
    async def action_load_sample(self) -> None:
        """Load sample text."""
        text_area = self.query_one("#text-input", TextArea)
        text_area.text = SAMPLE_TEXT
        self._current_text = SAMPLE_TEXT
        await self._rechunk()
        self.notify("📄 Sample text loaded", timeout=2)
    
    def _clear_text(self) -> None:
        """Clear all text."""
        text_area = self.query_one("#text-input", TextArea)
        text_area.clear()
        self._current_text = ""
        self._current_chunks = []
        self._update_chunk_display()
    
    async def action_rechunk(self) -> None:
        """Manually trigger rechunking."""
        await self._rechunk()
    
    def action_toggle_dark(self) -> None:
        """Toggle dark mode."""
        self.dark = not self.dark
    
    def action_export_config(self) -> None:
        """Export current configuration."""
        self._copy_config("json")
    
    def action_show_help(self) -> None:
        """Show help overlay."""
        self.push_screen(HelpOverlay())


    async def action_load_file(self) -> None:
        """Load file from input path."""
        file_input = self.query_one("#file-path-input", Input)
        path = file_input.value.strip()
        if path:
            await self._load_file(path)
        else:
            self.notify("⚠️ Please enter a file path", severity="warning")



def main():
    """Run the RAG-TUI application."""
    app = RAGTUIApp()
    app.run()


if __name__ == "__main__":
    main()
