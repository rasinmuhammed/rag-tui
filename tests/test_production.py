"""Comprehensive production-level tests for RAG-TUI v0.0.2-beta."""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path


# ============================================================================
# TEST 1: CHUNKING STRATEGIES
# ============================================================================

class TestChunkingStrategies:
    """Test all 6 chunking strategies."""
    
    def test_token_strategy(self):
        """Test token-based chunking."""
        from rag_tui.core.strategies import TokenStrategy
        
        strategy = TokenStrategy()
        text = "This is a test. " * 100  # ~400 words
        
        chunks = strategy.chunk(text, chunk_size=50, overlap=10)
        
        assert len(chunks) > 0, "Should produce chunks"
        assert all(hasattr(c, 'text') for c in chunks), "Chunks should have text"
        assert all(hasattr(c, 'start_pos') for c in chunks), "Chunks should have positions"
        print(f"✅ Token strategy: {len(chunks)} chunks")
    
    def test_sentence_strategy(self):
        """Test sentence-based chunking."""
        from rag_tui.core.strategies import SentenceStrategy
        
        strategy = SentenceStrategy()
        text = "First sentence. Second sentence. Third sentence. " * 20
        
        chunks = strategy.chunk(text, chunk_size=100, overlap=10)
        
        assert len(chunks) > 0, "Should produce chunks"
        print(f"✅ Sentence strategy: {len(chunks)} chunks")
    
    def test_paragraph_strategy(self):
        """Test paragraph-based chunking."""
        from rag_tui.core.strategies import ParagraphStrategy
        
        strategy = ParagraphStrategy()
        text = "Paragraph one with content.\n\nParagraph two with more content.\n\nParagraph three."
        
        chunks = strategy.chunk(text, chunk_size=500, overlap=50)
        
        assert len(chunks) > 0, "Should produce chunks"
        print(f"✅ Paragraph strategy: {len(chunks)} chunks")
    
    def test_recursive_strategy(self):
        """Test recursive chunking."""
        from rag_tui.core.strategies import RecursiveStrategy
        
        strategy = RecursiveStrategy()
        text = "Line one\nLine two\n\nNew paragraph\n\nAnother paragraph with more text."
        
        chunks = strategy.chunk(text, chunk_size=50, overlap=5)
        
        assert len(chunks) > 0, "Should produce chunks"
        print(f"✅ Recursive strategy: {len(chunks)} chunks")
    
    def test_fixed_chars_strategy(self):
        """Test fixed character chunking."""
        from rag_tui.core.strategies import FixedCharsStrategy
        
        strategy = FixedCharsStrategy()
        text = "A" * 2000  # 2000 characters
        
        chunks = strategy.chunk(text, chunk_size=100, overlap=10)  # 400 chars per chunk
        
        assert len(chunks) >= 1, "Should produce at least 1 chunk"
        print(f"✅ Fixed chars strategy: {len(chunks)} chunks")
    
    def test_chunking_engine_integration(self):
        """Test the chunking engine with all strategies."""
        from rag_tui.core.engine import ChunkingEngine
        from rag_tui.core.strategies import StrategyType
        
        engine = ChunkingEngine()
        text = "Test text for chunking. " * 50
        
        for strategy_type in StrategyType:
            if strategy_type == StrategyType.CUSTOM:
                continue  # Skip custom, needs function
            
            engine.set_strategy(strategy_type)
            chunks = engine.chunk_text(text, chunk_size=100, overlap=10)
            
            assert len(chunks) > 0, f"{strategy_type.value} should produce chunks"
        
        print("✅ All strategies work through engine")


# ============================================================================
# TEST 2: FILE HANDLER
# ============================================================================

class TestFileHandler:
    """Test file reading capabilities."""
    
    def test_read_txt_file(self):
        """Test reading .txt files."""
        from rag_tui.core.file_handler import read_file
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for txt file")
            temp_path = f.name
        
        try:
            content, info = read_file(temp_path)
            assert "Test content" in content
            assert info.extension == ".txt"
            print("✅ Read .txt file")
        finally:
            os.unlink(temp_path)
    
    def test_read_md_file(self):
        """Test reading .md files."""
        from rag_tui.core.file_handler import read_file
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Markdown Header\n\nContent here")
            temp_path = f.name
        
        try:
            content, info = read_file(temp_path)
            assert "# Markdown" in content
            assert info.extension == ".md"
            print("✅ Read .md file")
        finally:
            os.unlink(temp_path)
    
    def test_read_py_file(self):
        """Test reading .py files."""
        from rag_tui.core.file_handler import read_file
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def hello():\n    print('world')")
            temp_path = f.name
        
        try:
            content, info = read_file(temp_path)
            assert "def hello" in content
            assert info.extension == ".py"
            print("✅ Read .py file")
        finally:
            os.unlink(temp_path)
    
    def test_file_not_found(self):
        """Test file not found error."""
        from rag_tui.core.file_handler import read_file
        
        with pytest.raises(FileNotFoundError):
            read_file("/nonexistent/path/to/file.txt")
        
        print("✅ FileNotFoundError handled")
    
    def test_unsupported_extension(self):
        """Test unsupported file extension."""
        from rag_tui.core.file_handler import read_file
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write("content")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                read_file(temp_path)
            print("✅ Unsupported extension handled")
        finally:
            os.unlink(temp_path)
    
    def test_supported_extensions_list(self):
        """Test that SUPPORTED_EXTENSIONS is properly defined."""
        from rag_tui.core.file_handler import SUPPORTED_EXTENSIONS
        
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".md" in SUPPORTED_EXTENSIONS
        assert ".py" in SUPPORTED_EXTENSIONS
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert len(SUPPORTED_EXTENSIONS) >= 10
        print(f"✅ {len(SUPPORTED_EXTENSIONS)} file types supported")


# ============================================================================
# TEST 3: PROVIDERS
# ============================================================================

class TestProviders:
    """Test LLM provider implementations."""
    
    def test_provider_types_exist(self):
        """Test that all provider types are defined."""
        from rag_tui.core.providers import ProviderType
        
        assert ProviderType.OLLAMA.value == "ollama"
        assert ProviderType.OPENAI.value == "openai"
        assert ProviderType.GROQ.value == "groq"
        assert ProviderType.GOOGLE.value == "google"
        print("✅ All 4 provider types defined")
    
    def test_provider_configs_exist(self):
        """Test that all provider configs are defined."""
        from rag_tui.core.providers import PROVIDER_CONFIGS, ProviderType
        
        for provider_type in ProviderType:
            assert provider_type in PROVIDER_CONFIGS
            config = PROVIDER_CONFIGS[provider_type]
            assert config.name
            assert config.base_url
        
        print("✅ All provider configs defined")
    
    def test_get_provider_factory(self):
        """Test provider factory function."""
        from rag_tui.core.providers import get_provider, ProviderType
        
        for provider_type in ProviderType:
            provider = get_provider(provider_type)
            assert provider is not None
            assert hasattr(provider, 'check_connection')
            assert hasattr(provider, 'embed')
            assert hasattr(provider, 'generate')
        
        print("✅ Provider factory works for all types")
    
    def test_ollama_provider_methods(self):
        """Test Ollama provider has required methods."""
        from rag_tui.core.providers import OllamaProvider, PROVIDER_CONFIGS, ProviderType
        
        config = PROVIDER_CONFIGS[ProviderType.OLLAMA]
        provider = OllamaProvider(config)
        
        assert hasattr(provider, 'check_connection')
        assert hasattr(provider, 'embed')
        assert hasattr(provider, 'embed_batch')
        assert hasattr(provider, 'generate')
        assert hasattr(provider, 'stream_generate')
        assert hasattr(provider, 'build_rag_prompt')
        print("✅ OllamaProvider has all methods")
    
    def test_openai_provider_methods(self):
        """Test OpenAI provider has required methods."""
        from rag_tui.core.providers import OpenAIProvider, PROVIDER_CONFIGS, ProviderType
        
        config = PROVIDER_CONFIGS[ProviderType.OPENAI]
        provider = OpenAIProvider(config)
        
        assert hasattr(provider, 'check_connection')
        assert hasattr(provider, 'embed')
        assert hasattr(provider, 'generate')
        print("✅ OpenAIProvider has all methods")


# ============================================================================
# TEST 4: PRESETS
# ============================================================================

class TestPresets:
    """Test preset management."""
    
    def test_builtin_presets_exist(self):
        """Test that built-in presets are defined."""
        from rag_tui.core.presets import BUILTIN_PRESETS
        
        assert len(BUILTIN_PRESETS) >= 5
        
        names = [p.name for p in BUILTIN_PRESETS]
        assert "📝 Summarization" in names or "📚 Long Documents" in names
        assert "💻 Code Files" in names or "🎯 High Precision" in names
        
        print(f"✅ {len(BUILTIN_PRESETS)} built-in presets")
    
    def test_list_presets(self):
        """Test listing all presets."""
        from rag_tui.core.presets import list_presets
        
        presets = list_presets()
        assert len(presets) >= 5
        print(f"✅ list_presets returns {len(presets)} presets")
    
    def test_load_preset(self):
        """Test loading a preset by name."""
        from rag_tui.core.presets import load_preset
        
        preset = load_preset("📄 General Text")
        assert preset is not None
        assert preset.chunk_size == 256
        assert preset.overlap_percent == 10
        print("✅ load_preset works")
    
    def test_load_nonexistent_preset(self):
        """Test loading a non-existent preset."""
        from rag_tui.core.presets import load_preset
        
        preset = load_preset("Nonexistent Preset Name")
        assert preset is None
        print("✅ Non-existent preset returns None")
    
    def test_preset_dataclass(self):
        """Test Preset dataclass."""
        from rag_tui.core.presets import Preset
        
        preset = Preset(
            name="Test Preset",
            strategy="token",
            chunk_size=300,
            overlap_percent=20,
            description="Test description"
        )
        
        assert preset.name == "Test Preset"
        assert preset.strategy == "token"
        assert preset.chunk_size == 300
        print("✅ Preset dataclass works")


# ============================================================================
# TEST 5: METRICS & EXPORT
# ============================================================================

class TestMetricsExport:
    """Test metrics calculation and config export."""
    
    def test_chunk_config(self):
        """Test ChunkConfig dataclass."""
        from rag_tui.core.metrics import ChunkConfig
        
        config = ChunkConfig(
            strategy="token",
            chunk_size=200,
            overlap_percent=10,
            overlap_tokens=20
        )
        
        assert config.strategy == "token"
        assert config.chunk_size == 200
        print("✅ ChunkConfig works")
    
    def test_json_export(self):
        """Test JSON export."""
        from rag_tui.core.metrics import ChunkConfig, export_config
        
        config = ChunkConfig("token", 200, 10, 20)
        output = export_config(config, "json")
        
        assert '"strategy": "token"' in output
        assert '"chunk_size": 200' in output
        print("✅ JSON export works")
    
    def test_langchain_export(self):
        """Test LangChain export."""
        from rag_tui.core.metrics import ChunkConfig, export_config
        
        config = ChunkConfig("token", 200, 10, 20)
        output = export_config(config, "langchain")
        
        assert "langchain" in output.lower() or "RecursiveCharacterTextSplitter" in output
        print("✅ LangChain export works")
    
    def test_llamaindex_export(self):
        """Test LlamaIndex export."""
        from rag_tui.core.metrics import ChunkConfig, export_config
        
        config = ChunkConfig("token", 200, 10, 20)
        output = export_config(config, "llamaindex")
        
        assert "llama" in output.lower() or "SentenceSplitter" in output
        print("✅ LlamaIndex export works")
    
    def test_batch_metrics_calculation(self):
        """Test batch metrics calculation."""
        from rag_tui.core.metrics import QueryResult, calculate_batch_metrics
        
        results = [
            QueryResult("query1", [("chunk1", 0.9)], 0.9, 0.9),
            QueryResult("query2", [("chunk2", 0.7)], 0.7, 0.7),
            QueryResult("query3", [("chunk3", 0.3)], 0.3, 0.3),
        ]
        
        batch = calculate_batch_metrics(results)
        
        assert batch.total_queries == 3
        assert 0.0 <= batch.hit_rate <= 1.0
        assert batch.avg_top_score > 0
        print(f"✅ Batch metrics: {batch.total_queries} queries, {batch.hit_rate:.0%} hit rate")


# ============================================================================
# TEST 6: UI WIDGETS
# ============================================================================

class TestUIWidgets:
    """Test UI widget imports and basic structure."""
    
    def test_parameter_panel_import(self):
        """Test ParameterPanel can be imported."""
        from rag_tui.widgets.parameter_panel import ParameterPanel, ParameterControl
        
        assert ParameterPanel is not None
        assert ParameterControl is not None
        print("✅ ParameterPanel imports")
    
    def test_chunk_card_import(self):
        """Test ChunkCard can be imported."""
        from rag_tui.widgets.chunk_card import ChunkCard, ChunkList
        
        assert ChunkCard is not None
        assert ChunkList is not None
        print("✅ ChunkCard imports")
    
    def test_search_panel_import(self):
        """Test SearchPanel can be imported."""
        from rag_tui.widgets.search_panel import SearchPanel, ResultCard
        
        assert SearchPanel is not None
        assert ResultCard is not None
        print("✅ SearchPanel imports")
    
    def test_control_bar_import(self):
        """Test ControlBar can be imported."""
        from rag_tui.widgets.control_bar import ControlBar
        
        assert ControlBar is not None
        print("✅ ControlBar imports")
    
    def test_help_overlay_import(self):
        """Test HelpOverlay can be imported."""
        from rag_tui.widgets.help_overlay import HelpOverlay
        
        assert HelpOverlay is not None
        print("✅ HelpOverlay imports")


# ============================================================================
# TEST 7: INTEGRATION
# ============================================================================

class TestIntegration:
    """Test full application integration."""
    
    def test_app_import(self):
        """Test main app can be imported."""
        from rag_tui.app import RAGTUIApp, main
        
        assert RAGTUIApp is not None
        assert main is not None
        print("✅ App imports successfully")
    
    def test_app_instantiation(self):
        """Test app can be instantiated."""
        from rag_tui.app import RAGTUIApp
        
        app = RAGTUIApp()
        assert app is not None
        assert "RAG-TUI" in app.TITLE
        print("✅ App instantiates")
    
    def test_package_version(self):
        """Test package version."""
        import rag_tui
        
        assert rag_tui.__version__ == "0.0.2-beta"
        print(f"✅ Version: {rag_tui.__version__}")
    
    def test_vector_store(self):
        """Test vector store operations."""
        from rag_tui.core.vector import VectorStore
        import numpy as np
        
        store = VectorStore(embedding_dim=768)
        
        # Add some chunks
        chunks = ["chunk1", "chunk2", "chunk3"]
        embeddings = np.random.rand(3, 768).astype(np.float32)
        
        store.add_chunks(chunks, embeddings)
        assert store.count() == 3
        
        # Search
        query = np.random.rand(768).astype(np.float32)
        results = store.search(query, top_k=2)
        
        assert len(results) == 2
        print("✅ VectorStore works")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RAG-TUI v0.0.2-beta PRODUCTION TESTS")
    print("=" * 70 + "\n")
    
    # Run tests manually for detailed output
    tests = [
        TestChunkingStrategies(),
        TestFileHandler(),
        TestProviders(),
        TestPresets(),
        TestMetricsExport(),
        TestUIWidgets(),
        TestIntegration(),
    ]
    
    passed = 0
    failed = 0
    
    for test_class in tests:
        print(f"\n{'─' * 50}")
        print(f"Running: {test_class.__class__.__name__}")
        print('─' * 50)
        
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                try:
                    method = getattr(test_class, method_name)
                    method()
                    passed += 1
                except Exception as e:
                    print(f"❌ {method_name}: {e}")
                    failed += 1
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Ready for production.\n")
    else:
        print(f"\n⚠️  {failed} tests failed. Please fix before release.\n")
