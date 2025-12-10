# Changelog

All notable changes to RAG-TUI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.2-beta] - 2025-12-10

### Fixed
- **Embedding 500 Error (Bulletproof Fix)**: Completely resolved Ollama `500 Internal Server Error` that occurred when changing chunk parameters rapidly
  - **OllamaProvider**: 5 retries with aggressive exponential backoff (1-8 seconds)
  - **OllamaProvider**: Global async lock prevents concurrent embedding batches
  - **OllamaProvider**: 200ms delay between sequential requests
  - **App**: Automatic cancellation of in-flight embeddings when parameters change
  - **App**: Silent handling of cancellation (no error messages during rapid slider changes)
  - **App**: Graceful degradation - app continues working even if embeddings fail

### Changed
- `OllamaProvider.embed()` now accepts optional `max_retries` parameter (default: 3)
- `OllamaProvider.embed_batch()` now processes embeddings sequentially with rate limiting instead of all at once

### Technical Details
- Modified: `rag_tui/core/providers.py` - Added retry logic and rate limiting to Ollama embedding methods
- Modified: `rag_tui/core/llm.py` - Added semaphore-based concurrency control (for API consistency)

---

## [0.0.1-beta] - 2025-12-05

### Added
- Initial beta release
- Interactive TUI for RAG chunking visualization and debugging
- Multiple chunking strategies: Token-based, Sentence, Paragraph, Recursive, Fixed Characters, Custom
- Multi-provider LLM support: Ollama, OpenAI, Groq, Google Gemini
- Real-time chunk visualization with statistics
- Semantic search with vector store
- Batch query testing with metrics
- Configuration export (JSON, LangChain, LlamaIndex)
- Quick text cleaning utilities
- Custom chunker and cleaner support
- Preset configurations for common use cases
- Chat interface with RAG context
