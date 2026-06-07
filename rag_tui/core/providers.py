"""Multi-provider LLM support for RAG-TUI.

Supports: Ollama (local), OpenAI, Groq, Google Gemini
Auto-detects available providers or uses environment variables.
"""

import os
import asyncio
import httpx
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, AsyncIterator, Tuple
from enum import Enum


class ProviderType(Enum):
    """Available LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    GROQ = "groq"
    GOOGLE = "google"


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    name: str
    embedding_model: str
    llm_model: str
    embedding_dim: int
    base_url: str
    api_key: Optional[str] = None
    supports_embedding: bool = True
    supports_llm: bool = True


# Default configurations for each provider
PROVIDER_CONFIGS = {
    ProviderType.OLLAMA: ProviderConfig(
        name="Ollama (Local)",
        embedding_model="nomic-embed-text",
        llm_model="llama3.2:1b",
        embedding_dim=768,
        base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
        supports_embedding=True,
        supports_llm=True,
    ),
    ProviderType.OPENAI: ProviderConfig(
        name="OpenAI",
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o-mini",
        embedding_dim=1536,
        base_url="https://api.openai.com/v1",
        api_key=os.environ.get("OPENAI_API_KEY"),
        supports_embedding=True,
        supports_llm=True,
    ),
    ProviderType.GROQ: ProviderConfig(
        name="Groq (Fast)",
        embedding_model="",  # Groq doesn't support embeddings
        llm_model="llama-3.1-8b-instant",
        embedding_dim=0,
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY"),
        supports_embedding=False,
        supports_llm=True,
    ),
    ProviderType.GOOGLE: ProviderConfig(
        name="Google Gemini",
        embedding_model="text-embedding-004",
        llm_model="gemini-1.5-flash",
        embedding_dim=768,
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key=os.environ.get("GOOGLE_API_KEY"),
        supports_embedding=True,
        supports_llm=True,
    ),
}


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    # Subclasses set these to enable the shared embedding cache.
    _cache_provider_name: str = ""
    _cache_model_name: str = ""
    _emb_cache = None  # lazy EmbeddingCache instance

    def _get_cache(self):
        """Return a lazily-initialised EmbeddingCache, or None if disabled."""
        if not self._cache_provider_name or not self._cache_model_name:
            return None
        if self._emb_cache is None:
            try:
                from rag_tui.core.cache import EmbeddingCache
                self._emb_cache = EmbeddingCache(
                    self._cache_provider_name, self._cache_model_name
                )
            except Exception:
                pass
        return self._emb_cache

    @abstractmethod
    async def check_connection(self) -> bool:
        """Check if the provider is available."""
        pass

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        pass

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate a response."""
        pass

    @abstractmethod
    async def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        """Stream a response token by token."""
        pass

    def build_rag_prompt(self, query: str, context_chunks: List[str]) -> str:
        """Build a RAG prompt with context."""
        context = "\n\n---\n\n".join(context_chunks)
        return f"""Use the following context to answer the question. If the answer is not in the context, say so.

Context:
{context}

Question: {query}

Answer:"""


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=60.0)
        self._cache_provider_name = "ollama"
        self._cache_model_name = config.embedding_model
    
    async def check_connection(self) -> bool:
        try:
            response = await self.client.get(f"{self.config.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    # Class-level lock to prevent concurrent embedding batches
    _embedding_lock: asyncio.Lock = None
    
    async def embed(self, text: str, max_retries: int = 5) -> List[float]:
        """Generate embedding with robust retry logic for transient errors.
        
        Uses exponential backoff with longer delays to handle Ollama overload.
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self.client.post(
                    f"{self.config.base_url}/api/embeddings",
                    json={"model": self.config.embedding_model, "prompt": text},
                    timeout=30.0  # Short timeout per request
                )
                response.raise_for_status()
                return response.json()["embedding"]
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 500 and attempt < max_retries - 1:
                    # Ollama overwhelmed - use aggressive exponential backoff
                    wait_time = min(1.0 * (2 ** attempt), 8.0)  # 1s, 2s, 4s, 8s, 8s
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = min(1.0 * (2 ** attempt), 8.0)
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except asyncio.CancelledError:
                # Allow cancellation to propagate
                raise
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.0 * (2 ** attempt))
                else:
                    raise
        
        raise RuntimeError(f"Embedding failed after {max_retries} attempts: {last_error}")
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with cache support and bulletproof error handling."""
        cache = self._get_cache()

        # Serve hits from cache; only embed misses.
        results: List[Optional[List[float]]] = [None] * len(texts)
        miss_indices = []
        if cache:
            for i, text in enumerate(texts):
                cached = cache.get(text)
                if cached is not None:
                    results[i] = cached.tolist()
                else:
                    miss_indices.append(i)
        else:
            miss_indices = list(range(len(texts)))

        if not miss_indices:
            return results  # type: ignore[return-value]

        if OllamaProvider._embedding_lock is None:
            OllamaProvider._embedding_lock = asyncio.Lock()

        async with OllamaProvider._embedding_lock:
            fresh_embeddings = []
            for i, idx in enumerate(miss_indices):
                try:
                    await asyncio.sleep(0)
                    emb = await self.embed(texts[idx])
                    fresh_embeddings.append(emb)
                    results[idx] = emb
                    if i < len(miss_indices) - 1:
                        await asyncio.sleep(0.2)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    raise RuntimeError(f"Failed to embed chunk {idx + 1}/{len(texts)}: {e}")

        if cache and fresh_embeddings:
            miss_texts = [texts[i] for i in miss_indices]
            cache.put_batch(miss_texts, fresh_embeddings)

        return results  # type: ignore[return-value]
    
    async def generate(self, prompt: str) -> str:
        response = await self.client.post(
            f"{self.config.base_url}/api/generate",
            json={"model": self.config.llm_model, "prompt": prompt, "stream": False}
        )
        response.raise_for_status()
        return response.json()["response"]
    
    async def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        async with self.client.stream(
            "POST",
            f"{self.config.base_url}/api/generate",
            json={"model": self.config.llm_model, "prompt": prompt, "stream": True}
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=60.0,
            headers={"Authorization": f"Bearer {config.api_key}"}
        )
        self._cache_provider_name = "openai"
        self._cache_model_name = config.embedding_model
    
    async def check_connection(self) -> bool:
        if not self.config.api_key:
            return False
        try:
            response = await self.client.get(f"{self.config.base_url}/models")
            return response.status_code == 200
        except Exception:
            return False
    
    async def embed(self, text: str) -> List[float]:
        response = await self.client.post(
            f"{self.config.base_url}/embeddings",
            json={"model": self.config.embedding_model, "input": text}
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        cache = self._get_cache()
        results: List[Optional[List[float]]] = [None] * len(texts)
        miss_indices = []
        if cache:
            for i, text in enumerate(texts):
                cached = cache.get(text)
                if cached is not None:
                    results[i] = cached.tolist()
                else:
                    miss_indices.append(i)
        else:
            miss_indices = list(range(len(texts)))

        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            response = await self.client.post(
                f"{self.config.base_url}/embeddings",
                json={"model": self.config.embedding_model, "input": miss_texts},
            )
            response.raise_for_status()
            fresh = [d["embedding"] for d in response.json()["data"]]
            for idx, emb in zip(miss_indices, fresh):
                results[idx] = emb
            if cache:
                cache.put_batch(miss_texts, fresh)

        return results  # type: ignore[return-value]
    
    async def generate(self, prompt: str) -> str:
        response = await self.client.post(
            f"{self.config.base_url}/chat/completions",
            json={
                "model": self.config.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    async def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        async with self.client.stream(
            "POST",
            f"{self.config.base_url}/chat/completions",
            json={
                "model": self.config.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: ") and not line.endswith("[DONE]"):
                    import json
                    try:
                        data = json.loads(line[6:])
                        if data["choices"][0]["delta"].get("content"):
                            yield data["choices"][0]["delta"]["content"]
                    except Exception:
                        pass


class GroqProvider(LLMProvider):
    """Groq API provider (fast LLM, no embeddings)."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=60.0,
            headers={"Authorization": f"Bearer {config.api_key}"}
        )
    
    async def check_connection(self) -> bool:
        if not self.config.api_key:
            return False
        try:
            response = await self.client.get(f"{self.config.base_url}/models")
            return response.status_code == 200
        except Exception:
            return False
    
    async def embed(self, text: str) -> List[float]:
        raise NotImplementedError("Groq doesn't support embeddings. Use with Ollama or OpenAI for embeddings.")
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("Groq doesn't support embeddings.")
    
    async def generate(self, prompt: str) -> str:
        response = await self.client.post(
            f"{self.config.base_url}/chat/completions",
            json={
                "model": self.config.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    async def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        async with self.client.stream(
            "POST",
            f"{self.config.base_url}/chat/completions",
            json={
                "model": self.config.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: ") and not line.endswith("[DONE]"):
                    import json
                    try:
                        data = json.loads(line[6:])
                        if data["choices"][0]["delta"].get("content"):
                            yield data["choices"][0]["delta"]["content"]
                    except Exception:
                        pass


class GoogleProvider(LLMProvider):
    """Google Gemini API provider."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def check_connection(self) -> bool:
        if not self.config.api_key:
            return False
        try:
            url = f"{self.config.base_url}/models?key={self.config.api_key}"
            response = await self.client.get(url)
            return response.status_code == 200
        except Exception:
            return False
    
    async def embed(self, text: str) -> List[float]:
        url = f"{self.config.base_url}/models/{self.config.embedding_model}:embedContent?key={self.config.api_key}"
        response = await self.client.post(
            url,
            json={"model": f"models/{self.config.embedding_model}", "content": {"parts": [{"text": text}]}}
        )
        response.raise_for_status()
        return response.json()["embedding"]["values"]
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [await self.embed(text) for text in texts]
    
    async def generate(self, prompt: str) -> str:
        url = f"{self.config.base_url}/models/{self.config.llm_model}:generateContent?key={self.config.api_key}"
        response = await self.client.post(
            url,
            json={"contents": [{"parts": [{"text": prompt}]}]}
        )
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    
    async def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        # Google's streaming is complex, fall back to non-streaming
        result = await self.generate(prompt)
        for word in result.split():
            yield word + " "
            await asyncio.sleep(0.02)


# Provider factory
def get_provider(provider_type: ProviderType) -> LLMProvider:
    """Get a provider instance by type."""
    config = PROVIDER_CONFIGS[provider_type]
    
    if provider_type == ProviderType.OLLAMA:
        return OllamaProvider(config)
    elif provider_type == ProviderType.OPENAI:
        return OpenAIProvider(config)
    elif provider_type == ProviderType.GROQ:
        return GroqProvider(config)
    elif provider_type == ProviderType.GOOGLE:
        return GoogleProvider(config)
    else:
        raise ValueError(f"Unknown provider: {provider_type}")


async def detect_available_providers() -> List[Tuple[ProviderType, ProviderConfig]]:
    """Detect which providers are available."""
    available = []
    
    for provider_type, config in PROVIDER_CONFIGS.items():
        provider = get_provider(provider_type)
        if await provider.check_connection():
            available.append((provider_type, config))
    
    return available


async def get_best_provider() -> Tuple[Optional[LLMProvider], Optional[LLMProvider]]:
    """Get the best available embedding and LLM providers.
    
    Returns:
        Tuple of (embedding_provider, llm_provider)
    """
    embedding_provider = None
    llm_provider = None
    
    # Priority order for checking
    priority = [
        ProviderType.OLLAMA,  # Free, local
        ProviderType.OPENAI,  # Best quality
        ProviderType.GOOGLE,  # Free tier
        ProviderType.GROQ,    # Fast but no embeddings
    ]
    
    for provider_type in priority:
        config = PROVIDER_CONFIGS[provider_type]
        provider = get_provider(provider_type)
        
        if await provider.check_connection():
            if config.supports_embedding and embedding_provider is None:
                embedding_provider = provider
            if config.supports_llm and llm_provider is None:
                llm_provider = provider
        
        if embedding_provider and llm_provider:
            break
    
    return embedding_provider, llm_provider
