"""Async wrapper for Ollama LLM operations."""

import asyncio
from typing import AsyncIterator, Optional, List
from ollama import AsyncClient


class OllamaLLM:
    """Async wrapper for Ollama local LLM inference."""
    
    def __init__(
        self,
        model: str = "llama3.2:1b",  # Using your installed model
        embedding_model: str = "nomic-embed-text",
        host: str = "http://localhost:11434"
    ):
        """Initialize the Ollama client.
        
        Args:
            model: The LLM model to use for generation (default: llama3.2:1b)
            embedding_model: The model to use for embeddings (default: nomic-embed-text)
            host: Ollama server host (default: http://localhost:11434)
        """
        self.model = model
        self.embedding_model = embedding_model
        self.host = host
        self.client = AsyncClient(host=host)
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        try:
            response = await self.client.embeddings(
                model=self.embedding_model,
                prompt=text
            )
            return response["embedding"]
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e}")
    
    async def embed_batch(
        self,
        texts: List[str],
        max_concurrent: int = 3,
        max_retries: int = 3
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts with rate limiting.
        
        Args:
            texts: List of texts to embed
            max_concurrent: Maximum concurrent requests to Ollama (default: 3)
            max_retries: Maximum retry attempts per embedding (default: 3)
            
        Returns:
            List of embedding vectors
        """
        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def embed_with_retry(text: str, idx: int) -> List[float]:
            """Embed single text with retry logic."""
            async with semaphore:
                for attempt in range(max_retries):
                    try:
                        return await self.embed(text)
                    except Exception as e:
                        if attempt < max_retries - 1:
                            # Exponential backoff: 0.5s, 1s, 2s
                            await asyncio.sleep(0.5 * (2 ** attempt))
                        else:
                            raise RuntimeError(
                                f"Failed to embed chunk {idx + 1} after {max_retries} attempts: {e}"
                            )
        
        # Generate embeddings with rate limiting
        tasks = [embed_with_retry(text, i) for i, text in enumerate(texts)]
        return await asyncio.gather(*tasks)
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate a response from the LLM.
        
        Args:
            prompt: The user prompt
            system: Optional system prompt
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        try:
            options = {"temperature": temperature}
            if max_tokens:
                options["num_predict"] = max_tokens
            
            response = await self.client.generate(
                model=self.model,
                prompt=prompt,
                system=system,
                options=options
            )
            return response["response"]
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {e}")
    
    async def stream_generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncIterator[str]:
        """Stream a response from the LLM token by token.
        
        Args:
            prompt: The user prompt
            system: Optional system prompt
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens to generate
            
        Yields:
            Chunks of generated text
        """
        try:
            options = {"temperature": temperature}
            if max_tokens:
                options["num_predict"] = max_tokens
            
            stream = await self.client.generate(
                model=self.model,
                prompt=prompt,
                system=system,
                options=options,
                stream=True
            )
            
            async for chunk in stream:
                if "response" in chunk:
                    yield chunk["response"]
        except Exception as e:
            raise RuntimeError(f"Failed to stream response: {e}")
    
    async def chat(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Chat with the LLM using message history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        try:
            options = {"temperature": temperature}
            if max_tokens:
                options["num_predict"] = max_tokens
            
            response = await self.client.chat(
                model=self.model,
                messages=messages,
                options=options
            )
            return response["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Failed to chat: {e}")
    
    async def stream_chat(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncIterator[str]:
        """Stream a chat response token by token.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens to generate
            
        Yields:
            Chunks of generated text
        """
        try:
            options = {"temperature": temperature}
            if max_tokens:
                options["num_predict"] = max_tokens
            
            stream = await self.client.chat(
                model=self.model,
                messages=messages,
                options=options,
                stream=True
            )
            
            async for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Failed to stream chat: {e}")
    
    async def check_connection(self) -> bool:
        """Check if Ollama server is reachable.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            # Try to list models as a connection test
            await self.client.list()
            return True
        except Exception:
            return False
    
    def build_rag_prompt(
        self,
        query: str,
        context_chunks: List[str],
        system_prompt: Optional[str] = None
    ) -> str:
        """Build a RAG prompt with context chunks.
        
        Args:
            query: The user's question
            context_chunks: List of relevant text chunks
            system_prompt: Optional system instructions
            
        Returns:
            Formatted prompt string
        """
        if system_prompt is None:
            system_prompt = "You are a helpful assistant. Answer questions based on the provided context."
        
        context = "\n\n".join([
            f"Context {i+1}:\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        prompt = f"""{system_prompt}

{context}

Question: {query}

Answer based on the context provided above. If the answer cannot be found in the context, say so."""
        
        return prompt
