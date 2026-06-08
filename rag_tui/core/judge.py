"""LLM-as-judge evaluation for RAG retrieval quality.

Uses any provider's existing generate() method to produce real relevance
and faithfulness scores. Works fully locally with Ollama -- no data leaves
the machine. Other tools (Ragas, TruLens, DeepEval) require OpenAI for this.

Eval modes ranked by signal quality:
  JUDGE       -- LLM scores each chunk per query (most accurate)
  GROUND_TRUTH -- dataset provides the correct chunk (binary, no LLM needed)
  SIMILARITY  -- cosine similarity >= threshold (weakest, current default)
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from rag_tui.core.providers import LLMProvider


EVAL_MODE_JUDGE = "judge"
EVAL_MODE_GROUND_TRUTH = "ground_truth"
EVAL_MODE_SIMILARITY = "similarity"

_RELEVANCE_PROMPT = """\
Rate how relevant the following passage is to the question.

Question: {query}

Passage: {chunk}

Reply with a single decimal number from 0.0 to 1.0 and nothing else.
0.0 = completely irrelevant
0.5 = partially related but does not directly address the question
1.0 = directly answers or strongly supports answering the question

Number:"""

_FAITHFULNESS_PROMPT = """\
Can the following question be answered using only the provided context passages?

Question: {query}

Context:
{context}

Reply with a single decimal number from 0.0 to 1.0 and nothing else.
0.0 = the context does not contain the information needed
0.5 = the context has partial information
1.0 = the context fully contains what is needed to answer

Number:"""


def _parse_score(text: str, default: float = 0.5) -> float:
    m = re.search(r"\b(1\.0|0?\.\d+|[01])\b", text.strip())
    if not m:
        return default
    return max(0.0, min(1.0, float(m.group(1))))


class LLMJudge:
    """Score chunk relevance and context faithfulness using an LLM provider.

    The provider must implement generate(prompt: str) -> str (all built-in
    providers already do). Ollama is the recommended default for privacy.
    """

    def __init__(self, provider: "LLMProvider", max_concurrent: int = 4) -> None:
        self._provider = provider
        self._sem = asyncio.Semaphore(max_concurrent)

    async def score_relevance(self, query: str, chunk: str) -> float:
        """Return 0-1: how relevant is this chunk to the query?"""
        prompt = _RELEVANCE_PROMPT.format(query=query, chunk=chunk[:1500])
        try:
            async with self._sem:
                text = await self._provider.generate(prompt)
            return _parse_score(text)
        except Exception:
            return 0.5

    async def score_relevance_batch(
        self, query: str, chunks: List[str]
    ) -> List[float]:
        """Score every chunk for one query in parallel (semaphore-bounded)."""
        tasks = [self.score_relevance(query, c) for c in chunks]
        return list(await asyncio.gather(*tasks))

    async def score_faithfulness(self, query: str, chunks: List[str]) -> float:
        """Return 0-1: can this query be answered from these chunks?"""
        context = "\n\n".join(f"[{i + 1}] {c[:800]}" for i, c in enumerate(chunks))
        prompt = _FAITHFULNESS_PROMPT.format(query=query, context=context)
        try:
            async with self._sem:
                text = await self._provider.generate(prompt)
            return _parse_score(text)
        except Exception:
            return 0.5

    async def evaluate_query(
        self, query: str, chunks: List[str]
    ) -> tuple[List[float], float]:
        """Return (relevance_scores, faithfulness) for a query + chunk list."""
        relevance, faithfulness = await asyncio.gather(
            self.score_relevance_batch(query, chunks),
            self.score_faithfulness(query, chunks),
        )
        return relevance, faithfulness


def ground_truth_labels(
    chunks_retrieved: List[tuple[str, float]],
    relevant_chunk: Optional[str],
) -> Optional[List[float]]:
    """Build binary relevance labels from a ground-truth relevant_chunk string.

    Returns None when no ground truth is available (falls back to similarity).
    Matching is substring-based: a retrieved chunk is relevant if it contains
    at least 60% of the ground truth text (handles truncation gracefully).
    """
    if not relevant_chunk or not relevant_chunk.strip():
        return None
    gt = relevant_chunk.strip().lower()
    threshold = max(1, int(len(gt) * 0.6))
    labels = []
    for chunk_text, _ in chunks_retrieved:
        ct = chunk_text.lower()
        overlap = sum(1 for i in range(0, len(gt), 5) if gt[i:i + 20] in ct)
        score = min(1.0, (overlap * 20) / max(1, len(gt)))
        labels.append(score)
    return labels
