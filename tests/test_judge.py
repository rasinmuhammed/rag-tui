"""Tests for LLM-as-judge evaluation module."""
import asyncio
import pytest
from rag_tui.core.judge import (
    LLMJudge, ground_truth_labels, _parse_score,
    EVAL_MODE_JUDGE, EVAL_MODE_GROUND_TRUTH, EVAL_MODE_SIMILARITY,
)
from rag_tui.core.metrics import QueryResult, calculate_batch_metrics


class MockLLM:
    """Returns a fixed score string for all prompts."""
    def __init__(self, score: float = 0.8):
        self.score = score
        self.calls: list = []

    async def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return str(self.score)


# ---------------------------------------------------------------------------
# _parse_score
# ---------------------------------------------------------------------------

def test_parse_score_decimal():
    assert _parse_score("0.8") == pytest.approx(0.8)

def test_parse_score_one():
    assert _parse_score("1.0") == pytest.approx(1.0)

def test_parse_score_zero():
    assert _parse_score("0.0") == pytest.approx(0.0)

def test_parse_score_with_noise():
    assert _parse_score("The answer is 0.7 out of 1.0") == pytest.approx(0.7)

def test_parse_score_clamps_above_one():
    assert _parse_score("1.5") == pytest.approx(1.0)

def test_parse_score_default_on_garbage():
    assert _parse_score("sorry I cannot rate this") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------

def test_judge_score_relevance():
    llm = MockLLM(score=0.9)
    judge = LLMJudge(llm)
    score = asyncio.run(judge.score_relevance("What is RAG?", "RAG stands for Retrieval-Augmented Generation."))
    assert score == pytest.approx(0.9)
    assert len(llm.calls) == 1
    assert "What is RAG?" in llm.calls[0]

def test_judge_score_relevance_batch():
    llm = MockLLM(score=0.7)
    judge = LLMJudge(llm)
    scores = asyncio.run(judge.score_relevance_batch("What is RAG?", ["chunk A", "chunk B", "chunk C"]))
    assert len(scores) == 3
    assert all(s == pytest.approx(0.7) for s in scores)

def test_judge_faithfulness():
    llm = MockLLM(score=0.6)
    judge = LLMJudge(llm)
    score = asyncio.run(judge.score_faithfulness("What is RAG?", ["RAG is a technique...", "It uses vector search..."]))
    assert score == pytest.approx(0.6)
    assert "What is RAG?" in llm.calls[0]

def test_judge_evaluate_query_returns_labels_and_faithfulness():
    llm = MockLLM(score=0.85)
    judge = LLMJudge(llm)
    chunks = ["chunk one", "chunk two"]
    labels, faith = asyncio.run(judge.evaluate_query("query", chunks))
    assert len(labels) == 2
    assert faith == pytest.approx(0.85)

def test_judge_graceful_on_provider_error():
    class FailingLLM:
        async def generate(self, prompt):
            raise RuntimeError("connection refused")

    judge = LLMJudge(FailingLLM())
    score = asyncio.run(judge.score_relevance("q", "chunk"))
    assert score == pytest.approx(0.5)  # default on error


# ---------------------------------------------------------------------------
# ground_truth_labels
# ---------------------------------------------------------------------------

def test_ground_truth_labels_matching():
    chunks = [
        ("RAG stands for Retrieval-Augmented Generation.", 0.9),
        ("Python is a programming language.", 0.4),
    ]
    labels = ground_truth_labels(chunks, "RAG stands for Retrieval-Augmented Generation.")
    assert labels is not None
    assert labels[0] > labels[1]

def test_ground_truth_labels_none_when_no_gt():
    chunks = [("some chunk", 0.8)]
    assert ground_truth_labels(chunks, None) is None
    assert ground_truth_labels(chunks, "") is None
    assert ground_truth_labels(chunks, "   ") is None


# ---------------------------------------------------------------------------
# Metrics integration with labels
# ---------------------------------------------------------------------------

def test_metrics_use_labels_not_cosine():
    # Two chunks, both with low cosine similarity (0.2)
    # but labels say first chunk is highly relevant (0.9)
    results = [
        QueryResult(
            query="test",
            chunks_retrieved=[("relevant chunk", 0.2), ("irrelevant", 0.1)],
            top_score=0.2,
            avg_score=0.15,
            relevance_labels=[0.9, 0.1],
        )
    ]
    batch = calculate_batch_metrics(results, threshold=0.5)
    # With labels, MRR should be 1.0 (relevant chunk is rank 1)
    assert batch.mrr == pytest.approx(1.0)
    assert batch.hit_rate == pytest.approx(1.0)
    assert batch.eval_mode == EVAL_MODE_JUDGE

def test_metrics_fall_back_to_cosine_without_labels():
    results = [
        QueryResult(
            query="test",
            chunks_retrieved=[("chunk", 0.2)],
            top_score=0.2,
            avg_score=0.2,
        )
    ]
    batch = calculate_batch_metrics(results, threshold=0.5)
    # 0.2 < 0.5 threshold → not a hit
    assert batch.hit_rate == pytest.approx(0.0)
    assert batch.eval_mode == EVAL_MODE_SIMILARITY

def test_metrics_eval_mode_judge_when_all_labeled():
    results = [
        QueryResult("q1", [("c", 0.3)], 0.3, 0.3, relevance_labels=[0.8]),
        QueryResult("q2", [("c", 0.3)], 0.3, 0.3, relevance_labels=[0.2]),
    ]
    batch = calculate_batch_metrics(results)
    assert batch.eval_mode == EVAL_MODE_JUDGE

def test_faithfulness_in_batch():
    results = [
        QueryResult("q", [("c", 0.8)], 0.8, 0.8, faithfulness=0.9),
        QueryResult("q2", [("c2", 0.7)], 0.7, 0.7, faithfulness=0.6),
    ]
    batch = calculate_batch_metrics(results)
    assert batch.avg_faithfulness == pytest.approx(0.75)
