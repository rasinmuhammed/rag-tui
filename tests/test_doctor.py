"""Unit tests for the doctor module: query-free retrievability analysis.

Embeddings are constructed by hand so every detector can be exercised
deterministically without touching a provider.
"""

import numpy as np
import pytest

from rag_tui.core.doctor import (
    DoctorReport,
    Finding,
    _k_occurrence,
    _similarity_matrix,
    _skewness,
    analyze,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _healthy_embeddings(n: int, dim: int = 64, seed: int = 7) -> np.ndarray:
    """n well-separated unit vectors, i.e. a healthy corpus.

    Random vectors in high dimensions are near-orthogonal (peak off-diagonal
    similarity here is ~0.4), which is what a well-chunked document looks
    like. Seeded so the detectors are deterministic.
    """
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, dim)).astype(np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def _sentences(n: int) -> list:
    """Well-formed chunks: long enough, varied, ending in full stops."""
    vocabulary = [
        "retrieval augmented generation pipelines depend on careful document preparation",
        "vector similarity search returns candidates ranked by cosine distance metrics",
        "embedding models project natural language into dense numerical coordinates",
        "chunk boundaries determine which facts survive intact inside a single unit",
        "evaluation harnesses measure whether relevant passages reach the context window",
    ]
    return [f"Chunk {i}: {vocabulary[i % len(vocabulary)]} number {i}." for i in range(n)]


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

def test_similarity_matrix_is_symmetric_with_unit_diagonal():
    embeddings = _healthy_embeddings(6)
    sim = _similarity_matrix(embeddings)

    assert sim.shape == (6, 6)
    assert np.allclose(np.diag(sim), 1.0, atol=1e-5)
    assert np.allclose(sim, sim.T, atol=1e-5)
    assert sim.max() <= 1.0 and sim.min() >= -1.0


def test_similarity_matrix_handles_zero_vectors():
    embeddings = np.zeros((3, 8), dtype=np.float32)
    sim = _similarity_matrix(embeddings)

    assert not np.isnan(sim).any()


def test_k_occurrence_sums_to_n_times_k():
    embeddings = _healthy_embeddings(12)
    sim = _similarity_matrix(embeddings)
    occurrence, neighbors = _k_occurrence(sim, k=3)

    assert neighbors.shape == (12, 3)
    # Every chunk contributes exactly k votes.
    assert occurrence.sum() == 12 * 3


def test_k_occurrence_excludes_self():
    embeddings = _healthy_embeddings(10)
    sim = _similarity_matrix(embeddings)
    _, neighbors = _k_occurrence(sim, k=4)

    for i, row in enumerate(neighbors):
        assert i not in row


def test_k_occurrence_clamps_k_above_corpus_size():
    embeddings = _healthy_embeddings(3)
    sim = _similarity_matrix(embeddings)
    occurrence, neighbors = _k_occurrence(sim, k=50)

    assert neighbors.shape[1] == 2  # n - 1


def test_skewness_zero_for_uniform_distribution():
    assert _skewness(np.array([4.0, 4.0, 4.0, 4.0])) == 0.0


def test_skewness_positive_when_one_value_dominates():
    skew = _skewness(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 20.0]))
    assert skew > 1.0


# ---------------------------------------------------------------------------
# Healthy corpus produces no noise
# ---------------------------------------------------------------------------

def test_healthy_corpus_scores_high_and_reports_nothing():
    chunks = _sentences(20)
    embeddings = _healthy_embeddings(20)

    report = analyze(chunks, embeddings)

    assert isinstance(report, DoctorReport)
    assert report.findings == []
    assert report.retrievability_score >= 90
    assert report.grade == "A"


def test_empty_corpus_returns_empty_report():
    report = analyze([], np.zeros((0, 8)))

    assert report.total_chunks == 0
    assert report.findings == []
    assert report.grade == "F"


def test_mismatched_embedding_count_raises():
    with pytest.raises(ValueError, match="must match"):
        analyze(_sentences(3), _healthy_embeddings(5))


def test_single_chunk_does_not_crash():
    report = analyze(["A single well formed chunk of reasonable length goes here."],
                     _healthy_embeddings(1))

    assert report.total_chunks == 1


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def _kinds(report: DoctorReport) -> set:
    return {f.kind for f in report.findings}


def test_detects_identical_chunks_as_duplicates():
    chunks = _sentences(10)
    embeddings = _healthy_embeddings(10)
    # Make chunks 3 and 7 identical vectors.
    embeddings[7] = embeddings[3]

    report = analyze(chunks, embeddings)
    duplicates = [f for f in report.findings if f.kind == "duplicate"]

    assert len(duplicates) == 1
    assert set(duplicates[0].chunk_indices) == {3, 7}
    assert duplicates[0].stats["peak_similarity"] == pytest.approx(1.0, abs=1e-3)


def test_duplicate_cluster_groups_transitively():
    chunks = _sentences(10)
    embeddings = _healthy_embeddings(10)
    embeddings[4] = embeddings[2]
    embeddings[6] = embeddings[2]

    report = analyze(chunks, embeddings)
    duplicates = [f for f in report.findings if f.kind == "duplicate"]

    assert len(duplicates) == 1
    assert set(duplicates[0].chunk_indices) == {2, 4, 6}
    assert duplicates[0].stats["cluster_size"] == 3


def test_duplicate_threshold_is_respected():
    chunks = _sentences(10)
    embeddings = _healthy_embeddings(10)
    embeddings[7] = embeddings[3]

    loose = analyze(chunks, embeddings, duplicate_threshold=0.99)
    strict = analyze(chunks, embeddings, duplicate_threshold=1.01)

    assert "duplicate" in _kinds(loose)
    assert "duplicate" not in _kinds(strict)


# ---------------------------------------------------------------------------
# Hub detection
# ---------------------------------------------------------------------------

def test_detects_hub_chunk():
    n = 30
    chunks = _sentences(n)
    embeddings = _healthy_embeddings(n)
    # Chunk 0 becomes the centroid of everything: maximally hubby.
    embeddings[0] = embeddings.mean(axis=0)
    embeddings[0] /= np.linalg.norm(embeddings[0])

    report = analyze(chunks, embeddings)
    hubs = [f for f in report.findings if f.kind == "hub"]

    assert hubs, "centroid chunk should be flagged as a hub"
    assert hubs[0].chunk_indices == [0]
    assert hubs[0].stats["k_occurrence"] > 5
    assert report.hubness_skew > 0


def test_hub_check_skipped_on_tiny_corpus():
    chunks = _sentences(6)
    embeddings = _healthy_embeddings(6)
    embeddings[0] = embeddings.mean(axis=0)

    report = analyze(chunks, embeddings)

    assert "hub" not in _kinds(report)


# ---------------------------------------------------------------------------
# Orphan detection
# ---------------------------------------------------------------------------

def _shadowed_corpus():
    """Two tight clusters plus a satellite that no neighbourhood contains.

    A symmetric near-duplicate pair is *not* shadowed, because the two sit in
    each other's top-k, so both stay reachable. Shadowing needs asymmetry:
    clusters dense enough to fill their own top-k, and an outlier hanging off
    one of them that is therefore never anyone's neighbour.
    """
    n = 11
    embeddings = _healthy_embeddings(n)
    cluster_a, cluster_b = embeddings[0].copy(), embeddings[5].copy()

    for i in range(1, 5):
        vector = cluster_a + 0.02 * embeddings[i]
        embeddings[i] = vector / np.linalg.norm(vector)
    for i in range(6, 10):
        vector = cluster_b + 0.02 * embeddings[i]
        embeddings[i] = vector / np.linalg.norm(vector)

    satellite = cluster_a + 0.45 * embeddings[10]
    embeddings[10] = satellite / np.linalg.norm(satellite)

    return _sentences(n), embeddings


def test_shadowed_chunk_is_reported_with_its_shadower():
    chunks, embeddings = _shadowed_corpus()

    report = analyze(chunks, embeddings, neighbors=3)
    orphans = [f for f in report.findings if f.kind == "orphan"]

    assert len(orphans) == 1
    finding = orphans[0]
    assert finding.chunk_indices == [10]
    # Nearest neighbour is well above the shadow threshold, so the report
    # should name the chunk that is outranking it rather than call it isolated.
    assert finding.stats["nearest_similarity"] >= 0.88
    assert finding.stats["nearest_chunk"] in range(5)
    assert f"#{finding.stats['nearest_chunk']}" in finding.detail
    assert "wins every ranking" in finding.detail


def test_isolated_chunk_is_reported_as_a_content_island():
    chunks, embeddings = _shadowed_corpus()
    # Push the satellite far away from everything instead of near cluster A.
    embeddings[10] = _healthy_embeddings(1, seed=99)[0]

    report = analyze(chunks, embeddings, neighbors=3)
    orphans = [f for f in report.findings if f.kind == "orphan"]

    assert len(orphans) == 1
    finding = orphans[0]
    assert finding.chunk_indices == [10]
    assert finding.stats["nearest_similarity"] < 0.35
    assert "semantically adjacent" in finding.detail


def test_orphan_check_skipped_on_tiny_corpus():
    report = analyze(_sentences(4), _healthy_embeddings(4))

    assert "orphan" not in _kinds(report)


# ---------------------------------------------------------------------------
# Fracture detection
# ---------------------------------------------------------------------------

def test_detects_sentence_split_across_boundary():
    chunks = [
        "The retrieval system indexes every document and then",
        "ranks them by cosine similarity against the query.",
    ]
    report = analyze(chunks, _healthy_embeddings(2))
    fractures = [f for f in report.findings if f.kind == "fracture"]

    assert len(fractures) == 1
    assert fractures[0].chunk_indices == [0, 1]


def test_clean_sentence_boundary_is_not_flagged():
    chunks = [
        "The retrieval system indexes every document it receives.",
        "Ranking happens afterwards using cosine similarity scores.",
    ]
    report = analyze(chunks, _healthy_embeddings(2))

    assert "fracture" not in _kinds(report)


def test_heading_boundary_is_not_flagged_as_fracture():
    # A heading legitimately lacks terminal punctuation; the next chunk
    # starting uppercase means this is a real boundary, not a split sentence.
    chunks = [
        "Chapter Four: Retrieval Strategies",
        "Dense retrieval outperforms sparse methods on paraphrased queries.",
    ]
    report = analyze(chunks, _healthy_embeddings(2))

    assert "fracture" not in _kinds(report)


# ---------------------------------------------------------------------------
# Boilerplate detection
# ---------------------------------------------------------------------------

def _boilerplate(report: DoctorReport) -> list:
    return [f for f in report.findings if f.kind == "boilerplate"]


def test_detects_short_chunk():
    chunks = _sentences(8)
    chunks[5] = "Page 12"
    report = analyze(chunks, _healthy_embeddings(8))
    flagged = _boilerplate(report)

    assert len(flagged) == 1
    assert flagged[0].chunk_indices == [5]
    assert flagged[0].stats["min_chars"] == len("Page 12")


def test_short_chunks_are_grouped_into_one_finding():
    # Five stray headings are one systemic problem, not five separate ones.
    chunks = _sentences(12)
    for i in (0, 3, 5, 8, 11):
        chunks[i] = f"## Section {i}"

    report = analyze(chunks, _healthy_embeddings(12))
    flagged = _boilerplate(report)

    assert len(flagged) == 1
    assert flagged[0].chunk_indices == [0, 3, 5, 8, 11]
    assert flagged[0].stats["count"] == 5
    assert len(flagged[0].stats["chunks"]) == 5
    # Only the first few previews are inlined, the rest are summarised.
    assert "+1 more" in flagged[0].detail


def test_detects_repetitive_chunk():
    chunks = _sentences(8)
    chunks[3] = " ".join(["next page previous page"] * 12) + "."
    report = analyze(chunks, _healthy_embeddings(8))
    flagged = _boilerplate(report)

    assert len(flagged) == 1
    assert flagged[0].chunk_indices == [3]
    assert flagged[0].stats["min_diversity"] < 0.35


def test_short_and_repetitive_are_separate_findings():
    chunks = _sentences(10)
    chunks[2] = "p. 7"
    chunks[6] = " ".join(["next page previous page"] * 12) + "."

    report = analyze(chunks, _healthy_embeddings(10))
    flagged = _boilerplate(report)

    assert len(flagged) == 2
    assert {tuple(f.chunk_indices) for f in flagged} == {(2,), (6,)}


def test_short_chunk_threshold_is_configurable():
    chunks = _sentences(8)
    chunks[5] = "Page 12"

    lenient = analyze(chunks, _healthy_embeddings(8), boilerplate_chars=3)

    assert not _boilerplate(lenient)


# ---------------------------------------------------------------------------
# Scoring and serialisation
# ---------------------------------------------------------------------------

def test_defects_lower_the_score():
    chunks = _sentences(20)
    clean = analyze(chunks, _healthy_embeddings(20))

    broken_embeddings = _healthy_embeddings(20)
    for i in (5, 9, 13):
        broken_embeddings[i] = broken_embeddings[1]
    broken = analyze(chunks, broken_embeddings)

    assert broken.retrievability_score < clean.retrievability_score


def test_score_is_bounded():
    chunks = ["x"] * 40
    embeddings = np.ones((40, 8), dtype=np.float32)
    report = analyze(chunks, embeddings)

    assert 0.0 <= report.retrievability_score <= 100.0


def test_findings_sorted_by_severity():
    n = 30
    chunks = _sentences(n)
    chunks[11] = "p. 4"
    embeddings = _healthy_embeddings(n)
    embeddings[0] = embeddings.mean(axis=0)
    embeddings[0] /= np.linalg.norm(embeddings[0])

    report = analyze(chunks, embeddings)
    ranks = [{"critical": 0, "warning": 1, "info": 2}[f.severity] for f in report.findings]

    assert ranks == sorted(ranks)


def test_report_to_dict_is_json_serialisable():
    import json

    chunks = _sentences(15)
    embeddings = _healthy_embeddings(15)
    embeddings[4] = embeddings[1]

    report = analyze(chunks, embeddings, strategy="sentence", chunk_size=256)
    payload = report.to_dict()
    round_tripped = json.loads(json.dumps(payload))

    assert round_tripped["config"]["strategy"] == "sentence"
    assert round_tripped["config"]["chunk_size"] == 256
    assert round_tripped["total_chunks"] == 15
    assert "retrievability_score" in round_tripped
    assert isinstance(round_tripped["findings"], list)
    assert round_tripped["counts"]["duplicate"] == 1


def test_summary_mentions_score_and_grade():
    report = analyze(_sentences(20), _healthy_embeddings(20))
    summary = report.summary()

    assert "Retrievability" in summary
    assert report.grade in summary


def test_counts_match_findings():
    chunks = _sentences(20)
    chunks[7] = "n/a"
    embeddings = _healthy_embeddings(20)
    embeddings[12] = embeddings[3]

    report = analyze(chunks, embeddings)
    recomputed = {}
    for finding in report.findings:
        recomputed[finding.kind] = recomputed.get(finding.kind, 0) + 1

    assert report.counts == recomputed
