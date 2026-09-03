"""Query-free retrievability analysis for RAG corpora.

Every other RAG evaluation approach needs a labelled query set before it can
tell you anything. This module needs none. It embeds the chunks once and then
reads the *geometry* of the resulting vector space to find structural defects
that will hurt retrieval no matter what queries eventually arrive.

Diagnostics:
    hub           - chunk retrieved for far too many neighbourhoods
    orphan        - chunk no other chunk retrieves; unreachable or shadowed
    duplicate     - near-identical chunks splitting each other's ranking mass
    fracture      - chunk boundary that cuts a sentence in half
    boilerplate   - low-information filler occupying index space

Findings are risk signals derived from the embedding space, not proofs that a
given query will fail. They point at the chunks worth looking at first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Severity ordering, most severe first. Used for sorting and display.
SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}

# Defaults chosen to be quiet on healthy corpora: every threshold here was set
# so that a well-chunked document produces zero findings.
DEFAULT_NEIGHBORS = 5
DEFAULT_DUPLICATE_THRESHOLD = 0.95
DEFAULT_SHADOW_THRESHOLD = 0.88
DEFAULT_ISLAND_THRESHOLD = 0.35
DEFAULT_BOILERPLATE_CHARS = 80
DEFAULT_DIVERSITY_THRESHOLD = 0.35

_SENTENCE_END = re.compile(r"[.!?][\"')\]]*$")
_WORD = re.compile(r"[A-Za-z0-9']+")


@dataclass
class Finding:
    """A single defect detected in the corpus."""

    kind: str
    severity: str
    chunk_indices: List[int]
    message: str
    detail: str = ""
    suggestion: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "chunk_indices": self.chunk_indices,
            "message": self.message,
            "detail": self.detail,
            "suggestion": self.suggestion,
            "stats": self.stats,
        }


@dataclass
class DoctorReport:
    """Full retrievability report for one chunking configuration."""

    timestamp: str
    total_chunks: int
    strategy: str
    chunk_size: int
    overlap_percent: int
    neighbors: int
    retrievability_score: float
    grade: str
    hubness_skew: float
    findings: List[Finding] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    provider: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "total_chunks": self.total_chunks,
            "config": {
                "strategy": self.strategy,
                "chunk_size": self.chunk_size,
                "overlap_percent": self.overlap_percent,
                "neighbors": self.neighbors,
                "provider": self.provider,
            },
            "retrievability_score": round(self.retrievability_score, 1),
            "grade": self.grade,
            "hubness_skew": round(self.hubness_skew, 3),
            "counts": self.counts,
            "findings": [f.to_dict() for f in self.findings],
        }

    def summary(self) -> str:
        """One-line human summary."""
        if not self.findings:
            return (
                f"Retrievability {self.retrievability_score:.0f}/100 ({self.grade}). "
                f"No structural defects found across {self.total_chunks} chunks."
            )
        parts = [f"{n} {kind}" for kind, n in sorted(self.counts.items()) if n]
        return (
            f"Retrievability {self.retrievability_score:.0f}/100 ({self.grade}). "
            f"Found {', '.join(parts)} across {self.total_chunks} chunks."
        )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalise rows so the dot product equals cosine similarity."""
    matrix = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return the full chunk-to-chunk cosine similarity matrix."""
    normalized = _normalize(embeddings)
    sim = normalized @ normalized.T
    return np.clip(sim, -1.0, 1.0)


def _k_occurrence(sim: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Count how often each chunk lands in another chunk's top-k neighbours.

    This is the k-occurrence N_k from the hubness literature. In high
    dimensions its distribution becomes right-skewed: a few "hub" vectors sit
    in a disproportionate number of neighbourhoods and get retrieved for
    queries they have nothing to do with.

    Args:
        sim: Square cosine similarity matrix.
        k: Neighbourhood size.

    Returns:
        Tuple of (k_occurrence per chunk, top-k neighbour indices per chunk).
    """
    n = sim.shape[0]
    masked = sim.copy()
    np.fill_diagonal(masked, -np.inf)

    k = max(1, min(k, n - 1))
    # argpartition puts the k largest in the first k slots, then we order them.
    partitioned = np.argpartition(-masked, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    order = np.argsort(-masked[rows, partitioned], axis=1)
    neighbors = partitioned[rows, order]

    occurrence = np.bincount(neighbors.ravel(), minlength=n)
    return occurrence, neighbors


def _skewness(values: np.ndarray) -> float:
    """Standardised third moment. >1 indicates meaningful hubness."""
    if values.size < 3:
        return 0.0
    mean = float(values.mean())
    std = float(values.std())
    if std == 0:
        return 0.0
    return float(((values - mean) ** 3).mean() / (std ** 3))


class _UnionFind:
    """Minimal union-find, used to group duplicate chunks into clusters."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _preview(text: str, width: int = 60) -> str:
    """Collapse a chunk to a single short line for display."""
    flat = " ".join(text.split())
    return flat[:width] + ("…" if len(flat) > width else "")


def _index_list(indices: Sequence[int], limit: int = 8) -> str:
    """Render chunk indices as '#1, #2, #3 (+4 more)'."""
    shown = ", ".join(f"#{i}" for i in indices[:limit])
    extra = len(indices) - limit
    return f"{shown} (+{extra} more)" if extra > 0 else shown


def _check_hubs(
    chunks: Sequence[str],
    occurrence: np.ndarray,
    k: int,
) -> List[Finding]:
    """Flag chunks sitting in far more neighbourhoods than their share."""
    n = len(chunks)
    if n < 10:
        return []

    mean = float(occurrence.mean())
    std = float(occurrence.std())
    if std == 0:
        return []

    # Two conditions must both hold: statistically extreme (2 sigma) and at
    # least 3x the average neighbourhood count. Hubness is intrinsic to
    # high-dimensional space, so a clean corpus still has mild 2 sigma
    # outliers; the absolute floor keeps those out of the report.
    cutoff = max(mean + 2.0 * std, 3.0 * k)
    findings: List[Finding] = []

    for idx in np.argsort(-occurrence):
        count = int(occurrence[idx])
        if count < cutoff:
            break
        z = (count - mean) / std
        findings.append(
            Finding(
                kind="hub",
                severity="critical" if z >= 3.0 else "warning",
                chunk_indices=[int(idx)],
                message=f"Chunk #{idx} is a retrieval hub ({count} neighbourhoods, {z:.1f}σ above mean)",
                detail=(
                    f"\"{_preview(chunks[idx])}\" sits in the top-{k} neighbourhood of "
                    f"{count} of {n} chunks (average is {mean:.1f}). Vectors like this "
                    f"get returned for queries they are unrelated to, pushing the "
                    f"correct chunk out of top-k."
                ),
                suggestion=(
                    "Usually generic filler: a heading stack, boilerplate footer, or "
                    "an overly broad summary paragraph. Remove it from the corpus, or "
                    "split it so each piece carries specific content."
                ),
                stats={"k_occurrence": count, "z_score": round(float(z), 2)},
            )
        )

    return findings


def _check_orphans(
    chunks: Sequence[str],
    sim: np.ndarray,
    occurrence: np.ndarray,
    shadow_threshold: float,
    island_threshold: float,
) -> List[Finding]:
    """Flag chunks that no other chunk retrieves as a neighbour.

    Zero k-occurrence has two very different causes, and the fix differs, so
    the report distinguishes them by looking at the chunk's nearest neighbour:
    a high-similarity neighbour means the chunk is being *shadowed* by a
    near-twin; a low-similarity one means it is a content island.
    """
    n = len(chunks)
    if n < 5:
        return []

    masked = sim.copy()
    np.fill_diagonal(masked, -np.inf)

    findings: List[Finding] = []
    for idx in np.where(occurrence == 0)[0]:
        nearest = int(np.argmax(masked[idx]))
        nearest_sim = float(masked[idx][nearest])

        if nearest_sim >= shadow_threshold:
            detail = (
                f"\"{_preview(chunks[idx])}\". Nearest neighbour is chunk #{nearest} "
                f"at {nearest_sim:.2f} similarity. The two are close enough that #{nearest} "
                f"wins every ranking they both compete in, so this chunk is effectively "
                f"dead weight in the index."
            )
            suggestion = (
                f"Merge with chunk #{nearest}, or differentiate them. If both must "
                f"stay, they need distinct content, not overlapping restatements."
            )
            severity = "warning"
        elif nearest_sim < island_threshold:
            detail = (
                f"\"{_preview(chunks[idx])}\". Nearest neighbour is only {nearest_sim:.2f} "
                f"similar. Nothing in the corpus is semantically adjacent to it, so only "
                f"a query that nearly quotes it will pull it back."
            )
            suggestion = (
                "Check whether this is genuinely unique content (fine, but it needs "
                "targeted queries) or extraction noise: mangled tables, OCR artefacts, "
                "and stray markup all land here."
            )
            severity = "info"
        else:
            detail = (
                f"\"{_preview(chunks[idx])}\". Never appears in another chunk's "
                f"neighbourhood (nearest is #{nearest} at {nearest_sim:.2f})."
            )
            suggestion = (
                "Low reachability. Verify a representative query actually retrieves it."
            )
            severity = "info"

        findings.append(
            Finding(
                kind="orphan",
                severity=severity,
                chunk_indices=[int(idx)],
                message=f"Chunk #{idx} is unreachable from any neighbourhood",
                detail=detail,
                suggestion=suggestion,
                stats={
                    "nearest_chunk": nearest,
                    "nearest_similarity": round(nearest_sim, 4),
                },
            )
        )

    return findings


def _check_duplicates(
    chunks: Sequence[str],
    sim: np.ndarray,
    threshold: float,
) -> List[Finding]:
    """Group near-identical chunks that split each other's ranking mass."""
    n = len(chunks)
    if n < 2:
        return []

    upper = np.triu(sim, k=1)
    pairs = np.argwhere(upper >= threshold)
    if pairs.size == 0:
        return []

    uf = _UnionFind(n)
    for a, b in pairs:
        uf.union(int(a), int(b))

    clusters: Dict[int, List[int]] = {}
    for a, b in pairs:
        root = uf.find(int(a))
        members = clusters.setdefault(root, [])
        for node in (int(a), int(b)):
            if node not in members:
                members.append(node)

    findings: List[Finding] = []
    for members in clusters.values():
        members.sort()
        sims = [float(sim[i][j]) for i in members for j in members if i < j]
        peak = max(sims) if sims else float(threshold)
        listed = ", ".join(f"#{m}" for m in members[:8])
        if len(members) > 8:
            listed += f" (+{len(members) - 8} more)"

        findings.append(
            Finding(
                kind="duplicate",
                severity="warning" if len(members) > 2 else "info",
                chunk_indices=members,
                message=f"{len(members)} near-duplicate chunks ({listed})",
                detail=(
                    f"Peak similarity {peak:.3f}. \"{_preview(chunks[members[0]])}\" "
                    f"Duplicates compete for the same queries: each takes a slot in "
                    f"top-k, so a single query can burn its whole context budget on "
                    f"restatements of one fact while genuinely relevant chunks are cut."
                ),
                suggestion=(
                    "Deduplicate before indexing. If the repetition comes from overlap, "
                    "lower --overlap-percent; if it is in the source document, strip it "
                    "in the cleaning step."
                ),
                stats={"cluster_size": len(members), "peak_similarity": round(peak, 4)},
            )
        )

    return findings


def _check_fractures(chunks: Sequence[str]) -> List[Finding]:
    """Flag boundaries that cut a sentence in half.

    A chunk ending mid-sentence loses the second half of whatever claim it was
    making, and the chunk after it starts with a dangling fragment. Neither
    side embeds cleanly.
    """
    findings: List[Finding] = []

    for i in range(len(chunks) - 1):
        current = chunks[i].rstrip()
        following = chunks[i + 1].lstrip()
        if not current or not following:
            continue

        ends_mid_sentence = not _SENTENCE_END.search(current)
        starts_lowercase = following[0].islower()

        # Both signals together to keep the check quiet: lists, headings and
        # code blocks legitimately end without terminal punctuation.
        if ends_mid_sentence and starts_lowercase:
            tail = " ".join(current.split())[-40:]
            head = " ".join(following.split())[:40]
            findings.append(
                Finding(
                    kind="fracture",
                    severity="warning",
                    chunk_indices=[i, i + 1],
                    message=f"Boundary between #{i} and #{i + 1} splits a sentence",
                    detail=f"…{tail} ⁞ {head}…",
                    suggestion=(
                        "Switch to the 'sentence' or 'recursive' strategy, or raise "
                        "overlap so the full sentence survives in one of the two chunks."
                    ),
                    stats={"left": i, "right": i + 1},
                )
            )

    return findings


def _check_boilerplate(
    chunks: Sequence[str],
    min_chars: int,
    diversity_threshold: float,
) -> List[Finding]:
    """Flag chunks too short or too repetitive to carry retrievable meaning.

    Both problems are usually systemic (a heading level that got its own
    chunk, a footer repeated on every page), so each class is reported once
    with every affected chunk listed, rather than as dozens of near-identical
    findings the reader has to scroll past.
    """
    short: List[Tuple[int, int]] = []
    repetitive: List[Tuple[int, float]] = []

    for i, text in enumerate(chunks):
        stripped = text.strip()
        if not stripped:
            continue

        if len(stripped) < min_chars:
            short.append((i, len(stripped)))
            continue

        words = _WORD.findall(stripped.lower())
        if len(words) >= 20:
            diversity = len(set(words)) / len(words)
            if diversity < diversity_threshold:
                repetitive.append((i, diversity))

    findings: List[Finding] = []

    if short:
        indices = [i for i, _ in short]
        samples = "; ".join(f'#{i} "{_preview(chunks[i], 32)}"' for i, _ in short[:4])
        if len(short) > 4:
            samples += f"; +{len(short) - 4} more"
        findings.append(
            Finding(
                kind="boilerplate",
                severity="info",
                chunk_indices=indices,
                message=(
                    f"{len(short)} chunk{'s' if len(short) > 1 else ''} too short to be "
                    f"retrievable ({_index_list(indices)})"
                ),
                detail=f"Under {min_chars} characters. {samples}",
                suggestion=(
                    "Short fragments embed poorly and rarely rank. This usually means a "
                    "heading level is being split into its own chunk: merge them into "
                    "the section body, or drop them during cleaning."
                ),
                stats={
                    "count": len(short),
                    "min_chars": min(c for _, c in short),
                    "chunks": [{"chunk": i, "chars": c} for i, c in short],
                },
            )
        )

    if repetitive:
        indices = [i for i, _ in repetitive]
        samples = "; ".join(f'#{i} "{_preview(chunks[i], 32)}"' for i, _ in repetitive[:4])
        if len(repetitive) > 4:
            samples += f"; +{len(repetitive) - 4} more"
        findings.append(
            Finding(
                kind="boilerplate",
                severity="info",
                chunk_indices=indices,
                message=(
                    f"{len(repetitive)} highly repetitive "
                    f"chunk{'s' if len(repetitive) > 1 else ''} ({_index_list(indices)})"
                ),
                detail=f"Under {diversity_threshold:.0%} unique words. {samples}",
                suggestion=(
                    "Repeated tokens flatten the embedding toward the corpus average. "
                    "Often a table, nav bar, or template. Strip it during cleaning."
                ),
                stats={
                    "count": len(repetitive),
                    "min_diversity": round(min(d for _, d in repetitive), 3),
                    "chunks": [
                        {"chunk": i, "diversity": round(d, 3)} for i, d in repetitive
                    ],
                },
            )
        )

    return findings


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 55:
        return "D"
    return "F"


def _score(
    total_chunks: int,
    findings: Sequence[Finding],
    hubness_skew: float,
) -> float:
    """Composite 0-100 retrievability score.

    Penalties are rate-based so the score is comparable across corpus sizes,
    weighted by how directly each defect degrades retrieval.
    """
    if total_chunks == 0:
        return 0.0

    affected: Dict[str, set] = {}
    for finding in findings:
        affected.setdefault(finding.kind, set()).update(finding.chunk_indices)

    def rate(kind: str) -> float:
        return len(affected.get(kind, ())) / total_chunks

    penalty = (
        rate("orphan") * 35.0
        + rate("duplicate") * 25.0
        + rate("fracture") * 15.0
        + rate("boilerplate") * 10.0
        + rate("hub") * 40.0
    )

    # Hubness is a whole-corpus property, so it is scored off the distribution
    # rather than the count of flagged chunks.
    penalty += min(15.0, max(0.0, hubness_skew - 1.0) * 8.0)

    return float(max(0.0, min(100.0, 100.0 - penalty)))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def analyze(
    chunks: Sequence[str],
    embeddings: np.ndarray,
    *,
    strategy: str = "token",
    chunk_size: int = 200,
    overlap_percent: int = 10,
    neighbors: int = DEFAULT_NEIGHBORS,
    duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
    shadow_threshold: float = DEFAULT_SHADOW_THRESHOLD,
    island_threshold: float = DEFAULT_ISLAND_THRESHOLD,
    boilerplate_chars: int = DEFAULT_BOILERPLATE_CHARS,
    diversity_threshold: float = DEFAULT_DIVERSITY_THRESHOLD,
    provider: Optional[str] = None,
) -> DoctorReport:
    """Run every retrievability check against an already-embedded corpus.

    Args:
        chunks: The chunk texts.
        embeddings: Matrix of shape [n_chunks, embedding_dim].
        strategy: Chunking strategy used, recorded in the report.
        chunk_size: Chunk size used, recorded in the report.
        overlap_percent: Overlap used, recorded in the report.
        neighbors: Neighbourhood size k for the hubness computation.
        duplicate_threshold: Cosine similarity at or above which two chunks
            count as near-duplicates.
        shadow_threshold: Similarity above which an unreachable chunk is
            reported as shadowed by its nearest neighbour.
        island_threshold: Similarity below which an unreachable chunk is
            reported as an isolated content island.
        boilerplate_chars: Chunks shorter than this are flagged.
        diversity_threshold: Unique-word ratio below which a chunk is flagged
            as repetitive.
        provider: Embedding provider name, recorded in the report.

    Returns:
        A DoctorReport with findings sorted by severity.
    """
    total = len(chunks)
    timestamp = datetime.now().isoformat()

    if total == 0:
        return DoctorReport(
            timestamp=timestamp,
            total_chunks=0,
            strategy=strategy,
            chunk_size=chunk_size,
            overlap_percent=overlap_percent,
            neighbors=neighbors,
            retrievability_score=0.0,
            grade="F",
            hubness_skew=0.0,
            provider=provider,
        )

    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.shape[0] != total:
        raise ValueError(
            f"Got {total} chunks but {matrix.shape[0]} embeddings; they must match."
        )

    findings: List[Finding] = []

    # Textual checks need no embeddings and work even on a single chunk.
    findings.extend(_check_fractures(chunks))
    findings.extend(_check_boilerplate(chunks, boilerplate_chars, diversity_threshold))

    hubness_skew = 0.0
    if total >= 2:
        sim = _similarity_matrix(matrix)
        occurrence, _ = _k_occurrence(sim, neighbors)
        hubness_skew = _skewness(occurrence.astype(np.float64))

        findings.extend(_check_hubs(chunks, occurrence, neighbors))
        findings.extend(
            _check_orphans(chunks, sim, occurrence, shadow_threshold, island_threshold)
        )
        findings.extend(_check_duplicates(chunks, sim, duplicate_threshold))

    findings.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 3), f.chunk_indices))

    counts: Dict[str, int] = {}
    for finding in findings:
        counts[finding.kind] = counts.get(finding.kind, 0) + 1

    score = _score(total, findings, hubness_skew)

    return DoctorReport(
        timestamp=timestamp,
        total_chunks=total,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap_percent=overlap_percent,
        neighbors=neighbors,
        retrievability_score=score,
        grade=_grade(score),
        hubness_skew=hubness_skew,
        findings=findings,
        counts=counts,
        provider=provider,
    )
