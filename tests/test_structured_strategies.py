"""Unit tests for the markdown and hierarchical chunking strategies."""

import pytest

from rag_tui.core.strategies import (
    HierarchicalStrategy,
    MarkdownStrategy,
    StrategyType,
    get_strategy,
)

DOC = """# Handbook

Opening paragraph that introduces the whole handbook to a new reader.

## Billing

Invoices go out on the first of the month and cover the previous period.

### Refunds

Refunds are available within thirty days of the original charge.

## Rate limits

The API permits one thousand requests per minute for every project.
"""


def _md(text, chunk_size=200, overlap=20):
    return MarkdownStrategy().chunk(text, chunk_size, overlap)


def _body(chunk):
    """The chunk text with the heading trail stripped back off."""
    return chunk.text[chunk.metadata["prefix_chars"]:]


# ---------------------------------------------------------------------------
# Markdown: heading structure
# ---------------------------------------------------------------------------

def test_registered_in_strategy_table():
    assert isinstance(get_strategy(StrategyType.MARKDOWN), MarkdownStrategy)
    assert isinstance(get_strategy(StrategyType.HIERARCHICAL), HierarchicalStrategy)


def test_each_section_becomes_a_chunk():
    chunks = _md(DOC)
    paths = [c.metadata["heading_path"] for c in chunks]

    assert paths == [
        ["Handbook"],
        ["Handbook", "Billing"],
        ["Handbook", "Billing", "Refunds"],
        ["Handbook", "Rate limits"],
    ]


def test_heading_trail_is_prefixed_to_the_text():
    chunks = _md(DOC)
    refunds = next(c for c in chunks if c.metadata["heading_path"][-1] == "Refunds")

    assert refunds.text.startswith("Handbook > Billing > Refunds")
    assert "thirty days" in refunds.text


def test_dropping_back_a_level_pops_the_deeper_headings():
    """After an h3, the next h2 must not inherit the h3 in its trail."""
    chunks = _md(DOC)
    rate_limits = next(c for c in chunks if c.metadata["heading_path"][-1] == "Rate limits")

    assert rate_limits.metadata["heading_path"] == ["Handbook", "Rate limits"]
    assert "Refunds" not in rate_limits.text


def test_no_chunk_is_a_bare_heading():
    """A lone '## Rate limits' is a chunk no query will ever retrieve."""
    for chunk in _md(DOC):
        assert len(_body(chunk).strip()) > 30


def test_heading_depth_recorded():
    chunks = _md(DOC)

    assert [c.metadata["heading_depth"] for c in chunks] == [1, 2, 3, 2]


def test_closed_atx_headings_are_stripped():
    chunks = _md("## Billing ##\n\nInvoices go out on the first of each month here.\n")

    assert chunks[0].metadata["heading_path"] == ["Billing"]


def test_deep_headings_beyond_six_hashes_are_not_headings():
    text = "# Real\n\nBody text that is long enough to survive the checks here.\n\n####### Not a heading\n"
    chunks = _md(text)

    assert all(c.metadata["heading_path"] == ["Real"] for c in chunks)


# ---------------------------------------------------------------------------
# Markdown: code fences
# ---------------------------------------------------------------------------

def test_hashes_inside_a_code_fence_are_not_headings():
    text = """# Setup

Install the package and then run the following snippet to get started.

```python
# This is a comment, not a heading
value = 1
```

More prose after the fence to keep the section substantial enough.
"""
    chunks = _md(text)

    assert all(c.metadata["heading_path"] == ["Setup"] for c in chunks)
    assert "not a heading" in " ".join(c.text for c in chunks)


def test_tilde_fences_are_handled_too():
    text = """# Setup

Some introductory prose that runs on for a reasonable length here.

~~~
# not a heading either
~~~

Closing prose that also runs on for a reasonable length here.
"""
    assert all(c.metadata["heading_path"] == ["Setup"] for c in _md(text))


# ---------------------------------------------------------------------------
# Markdown: offsets and edge cases
# ---------------------------------------------------------------------------

def test_positions_map_back_into_the_source():
    """The prefix is added context, so offsets must point at the real body."""
    for chunk in _md(DOC):
        assert DOC[chunk.start_pos:chunk.end_pos] == _body(chunk)


def test_prefix_chars_is_the_length_of_the_added_trail():
    for chunk in _md(DOC):
        trail = " > ".join(chunk.metadata["heading_path"])
        assert chunk.metadata["prefix_chars"] == len(trail) + 2  # trail + blank line


def test_long_section_splits_but_keeps_the_same_trail():
    body = "\n\n".join(f"Paragraph number {i} with enough words to take up room." for i in range(40))
    chunks = _md(f"# Big\n\n## Section\n\n{body}\n", chunk_size=40)

    assert len(chunks) > 1
    assert all(c.metadata["heading_path"] == ["Big", "Section"] for c in chunks)


def test_document_without_headings_falls_back():
    text = "Just prose with no headings at all.\n\nA second paragraph follows it.\n"
    chunks = _md(text)

    assert chunks
    assert "Just prose" in chunks[0].text


def test_empty_document_produces_nothing():
    assert _md("") == []
    assert _md("   \n\n  \n") == []


def test_preamble_before_the_first_heading_is_kept():
    text = "Front matter that appears before any heading in the document.\n\n# Later\n\nBody here now.\n"
    chunks = _md(text)

    assert "Front matter" in chunks[0].text
    assert chunks[0].metadata["heading_path"] == []


# ---------------------------------------------------------------------------
# Hierarchical
# ---------------------------------------------------------------------------

PROSE = " ".join(
    f"Sentence number {i} carries a distinct fact about retrieval systems." for i in range(60)
)


def _hier(text, chunk_size=30, overlap=0):
    return HierarchicalStrategy().chunk(text, chunk_size, overlap)


def test_children_are_returned_not_parents():
    chunks = _hier(PROSE)
    child_len = max(len(c.text) for c in chunks)
    parent_len = max(len(c.metadata["parent_text"]) for c in chunks)

    assert parent_len > child_len


def test_every_child_lives_inside_its_own_parent():
    for chunk in _hier(PROSE):
        assert chunk.text in chunk.metadata["parent_text"]


def test_parents_group_multiple_children():
    chunks = _hier(PROSE)
    parents = {c.metadata["parent_index"] for c in chunks}

    assert len(parents) >= 2
    assert len(chunks) > len(parents)


def test_parent_index_is_contiguous_from_zero():
    chunks = _hier(PROSE)
    parents = sorted({c.metadata["parent_index"] for c in chunks})

    assert parents == list(range(len(parents)))


def test_children_in_parent_count_is_accurate():
    chunks = _hier(PROSE)
    actual = {}
    for chunk in chunks:
        actual[chunk.metadata["parent_index"]] = actual.get(chunk.metadata["parent_index"], 0) + 1

    for chunk in chunks:
        assert chunk.metadata["children_in_parent"] == actual[chunk.metadata["parent_index"]]


def test_child_offsets_map_back_into_the_source():
    for chunk in _hier(PROSE):
        assert PROSE[chunk.start_pos:chunk.end_pos] == chunk.text


def test_parent_span_contains_every_child_span():
    for chunk in _hier(PROSE):
        assert chunk.metadata["parent_start"] <= chunk.start_pos
        assert chunk.end_pos <= chunk.metadata["parent_end"]


def test_child_size_tracks_chunk_size():
    small = max(len(c.text) for c in _hier(PROSE, chunk_size=20))
    large = max(len(c.text) for c in _hier(PROSE, chunk_size=60))

    assert large > small


def test_short_text_yields_one_child():
    chunks = _hier("A single short sentence about retrieval.")

    assert len(chunks) == 1
    assert chunks[0].metadata["parent_index"] == 0


def test_empty_text_yields_nothing():
    assert _hier("") == []
    assert _hier("   \n  ") == []


# ---------------------------------------------------------------------------
# Both, through the engine
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", [StrategyType.MARKDOWN, StrategyType.HIERARCHICAL])
def test_engine_can_drive_the_new_strategies(strategy):
    from rag_tui.core.engine import ChunkingEngine

    chunks = ChunkingEngine().chunk_text(DOC, chunk_size=100, overlap=10, strategy_type=strategy)

    assert chunks
    assert all(isinstance(t, str) and t.strip() for t, _, _ in chunks)
