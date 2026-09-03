"""Unit tests for the MCP server: tool registration, the text/path contract,
error handling, and the output caps that keep a call from flooding an agent's
context window.

Every call passes provider="local" so the suite never depends on Ollama being
reachable and stays fast; the built-in lexical embedder is enough to exercise
every code path here.
"""

import json

import pytest

mcp = pytest.importorskip("mcp", reason="mcp extra not installed (pip install 'rag-tui[mcp]')")

from rag_tui.mcp_server import create_server  # noqa: E402

DOC = """# Handbook

## Billing

Invoices go out on the first of the month and cover the previous period.

## Billing

Invoices go out on the first of the month and cover the previous period.

## Rate limits

The API permits one thousand requests per minute for every project.
"""


@pytest.fixture
def server():
    return create_server()


async def _call(server, name, **kwargs):
    """Call a tool exactly as a real client would, and return the parsed dict."""
    result = await server.call_tool(name, kwargs)
    assert not result.is_error, f"{name} reported an error: {result.content}"
    return json.loads(result.content[0].text)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

EXPECTED_TOOLS = {
    "list_strategies",
    "diagnose_corpus",
    "chunk_document",
    "evaluate_retrieval",
    "optimize_chunking",
    "export_chunking_config",
}


@pytest.mark.asyncio
async def test_every_expected_tool_is_registered(server):
    tools = await server.list_tools()
    assert {t.name for t in tools} == EXPECTED_TOOLS


@pytest.mark.asyncio
async def test_every_tool_has_a_real_description(server):
    """Guards against the f-string docstring trap: an f-string as a function's
    first statement is not a docstring at all, so __doc__ comes back None and
    the tool would register with no description for the calling agent."""
    tools = await server.list_tools()
    for tool in tools:
        assert tool.description and len(tool.description.strip()) > 20, tool.name


@pytest.mark.asyncio
async def test_no_tool_advertises_a_rigid_output_schema(server):
    """These all return open-ended dicts; structured_output=False keeps the
    SDK from wrapping them in {"result": ...} for structuredContent, which
    would silently disagree with the plain dict shown in the text content."""
    tools = await server.list_tools()
    for tool in tools:
        assert tool.output_schema is None, tool.name


# ---------------------------------------------------------------------------
# The shared text/path contract
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_neither_text_nor_path_is_an_error(server):
    r = await _call(server, "diagnose_corpus")
    assert "error" in r and "exactly one" in r["error"]


@pytest.mark.asyncio
async def test_both_text_and_path_is_an_error(server):
    r = await _call(server, "diagnose_corpus", text="hello", path="/tmp/doesnotmatter.md")
    assert "error" in r and "exactly one" in r["error"]


@pytest.mark.asyncio
async def test_missing_file_is_a_clean_error_not_a_crash(server):
    r = await _call(server, "diagnose_corpus", path="/definitely/not/a/real/file.md")
    assert "error" in r


@pytest.mark.asyncio
async def test_path_reads_a_real_file(server, tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text(DOC)
    r = await _call(server, "chunk_document", path=str(doc), strategy="markdown")
    assert r["stats"]["total_chunks"] > 0


@pytest.mark.asyncio
async def test_text_is_used_directly(server):
    r = await _call(server, "chunk_document", text=DOC, strategy="markdown")
    assert r["stats"]["total_chunks"] > 0


# ---------------------------------------------------------------------------
# list_strategies
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_strategies_excludes_custom(server):
    r = await _call(server, "list_strategies")
    names = {s["name"] for s in r["strategies"]}
    assert "custom" not in names
    assert {"token", "markdown", "hierarchical"} <= names


# ---------------------------------------------------------------------------
# diagnose_corpus
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_diagnose_corpus_finds_the_planted_duplicate(server):
    r = await _call(server, "diagnose_corpus", text=DOC, strategy="paragraph",
                     chunk_size=20, provider="local")
    assert r["retrievability_score"] < 100
    assert any(f["kind"] == "duplicate" for f in r["findings"])
    # The report names which embedder actually produced these numbers.
    assert r["config"]["provider"]


@pytest.mark.asyncio
async def test_diagnose_corpus_max_findings_caps_the_list(server):
    full = await _call(server, "diagnose_corpus", text=DOC, strategy="paragraph",
                        chunk_size=20, provider="local", max_findings=1000)
    capped = await _call(server, "diagnose_corpus", text=DOC, strategy="paragraph",
                          chunk_size=20, provider="local", max_findings=1)

    assert len(capped["findings"]) == 1
    assert capped.get("findings_omitted") == len(full["findings"]) - 1


@pytest.mark.asyncio
async def test_diagnose_corpus_fail_under_gate(server):
    strict = await _call(server, "diagnose_corpus", text=DOC, provider="local", fail_under=99)
    lenient = await _call(server, "diagnose_corpus", text=DOC, provider="local", fail_under=1)

    assert strict["passed"] is False
    assert lenient["passed"] is True


@pytest.mark.asyncio
async def test_diagnose_corpus_without_fail_under_has_no_passed_key(server):
    r = await _call(server, "diagnose_corpus", text=DOC, provider="local")
    assert "passed" not in r


# ---------------------------------------------------------------------------
# chunk_document
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chunk_document_stats_reflect_the_whole_document_even_when_capped(server):
    text = "\n\n".join(f"Paragraph {i} with some words in it for bulk." for i in range(50))
    r = await _call(server, "chunk_document", text=text, strategy="paragraph",
                     chunk_size=10, max_chunks=3)

    assert len(r["chunks"]) == 3
    assert r["stats"]["total_chunks"] > 3
    assert r["chunks_omitted"] == r["stats"]["total_chunks"] - 3


@pytest.mark.asyncio
async def test_chunk_document_truncates_long_chunk_text(server):
    text = "word " * 500
    r = await _call(server, "chunk_document", text=text, chunk_size=400,
                     max_chunks=1, max_chars_per_chunk=50)

    assert len(r["chunks"][0]["text"]) > 50  # includes the "[N more chars]" suffix
    assert "more chars" in r["chunks"][0]["text"]


@pytest.mark.asyncio
async def test_chunk_document_no_omission_key_when_nothing_is_cut(server):
    r = await _call(server, "chunk_document", text="one short paragraph here.", max_chunks=50)
    assert "chunks_omitted" not in r


@pytest.mark.asyncio
async def test_hierarchical_metadata_survives_the_mcp_layer(server):
    """Regression test: api.chunk() used to call chunk_text(), which discards
    ChunkResult.metadata entirely. Hierarchical's whole point is the parent
    window it attaches per chunk, so losing metadata made it useless here."""
    r = await _call(server, "chunk_document", text=DOC, strategy="hierarchical", chunk_size=20)

    meta = r["chunks"][0]["metadata"]
    assert "parent_text" in meta
    assert "parent_index" in meta
    assert r["chunks"][0]["text"] in meta["parent_text"]


@pytest.mark.asyncio
async def test_hierarchical_parent_text_is_also_truncated(server):
    r = await _call(server, "chunk_document", text=DOC, strategy="hierarchical",
                     chunk_size=20, max_chars_per_chunk=20)
    meta = r["chunks"][0]["metadata"]
    assert len(meta["parent_text"]) > 20  # includes the truncation suffix
    assert "more chars" in meta["parent_text"]


@pytest.mark.asyncio
async def test_markdown_heading_path_survives_the_mcp_layer(server):
    r = await _call(server, "chunk_document", text=DOC, strategy="markdown", chunk_size=100)
    paths = [c["metadata"]["heading_path"] for c in r["chunks"]]
    assert ["Handbook", "Billing"] in paths


# ---------------------------------------------------------------------------
# evaluate_retrieval
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluate_retrieval_rejects_empty_queries(server):
    r = await _call(server, "evaluate_retrieval", text=DOC, queries=[])
    assert "error" in r


@pytest.mark.asyncio
async def test_evaluate_retrieval_reports_all_five_metrics(server):
    r = await _call(server, "evaluate_retrieval", text=DOC,
                     queries=["what are the billing dates", "how many requests per minute"],
                     provider="local")

    for key in ("hit_rate", "mrr", "ndcg_at_k", "recall_at_k", "precision_at_k"):
        assert key in r["metrics"]
    assert r["config"]["provider"]


@pytest.mark.asyncio
async def test_evaluate_retrieval_bad_file_is_an_error(server):
    r = await _call(server, "evaluate_retrieval", path="/nope.md", queries=["x"])
    assert "error" in r


# ---------------------------------------------------------------------------
# optimize_chunking
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_optimize_chunking_rejects_empty_queries(server):
    r = await _call(server, "optimize_chunking", text=DOC, queries=[])
    assert "error" in r


@pytest.mark.asyncio
async def test_optimize_chunking_caps_results_but_keeps_the_true_total(server):
    r = await _call(
        server, "optimize_chunking", text=DOC,
        queries=["billing dates", "rate limits"],
        strategies=["token", "paragraph"], sizes=[30, 60], overlap_percents=[10],
        provider="local", max_results=1,
    )

    assert r["total_configs_tested"] == 4  # 2 strategies x 2 sizes x 1 overlap
    assert len(r["all_results"]) == 1
    assert r["results_omitted"] == 3
    assert r["best"]["strategy"] in ("token", "paragraph")
    assert r["provider"]


@pytest.mark.asyncio
async def test_optimize_chunking_no_omission_key_when_nothing_is_cut(server):
    r = await _call(
        server, "optimize_chunking", text=DOC, queries=["billing dates"],
        strategies=["token"], sizes=[30], overlap_percents=[10],
        provider="local", max_results=50,
    )
    assert "results_omitted" not in r


# ---------------------------------------------------------------------------
# export_chunking_config
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_export_langchain(server):
    r = await _call(server, "export_chunking_config", format="langchain", chunk_size=300)
    assert "RecursiveCharacterTextSplitter" in r["code"]
    assert r["format"] == "langchain"


@pytest.mark.asyncio
async def test_export_llamaindex(server):
    r = await _call(server, "export_chunking_config", format="llamaindex")
    assert "SentenceSplitter" in r["code"]


@pytest.mark.asyncio
async def test_export_json(server):
    r = await _call(server, "export_chunking_config", format="json", chunk_size=256)
    parsed = json.loads(r["code"])
    assert parsed["chunk_size"] == 256


@pytest.mark.asyncio
async def test_export_bad_format_is_an_error_not_a_crash(server):
    r = await _call(server, "export_chunking_config", format="bogus")
    assert "error" in r
