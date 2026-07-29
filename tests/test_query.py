"""Mocked unit tests for ``query.py``.

These tests patch ``pinecone.Pinecone`` and ``embedder.embed_text`` at
import time so ``query.py`` can be imported without real credentials or a
local model download, then assert each helper assembles the expected
``idx.documents.search`` request and returns a ``SearchResult`` exposing
matches + the call that was made.

Run with::

    pytest tests/test_query.py -v

These run offline; the live counterparts in ``tests/test_live.py`` hit
the real index and require API keys.
"""

from __future__ import annotations

import importlib
import sys
from unittest import mock

import pytest


def _fresh_query_module(search_matches=None, search_side_effect=None):
    """Import (or re-import) ``query`` with both dependencies fully mocked.

    ``search_matches`` injects a list of fake match objects on the
    mocked ``documents.search`` response — used by helpers that read
    matches back from the response (e.g. the
    ``search_filter_visual(filter_terms=[])`` short-circuit path that
    delegates to ``search_visual``).

    ``search_side_effect``, if given, is a list of responses returned on
    successive ``documents.search`` calls in order — used by
    ``search_hybrid_rrf``, which issues two calls (text, then dense) and
    needs each to return a different match set.
    """
    matches = list(search_matches or [])
    response = mock.MagicMock(matches=matches)
    fake_search = mock.MagicMock(return_value=response)
    if search_side_effect is not None:
        fake_search.side_effect = search_side_effect

    fake_idx = mock.MagicMock()
    fake_idx.documents.search = fake_search

    fake_pc = mock.MagicMock()
    fake_pc.preview.index.return_value = fake_idx

    fake_embed_text = mock.MagicMock(return_value=[0.1, 0.2, 0.3])

    pinecone_mod = mock.MagicMock()
    pinecone_mod.Pinecone.return_value = fake_pc

    embedder_mod = mock.MagicMock()
    embedder_mod.embed_text = fake_embed_text

    patches = {
        "pinecone": pinecone_mod,
        "embedder": embedder_mod,
    }
    with mock.patch.dict(sys.modules, patches):
        if "query" in sys.modules:
            del sys.modules["query"]
        query = importlib.import_module("query")

    return query, fake_search, fake_embed_text


def _fake_match(_id: str, body: str):
    """Stand-in for a Pinecone match object — supports ``.get(field)``
    the way ``search_filter_visual`` calls it."""
    m = mock.MagicMock()
    m._id = _id
    m.get.side_effect = lambda field, default=None: {"body": body}.get(field, default)
    return m


def test_search_text_builds_single_field_score_by():
    query, fake_search, _ = _fresh_query_module()
    result = query.search_text("woodpecker", field="body", top_k=7)
    _, kwargs = fake_search.call_args
    assert kwargs["namespace"] == "birds"
    assert kwargs["top_k"] == 7
    assert kwargs["score_by"] == [
        {"type": "text", "field": "body", "query": "woodpecker"}
    ]
    assert kwargs["include_fields"] == ["bird_name", "intro", "body"]
    # SearchResult contract: kwargs round-tripped, code populated.
    assert result.kwargs == kwargs
    assert "documents.search" in result.code
    assert "woodpecker" in result.code


def test_search_text_multi_blends_three_fields():
    query, fake_search, _ = _fresh_query_module()
    result = query.search_text_multi("red wings", top_k=5)
    _, kwargs = fake_search.call_args
    assert kwargs["top_k"] == 5
    fields = [s["field"] for s in kwargs["score_by"]]
    assert fields == ["bird_name", "intro", "body"]
    assert all(s["type"] == "text" and s["query"] == "red wings" for s in kwargs["score_by"])
    assert "score_by" in result.code


def test_search_text_multi_per_field_dict_emits_one_clause_per_field():
    """Dict form: only fields with non-empty queries get a score_by clause,
    each carrying its own query string."""
    query, fake_search, _ = _fresh_query_module()
    query.search_text_multi(
        {"bird_name": "swallow", "intro": "  ", "body": "in mountains"},
        top_k=4,
    )
    _, kwargs = fake_search.call_args
    assert kwargs["top_k"] == 4
    # `intro` is whitespace-only → skipped. Only bird_name + body should fire.
    clauses = kwargs["score_by"]
    assert [c["field"] for c in clauses] == ["bird_name", "body"]
    assert all(c["type"] == "text" for c in clauses)
    by_field = {c["field"]: c["query"] for c in clauses}
    assert by_field == {"bird_name": "swallow", "body": "in mountains"}


def test_search_text_multi_dict_all_empty_raises():
    """Dict form with every value blank should fail loudly rather than
    silently fall through to an empty score_by."""
    query, fake_search, _ = _fresh_query_module()
    with pytest.raises(ValueError):
        query.search_text_multi({"bird_name": "", "intro": "   ", "body": ""})
    fake_search.assert_not_called()


def test_search_text_phrase_uses_query_string_with_quotes():
    """Single-field phrase mode wraps the query as field:("…")."""
    query, fake_search, _ = _fresh_query_module()
    query.search_text_phrase("state bird of seven", field="body", top_k=10)
    _, kwargs = fake_search.call_args
    assert kwargs["top_k"] == 10
    assert kwargs["score_by"] == [
        {"type": "query_string", "query": 'body:("state bird of seven")'}
    ]


def test_search_text_phrase_multi_ors_across_fields():
    """``field='multi'`` ORs the phrase across bird_name / intro / body."""
    query, fake_search, _ = _fresh_query_module()
    query.search_text_phrase("Miracle of the Gulls", field="multi")
    _, kwargs = fake_search.call_args
    q_str = kwargs["score_by"][0]["query"]
    assert kwargs["score_by"][0]["type"] == "query_string"
    assert 'bird_name:("Miracle of the Gulls")' in q_str
    assert 'intro:("Miracle of the Gulls")' in q_str
    assert 'body:("Miracle of the Gulls")' in q_str
    assert " OR " in q_str


def test_search_query_string_passes_lucene_through_unchanged():
    query, fake_search, _ = _fresh_query_module()
    raw = "body:(+illinois +cardinal -opinion)"
    query.search_query_string(raw, top_k=8)
    _, kwargs = fake_search.call_args
    assert kwargs["top_k"] == 8
    assert kwargs["score_by"] == [{"type": "query_string", "query": raw}]


def test_search_visual_uses_dense_vector_on_image_embedding():
    query, fake_search, fake_embed_text = _fresh_query_module()
    query.search_visual("red round bird", top_k=3)
    fake_embed_text.assert_called_once()
    _, kwargs = fake_search.call_args
    assert kwargs["top_k"] == 3
    signal = kwargs["score_by"][0]
    assert signal["type"] == "dense_vector"
    assert signal["field"] == "image_embedding"
    assert signal["values"] == [0.1, 0.2, 0.3]


def test_search_filter_visual_uses_match_all_filter():
    """$match_all + dense_vector → single Pinecone call."""
    query, fake_search, fake_embed_text = _fresh_query_module()
    result = query.search_filter_visual(
        filter_terms=["illinois", "forest"],
        visual_q="red bird with black wings",
        filter_field="body",
        top_k=4,
    )
    fake_embed_text.assert_called_once()
    assert fake_search.call_count == 1
    _, kwargs = fake_search.call_args
    assert kwargs["top_k"] == 4
    assert kwargs["filter"] == {"body": {"$match_all": "illinois forest"}}
    signal = kwargs["score_by"][0]
    assert signal["type"] == "dense_vector"
    assert signal["field"] == "image_embedding"
    assert "$match_all" in result.code
    assert result.extra_code == ""


def test_search_filter_visual_match_any_mode():
    """``mode='any'`` switches the filter operator to ``$match_any``."""
    query, fake_search, _ = _fresh_query_module()
    query.search_filter_visual(
        filter_terms=["epaulet", "yellow"],
        visual_q="black bird with bright spots on wings",
        mode="any",
    )
    _, kwargs = fake_search.call_args
    assert kwargs["filter"] == {"body": {"$match_any": "epaulet yellow"}}


def test_search_filter_visual_match_phrase_mode():
    """``mode='phrase'`` switches the filter operator to ``$match_phrase``,
    preserving term order (unlike ``all``/``any``, where order is
    irrelevant)."""
    query, fake_search, _ = _fresh_query_module()
    query.search_filter_visual(
        filter_terms=["yellow wing bar"],
        visual_q="black bird with bright spots on wings",
        mode="phrase",
    )
    _, kwargs = fake_search.call_args
    assert kwargs["filter"] == {"body": {"$match_phrase": "yellow wing bar"}}


def test_search_filter_visual_invalid_mode_raises():
    query, fake_search, _ = _fresh_query_module()
    with pytest.raises(ValueError):
        query.search_filter_visual(
            filter_terms=["illinois"],
            visual_q="red bird",
            mode="bogus",
        )
    fake_search.assert_not_called()


def test_search_filter_visual_pure_visual_when_no_terms():
    """Empty / whitespace-only filter terms = pure visual search;
    falls through to ``search_visual`` (single dense call, no filter)."""
    query, fake_search, _ = _fresh_query_module(
        search_matches=[_fake_match("cardinal", "anything")]
    )
    result = query.search_filter_visual(
        filter_terms=["   ", ""],
        visual_q="red bird",
    )
    _, kwargs = fake_search.call_args
    assert "filter" not in kwargs
    assert kwargs["score_by"][0]["type"] == "dense_vector"
    assert [m._id for m in result.matches] == ["cardinal"]


def test_rrf_combine_sums_reciprocal_ranks():
    """A doc present (and decently ranked) in every ranking should beat one
    that's #1 in only a single ranking — the whole point of RRF."""
    query, _, _ = _fresh_query_module()
    scores = query._rrf_combine([["A", "B"], ["B", "C"]], k=60)
    assert scores["A"] == pytest.approx(1 / 61)
    assert scores["C"] == pytest.approx(1 / 62)
    assert scores["B"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["B"] > scores["A"] > scores["C"]


def test_search_hybrid_rrf_fuses_two_independent_calls():
    """Issues a text call then a dense call, and fuses their rankings —
    a doc absent from both individual top spots (but present in both
    lists) should still win the fused ranking."""
    text_response = mock.MagicMock(
        matches=[_fake_match("A", "a"), _fake_match("B", "b")]
    )
    dense_response = mock.MagicMock(
        matches=[_fake_match("B", "b"), _fake_match("C", "c")]
    )
    query, fake_search, fake_embed_text = _fresh_query_module(
        search_side_effect=[text_response, dense_response]
    )

    result = query.search_hybrid_rrf(
        "some text", "some visual", field="body", top_k=3, fetch_k=10
    )

    assert fake_search.call_count == 2
    fake_embed_text.assert_called_once()

    text_kwargs = fake_search.call_args_list[0].kwargs
    dense_kwargs = fake_search.call_args_list[1].kwargs
    assert text_kwargs["score_by"][0]["type"] == "text"
    assert text_kwargs["score_by"][0]["field"] == "body"
    assert dense_kwargs["score_by"][0]["type"] == "dense_vector"
    assert dense_kwargs["score_by"][0]["field"] == "image_embedding"

    # B ranks #2 in text and #1 in dense — decent on both beats being #1
    # on only one (A).
    assert [m._id for m in result.matches] == ["B", "A", "C"]
    assert result.kwargs == text_kwargs
    assert "rrf_combine" in result.extra_code
