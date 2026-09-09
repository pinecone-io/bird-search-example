"""Mocked unit tests for ``query.py``.

These tests patch ``pinecone.Pinecone`` and ``google.genai.Client`` at
import time so ``query.py`` can be imported without real credentials,
then assert each helper assembles the expected ``idx.documents.search``
request and returns a ``SearchResult`` exposing matches + the call
that was made.

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


def _fresh_query_module(search_matches=None):
    """Import (or re-import) ``query`` with both clients fully mocked.

    ``search_matches`` injects a list of fake match objects on the
    mocked ``documents.search`` response — used by helpers that read
    matches back from the response (e.g. the
    ``search_filter_visual(filter_terms=[])`` short-circuit path that
    delegates to ``search_visual``).
    """
    matches = list(search_matches or [])
    response = mock.MagicMock(matches=matches)
    fake_search = mock.MagicMock(return_value=response)

    fake_idx = mock.MagicMock()
    fake_idx.documents.search = fake_search

    fake_pc = mock.MagicMock()
    fake_pc.index.return_value = fake_idx

    fake_embedding = mock.MagicMock()
    fake_embedding.values = [0.1, 0.2, 0.3]
    fake_embed_response = mock.MagicMock(embeddings=[fake_embedding])

    fake_gem = mock.MagicMock()
    fake_gem.models.embed_content.return_value = fake_embed_response

    pinecone_mod = mock.MagicMock()
    pinecone_mod.Pinecone.return_value = fake_pc

    google_mod = mock.MagicMock()
    google_mod.genai.Client.return_value = fake_gem

    patches = {
        "pinecone": pinecone_mod,
        "google": google_mod,
        "google.genai": google_mod.genai,
    }
    with mock.patch.dict(sys.modules, patches):
        if "query" in sys.modules:
            del sys.modules["query"]
        query = importlib.import_module("query")

    return query, fake_search, fake_gem


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
        {"type": "text", "fields": ["body"], "query": "woodpecker"}
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
    fields = [s["fields"][0] for s in kwargs["score_by"]]
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
    assert [c["fields"][0] for c in clauses] == ["bird_name", "body"]
    assert all(c["type"] == "text" for c in clauses)
    by_field = {c["fields"][0]: c["query"] for c in clauses}
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
    query, fake_search, fake_gem = _fresh_query_module()
    query.search_visual("red round bird", top_k=3)
    fake_gem.models.embed_content.assert_called_once()
    _, kwargs = fake_search.call_args
    assert kwargs["top_k"] == 3
    signal = kwargs["score_by"][0]
    assert signal["type"] == "dense_vector"
    assert signal["field"] == "image_embedding"
    assert signal["values"] == [0.1, 0.2, 0.3]


def test_search_filter_visual_uses_match_all_filter():
    """$match_all + dense_vector → single Pinecone call."""
    query, fake_search, fake_gem = _fresh_query_module()
    result = query.search_filter_visual(
        filter_terms=["illinois", "forest"],
        visual_q="red bird with black wings",
        filter_field="body",
        top_k=4,
    )
    fake_gem.models.embed_content.assert_called_once()
    assert fake_search.call_count == 1
    _, kwargs = fake_search.call_args
    assert kwargs["top_k"] == 4
    assert kwargs["filter"] == {"body": {"$match_all": "illinois forest"}}
    signal = kwargs["score_by"][0]
    assert signal["type"] == "dense_vector"
    assert signal["field"] == "image_embedding"
    assert "$match_all" in result.code
    assert result.extra_code == ""


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
