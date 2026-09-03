"""Live integration tests — every example in ``demo-queries.md`` against
the real ``bird-search-fts`` index.

Each test does two things:
1. **Asserts request shape** on ``result.kwargs`` so an SDK bump that
   silently rewrites the ``score_by`` / ``filter`` payload fails loudly here.
2. **Asserts response signal** against the live index — the curated corpus
   makes top-k membership stable enough to lock in (we don't pin exact
   ranks, just "the obvious bird shows up in the top N").

Skipped when ``PINECONE_API_KEY`` or ``GOOGLE_API_KEY`` is unset (see
``tests/conftest.py``). Run with::

    pytest tests/test_live.py -v
"""

from __future__ import annotations

from query import (
    search_filter_visual,
    search_query_string,
    search_text,
    search_text_multi,
    search_text_phrase,
    search_visual,
)


# ---------------------------------------------------------------------------
# Tiny match-introspection helpers. The Pinecone SDK exposes match
# fields via ``.get(field)`` and the doc id as ``._id``; mirror what app.py
# / query.py read at runtime.
# ---------------------------------------------------------------------------

def _slugs(result, n: int | None = None):
    matches = result.matches if n is None else result.matches[:n]
    return [m._id for m in matches]


def _body(match) -> str:
    return (match.get("body") or "").lower()


def _slug_contains(result, needle: str, top_n: int = 5) -> bool:
    needle_lc = needle.lower()
    return any(needle_lc in s.lower() for s in _slugs(result, top_n))


# ===========================================================================
# 1. Text FTS — keyword precision
# ===========================================================================

def test_phrase_state_bird_of_seven(live_index):
    """1a — phrase ON, `state bird of seven` (body) → northern cardinal."""
    result = search_text_phrase("state bird of seven", field="body", top_k=10)

    # Request shape — query_string with quoted body clause.
    assert result.kwargs["score_by"] == [
        {"type": "query_string", "query": 'body:("state bird of seven")'}
    ]
    assert result.kwargs["namespace"] == "birds"

    # Response — cardinal in the top 5.
    assert len(result.matches) > 0
    assert _slug_contains(result, "cardinal", top_n=5), (
        f"expected a cardinal in top-5 slugs; got {_slugs(result, 5)}"
    )


def test_multi_miracle_of_the_gulls(live_index):
    """1b — phrase OFF, `Miracle of the Gulls` (multi) → California gull."""
    result = search_text_multi("Miracle of the Gulls", top_k=10)

    fields = [s["fields"][0] for s in result.kwargs["score_by"]]
    assert fields == ["bird_name", "intro", "body"]
    assert all(s["type"] == "text" for s in result.kwargs["score_by"])
    assert all(
        s["query"] == "Miracle of the Gulls" for s in result.kwargs["score_by"]
    )

    assert len(result.matches) > 0
    assert _slug_contains(result, "gull", top_n=5), (
        f"expected a gull in top-5 slugs; got {_slugs(result, 5)}"
    )


def test_phrase_national_bird_of_united_states(live_index):
    """1c — phrase ON, `national bird of the United States` (body) → bald eagle."""
    result = search_text_phrase(
        "national bird of the United States", field="body", top_k=10
    )

    assert result.kwargs["score_by"][0]["type"] == "query_string"
    assert (
        'body:("national bird of the United States")'
        in result.kwargs["score_by"][0]["query"]
    )

    assert len(result.matches) > 0
    assert _slug_contains(result, "eagle", top_n=5), (
        f"expected an eagle in top-5 slugs; got {_slugs(result, 5)}"
    )


def test_text_mormon_crickets(live_index):
    """1d — phrase OFF, `Mormon crickets` (body) → California gull."""
    result = search_text("Mormon crickets", field="body", top_k=10)

    assert result.kwargs["score_by"] == [
        {"type": "text", "fields": ["body"], "query": "Mormon crickets"}
    ]

    assert len(result.matches) > 0
    assert _slug_contains(result, "gull", top_n=5), (
        f"expected a gull in top-5 slugs; got {_slugs(result, 5)}"
    )


def test_multi_per_field_swallow_in_mountains(live_index):
    """1e — per-field multi: bird_name=swallow + body=in mountains."""
    result = search_text_multi(
        {"bird_name": "swallow", "body": "in mountains"}, top_k=10
    )

    # Two clauses (intro is omitted), each carrying its own query.
    clauses = result.kwargs["score_by"]
    assert [c["fields"][0] for c in clauses] == ["bird_name", "body"]
    assert all(c["type"] == "text" for c in clauses)
    by_field = {c["fields"][0]: c["query"] for c in clauses}
    assert by_field == {"bird_name": "swallow", "body": "in mountains"}

    assert len(result.matches) > 0
    # At least one swallow species should reach the top 5 because both
    # signals favour swallows.
    assert _slug_contains(result, "swallow", top_n=5), (
        f"expected a swallow in top-5; got {_slugs(result, 5)}"
    )


# ===========================================================================
# 2. Visual — typed description embedded against image vectors
# ===========================================================================

def test_visual_pink_wading_bird(live_index):
    """2a — `tall pink wading bird with long curved neck` → flamingo."""
    result = search_visual(
        "tall pink wading bird with long curved neck", top_k=10
    )
    signal = result.kwargs["score_by"][0]
    assert signal["type"] == "dense_vector"
    assert signal["field"] == "image_embedding"

    assert len(result.matches) > 0
    assert _slug_contains(result, "flamingo", top_n=5), (
        f"expected a flamingo in top-5; got {_slugs(result, 5)}"
    )


def test_visual_clown_beak(live_index):
    """2b — `huge colorful beak like a clown` → puffin."""
    result = search_visual(
        "black and white bird with a huge colorful beak like a clown", top_k=10
    )
    assert result.kwargs["score_by"][0]["type"] == "dense_vector"

    assert len(result.matches) > 0
    assert _slug_contains(result, "puffin", top_n=5), (
        f"expected a puffin in top-5; got {_slugs(result, 5)}"
    )


def test_visual_white_owl(live_index):
    """2c — `white owl with yellow eyes perched on snow` → snowy owl."""
    result = search_visual(
        "white owl with yellow eyes perched on snow", top_k=10
    )

    assert len(result.matches) > 0
    assert _slug_contains(result, "owl", top_n=5), (
        f"expected an owl in top-5; got {_slugs(result, 5)}"
    )


def test_visual_iridescent_hummingbird(live_index):
    """2d — `small iridescent green bird hovering at a flower` → hummingbird.

    Many hummingbird-family species are named after visual traits (mango,
    emerald, carib, goldentail, sapphire) rather than carrying 'hummingbird'
    in the slug, so check the top result's body text instead — every
    hummingbird article describes itself as a hummingbird in prose."""
    result = search_visual(
        "small iridescent green bird hovering at a flower", top_k=10
    )

    assert len(result.matches) > 0
    top_body = _body(result.matches[0])
    assert "hummingbird" in top_body, (
        f"expected a hummingbird at #1; got {_slugs(result, 5)} "
        f"(top body did not mention 'hummingbird')"
    )


# ===========================================================================
# 3. Combined — $match_all filter + dense visual rerank
# ===========================================================================

def test_visual_red_black_demo_flip_precondition(live_index):
    """The 'Bonus' beat in demo-queries.md hinges on most of the visual
    top-10 for `red bird with black wings` *not* mentioning Illinois — so
    that switching on the `$match_all: "illinois"` filter visibly
    rearranges the leaderboard. This test pins that pre-condition: at
    most 3 of the top-10 visual hits should have 'illinois' in their
    body (today the count is 1, only Red-winged blackbird). If a future
    scoring change pushes 4+ Illinois-mentioning birds into the visual
    top-10, the demo's A/B beat gets weaker and the docs need a refresh."""
    result = search_visual("red bird with black wings", top_k=10)
    assert len(result.matches) >= 5

    illinois_hits = sum(1 for m in result.matches if "illinois" in _body(m))
    assert illinois_hits <= 3, (
        f"demo flip eroded — {illinois_hits}/10 visual top hits mention "
        f"'illinois' (limit 3). Top slugs: {_slugs(result, 10)}"
    )


def test_combined_illinois_red_black(live_index):
    """3a — must-mention `illinois`, rank by `red bird with black wings`.
    Top hit should be cardinal or red-winged blackbird. Server-side
    $match_all means every match's body must contain 'illinois'."""
    result = search_filter_visual(
        filter_terms=["illinois"],
        visual_q="red bird with black wings",
        filter_field="body",
        top_k=10,
    )
    assert len(result.matches) > 0
    assert "$match_all" in result.code
    for m in result.matches:
        assert "illinois" in _body(m), (
            f"match {m._id} bypassed $match_all filter — body lacks 'illinois'"
        )
    top5 = " ".join(_slugs(result, 5)).lower()
    assert "cardinal" in top5 or "blackbird" in top5, (
        f"expected cardinal or blackbird in top-5; got {_slugs(result, 5)}"
    )


def test_combined_mormon_white_gull(live_index):
    """3b — must-mention `mormon`, rank by `white gull with gray wings`."""
    result = search_filter_visual(
        filter_terms=["mormon"],
        visual_q="white gull with gray wings",
        filter_field="body",
        top_k=10,
    )
    assert len(result.matches) > 0
    assert _slug_contains(result, "gull", top_n=5), (
        f"expected a gull in top-5; got {_slugs(result, 5)}"
    )
    for m in result.matches:
        assert "mormon" in _body(m)


def test_combined_tundra_arctic_white(live_index):
    """3c — must-mention `tundra arctic`, rank by `large white bird`."""
    result = search_filter_visual(
        filter_terms=["tundra", "arctic"],
        visual_q="large white bird",
        filter_field="body",
        top_k=10,
    )
    assert len(result.matches) > 0
    assert "$match_all" in result.code
    for m in result.matches:
        text = _body(m)
        assert "tundra" in text and "arctic" in text, (
            f"match {m._id} bypassed $match_all — body missing tundra/arctic"
        )


# ===========================================================================
# 4. Boolean — raw Lucene query_string
# ===========================================================================

def test_boolean_phrase_state_bird(live_index):
    """4a — `body:("state bird of seven")` → cardinal."""
    raw = 'body:("state bird of seven")'
    result = search_query_string(raw, top_k=10)
    assert result.kwargs["score_by"] == [{"type": "query_string", "query": raw}]
    assert len(result.matches) > 0
    assert _slug_contains(result, "cardinal", top_n=5)


def test_boolean_boost_eagle(live_index):
    """4b — `body:(eagle^3 OR hawk OR raptor)` → eagles dominate top hits."""
    raw = "body:(eagle^3 OR hawk OR raptor)"
    result = search_query_string(raw, top_k=10)
    assert result.kwargs["score_by"][0]["query"] == raw
    assert len(result.matches) > 0
    # With ^3 boost, an eagle article should be in the top 3.
    top3 = " ".join(_slugs(result, 3)).lower()
    assert "eagle" in top3, f"expected eagle in top-3; got {_slugs(result, 3)}"


def test_boolean_slop_northern_cardinal(live_index):
    """4c — `body:("northern cardinal"~3)` → cardinal."""
    raw = 'body:("northern cardinal"~3)'
    result = search_query_string(raw, top_k=10)
    assert result.kwargs["score_by"][0]["query"] == raw
    assert len(result.matches) > 0
    assert _slug_contains(result, "cardinal", top_n=5)


def test_boolean_cross_field_swallow(live_index):
    """4d — `bird_name:(swallow*) OR body:(swallow)` → swallows."""
    raw = "bird_name:(swallow*) OR body:(swallow)"
    result = search_query_string(raw, top_k=10)
    assert result.kwargs["score_by"][0]["query"] == raw
    assert len(result.matches) > 0
    # At least one top-5 result should be a swallow species.
    assert _slug_contains(result, "swallow", top_n=5), (
        f"expected a swallow in top-5; got {_slugs(result, 5)}"
    )


def test_boolean_required_excluded(live_index):
    """4e — `body:(+illinois +cardinal -opinion)` → no match contains 'opinion'."""
    raw = "body:(+illinois +cardinal -opinion)"
    result = search_query_string(raw, top_k=10)
    assert result.kwargs["score_by"][0]["query"] == raw
    assert len(result.matches) > 0
    for m in result.matches:
        assert "opinion" not in _body(m), (
            f"match {m._id} body still contains 'opinion' despite -opinion"
        )
