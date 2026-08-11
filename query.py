"""Bird Search v2 — search helpers.

Thin wrappers over `idx.documents.search(...)` for each of the UI tabs:

    search_text           single-field FTS, BM25 token-OR (`type: "text"`)
    search_text_multi     blended multi-field FTS (bird_name + intro + body)
    search_text_phrase    exact-phrase FTS via `query_string` with quotes
    search_query_string   raw Lucene query_string (boolean / boost / slop / …)
    search_visual         Gemini-2 text embed scored against stored image
                          embeddings (cross-modal)
    search_filter_visual  $match_all body filter + dense visual rerank in
                          one Pinecone call
    search_hybrid_rrf     BM25 text search + dense visual search, fused
                          client-side via Reciprocal Rank Fusion (Pinecone currently
                          has no built-in cross-call fusion)

Every helper returns a ``SearchResult`` with:

- ``matches`` — list of match objects (drop-in for the SDK response).
- ``kwargs`` — the dict actually passed to ``documents.search(**kwargs)``.
- ``code`` — a Python source snippet of that call, ready for ``st.code(...)``
  in the UI so demo viewers can see exactly what hit Pinecone.
- ``extra_code`` — optional post-call Python (currently always empty;
  reserved for any future helper that does client-side processing on top
  of the SDK response).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone

from google import genai
from google.genai import types as genai_types


INDEX = "bird-search-fts"
NAMESPACE = "birds"

GEMINI_MODEL = "gemini-embedding-2"

GEMINI_EMBED_DIMENSIONS = 768

INCLUDE_FIELDS = ["bird_name", "intro", "body"]


pc = Pinecone(source_tag="pinecone:bird_search_example")
gem = genai.Client()



_EMBED_CONFIG = genai_types.EmbedContentConfig(
    output_dimensionality=GEMINI_EMBED_DIMENSIONS,
)


def embed_text(text: str) -> list[float]:
    """Embed a text query into Gemini-2's multimodal space, truncated to
    GEMINI_EMBED_DIMENSIONS."""
    resp = gem.models.embed_content(
        model=GEMINI_MODEL, contents=text, config=_EMBED_CONFIG
    )
    return list(resp.embeddings[0].values)


def _index():
    """Per-call index handle"""
    return pc.preview.index(name=INDEX)


# -----------------------------------------------------------------------------
# Result wrapper — exposes matches plus a printable copy of the call
# -----------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Drop-in for the SDK response, with the call exposed for the UI.

    ``code`` is a Python snippet of the ``documents.search(...)`` call we just
    made — long dense-vector ``values`` lists are abbreviated so the snippet
    stays readable. ``extra_code`` carries any post-call processing (e.g. the
    client-side substring filter in ``search_filter_visual``).
    """
    matches: list
    kwargs: dict
    code: str
    extra_code: str = ""


def _abbreviate_vectors(payload: Any) -> Any:
    """Walk a deep-copied payload and replace ``values`` arrays with a placeholder
    so the rendered call doesn't dump 768 floats into the demo UI."""
    if isinstance(payload, dict):
        out: dict[str, Any] = {}
        for k, v in payload.items():
            if (
                k == "values"
                and isinstance(v, list)
                and v
                and all(isinstance(x, (int, float)) for x in v)
            ):
                out[k] = f"<{len(v)}-dim Gemini embedding>"
            elif k == "sparse_values" and isinstance(v, dict):
                out[k] = "<sparse {indices, values}>"
            else:
                out[k] = _abbreviate_vectors(v)
        return out
    if isinstance(payload, list):
        return [_abbreviate_vectors(x) for x in payload]
    return payload


def _format_search_call(kwargs: dict) -> str:
    """Render a ``documents.search(**kwargs)`` call as Python-ish source.

    JSON dump is good enough for our payloads (no None/booleans inside) and
    keeps quoting predictable. Long vector lists are abbreviated upstream.
    """
    display = _abbreviate_vectors(copy.deepcopy(kwargs))
    parts = ["idx.documents.search("]
    for key, val in display.items():
        val_repr = json.dumps(val, indent=2, ensure_ascii=False)
        val_repr = val_repr.replace("\n", "\n    ")
        parts.append(f"    {key}={val_repr},")
    parts.append(")")
    return "\n".join(parts)


def _execute(kwargs: dict, *, extra_code: str = "") -> SearchResult:
    resp = _index().documents.search(**kwargs)
    return SearchResult(
        matches=list(resp.matches),
        kwargs=kwargs,
        code=_format_search_call(kwargs),
        extra_code=extra_code,
    )


# -----------------------------------------------------------------------------
# Search helpers
# -----------------------------------------------------------------------------

def search_text(q: str, field: str = "body", top_k: int = 10) -> SearchResult:
    """Single-field FTS over one of bird_name / intro / body. BM25 token-OR.

    This is the most basic way to search. Each word gains an "OR", so results will
    contain any of the query words. Notably, this does not preserve phrases or word order.

    Multi-word queries match documents that contain *any* of the words —
    documents matching more words score higher. For phrase semantics
    (adjacency required), use :func:`search_text_phrase` instead.
    """
    return _execute({
        "namespace": NAMESPACE,
        "top_k": top_k,
        "score_by": [{"type": "text", "field": field, "query": q}],
        "include_fields": INCLUDE_FIELDS,
    })


_MULTI_FIELDS = ("bird_name", "intro", "body")


def search_text_multi(
    queries: str | dict[str, str], top_k: int = 10
) -> SearchResult:
    """Blended multi-field FTS — rewards docs relevant in multiple fields.

    Two call shapes:

    - **``str``** — broadcast the same query to all three text fields
      (``bird_name``, ``intro``, ``body``). Server combines per-field BM25
      scores into one per-doc relevance score; documents that match in more
      fields rank higher.
    - **``dict[str, str]``** — per-field queries. Each ``(field, query)``
      pair becomes its own ``text`` clause; empty / whitespace-only values
      are skipped. Lets you combine signals like
      ``{"bird_name": "swallow", "body": "in mountains"}`` without forcing
      one search string to satisfy every field.

    For phrase semantics see :func:`search_text_phrase` with
    ``field='multi'``.
    """
    if isinstance(queries, str):
        clauses = [
            {"type": "text", "field": f, "query": queries}
            for f in _MULTI_FIELDS
        ]
    else:
        clauses = [
            {"type": "text", "field": f, "query": q.strip()}
            for f, q in queries.items()
            if q and q.strip()
        ]
        if not clauses:
            raise ValueError(
                "search_text_multi: dict form needs at least one non-empty query"
            )

    return _execute({
        "namespace": NAMESPACE,
        "top_k": top_k,
        "score_by": clauses,
        "include_fields": INCLUDE_FIELDS,
    })


def search_text_phrase(q: str, field: str = "body", top_k: int = 10) -> SearchResult:
    """Exact-phrase FTS via Lucene ``query_string`` with quotes.
    A phrase here means a series of words that you want to find together in the search results.


    Use this for queries whose meaning lives in word order or adjacency
    (e.g. "state bird of seven", "national bird of the United States").
    BM25 token-OR over the same query underrates the right document because
    each common token (state, bird, of, seven) is too abundant on its own.

    With ``field='multi'`` the phrase is OR-ed across all three fields:
    ``bird_name:("…") OR intro:("…") OR body:("…")``.
    """
    safe = q.replace('"', r'\"')
    if field == "multi":
        clauses = " OR ".join(
            f'{f}:("{safe}")' for f in ("bird_name", "intro", "body")
        )
        query_str = clauses
    else:
        query_str = f'{field}:("{safe}")'
    return _execute({
        "namespace": NAMESPACE,
        "top_k": top_k,
        "score_by": [{"type": "query_string", "query": query_str}],
        "include_fields": INCLUDE_FIELDS,
    })


def search_query_string(query: str, top_k: int = 10) -> SearchResult:
    """Raw Lucene ``query_string`` — boolean ops, boost, slop, phrase prefix.

    For more complicated queries, you can write raw Lucene syntax in query_string, to query Pinecone.
    This allows for loads of operations that finely give you control over how the search works. 
    For example, you can use boolean operators (AND, OR, NOT), boost certain terms with ^ (i.e increase the importance of), 
    specify phrase slop with ~, and even do prefix matching for autocomplete-like behavior.

    The user types Lucene directly. Examples:
      ``body:("state bird of seven")``, which is a phrase query like search_text_phrase but with explicit syntax.
      ``body:(+illinois +cardinal -opinion)``, which requires "illinois" and "cardinal" but excludes "opinion".
      ``body:(eagle^3 OR hawk OR raptor)`` which boosts "eagle" matches above "hawk" or "raptor".
      ``body:("northern cardinal"~3)`` which allows "northern" and "cardinal" to be separated by up to 3 other words.
      ``body:("james w"*)``    # phrase-prefix (autocomplete-ish)
    Field names available: ``bird_name``, ``intro``, ``body``.
    """
    return _execute({
        "namespace": NAMESPACE,
        "top_k": top_k,
        "score_by": [{"type": "query_string", "query": query}],
        "include_fields": INCLUDE_FIELDS,
    })


def search_visual(q: str, top_k: int = 10) -> SearchResult:
    """Cross-modal: embed TEXT query via Gemini-2, score against stored IMAGE
    embeddings. Both the text and the image land in Gemini-2's shared space."""
    emb = embed_text(q)
    return _execute({
        "namespace": NAMESPACE,
        "top_k": top_k,
        "score_by": [
            {"type": "dense_vector", "field": "image_embedding", "values": emb}
        ],
        "include_fields": INCLUDE_FIELDS,
    })


def search_filter_visual(
    filter_terms: list[str],
    visual_q: str,
    filter_field: str = "body",
    top_k: int = 10,
) -> SearchResult:
    """Compound query: require every term in ``filter_terms`` to appear in
    ``filter_field`` (server-side ``$match_all``) and rank survivors by
    dense-vector similarity to ``visual_q``'s Gemini-2 embedding. One
    Pinecone call — hard filter + visual rerank.

    Example: ``filter_terms=["illinois"]`` on ``body`` plus
    ``visual_q="red bird with black wings"`` → Illinois-range birds
    ranked by visual similarity to that description.

    **UI vs library behavior.** The Combined tab in ``app.py`` requires
    *both* a non-empty filter and a non-empty visual description before
    it will call this helper — it shows a warning otherwise. Programmatic
    callers, however, can pass an empty / whitespace-only
    ``filter_terms`` list, in which case this helper short-circuits to a
    pure visual search via :func:`search_visual`. This split is
    intentional: the UI guards the cross-modal demo beat (filter + rerank
    is the headline), while the library stays composable.

    The helper used to fall back to a wide dense fetch + Python-side
    substring filter for backends that hadn't picked up ``$match_all``
    yet; that fallback was removed once preprod accepted the operator.
    If a future backend regresses on ``$match_all``, this helper will
    raise rather than degrade — the live tests will catch it first.
    """
    emb = embed_text(visual_q)
    required = [t.strip() for t in filter_terms if t.strip()]
    if not required:
        return search_visual(visual_q, top_k=top_k)

    return _execute({
        "namespace": NAMESPACE,
        "top_k": top_k,
        "filter": {filter_field: {"$match_all": " ".join(required)}},
        "score_by": [
            {"type": "dense_vector", "field": "image_embedding", "values": emb}
        ],
        "include_fields": INCLUDE_FIELDS,
    })


# -----------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF) — client-side hybrid search.
# -----------------------------------------------------------------------------
#
# Pinecone currently has no built-in way to fuse two independently-issued searches
# (unlike search_text_multi's single-call multi-field blend, which is
# Pinecone's own internal per-call combination of same-shaped scores). BM25
# text scores and cosine dense-vector scores also live on incomparable
# scales, so summing them directly wouldn't be meaningful even in one call.
# RRF sidesteps both problems by fusing *rankings* instead of raw scores.

def _rrf_combine(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: for each ranking (a list of doc ids in rank
    order), add ``1 / (k + rank)`` to that doc's score, rank starting at 1.
    A doc absent from a ranking contributes 0 from it. ``k=60`` is the
    default from the original RRF paper (Cormack et al., 2009) — a larger
    k discounts how much any single ranking's #1 spot dominates the fused
    order."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class _RRFMatch:
    """Wraps an SDK match, exposing its fused RRF score as ``.score``
    instead of the match's original per-list score, so generic score
    display (the UI, ``highlight_for_field``, etc.) shows the number the
    fused list is actually sorted by."""

    def __init__(self, match: Any, score: float):
        self._match = match
        self._id = match._id
        self.score = score

    def get(self, *args: Any, **kwargs: Any) -> Any:
        return self._match.get(*args, **kwargs)


def search_hybrid_rrf(
    text_q: str,
    visual_q: str,
    field: str = "body",
    top_k: int = 10,
    fetch_k: int = 50,
    k: int = 60,
) -> SearchResult:
    """Hybrid search via Reciprocal Rank Fusion: run a BM25 text search and
    a dense-vector visual search independently, then merge their rankings
    client-side (see :func:`_rrf_combine`).

    ``fetch_k`` controls how deep each individual ranking is fetched before
    fusing (must be >= top_k). A doc ranked outside ``fetch_k`` in one list
    contributes 0 from that list, so raising ``fetch_k`` lets a
    weaker-but-present signal still count toward the fused score.
    """
    idx = _index()

    text_kwargs = {
        "namespace": NAMESPACE,
        "top_k": fetch_k,
        "score_by": [{"type": "text", "field": field, "query": text_q}],
        "include_fields": INCLUDE_FIELDS,
    }
    text_matches = list(idx.documents.search(**text_kwargs).matches)

    emb = embed_text(visual_q)
    dense_kwargs = {
        "namespace": NAMESPACE,
        "top_k": fetch_k,
        "score_by": [
            {"type": "dense_vector", "field": "image_embedding", "values": emb}
        ],
        "include_fields": INCLUDE_FIELDS,
    }
    dense_matches = list(idx.documents.search(**dense_kwargs).matches)

    rrf_scores = _rrf_combine(
        [[m._id for m in text_matches], [m._id for m in dense_matches]], k=k
    )
    by_id = {m._id: m for m in text_matches}
    for m in dense_matches:
        by_id.setdefault(m._id, m)
    fused_ids = sorted(by_id, key=lambda i: rrf_scores[i], reverse=True)[:top_k]
    matches = [_RRFMatch(by_id[doc_id], rrf_scores[doc_id]) for doc_id in fused_ids]

    extra_code = (
        "# Pinecone currently has no built-in fusion across separate calls — merge two\n"
        "# independently-issued rankings client-side via Reciprocal Rank Fusion:\n"
        "dense_response = idx.documents.search(\n"
        f"    namespace={NAMESPACE!r},\n"
        f"    top_k={fetch_k},\n"
        '    score_by=[{"type": "dense_vector", "field": "image_embedding", '
        f'"values": <{len(emb)}-dim embedding>}}],\n'
        f"    include_fields={INCLUDE_FIELDS!r},\n"
        ")\n"
        "\n"
        f"def rrf_combine(rankings, k={k}):\n"
        "    scores = {}\n"
        "    for ranking in rankings:\n"
        "        for rank, doc_id in enumerate(ranking, start=1):\n"
        "            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank)\n"
        "    return scores\n"
        "\n"
        "rrf_scores = rrf_combine([\n"
        "    [m._id for m in text_response.matches],\n"
        "    [m._id for m in dense_response.matches],\n"
        "])\n"
        f"fused = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:{top_k}]"
    )

    return SearchResult(
        matches=matches,
        kwargs=text_kwargs,
        code=_format_search_call(text_kwargs),
        extra_code=extra_code,
    )
