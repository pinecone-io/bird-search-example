"""Bird Search v2 — search helpers.

Thin wrappers over `idx.documents.search(...)` for each of the UI tabs:

    search_text           single-field FTS, BM25 token-OR (`type: "text"`)
    search_text_multi     blended multi-field FTS (bird_name + intro + body)
    search_text_phrase    exact-phrase FTS via `query_string` with quotes
    search_query_string   raw Lucene query_string (boolean / boost / slop / …)
    search_visual         Gemini-2 text embed scored against stored image
                          embeddings (cross-modal)
    search_filter_visual  dense visual rank + client-side body keyword guardrail


Every helper returns a ``SearchResult`` with three things:

- ``matches`` — list of match objects (drop-in for the previous SDK response).
- ``kwargs`` — the dict actually passed to ``documents.search(**kwargs)``.
- ``code`` — a Python source snippet of that call, ready for ``st.code(...)``
  in the UI so demo viewers can see exactly what hit Pinecone.

``search_filter_visual`` also sets ``extra_code`` describing the Python-side
filter step (until the operator lands and we can collapse it back to one call).
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


pc = Pinecone(additional_headers={"x-environment": "preprod-aws-0"})
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
    candidate_pool: int = 200,
) -> SearchResult:
    """
    This function handles the semantic-visual only search path, and the combined full text filter path.

    The combined text path works kinda like a filter + dense search, where the search space is constrained
    first by a hard filter on the text fields, and then the remaining are ranked by visual similarity to the query.

    You can imagine this being pretty handy for a situation where you see a bird in a certain place, or a location,
    and you restrict your search only to those places, using the description you saw.

    
    Compound query: require all filter_terms in a text field + rank by
    dense-vector similarity to visual_q's Gemini-2 embedding. Example:
    filter=["illinois"] on body + visual_q="red bird with black wings" →
    Illinois-range birds ranked by visual similarity to that description.

    **Primary path** — public-preview docs (2026-01.alpha) document a
    ``$match_all`` filter operator that takes a space-separated string of
    required tokens. Combined with a ``dense_vector`` score_by, that gives
    us the intended single-call shape: server-side hard filter, dense
    rerank.

    **Fallback path** — if the primary call raises (e.g. the backend we're
    pointed at hasn't picked up ``$match_all`` yet), we degrade to a wide
    dense fetch and a Python-side substring filter. The returned
    ``SearchResult.code`` reflects whichever path actually ran, and the
    fallback also stashes the API error in ``extra_code`` so the demo UI
    is honest about what happened.

    Empty ``filter_terms`` short-circuits to pure visual search.
    """
    emb = embed_text(visual_q)
    required = [t.strip() for t in filter_terms if t.strip()]
    if not required:
        return search_visual(visual_q, top_k=top_k)

    # ---- Primary: single-call form with $match_all ------------------------
    primary_kwargs = {
        "namespace": NAMESPACE,
        "top_k": top_k,
        "filter": {filter_field: {"$match_all": " ".join(required)}},
        "score_by": [
            {"type": "dense_vector", "field": "image_embedding", "values": emb}
        ],
        "include_fields": INCLUDE_FIELDS,
    }
    try:
        resp = _index().documents.search(**primary_kwargs)
        return SearchResult(
            matches=list(resp.matches),
            kwargs=primary_kwargs,
            code=_format_search_call(primary_kwargs),
        )
    except Exception as exc:
        primary_error = str(exc)

    # ---- Fallback: dense fetch + client-side substring filter -------------
    fallback_kwargs = {
        "namespace": NAMESPACE,
        "top_k": candidate_pool,
        "score_by": [
            {"type": "dense_vector", "field": "image_embedding", "values": emb}
        ],
        "include_fields": INCLUDE_FIELDS,
    }
    resp = _index().documents.search(**fallback_kwargs)

    required_lc = [t.lower() for t in required]
    survivors: list[Any] = []
    for m in resp.matches:
        text = (m.get(filter_field) or "").lower()
        if all(term in text for term in required_lc):
            survivors.append(m)
        if len(survivors) >= top_k:
            break

    extra = (
        f"# $match_all rejected by backend: {primary_error[:160]}\n"
        "# Falling back to wide dense fetch + Python-side substring filter.\n"
        "survivors = [\n"
        "    m for m in resp.matches\n"
        f"    if all(t in (m.get({filter_field!r}) or '').lower() for t in {required_lc!r})\n"
        f"][:{top_k}]"
    )

    return SearchResult(
        matches=survivors,
        kwargs=fallback_kwargs,
        code=_format_search_call(fallback_kwargs),
        extra_code=extra,
    )
