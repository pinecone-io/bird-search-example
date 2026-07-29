"""Streamlit UI for Bird Search v2.

Five search modes over one Pinecone preview FTS index:
  - Text FTS: BM25 token-OR over ``bird_name`` / ``intro`` / ``body``, either
    per-field or blended (``multi``). The blended mode also supports
    different queries per field (e.g. bird_name=swallow + body=mountains).
    An "exact phrase" toggle routes the query through Lucene ``query_string``
    when adjacency matters.
  - Visual: typed description embedded via the configured embedding
    provider (local SigLIP by default, or Gemini — see embedder.py),
    scored against each bird's primary image vector in the same
    multimodal space.
  - Lucene: raw ``query_string`` — boolean operators, required /
    excluded terms, term boosts, phrase slop, phrase prefixes, cross-field.
  - Combined: server-side ``$match_all`` filter on ``body`` (every required
    term must appear) plus dense-vector ranking on the image embedding —
    one round trip.
  - Hybrid (RRF): text search and visual search run independently, then
    fused client-side via Reciprocal Rank Fusion — Pinecone has no
    built-in way to combine two separately-issued searches.

Each tab renders the actual ``documents.search(...)`` call beneath the
results so viewers can see exactly what hit Pinecone.
"""

from __future__ import annotations

import html
import json
import os
import pathlib
import re

import streamlit as st

from query import (
    search_filter_visual,
    search_hybrid_rrf,
    search_query_string,
    search_text,
    search_text_multi,
    search_text_phrase,
    search_visual,
)

# ---------------------------------------------------------------------------
# Data location + metadata (loaded once at startup).
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = pathlib.Path(__file__).resolve().parent / "parsed_birds"
BIRD_DATA_DIR = pathlib.Path(
    os.environ.get("BIRD_DATA_DIR", str(_DEFAULT_DATA_DIR))
).expanduser()
IMAGES_DIR = BIRD_DATA_DIR / "images"
METADATA_PATH = BIRD_DATA_DIR / "parsing_metadata.json"


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text())


METADATA = load_metadata()


# ---------------------------------------------------------------------------
# Query-word highlighter.
#
# Highlight tokens are derived from the structured request we sent to
# Pinecone (``SearchResult.kwargs``) rather than the raw user input string,
# so per-field clauses light up only their own field — e.g.
# ``bird_name:(swallow*) OR body:(eagle)`` highlights ``swallow`` in the
# title and ``eagle`` in the body, not vice versa. Filter operators
# ($match_all / $match_phrase / $match_any) contribute tokens too.
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"([A-Za-z][A-Za-z\-']*)")
_LUCENE_OPERATORS = {"and", "or", "not", "to"}
# ``field:(...)`` or ``field:value`` scoped clause inside a query_string.
_FIELD_CLAUSE_RE = re.compile(
    r"(\w+):(?:\(([^)]*)\)|((?:\"[^\"]*\")|\S+))"
)
# Term inside a Lucene chunk. The optional sign captures ``+`` / ``-`` so
# excluded terms can be dropped from the highlight set.
_TERM_TOKEN_RE = re.compile(r"([+\-]?)([A-Za-z][A-Za-z\-']*)")


def _tokens_from_lucene_chunk(chunk: str) -> set[str]:
    """Extract positive word tokens from a Lucene chunk like
    ``+illinois +cardinal -opinion`` or ``"northern cardinal"~3``.

    Drops excluded terms (``-foo``), boolean operators (``AND/OR/NOT/TO``)
    and any non-alphabetic suffixes (boost ``^3``, slop ``~3``,
    wildcard ``*``)."""
    out: set[str] = set()
    for sign, word in _TERM_TOKEN_RE.findall(chunk):
        if sign == "-":
            continue
        wl = word.lower()
        if wl in _LUCENE_OPERATORS or len(word) < 2:
            continue
        out.add(wl)
    return out


def _tokens_for_field_from_query_string(qs: str, target_field: str) -> set[str]:
    """Pull tokens scoped to ``target_field`` out of a raw Lucene
    ``query_string``. Tokens written without a field prefix are unscoped
    and contribute to every text field's highlight set."""
    out: set[str] = set()
    consumed: list[tuple[int, int]] = []
    for m in _FIELD_CLAUSE_RE.finditer(qs):
        scoped_field = m.group(1)
        body = m.group(2) if m.group(2) is not None else (m.group(3) or "")
        consumed.append(m.span())
        if scoped_field == target_field:
            out |= _tokens_from_lucene_chunk(body.strip('"'))

    # Anything outside ``field:(...)`` blocks is unscoped — applies to every
    # text field. Slice those leftover spans back out of ``qs``.
    leftover_parts: list[str] = []
    last = 0
    for s, e in consumed:
        if s > last:
            leftover_parts.append(qs[last:s])
        last = e
    if last < len(qs):
        leftover_parts.append(qs[last:])
    leftover = " ".join(leftover_parts).strip()
    if leftover:
        out |= _tokens_from_lucene_chunk(leftover)
    return out


def _filter_tokens_for_field(filter_dict, target_field: str) -> set[str]:
    """Walk a Pinecone ``filter`` dict for text-match operators keyed at
    ``target_field`` and return the highlight tokens they imply."""
    if not isinstance(filter_dict, dict):
        return set()
    out: set[str] = set()
    for k, v in filter_dict.items():
        if k in ("$and", "$or") and isinstance(v, list):
            for child in v:
                out |= _filter_tokens_for_field(child, target_field)
            continue
        if k == "$not":
            # Excluded clause — don't highlight anything from it.
            continue
        if k == target_field and isinstance(v, dict):
            for op in ("$match_phrase", "$match_all", "$match_any"):
                if op in v and isinstance(v[op], str):
                    out |= _tokens_from_lucene_chunk(v[op])
    return out


def tokens_for_field(result, field: str) -> set[str]:
    """Tokens that should highlight inside the document's ``field`` text,
    based on the ``score_by`` clauses + ``filter`` operators in the call
    we just made. Returns an empty set when no text-shaped clause targets
    this field (e.g. a pure ``dense_vector`` search)."""
    if result is None:
        return set()
    kwargs = getattr(result, "kwargs", None) or {}
    out: set[str] = set()

    for clause in kwargs.get("score_by") or []:
        ctype = clause.get("type")
        if ctype == "text":
            # Public-preview docs use ``fields: [...]`` (array); preprod
            # still accepts the older singular ``field``. Honor both.
            scoped = clause.get("fields")
            if scoped is None:
                f = clause.get("field")
                scoped = [f] if f else []
            if field in scoped:
                out |= _tokens_from_lucene_chunk(clause.get("query") or "")
        elif ctype == "query_string":
            qs = clause.get("query") or ""
            out |= _tokens_for_field_from_query_string(qs, field)
        # dense_vector / sparse_vector contribute nothing to text highlights.

    out |= _filter_tokens_for_field(kwargs.get("filter"), field)
    return out


def _highlight_with_tokens(text: str, tokens: set[str]) -> str:
    """Wrap any word in ``text`` whose lowercase form prefix-matches a
    token in ``tokens``. The prefix heuristic (>= 3 char overlap) keeps
    stemmed FTS hits visible (``peck`` lights up ``pecks``/``pecking``).
    Whitespace and punctuation are preserved; all output is HTML-escaped.
    Render with ``st.markdown(..., unsafe_allow_html=True)``."""
    if not tokens:
        return html.escape(text)

    def is_match(word_lc: str) -> bool:
        for q in tokens:
            if q == word_lc:
                return True
            shorter, longer = (q, word_lc) if len(q) <= len(word_lc) else (word_lc, q)
            if len(shorter) >= 3 and longer.startswith(shorter):
                return True
        return False

    parts: list[str] = []
    last = 0
    for m in _WORD_RE.finditer(text):
        if m.start() > last:
            parts.append(html.escape(text[last : m.start()]))
        word = m.group(0)
        if is_match(word.lower()):
            parts.append(f"<mark>{html.escape(word)}</mark>")
        else:
            parts.append(html.escape(word))
        last = m.end()
    if last < len(text):
        parts.append(html.escape(text[last:]))
    return "".join(parts)


def highlight_for_field(text: str, field: str, result) -> str:
    """Highlight a document's ``field`` text using only the tokens
    targeted at that field by the search call. Returns HTML-escaped
    output safe to render with ``unsafe_allow_html=True``."""
    return _highlight_with_tokens(text, tokens_for_field(result, field))


# ---------------------------------------------------------------------------
# Shared result card renderer.
# ---------------------------------------------------------------------------

def _thumbnail_path(slug: str) -> pathlib.Path | None:
    entry = METADATA.get(slug)
    if not entry:
        return None
    images = entry.get("images") or []
    if not images:
        return None
    local_path = images[0].get("local_path")
    if not local_path:
        return None
    return IMAGES_DIR / local_path


def _render_call_snippet(response) -> None:
    """Show the actual ``documents.search(...)`` call (and any post-call
    Python step) for the just-completed search, as a collapsible toggle so
    viewers can pop it open to see what hit Pinecone. Reads ``response.code``
    and ``response.extra_code`` populated by the helpers in ``query.py``."""
    code = getattr(response, "code", None)
    if not code:
        return
    with st.expander("What we sent to Pinecone", expanded=False):
        st.code(code, language="python")
        extra = getattr(response, "extra_code", "") or ""
        if extra:
            st.code(extra, language="python")


def _render_compact_ranking(
    label: str, matches: list, new_ids: set[str] | None = None
) -> None:
    """Compact ranked list (name + score only) for side-by-side comparison
    views — full result cards (thumbnail, intro, body expander) are too
    heavy to repeat three times over. ``new_ids`` marks entries with 🆕
    that don't appear in the other lists being compared against."""
    st.markdown(f"**{label}**")
    if not matches:
        st.caption("No matches.")
        return
    for i, m in enumerate(matches, start=1):
        name = m.get("bird_name") or m._id.replace("_", " ")
        score = getattr(m, "score", None)
        score_str = f"{score:.3f}" if score is not None else "—"
        marker = " 🆕" if new_ids and m._id in new_ids else ""
        st.markdown(f"{i}. {name} — `{score_str}`{marker}")


def render_results(response) -> None:
    # Call snippet first — clickable toggle above the results so demo
    # viewers can see what was sent before they scroll the matches.
    _render_call_snippet(response)

    matches = getattr(response, "matches", None) or []
    if not matches:
        st.info("No matches.")
        return

    for doc in matches:
        slug = doc._id
        bird_name = doc.get("bird_name") or slug.replace("_", " ")
        score = getattr(doc, "score", None)

        with st.container(border=True):
            score_badge = (
                f'<span class="bird-card-score">score {score:.3f}</span>'
                if score is not None else ""
            )
            # Highlight bird_name only if the search actually scored on
            # bird_name (e.g. multi-field with bird_name="swallow").
            title_html = highlight_for_field(bird_name, "bird_name", response)
            st.markdown(
                f'<div class="bird-card-header">'
                f'<span class="bird-card-title">{title_html}</span>'
                f'{score_badge}'
                f'</div>',
                unsafe_allow_html=True,
            )
            cols = st.columns([1, 3])
            thumb = _thumbnail_path(slug)
            with cols[0]:
                if thumb and thumb.exists():
                    try:
                        st.image(str(thumb), use_container_width=True)
                    except Exception:
                        pass
            with cols[1]:
                intro = doc.get("intro") or ""
                body = doc.get("body") or ""

                if intro:
                    st.markdown(
                        highlight_for_field(intro, "intro", response),
                        unsafe_allow_html=True,
                    )

                if body:
                    with st.expander("Full article body"):
                        st.markdown(
                            highlight_for_field(body, "body", response),
                            unsafe_allow_html=True,
                        )


# ---------------------------------------------------------------------------
# Page layout.
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Bird Search", layout="wide")

logo_path = pathlib.Path(__file__).parent / "Pinecone-Primary-Logo-Black.png"
if logo_path.exists():
    st.logo(str(logo_path))



st.markdown(
    """
<style>
/* Chip styling for example-preset (secondary) buttons. */
.stButton > button[kind="secondary"] {
    height: auto;
    min-height: 0;
    padding: 0.25rem 0.85rem;
    font-size: 0.85rem;
    font-weight: 500;
    border-radius: 999px;
    border-color: rgba(49, 51, 63, 0.18);
    color: rgba(49, 51, 63, 0.78);
    line-height: 1.35;
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(49, 51, 63, 0.04);
    border-color: rgba(49, 51, 63, 0.32);
    color: rgba(49, 51, 63, 1);
}

/* Primary (Search) buttons — keep weight, just align with the page rhythm. */
.stButton > button[kind="primary"] {
    padding: 0.45rem 1.4rem;
    font-weight: 600;
}

/* Result card: bird-name left, score-pill right, neutral border. */
.bird-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin: 0 0 0.5rem 0;
}
.bird-card-title {
    font-size: 1.18rem;
    font-weight: 600;
    color: rgba(49, 51, 63, 0.95);
    line-height: 1.3;
}
.bird-card-score {
    background: rgba(49, 51, 63, 0.07);
    color: rgba(49, 51, 63, 0.78);
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}

/* <mark> highlights — softer than browser yellow. Text forced dark since
   a light-yellow chip reads best with dark text in either theme. */
mark {
    background: rgba(255, 213, 79, 0.45);
    color: rgba(49, 51, 63, 0.95);
    padding: 0 0.1rem;
    border-radius: 2px;
}

/* Streamlit doesn't expose its active theme as CSS custom properties, so
   dark-mode support for the hardcoded colors above follows the OS-level
   preference directly (this matches Streamlit's own default "Use system
   setting" theme, though not a manual in-app override away from it). */
@media (prefers-color-scheme: dark) {
    .stButton > button[kind="secondary"] {
        border-color: rgba(250, 250, 250, 0.18);
        color: rgba(250, 250, 250, 0.78);
    }
    .stButton > button[kind="secondary"]:hover {
        background: rgba(250, 250, 250, 0.08);
        border-color: rgba(250, 250, 250, 0.32);
        color: rgba(250, 250, 250, 1);
    }
    .bird-card-title {
        color: rgba(250, 250, 250, 0.95);
    }
    .bird-card-score {
        background: rgba(250, 250, 250, 0.12);
        color: rgba(250, 250, 250, 0.78);
    }
}
</style>
    """,
    unsafe_allow_html=True,
)

st.title("Bird Search")

# ---------------------------------------------------------------------------
# Top of page: description on the left, live index schema on the right.
# Showing the schema up-front lets the audience see exactly what fields are
# searchable / filterable / vector before they touch any tab. The block is
# the same shape ``build_index.py`` actually sends.
# ---------------------------------------------------------------------------

INDEX_SCHEMA_SOURCE = '''SchemaBuilder()
  .add_string_field("bird_name",
                    full_text_search={"language": "en"})
  .add_string_field("intro",
                    full_text_search={"language": "en"})
  .add_string_field("body",
                    full_text_search={"language": "en",
                                      "stemming": True})
  .add_dense_vector_field("image_embedding",
                          dimension=768, metric="cosine")
  .build()'''

intro_col, schema_col = st.columns([3, 2])
with intro_col:
    st.write(
        "Search ~2,079 North American birds five ways: **Text FTS** "
        "(keyword search over the article fields), **Visual** (type what "
        "the bird looks like — matched against each bird's photo in a "
        "shared text/image embedding space), **Lucene** (raw "
        "`query_string` for boost / slop / phrase prefix / cross-field "
        "queries), **Combined** (visual ranking narrowed by required "
        "keywords on the article body), and **Hybrid (RRF)** (text and "
        "visual search fused client-side via Reciprocal Rank Fusion)."
    )
    st.write(
        "Each tab also shows the actual `documents.search(...)` call beneath "
        "its results, so you can see exactly what hit Pinecone."
    )
    # "How to construct…" lives inside the left column so it fills the
    # L-shaped whitespace next to the schema block (which is the taller
    # of the two columns).
    st.markdown(
        """
**How to construct a new query**

1. **Pick the tab that matches your signal.** Words you'd find *in the article* → **Text FTS** or **Lucene**. Words describing what the *bird looks like* → **Visual**. Both at once → **Combined**.
2. **Write the query**, using the example buttons under each tab as templates.
3. **Open "What we sent to Pinecone"** above the results to see the exact `documents.search(...)` call. Every example here is reproducible from that snippet.
"""
    )
with schema_col:
    st.markdown("**Index schema**")
    st.code(INDEX_SCHEMA_SOURCE, language="python")
    st.caption(
        "Index `bird-search-fts` · namespace `birds` · "
        "one doc per bird, ~2,079 docs total."
    )


# ---------------------------------------------------------------------------
# Tab quick reference — full-width below the two-col block, since the table
# itself is wide. Lets viewers orient before clicking into a tab.
# ---------------------------------------------------------------------------

st.markdown(
    """
**Tab quick reference**

| Tab | API shape | Pick when… | Canonical example |
|---|---|---|---|
| **Text FTS** | `score_by=[{"type": "text", "field": …}]` (or `query_string` when phrase ON) | You can name a token in the article. Use `multi` with per-field queries to combine signals. | `Mormon crickets` (body) → California gull |
| **Visual** | `score_by=[{"type": "dense_vector", "field": "image_embedding"}]` | You can describe the bird's appearance but not the article's vocabulary. | `tall pink wading bird with long curved neck` → American flamingo |
| **Lucene** | `score_by=[{"type": "query_string"}]` (raw Lucene) | You need boosts, slop, phrase prefixes, exclusions, or cross-field clauses. | `body:(eagle^3 OR hawk OR raptor)` → eagles dominate |
| **Combined** | `filter={"body": {"$match_all": …}}` + dense `score_by` | You need both — a hard text gate and visual rerank. | `swoop, illinois` + `black bird with bright spots on wings` → Red-winged blackbird |
| **Hybrid (RRF)** | Two separate calls (`text` + `dense_vector`), fused client-side | Neither signal alone is decisive, but *both* being decent should count for something. | `songbird territory` + `bright red bird` → Northern cardinal (outside both individual top 5s) |
"""
)


# ---------------------------------------------------------------------------
# Example-query helper. Each tab calls `_example_buttons` with a list of
# preset specs; clicking a button writes them into the session_state keys
# the tab's widgets bind to, then sets a one-shot auto-run flag picked up
# by the tab's search block on the next render.
# ---------------------------------------------------------------------------

def _example_buttons(label_prefix: str, presets: list[dict]) -> None:
    st.markdown("**Try an example:**")
    cols = st.columns(len(presets))
    for i, preset in enumerate(presets):
        if cols[i].button(preset["label"], key=f"{label_prefix}_ex_{i}"):
            for state_key, value in preset["state"].items():
                st.session_state[state_key] = value
            st.session_state[f"{label_prefix}_run_now"] = True
            st.rerun()


def _consume_auto_run(label_prefix: str) -> bool:
    """One-shot read of the auto-run flag set by ``_example_buttons``."""
    return bool(st.session_state.pop(f"{label_prefix}_run_now", False))


tab_text, tab_visual, tab_boolean, tab_combined, tab_hybrid = st.tabs(
    ["Text FTS", "Visual", "Lucene", "Combined", "Hybrid (RRF)"]
)

# --- Tab 1: Text FTS ------------------------------------------------------
with tab_text:
    st.header("Text FTS")
    st.markdown(
        "BM25 keyword scoring against your chosen field.\n\n"
        "- **Pick a field** to scope the search: `body` (article prose), "
        "`intro` (Wikipedia lede), or `bird_name` (species name).\n"
        "- **`multi`** searches all three at once — and lets you write a "
        "*different* query per field, e.g. `bird_name=swallow` + "
        "`body=in mountains`. The server combines per-field scores into "
        "one rank, so birds that match in more fields rise.\n"
        "- **Phrase OFF** (default) is BM25 token-OR: each word scores "
        "independently. Best for queries with rare, distinctive tokens "
        "(`Mormon crickets`).\n"
        "- **Phrase ON** wraps your query in quotes and routes through "
        "Lucene `query_string`, requiring word adjacency. Use it when "
        "meaning lives in word order (`state bird of seven`)."
    )

    _example_buttons("text", [
        {
            "label": "Mormon crickets · body",
            "state": {"text_query": "Mormon crickets",
                      "text_field": "body",
                      "text_phrase": False},
        },
        {
            "label": "Miracle of the Gulls · multi (broadcast)",
            "state": {"text_query": "Miracle of the Gulls",
                      "multi_bird_name": "Miracle of the Gulls",
                      "multi_intro": "Miracle of the Gulls",
                      "multi_body": "Miracle of the Gulls",
                      "text_field": "multi",
                      "text_phrase": False},
        },
        {
            "label": "swallow + mountains · multi (per-field)",
            "state": {"multi_bird_name": "swallow",
                      "multi_intro": "",
                      "multi_body": "in mountains",
                      "text_field": "multi",
                      "text_phrase": False},
        },
        {
            "label": "state bird of seven · phrase",
            "state": {"text_query": "state bird of seven",
                      "text_field": "body",
                      "text_phrase": True},
        },
        {
            "label": "national bird of the United States · phrase",
            "state": {"text_query": "national bird of the United States",
                      "text_field": "body",
                      "text_phrase": True},
        },
    ])

    field_choice = st.radio(
        "Field",
        options=["body", "intro", "bird_name", "multi"],
        index=0,
        horizontal=True,
        key="text_field",
    )
    phrase_match = st.checkbox(
        "Match as exact phrase",
        value=False,
        key="text_phrase",
        help=(
            "Off (default): Pinecone `type: \"text\"` — BM25 token-OR. Each "
            "word scores independently.\n\n"
            "On: routes through `type: \"query_string\"` with the query "
            "wrapped in quotes (`field:(\"…\")`), so adjacency is required.\n\n"
            "Phrase mode uses the single Query input below regardless of "
            "field choice (and OR's across all three fields when "
            "field=multi)."
        ),
    )

    # Multi mode (non-phrase) opens three per-field inputs so the user can
    # combine signals like bird_name=swallow + body=in mountains. Every
    # other mode uses the single Query input.
    in_per_field_multi = field_choice == "multi" and not phrase_match
    if in_per_field_multi:
        st.markdown("**Per-field queries** — leave any field empty to skip it.")
        cols = st.columns(3)
        cols[0].text_input(
            "bird_name", key="multi_bird_name",
            placeholder="e.g. swallow",
        )
        cols[1].text_input(
            "intro", key="multi_intro",
            placeholder="(optional)",
        )
        cols[2].text_input(
            "body", key="multi_body",
            placeholder="e.g. in mountains",
        )
        query = ""  # not used in this branch
    else:
        query = st.text_input(
            "Query",
            placeholder="e.g. bright red wings pecks wood",
            key="text_query",
        )

    if st.button("Search", key="text_btn", type="primary") or _consume_auto_run("text"):
        if phrase_match:
            if query.strip():
                with st.spinner("Searching…"):
                    response = search_text_phrase(query, field=field_choice)
                render_results(response)
            else:
                st.warning("Enter a query.")
        elif in_per_field_multi:
            multi_queries = {
                "bird_name": st.session_state.get("multi_bird_name", ""),
                "intro": st.session_state.get("multi_intro", ""),
                "body": st.session_state.get("multi_body", ""),
            }
            non_empty = {f: q for f, q in multi_queries.items() if q.strip()}
            if non_empty:
                with st.spinner("Searching…"):
                    response = search_text_multi(non_empty)
                render_results(response)
            else:
                st.warning("Fill at least one of bird_name / intro / body.")
        else:
            if query.strip():
                with st.spinner("Searching…"):
                    response = search_text(query, field=field_choice)
                render_results(response)
            else:
                st.warning("Enter a query.")

# --- Tab 2: Visual --------------------------------------------------------
with tab_visual:
    st.header("Visual")
    st.markdown(
        "Cross-modal search — type **what the bird looks like** in plain "
        "English. Your text is embedded via the configured embedding "
        "provider and scored against each bird's primary photo embedding "
        "(same multimodal space). No keywords required: phrasing like "
        "*clown beak* or *roundest birds ever that are red* lands on the "
        "right photos even when the article never uses those words."
    )

    _example_buttons("visual", [
        {
            "label": "tall pink wading bird with long curved neck",
            "state": {"visual_query": "tall pink wading bird with long curved neck"},
        },
        {
            "label": "huge colorful beak like a clown",
            "state": {"visual_query": "black and white bird with a huge colorful beak like a clown"},
        },
        {
            "label": "white owl with yellow eyes on snow",
            "state": {"visual_query": "white owl with yellow eyes perched on snow"},
        },
        {
            "label": "iridescent green hovering at a flower",
            "state": {"visual_query": "small iridescent green bird hovering at a flower"},
        },
        {
            "label": "black bird with bright spots on wings",
            "state": {"visual_query": "black bird with bright spots on wings"},
        },
    ])

    visual_query = st.text_input(
        "Description",
        placeholder="e.g. roundest birds ever that are red",
        key="visual_query",
    )
    if st.button("Search", key="visual_btn", type="primary") or _consume_auto_run("visual"):
        if visual_query.strip():
            with st.spinner("Embedding and searching…"):
                response = search_visual(visual_query)
            render_results(response)
        else:
            st.warning("Enter a description.")

# --- Tab 3: Lucene (raw query_string) ------------------------------------
with tab_boolean:
    st.header("Lucene")
    st.markdown(
        "Write Lucene `query_string` directly when the Text FTS tab can't "
        "express what you want. Reach for this tab for **boosts** "
        "(`eagle^3` weights eagle 3× over peers), **phrase slop** "
        "(`\"northern cardinal\"~3` allows 3 tokens between the words), "
        "**phrase prefixes** (`\"james w\"*`), **required / excluded** "
        "terms (`+illinois -opinion`), and **cross-field clauses** "
        "(`bird_name:(swallow*) OR body:(swallow)`). Field names on this "
        "index: `bird_name`, `intro`, `body`."
    )

    _example_buttons("boolean", [
        {
            "label": 'phrase: "state bird of seven"',
            "state": {"boolean_query": 'body:("state bird of seven")'},
        },
        {
            "label": "boost: eagle^3 OR hawk OR raptor",
            "state": {"boolean_query": "body:(eagle^3 OR hawk OR raptor)"},
        },
        {
            "label": 'slop: "northern cardinal"~3',
            "state": {"boolean_query": 'body:("northern cardinal"~3)'},
        },
        {
            "label": "cross-field: bird_name OR body",
            "state": {"boolean_query": "bird_name:(swallow*) OR body:(swallow)"},
        },
        {
            "label": "+required −excluded",
            "state": {"boolean_query": "body:(+illinois +cardinal -opinion)"},
        },
    ])

    boolean_q = st.text_input(
        "Lucene query_string",
        placeholder='body:("state bird of seven")',
        key="boolean_query",
    )
    if st.button("Search", key="boolean_btn", type="primary") or _consume_auto_run("boolean"):
        if boolean_q.strip():
            with st.spinner("Searching…"):
                response = search_query_string(boolean_q)
            render_results(response)
        else:
            st.warning("Enter a Lucene query_string.")

# --- Tab 4: Combined ------------------------------------------------------
with tab_combined:
    st.header("Combined")
    st.markdown(
        "The headline cross-modal query — **filter by text, rerank by "
        "image, in one round trip**. Sent as `filter={\"body\": "
        "{\"$match_all\": \"…\"}}` (server-side hard filter; every required "
        "term must appear in the article body) plus a `dense_vector` "
        "`score_by` clause on `image_embedding` (the configured embedding "
        "provider ranks each survivor by visual similarity to your "
        "description). The "
        "demo flip: visual-only `black bird with bright spots on wings` "
        "lands on yellow-shouldered & tricolored blackbirds, antwrens, "
        "starlings; add `swoop, illinois` as required terms and the "
        "**Red-winged blackbird** leaps to #1 — its red shoulder patches "
        "*are* the territorial-defense markings the article describes."
    )

    _example_buttons("combined", [
        {
            "label": "swoop, illinois + black bird with bright spots on wings",
            "state": {"combined_filter": "swoop, illinois",
                      "combined_visual": "black bird with bright spots on wings"},
        },
        {
            "label": "mormon + white gull with gray wings",
            "state": {"combined_filter": "mormon",
                      "combined_visual": "white gull with gray wings"},
        },
        {
            "label": "tundra, arctic + large white bird",
            "state": {"combined_filter": "tundra, arctic",
                      "combined_visual": "large white bird"},
        },
    ])

    filter_raw = st.text_input(
        "Must mention (comma-separated terms)",
        placeholder="swoop, illinois",
        key="combined_filter",
    )
    visual_q = st.text_input(
        "Describe appearance",
        placeholder="black bird with bright spots on wings",
        key="combined_visual",
    )
    if st.button("Search", key="combined_btn", type="primary") or _consume_auto_run("combined"):
        filter_terms = [t.strip() for t in filter_raw.split(",") if t.strip()]
        if filter_terms and visual_q.strip():
            with st.spinner("Filtering and ranking…"):
                response = search_filter_visual(
                    filter_terms=filter_terms,
                    visual_q=visual_q,
                )
            render_results(response)
        else:
            st.warning("Enter at least one filter term and an appearance description.")

# --- Tab 5: Hybrid (RRF) ---------------------------------------------------
with tab_hybrid:
    st.header("Hybrid (RRF)")
    st.markdown(
        "Pinecone doesn't fuse two independently-issued searches for you — "
        "BM25 text scores and cosine visual scores live on incomparable "
        "scales, so summing them directly wouldn't mean anything even in "
        "one call. **Reciprocal Rank Fusion (RRF)** sidesteps that: run "
        "each search separately, then combine their *rankings* — "
        "`score(doc) = Σ 1/(k + rank)` summed across every ranking a doc "
        "appears in (`k=60`).\n\n"
        "The demo flip: text alone (`songbird territory`) and visual alone "
        "(`bright red bird`) both miss the **Northern cardinal** in their "
        "own top 5 (it's #19 on text, #6 on visual) — merely decent on "
        "*both* signals beats being #1 on only one, so fused, it's #1."
    )

    _example_buttons("hybrid", [
        {
            "label": "songbird territory + bright red bird",
            "state": {"hybrid_text": "songbird territory",
                      "hybrid_visual": "bright red bird"},
        },
        {
            "label": "coastal cliffs + gray and white seabird",
            "state": {"hybrid_text": "coastal cliffs",
                      "hybrid_visual": "gray and white seabird"},
        },
        {
            "label": "desert + small gray bird",
            "state": {"hybrid_text": "desert",
                      "hybrid_visual": "small gray bird"},
        },
    ])

    hybrid_text_q = st.text_input(
        "Text query (BM25 over body)",
        placeholder="songbird territory",
        key="hybrid_text",
    )
    hybrid_visual_q = st.text_input(
        "Describe appearance",
        placeholder="bright red bird",
        key="hybrid_visual",
    )
    if st.button("Search", key="hybrid_btn", type="primary") or _consume_auto_run("hybrid"):
        if hybrid_text_q.strip() and hybrid_visual_q.strip():
            with st.spinner("Running text + visual searches, then fusing…"):
                text_only = search_text(hybrid_text_q, field="body", top_k=5)
                visual_only = search_visual(hybrid_visual_q, top_k=5)
                fused = search_hybrid_rrf(
                    hybrid_text_q, hybrid_visual_q, field="body",
                    top_k=10, fetch_k=50,
                )

            text_ids = {m._id for m in text_only.matches}
            visual_ids = {m._id for m in visual_only.matches}
            new_ids = {m._id for m in fused.matches} - text_ids - visual_ids

            st.markdown(
                "**Without RRF** — each signal searched alone, top 5:"
            )
            cols = st.columns(2)
            with cols[0]:
                _render_compact_ranking("Text only", text_only.matches)
            with cols[1]:
                _render_compact_ranking("Visual only", visual_only.matches)

            st.markdown(
                "**With RRF** — both signals fused, top 10 "
                "(🆕 = wasn't in either list above):"
            )
            _render_compact_ranking("RRF fused", fused.matches, new_ids=new_ids)

            st.divider()
            render_results(fused)
        else:
            st.warning("Enter both a text query and an appearance description.")
