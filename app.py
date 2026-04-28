"""Streamlit UI for Bird Search v2.

Four search modes over one Pinecone preview FTS index:
  - Text FTS: BM25 token-OR over ``bird_name`` / ``intro`` / ``body``
    (per-field or blended multi-field), with an opt-in "exact phrase" toggle
    that routes through Lucene ``query_string``.
  - Visual: typed description embedded via Gemini Embedding 2, scored against
    each bird's image vector.
  - Combined: dense-vector ranking on the image embedding, then a client-side
    substring filter on ``body`` for every required term. (Will switch to a
    server-side ``$matches_all`` hard filter once that operator ships in
    preview — see ``query.search_filter_visual``.)
  - Boolean: raw Lucene ``query_string`` — the user types boolean / phrase /
    boost / slop / phrase-prefix expressions directly.

Each tab also renders a "what we sent to Pinecone" code block under the
results so demo viewers can see the actual ``documents.search(...)`` call.

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
    search_query_string,
    search_text,
    search_text_multi,
    search_text_phrase,
    search_visual,
)

# ---------------------------------------------------------------------------
# Data location + metadata (loaded once at startup).
# ---------------------------------------------------------------------------

BIRD_DATA_DIR = pathlib.Path(
    os.environ.get("BIRD_DATA_DIR", "../old-code/parsed_birds")
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
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"([A-Za-z][A-Za-z\-']*)")
# Words to ignore when extracting highlight terms from a Lucene-flavored query.
# Otherwise "body:(machine AND learning)" would also light up "and" everywhere.
_LUCENE_OPERATORS = {"and", "or", "not", "to"}


def highlight_matching_words(text: str, query: str) -> str:
    """Wrap query words (and simple stemming variants) in ``<mark>`` tags.

    Matching is case-insensitive and uses a prefix heuristic: a query word
    highlights any body word that shares a prefix of at least 3 characters
    (so ``peck`` lights up ``pecks``/``pecking``, matching how FTS stems on
    the ``body`` field). Whitespace, punctuation, and paragraph breaks are
    preserved; all text is HTML-escaped.

    Word extraction goes through ``_WORD_RE`` so Lucene punctuation
    (``body:(+illinois -opinion)``, quotes, parens, ``^N`` boosts) is
    stripped — only the actual search terms get highlighted.

    Render with ``st.markdown(..., unsafe_allow_html=True)``.
    """
    query_words = [
        m.group(0).lower()
        for m in _WORD_RE.finditer(query)
        if len(m.group(0)) >= 2 and m.group(0).lower() not in _LUCENE_OPERATORS
    ]
    if not query_words:
        return html.escape(text)

    def is_match(word_lc: str) -> bool:
        for q in query_words:
            if q == word_lc:
                return True
            shorter, longer = (q, word_lc) if len(q) <= len(word_lc) else (word_lc, q)
            if len(shorter) >= 3 and longer.startswith(shorter):
                return True
        return False

    def replace(m: re.Match[str]) -> str:
        word = m.group(0)
        if is_match(word.lower()):
            return f"<mark>{html.escape(word)}</mark>"
        return html.escape(word)

    # Escape non-word spans separately so we don't double-escape the matched
    # words (which already get escaped inside ``replace``).
    parts: list[str] = []
    last = 0
    for m in _WORD_RE.finditer(text):
        if m.start() > last:
            parts.append(html.escape(text[last : m.start()]))
        parts.append(replace(m))
        last = m.end()
    if last < len(text):
        parts.append(html.escape(text[last:]))
    return "".join(parts)


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


def render_results(response, highlight_query: str = "") -> None:
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
            st.markdown(
                f'<div class="bird-card-header">'
                f'<span class="bird-card-title">{html.escape(bird_name)}</span>'
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
                has_highlight = bool(highlight_query.strip())

                if intro:
                    if has_highlight:
                        st.markdown(highlight_matching_words(intro, highlight_query), unsafe_allow_html=True)
                    else:
                        st.markdown(intro)

                if body:
                    with st.expander("Full article body"):
                        if has_highlight:
                            st.markdown(highlight_matching_words(body, highlight_query), unsafe_allow_html=True)
                        else:
                            st.markdown(body)


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

/* <mark> highlights — softer than browser yellow. */
mark {
    background: rgba(255, 213, 79, 0.45);
    padding: 0 0.1rem;
    border-radius: 2px;
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
        "Search ~2,079 North American birds four ways: **Text FTS** "
        "(keyword search over the article fields), **Visual** (type what "
        "the bird looks like — matched against each bird's photo in Gemini "
        "Embedding 2's shared text/image space), **Combined** (visual "
        "ranking narrowed by required keywords on the article body), and "
        "**Boolean** (raw Lucene `query_string` for boost / slop / phrase "
        "prefix / cross-field queries)."
    )
    st.write(
        "Each tab also shows the actual `documents.search(...)` call beneath "
        "its results, so you can see exactly what hit Pinecone."
    )
with schema_col:
    st.markdown("**Index schema**")
    st.code(INDEX_SCHEMA_SOURCE, language="python")
    st.caption(
        "Index `bird-search-fts` · namespace `birds` · "
        "one doc per bird, ~2,079 docs total."
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


tab_text, tab_visual, tab_combined, tab_boolean, tab_about = st.tabs(
    ["Text FTS", "Visual", "Combined", "Boolean", "About"]
)

# --- Tab 1: Text FTS ------------------------------------------------------
with tab_text:
    st.header("Text FTS")
    st.write(
        "Keyword search over one field, or blended across all three. "
        "`multi` rewards birds whose article is relevant in `bird_name`, "
        "`intro`, and `body` together. Toggle **Match as exact phrase** for "
        "queries whose meaning hinges on word adjacency (e.g. `state bird "
        "of seven` — token-OR ranks the cardinal off the page; phrase mode "
        "lands it at #1)."
    )

    _example_buttons("text", [
        {
            "label": "Mormon crickets · body",
            "state": {"text_query": "Mormon crickets",
                      "text_field": "body",
                      "text_phrase": False},
        },
        {
            "label": "Miracle of the Gulls · multi",
            "state": {"text_query": "Miracle of the Gulls",
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
            "wrapped in quotes (`field:(\"…\")`), so adjacency is required."
        ),
    )
    query = st.text_input(
        "Query",
        placeholder="e.g. bright red wings pecks wood",
        key="text_query",
    )
    if st.button("Search", key="text_btn", type="primary") or _consume_auto_run("text"):
        if query.strip():
            with st.spinner("Searching…"):
                if phrase_match:
                    response = search_text_phrase(query, field=field_choice)
                elif field_choice == "multi":
                    response = search_text_multi(query)
                else:
                    response = search_text(query, field=field_choice)
            render_results(response, highlight_query=query)
        else:
            st.warning("Enter a query.")

# --- Tab 2: Visual --------------------------------------------------------
with tab_visual:
    st.header("Visual")
    st.write(
        "Describe what the bird looks like. Your description is embedded "
        "with Gemini Embedding 2 and matched against each bird's photo."
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

# --- Tab 3: Combined ------------------------------------------------------
with tab_combined:
    st.header("Combined")
    st.write(
        "Rank birds by visual similarity to your description, then keep only "
        "those whose article `body` contains every required term "
        "(case-insensitive substring match). Today this is post-filtered "
        "client-side over the top visual neighbours; once Pinecone's "
        "`$matches_all` operator ships in preview it will become a server-side "
        "hard filter."
    )

    _example_buttons("combined", [
        {
            "label": "illinois + red bird with black wings",
            "state": {"combined_filter": "illinois",
                      "combined_visual": "red bird with black wings"},
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
        placeholder="illinois",
        key="combined_filter",
    )
    visual_q = st.text_input(
        "Describe appearance",
        placeholder="red bird with black wings",
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
            render_results(response, highlight_query=filter_raw)
        else:
            st.warning("Enter at least one filter term and an appearance description.")

# --- Tab 4: Boolean (raw query_string) -----------------------------------
with tab_boolean:
    st.header("Boolean / Lucene")
    st.markdown(
        "Drive Pinecone's `query_string` mode directly. Supports "
        "`AND` / `OR` / `NOT` / `+required` / `-excluded` / `\"phrase\"` / "
        "`term^N` (boost) / `\"phrase\"~N` (slop) / `\"machine lea\"*` "
        "(phrase prefix). Field names available on this index: "
        "`bird_name`, `intro`, `body`."
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
            render_results(response, highlight_query=boolean_q)
        else:
            st.warning("Enter a Lucene query_string.")

# --- Tab 5: About ---------------------------------------------------------
with tab_about:
    st.header("About")
    st.markdown(
        """
**Text FTS** — BM25 keyword search over the three article fields
(`bird_name`, `intro`, `body`) using Pinecone's preview `type: "text"`
scoring. Pick a single field for precise matching, or `multi` to blend
all three and reward documents relevant in more than one place. Toggle
**Match as exact phrase** to flip to `type: "query_string"` with the query
wrapped in quotes — needed for queries whose meaning lives in adjacency
(e.g. `state bird of seven`).

- Example (token-OR, body): `bright red wings pecks wood` → woodpeckers.
- Example (phrase, body): `state bird of seven` → northern cardinal.
- Example (bird_name): `wren` → wrens.

**Visual** — Describe what the bird looks like in plain text. The
description is embedded with Gemini Embedding 2 and scored against each
bird's primary photo embedding (same multimodal space).

- Example: `roundest birds ever that are red` → visually round red birds,
  even when the article doesn't contain those exact words.

**Combined** — Visual ranking plus a keyword guardrail on `body`. The
visual rank picks shape and colour; the keyword filter ensures the article
is *about* the right place / context.

- Example: must-mention `illinois`, describe `red bird with black wings` →
  Illinois-range red+black birds (cardinal, red-winged blackbird).

**Boolean** — Raw Lucene `query_string`. Use for boolean operators
(`AND`/`OR`/`NOT`), required (`+`) / excluded (`-`) terms, exact phrases
(`"…"`), phrase slop (`"…"~N`), term boosts (`term^N`), phrase-prefix
matches (`"machine lea"*`), and cross-field queries
(`title:(…) OR body:(…)`).

Each tab shows the actual `documents.search(...)` call under its results,
so you can see exactly what hit Pinecone.
        """
    )
