# Bird Search v2

A demo app showcasing [Pinecone Full-Text Search](https://docs.pinecone.io/guides/search/full-text-search) combined with multimodal vector search, over a corpus of ~2,079 North American bird Wikipedia articles - one document per bird. The embedding model is swappable between a local model and Gemini Embedding 2 (see [Embedding provider](#embedding-provider)).

## Prerequisites

- Python 3.10+
- [Pinecone](https://app.pinecone.io) account and API key
- One of:
  - **Local embeddings (default)** — no extra account needed; ~400MB free disk space for the [SigLIP](https://huggingface.co/google/siglip-base-patch16-224) model weights (downloaded once from the Hugging Face Hub on first run)
  - **Gemini embeddings** — a [Google AI Studio](https://aistudio.google.com) API key

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your Pinecone API key:
   ```
   PINECONE_API_KEY=...
   ```
   Leave `EMBED_PROVIDER` at its default (`local`) to run fully offline, or see [Embedding provider](#embedding-provider) to use Gemini instead.
3. Build the index. Start with a small sample to verify everything works:
   ```
   python build_index.py --sample 50
   ```
   Then ingest the full corpus (~2,079 birds):
   ```
   python build_index.py --sample 0
   ```
4. Run the app:
   ```
   streamlit run app.py
   ```

## Dataset

The bird dataset lives at `parsed_birds/` (~58 MB, committed to the repo). It contains ~2,079 North American bird articles scraped from Wikipedia, structured as:

- `parsing_metadata.json` — index of all birds with image metadata
- `text/<slug>.txt` — full article text per bird
- `images/<slug>/<slug>_1.jpg` — primary photo per bird

Each bird is stored in Pinecone as a single document with three text fields (`bird_name`, `intro`, `body`) and one dense vector field (`image_embedding`). Set `BIRD_DATA_DIR` in your environment to override the default data path.

## Embedding provider

Both tabs that touch `image_embedding` (**Visual** and **Combined**) rely on a multimodal model that embeds text and images into the *same* vector space — a text description like *"tall pink wading bird"* produces a vector directly comparable to the vector computed from a bird's photo at index time, no separate image captioning or two-stage pipeline needed. Which model does that embedding is switchable via `EMBED_PROVIDER` in `.env` (see `embedder.py`); either way the output is 768 dimensions with cosine similarity, so switching providers never requires a Pinecone schema change.

| | `EMBED_PROVIDER=local` (default) | `EMBED_PROVIDER=gemini` |
|---|---|---|
| Model | [SigLIP](https://huggingface.co/google/siglip-base-patch16-224) via `transformers` + `torch` | [Gemini Embedding 2](https://ai.google.dev/gemini-api/docs/embeddings) via `google-genai` |
| Runs | Fully locally, offline after the first run | Remote API call |
| Setup | Weights (~400MB) download once from the Hugging Face Hub | Requires `GOOGLE_API_KEY` |
| Rate limits | None | Subject to Gemini API rate limits (`build_index.py` retries with backoff) |

All image embeddings are precomputed during `build_index.py` and cached at `embeddings-cache.jsonl`, tagged with the model name — switching `EMBED_PROVIDER` automatically invalidates the old provider's cached rows and re-embeds with the new one on the next ingest.

## Search tabs

### Text FTS
BM25 keyword scoring against `body`, `intro`, or `bird_name`. A `multi` mode searches all three fields at once and lets you write a different query per field (e.g. `bird_name=swallow` + `body=in mountains`). Toggle **Phrase** to require exact word adjacency via Lucene `query_string`.

### Visual
Type a description of what a bird looks like. The query is embedded via the configured embedding provider (see [Embedding provider](#embedding-provider)) and scored against each bird's stored image vector. Finds birds by appearance even when the article never uses your exact words.

### Combined
A text-match `filter` on `body` combined with dense-vector visual reranking — in a single Pinecone round trip. Use when you need both a hard text gate and visual ranking. Pinecone has three text-match filter operators, each a different precision/recall trade-off, all exposed in the tab via a mode toggle:

| Mode | Operator | Semantics |
|---|---|---|
| All terms | `$match_all` | Every term must appear, in any order. The precise default. |
| Any term | `$match_any` | At least one term must appear. Widens the candidate pool when you're unsure which term the article uses. |
| Exact phrase | `$match_phrase` | The words must appear adjacent, in order. Stricter than "All terms" — rejects documents where the words merely co-occur without meaning the same thing together. |

**Example** (verified live) — visual-only `black bird with bright spots on wings` ranks the Northern cardinal's cousin the Red-winged blackbird #19, outside the top 10:

| Filter | Mode | Rank of Red-winged blackbird |
|---|---|---|
| *(none — visual only)* | — | #19 |
| `epaulet, yellow` | All terms | **#1** |
| `epaulet, yellow` (same terms) | Any term | #7 |
| `yellow wing bar` | Exact phrase | **#1** |

"All terms" on `epaulet, yellow` (the red shoulder patch — an "epaulet" — and yellow wing bar the article describes) narrows the candidates enough for the visual rerank to nail it at #1. Switching the *same two terms* to "Any term" widens the filter to admit any bird mentioning either word, diluting the pool enough that the visual signal alone can't recover the precision. "Exact phrase" on the literal phrase `yellow wing bar` is narrow enough on its own to also land at #1.

### Lucene
Raw Lucene `query_string` — the full syntax Pinecone's `query_string` ranking type supports (see [full-text search docs](https://docs.pinecone.io/guides/search/full-text-search)):

| Feature | Syntax | Example |
|---|---|---|
| Boolean AND / OR / NOT | `AND` `OR` `NOT` | `body:(mountain AND eagle)` |
| Required / excluded | `+term` `-term` | `body:(+illinois +cardinal -opinion)` |
| Term boost | `term^N` | `body:(eagle^3 OR hawk OR raptor)` |
| Exact phrase | `"words"` | `body:("state bird of seven")` |
| Phrase slop | `"phrase"~N` | `body:("northern cardinal"~3)` |
| Phrase prefix | `"words"*` | `body:("james w"*)` |
| Fuzzy (typo-tolerant) | `term~` or `term~N` | `bird_name:(cardnal~1)` → Northern cardinal, despite the missing letter |
| Regex | `field:/pattern/` | `bird_name:/.*owl.*/` → every species with "owl" in the name |
| Grouping | `(expr)` | `body:((eagle OR hawk) AND mountain)` |
| Cross-field | `f1:(…) OR f2:(…)` | `bird_name:(swallow*) OR body:(swallow)` |

Two nuances worth knowing:
- **Fuzzy edit distance** is automatic with `term~` (exact for words under 4 letters, 1 edit for 4–7, 2 edits for 8+), or fixed with `term~N` (0–2). No SDK version bump needed — `query_string` is a plain string the server parses, so any syntax the API supports works today through `search_query_string`, regardless of the pinned client SDK version.
- **Regex matches the *indexed* token, not your original spelling.** `body` has stemming on (see the schema below), so "woodpecker" is indexed as its stem ("woodpeck") and `body:/woodpecker/` matches nothing. `bird_name`/`intro` aren't stemmed, so regex there matches the lowercased surface word directly.

### Hybrid (RRF)
Pinecone has no built-in way to fuse two independently-issued searches (unlike the Combined tab's single-call filter+rerank, or `multi`'s single-call multi-field blend — both server-side). This tab runs a BM25 text search and a dense-vector visual search separately, then merges their *rankings* client-side via [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf): `score(doc) = Σ 1/(k + rank)` summed across every ranking a doc appears in (`k=60`).

**Why fuse rankings instead of just picking one search's results?** BM25 and cosine-similarity scores live on different, incomparable scales, so averaging or summing the raw numbers wouldn't mean anything — and "whichever search's #1 result wins" throws away every bird that's a *strong* match on one signal but only *plausible* on the other. RRF sidesteps both problems by only ever looking at rank position, which makes the two searches comparable, and a bird that's decently ranked on *both* signals typically outscores one that's #1 on only a single axis.

**Example** — text `songbird territory` + visual `bright red bird` (verified against the live index):

| Search | Top result | Rank of Northern cardinal |
|---|---|---|
| Text only | Lark bunting | #19 (outside top 5) |
| Visual only | Red warbler | #6 (outside top 5) |
| **RRF fused** | **Northern cardinal** | **#1** |

Neither search alone puts the Northern cardinal in its own top 5 — it's a middling text match and a near-miss visual match. But `1/(60+19) + 1/(60+6) ≈ 0.0278` beats every bird that only ranks well on a *single* signal, because none of them accumulate score from a second ranking. That's the concrete benefit: RRF surfaces the bird both signals agree is plausible, even when neither is confident enough on its own to rank it highly — exactly the class of result a single-signal search misses. The tab shows the text-only and visual-only rankings side by side with the fused result (marking 🆕 anything that wasn't in either individual top list) so this effect is visible, not just asserted.

Each tab shows the exact `documents.search(...)` call beneath the results so you can see what was sent to Pinecone.

### Coverage vs. the Pinecone FTS docs

The [full-text search docs](https://docs.pinecone.io/guides/search/full-text-search) describe four `score_by` ranking types and three text-match filter operators. All of them are demonstrated somewhere in this app, except one that doesn't apply to this schema:

| Docs feature | Demonstrated in |
|---|---|
| `type: "text"` (BM25, single field) | Text FTS |
| `type: "text"` (multi-field blend) | Text FTS → `multi` |
| `type: "query_string"` (Lucene: boolean, boost, phrase, slop, prefix, fuzzy, regex, grouping, cross-field) | Text FTS → Phrase; Lucene |
| `type: "dense_vector"` | Visual, Combined, Hybrid (RRF) |
| `type: "sparse_vector"` | *Not demonstrated* — this schema has no sparse field (no learned-sparse or keyword-weighted embeddings in this corpus); the mechanics are otherwise identical to `dense_vector`, just with `sparse_values={"indices": …, "values": …}` instead of `values`. |
| `filter.$match_all` | Combined → All terms |
| `filter.$match_any` | Combined → Any term |
| `filter.$match_phrase` | Combined → Exact phrase |
| Filter + rank combined in one call | Combined |
| Fusing two separately-issued rankings | Hybrid (RRF) — Pinecone has no server-side equivalent, hence the client-side RRF |

**Two distinct ways to match a phrase**, easy to conflate: `query_string` phrase syntax (`field:("…")`, used by Text FTS's Phrase toggle and the Lucene tab) *ranks* by how well the phrase matches as part of `score_by`; `$match_phrase` (used by Combined) is a hard *filter* — it doesn't rank on its own, it only decides which documents are eligible for whatever `score_by` clause runs alongside it.

## Learn more

- [Full-text search — Pinecone docs](https://docs.pinecone.io/guides/search/full-text-search)
- [Full-text search notebook](https://colab.research.google.com/github/pinecone-io/examples/blob/master/docs/full-text-search.ipynb)

## License

MIT
