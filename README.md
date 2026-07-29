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
A `$match_all` filter on `body` (every required keyword must appear in the article) combined with dense-vector visual reranking — in a single Pinecone round trip. Use when you need both a hard text gate and visual ranking.

### Lucene
Raw Lucene `query_string` for advanced queries: boolean operators (`+required -excluded`), term boosts (`eagle^3`), phrase slop (`"northern cardinal"~3`), phrase prefixes, and cross-field clauses.

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

## Learn more

- [Full-text search — Pinecone docs](https://docs.pinecone.io/guides/search/full-text-search)
- [Full-text search notebook](https://colab.research.google.com/github/pinecone-io/examples/blob/master/docs/full-text-search.ipynb)

## License

MIT
