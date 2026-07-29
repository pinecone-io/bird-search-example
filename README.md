# Bird Search v2

A demo app showcasing [Pinecone Full-Text Search](https://docs.pinecone.io/guides/search/full-text-search) combined with multimodal vector search using a local [SigLIP](https://huggingface.co/google/siglip-base-patch16-224) model, over a corpus of ~2,079 North American bird Wikipedia articles - one document per bird.

## Prerequisites

- Python 3.10+
- [Pinecone](https://app.pinecone.io) account and API key
- ~400MB free disk space for the local SigLIP model weights (downloaded once from the Hugging Face Hub on first run)

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your API key:
   ```
   PINECONE_API_KEY=...
   ```
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

## Why SigLIP

[SigLIP](https://huggingface.co/google/siglip-base-patch16-224) is a multimodal model that embeds both text and images into the same vector space. This makes cross-modal search possible: a text description like *"tall pink wading bird"* produces a vector that is directly comparable to the vector computed from a bird's photo at index time — no separate image captioning or two-stage pipeline needed. It runs entirely locally via `transformers` + `torch` (see `embedder.py`) — weights download once from the Hugging Face Hub and are cached afterward, with no API key and no rate limits. All image embeddings are precomputed during `build_index.py` and stored in Pinecone at 768 dimensions with cosine similarity.

## Search tabs

### Text FTS
BM25 keyword scoring against `body`, `intro`, or `bird_name`. A `multi` mode searches all three fields at once and lets you write a different query per field (e.g. `bird_name=swallow` + `body=in mountains`). Toggle **Phrase** to require exact word adjacency via Lucene `query_string`.

### Visual
Type a description of what a bird looks like. The query is embedded via the local SigLIP model and scored against each bird's stored image vector. Finds birds by appearance even when the article never uses your exact words.

### Combined
A `$match_all` filter on `body` (every required keyword must appear in the article) combined with dense-vector visual reranking — in a single Pinecone round trip. Use when you need both a hard text gate and visual ranking.

### Lucene
Raw Lucene `query_string` for advanced queries: boolean operators (`+required -excluded`), term boosts (`eagle^3`), phrase slop (`"northern cardinal"~3`), phrase prefixes, and cross-field clauses.

Each tab shows the exact `documents.search(...)` call beneath the results so you can see what was sent to Pinecone.

## Learn more

- [Full-text search — Pinecone docs](https://docs.pinecone.io/guides/search/full-text-search)
- [Full-text search notebook](https://colab.research.google.com/github/pinecone-io/examples/blob/master/docs/full-text-search.ipynb)

## License

MIT
