# Bird Search v2

Streamlit app showcasing Pinecone's preview FTS API + Gemini Embedding 2 (multimodal) over a corpus of ~2,079 North American bird Wikipedia articles.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in `PINECONE_API_KEY` and `GOOGLE_API_KEY`.
3. Build the index with a small sample first:
   ```
   python build_index.py --sample 50
   ```
   Then scale up: `python build_index.py --sample 0` (full ~2,079 birds).
4. Run the app: `streamlit run app.py`.

## Data

The bird dataset lives at `parsed_birds/` in this repo (~58 MB, committed): `parsing_metadata.json`, `text/<slug>.txt` per bird, and `images/<slug>/<slug>_1.jpg` per bird. ~2,079 entries scraped from North American bird Wikipedia articles. Set `BIRD_DATA_DIR` to override the path if you keep the data elsewhere.
## Tabs

- **Text FTS** — keyword and multi-field search over `bird_name` / `intro` / `body`.
- **Visual** — typed description → Gemini-2 text embedding → scored against each bird's image vector.
- **Combined** — server-side `$match_all` filter on `body` (every required term must appear) plus dense-vector ranking on `image_embedding` — one Pinecone round trip. E.g. "must mention illinois" + "describe red bird with black wings".
- **About** — examples.

## License

MIT
