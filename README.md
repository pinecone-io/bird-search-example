# Bird Search v2

Streamlit app showcasing Pinecone's preview FTS API + Gemini Embedding 2 (multimodal) over a corpus of ~2,079 North American bird Wikipedia articles.

## Setup

1. Install the Pinecone preview SDK 
   ```
   pip install -r requirements.txt (need to update)
   ```
   `requirements.txt` includes the preview SDK's runtime transitives (`httpx[http2]`, `msgspec`, `orjson`) so a fresh checkout boots without "module not found" errors.
2. Copy `.env.example` to `.env` and fill in `PINECONE_API_KEY` and `GOOGLE_API_KEY`.
3. Build the index with a small sample first:
   ```
   python build_index.py --sample 50
   ```
   Then scale up: `python build_index.py --sample 0` (full ~2,079 birds).
4. Run the app: `streamlit run app.py`.

## Data

The bird dataset (`parsed_birds/`) lives....
## Tabs

- **Text FTS** — keyword and multi-field search over `bird_name` / `intro` / `body`.
- **Visual** — typed description → Gemini-2 text embedding → scored against each bird's image vector.
- **Combined** — dense-vector rank via Gemini-2, narrowed by a client-side substring filter on `body` for required terms. Will swap to a server-side `$matches_all` hard filter once that operator is live in preview. E.g. "must mention illinois" + "describe red bird with black wings".
- **About** — examples.

