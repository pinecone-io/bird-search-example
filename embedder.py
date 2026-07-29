"""Joint image/text embedding — provider selected via EMBED_PROVIDER.

Bird Search v2 needs a single vector space that both a bird's photo (at
index time) and a user's typed description (at query time) land in, so a
text query can be scored directly against stored image vectors
(``search_visual`` / ``search_filter_visual`` in ``query.py``). Two
interchangeable backends provide that shared space:

    local   (default) — _local_backend.py, SigLIP via transformers/torch,
                         fully offline after the first weights download,
                         no API key, no rate limit.
    gemini            — _gemini_backend.py, the Gemini API via
                         google-genai. Requires GOOGLE_API_KEY; subject to
                         Gemini's rate limits.

Both output 768-dim vectors, so switching providers needs no Pinecone
schema change. Set EMBED_PROVIDER=gemini in .env to opt into the Gemini
backend; anything else (including unset) uses the local one.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
load_dotenv()

EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "local").strip().lower()

if EMBED_PROVIDER == "local":
    from _local_backend import embed_image, embed_text, EMBED_MODEL_NAME, EMBED_DIMENSIONS
elif EMBED_PROVIDER == "gemini":
    from _gemini_backend import embed_image, embed_text, EMBED_MODEL_NAME, EMBED_DIMENSIONS
else:
    raise ValueError(
        f"Unknown EMBED_PROVIDER={EMBED_PROVIDER!r}; expected 'local' or 'gemini'."
    )

__all__ = [
    "embed_image",
    "embed_text",
    "EMBED_MODEL_NAME",
    "EMBED_DIMENSIONS",
    "EMBED_PROVIDER",
]
