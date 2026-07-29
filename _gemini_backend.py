"""Gemini Embedding 2 backend — select via EMBED_PROVIDER=gemini (see
``embedder.py``).

Calls the Gemini API (``google-genai`` SDK) to embed images and text into
a shared 768-dim multimodal space. Requires GOOGLE_API_KEY. Subject to
Gemini's API rate limits, so ``embed_image`` retries transient failures
(429s/5xxs) with exponential backoff + jitter — build_index.py calls it in
bulk across concurrent threads during ingestion. ``embed_text`` is a
single interactive query-time call and doesn't retry.
"""

from __future__ import annotations

import random
import sys
import time

from PIL import Image

from google import genai
from google.genai import types as genai_types

EMBED_MODEL_NAME = "gemini-embedding-2"
EMBED_DIMENSIONS = 768

EMBED_MAX_RETRIES = 5
EMBED_BASE_BACKOFF_S = 2.0

_client = genai.Client()  # reads GOOGLE_API_KEY

_EMBED_CONFIG = genai_types.EmbedContentConfig(
    output_dimensionality=EMBED_DIMENSIONS,
)


def embed_image(pil_image: Image.Image) -> list[float]:
    """Embed a PIL image into Gemini-2's multimodal space, retrying
    transient API errors (429s/5xxs) with backoff + jitter.

    Raises the last exception if every attempt fails.
    """
    last_exc: Exception | None = None
    for attempt in range(EMBED_MAX_RETRIES):
        try:
            resp = _client.models.embed_content(
                model=EMBED_MODEL_NAME, contents=pil_image, config=_EMBED_CONFIG
            )
            return list(resp.embeddings[0].values)
        except Exception as exc:
            last_exc = exc
            if attempt == EMBED_MAX_RETRIES - 1:
                break
            sleep_s = EMBED_BASE_BACKOFF_S * (2 ** attempt) + random.uniform(0, 1)
            print(
                f"  embed retry in {sleep_s:.1f}s (attempt {attempt + 1}"
                f"/{EMBED_MAX_RETRIES}): {exc}",
                file=sys.stderr,
            )
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def embed_text(text: str) -> list[float]:
    """Embed a text query into Gemini-2's multimodal space."""
    resp = _client.models.embed_content(
        model=EMBED_MODEL_NAME, contents=text, config=_EMBED_CONFIG
    )
    return list(resp.embeddings[0].values)
