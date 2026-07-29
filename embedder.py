"""Local joint image/text embedding model.

Bird Search v2 needs a single vector space that both a bird's photo (at
index time) and a user's typed description (at query time) land in, so a
text query can be scored directly against stored image vectors
(``search_visual`` / ``search_filter_visual`` in ``query.py``). That rules
out a text-only embedding model — this wraps SigLIP
(https://huggingface.co/google/siglip-base-patch16-224), a joint image/text
model that runs fully locally via ``transformers`` + ``torch``. Weights
(~400MB) download once from the Hugging Face Hub on first use and are
cached locally afterward — no API key, no rate limit.

Output dimension matches the previous Gemini Embedding 2 setup (768), so no
Pinecone schema change is required.
"""

from __future__ import annotations

import threading

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

EMBED_MODEL_NAME = "google/siglip-base-patch16-224"
EMBED_DIMENSIONS = 768

_lock = threading.Lock()
_model = None
_processor = None


def _get_model_and_processor():
    global _model, _processor
    if _model is None:
        with _lock:
            if _model is None:
                _processor = AutoProcessor.from_pretrained(EMBED_MODEL_NAME)
                model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
                model.eval()
                _model = model
    return _model, _processor


def _normalize(vec: torch.Tensor) -> list[float]:
    vec = vec / vec.norm(p=2, dim=-1, keepdim=True)
    return vec.squeeze(0).tolist()


def embed_image(pil_image: Image.Image) -> list[float]:
    """Embed a PIL image into SigLIP's shared image/text space."""
    model, processor = _get_model_and_processor()
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return _normalize(features.pooler_output)


def embed_text(text: str) -> list[float]:
    """Embed a text string into SigLIP's shared image/text space."""
    model, processor = _get_model_and_processor()
    inputs = processor(
        text=[text], return_tensors="pt", padding="max_length", truncation=True
    )
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return _normalize(features.pooler_output)
