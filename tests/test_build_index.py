"""Mocked unit tests for ``build_index.py``.

Patches ``pinecone.Pinecone`` at import time (mirroring ``test_query.py``)
so the module can be imported without real credentials, then exercises
``truncate_body`` directly — it exists specifically to keep every bird's
``body`` field under Pinecone's full-text-search limits (10,000 tokens /
100,000 bytes), which a handful of long Wikipedia articles (large
raptors/owls) exceed at full-corpus scale.

Run with::

    pytest tests/test_build_index.py -v
"""

from __future__ import annotations

import importlib
import sys
from unittest import mock

import pytest


def _fresh_build_index_module():
    """Import (or re-import) ``build_index`` with the Pinecone client mocked."""
    fake_pc = mock.MagicMock()
    pinecone_mod = mock.MagicMock()
    pinecone_mod.Pinecone.return_value = fake_pc

    preview_mod = mock.MagicMock()

    patches = {"pinecone": pinecone_mod, "pinecone.preview": preview_mod}
    with mock.patch.dict(sys.modules, patches):
        if "build_index" in sys.modules:
            del sys.modules["build_index"]
        build_index = importlib.import_module("build_index")

    return build_index


@pytest.fixture(scope="module")
def build_index():
    return _fresh_build_index_module()


def test_truncate_body_leaves_short_text_unchanged(build_index):
    body = "\n\n".join(f"Paragraph {i} with a few words." for i in range(5))
    assert build_index.truncate_body(body) == body


def test_truncate_body_caps_word_count(build_index):
    paragraphs = ["word " * 500 for _ in range(30)]  # 500 words each, 15,000 total
    body = "\n\n".join(paragraphs)
    truncated = build_index.truncate_body(body)
    assert len(truncated.split()) <= build_index.MAX_BODY_WORDS


def test_truncate_body_caps_byte_size_for_single_oversized_paragraph(build_index):
    # One giant paragraph (no break to truncate on) must still be hard-capped.
    body = "word " * 200_000
    truncated = build_index.truncate_body(body)
    assert len(truncated.encode("utf-8")) <= build_index.MAX_BODY_BYTES
