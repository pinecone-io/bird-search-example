"""Shared pytest fixtures.

Live tests under ``tests/test_live.py`` exercise the real
``bird-search-fts`` index. They are not run in CI — they exist so we can
sanity-check every example in ``demo-queries.md`` against the live index
before/after a Pinecone SDK bump.

The ``live_index`` fixture short-circuits the whole live test module when
``PINECONE_API_KEY`` or ``GOOGLE_API_KEY`` is unset, so plain
``pytest tests/`` on a machine without keys still passes (the mocked
unit tests in ``test_query.py`` stay green).

Run live tests with::

    pytest tests/test_live.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# Make `query` and `app` importable when pytest runs from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(scope="session")
def live_index():
    """Real preprod index handle. Skips when API keys are absent."""
    missing = [
        k for k in ("PINECONE_API_KEY", "GOOGLE_API_KEY") if not os.getenv(k)
    ]
    if missing:
        pytest.skip(f"Live tests require {', '.join(missing)}.")

    from query import _index

    idx = _index()
    return idx
