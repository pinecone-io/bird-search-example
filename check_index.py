"""One-shot check: which slugs are in the index vs. what we tried to upsert.

Compares the IDs in the live FTS index against the slugs we have cached
embeddings for (which is exactly what the last build_index.py run tried to
upsert). Prints any gaps so you can decide whether to repair.

Run from the project dir:  python check_index.py
"""

import json
import pathlib

from query import pc, INDEX, NAMESPACE

idx = pc.preview.index(name=INDEX)

cache_path = pathlib.Path(__file__).resolve().parent / "embeddings-cache.jsonl"
expected: set[str] = set()
with cache_path.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        expected.add(json.loads(line)["slug"])

print(f"Expected slugs (from cache): {len(expected):,}")

present: set[str] = set()
slugs = sorted(expected)
for i in range(0, len(slugs), 100):
    chunk = slugs[i : i + 100]
    resp = idx.documents.fetch(
        namespace=NAMESPACE, ids=chunk, include_fields=[]
    )
    present.update(resp.documents.keys())

missing = expected - present
print(f"Present in index: {len(present):,}")
print(f"Missing:          {len(missing):,}")
if missing:
    print("First 20 missing slugs:")
    for s in sorted(missing)[:20]:
        print(f"  {s}")
