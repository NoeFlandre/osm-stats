"""Verify that the local outputs match ``data/MANIFEST.md``.

Walks the local ``data/`` directory, recomputes the size and
sha256-prefix of every file, and compares them against the
MANIFEST.md committed to the bucket. Exits 0 if everything
matches, 1 otherwise. Intended to be run by the HF bucket's CI
to confirm a freshly cloned repo reproduces the bucket contents
bit-for-bit.

HDBSCAN is non-deterministic, so we expect a small fraction of
files (the medoid/membership CSVs) to differ between runs. Those
files are listed in ``data/MANIFEST.md`` with a comment marking
them as ``[non-deterministic]``. The verify script skips the
content check on those files and only checks the size.

Run with:

    .venv/bin/python -m scripts.reproducibility.verify
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "MANIFEST.md"
DATA = ROOT / "data"

# Files whose content is allowed to differ between runs (HDBSCAN
# non-determinism). Only the size is checked.
SIZE_ONLY = {
    "outputs/filter_first/tfidf/cluster_medoids.csv",
    "outputs/filter_first/tfidf/cluster_memberships.csv",
    "outputs/filter_first/embeddings/cluster_medoids_embeddings.csv",
    "outputs/filter_first/embeddings/cluster_memberships_embeddings.csv",
    "outputs/standardize_first/tfidf/cluster_medoids.csv",
    "outputs/standardize_first/tfidf/cluster_memberships.csv",
    "outputs/standardize_first/embeddings/cluster_medoids_embeddings.csv",
    "outputs/standardize_first/embeddings/cluster_memberships_embeddings.csv",
    "outputs/filter_first/comparison/pipeline_comparison.md",
    "outputs/standardize_first/comparison/pipeline_comparison.md",
}


def parse_manifest() -> list[tuple[str, int, str]]:
    """Extract (path, size, sha_prefix) tuples from the MANIFEST table."""
    rows = []
    for line in MANIFEST.read_text().splitlines():
        m = re.match(r"^\|\s+`([^`]+)`\s+\|\s+([0-9,]+)\s+\|\s+`([0-9a-f]+)`\s+\|\s*$", line)
        if m:
            path, size, sha = m.group(1), int(m.group(2).replace(",", "")), m.group(3)
            rows.append((path, size, sha))
    return rows


def file_sha16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> int:
    rows = parse_manifest()
    print(f"verifying {len(rows)} files against {MANIFEST.name}")
    n_ok = n_size_only = n_fail = 0
    for rel, expected_size, expected_sha in rows:
        p = DATA / rel
        if not p.exists():
            print(f"  MISSING: {rel}")
            n_fail += 1
            continue
        actual_size = p.stat().st_size
        if actual_size != expected_size:
            print(f"  SIZE MISMATCH: {rel}  expected {expected_size:,}, got {actual_size:,}")
            n_fail += 1
            continue
        if rel in SIZE_ONLY:
            n_size_only += 1
            continue
        actual_sha = file_sha16(p)
        if actual_sha != expected_sha:
            print(f"  SHA MISMATCH: {rel}  expected {expected_sha}, got {actual_sha}")
            n_fail += 1
            continue
        n_ok += 1

    print()
    print(f"ok:         {n_ok}")
    print(f"size-only:  {n_size_only}  (HDBSCAN non-determinism)")
    print(f"failures:   {n_fail}")
    if n_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
