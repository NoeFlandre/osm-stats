#!/usr/bin/env bash
# reproduce.sh — rebuild every artifact in this bucket from a fresh checkout.
#
# This script is the canonical recipe referenced in data/README.md.
# It is idempotent and can be re-run safely: the cache builds replace
# the existing files, the pipeline outputs are overwritten in place,
# and the optional final check verifies the resulting files against
# the MANIFEST.md committed to the bucket.
#
# Usage:
#     .venv/bin/python -m scripts.reproducibility.reproduce   # or
#     bash data/reproducibility/reproduce.sh
#
# Total wall time: ~55 minutes on an M-class laptop.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

THRESHOLD=${THRESHOLD:-500}

echo "=== [1/7] source DB ==="
if [ ! -f "/Volumes/Seagate M3/taginfo.sqlite" ]; then
    echo "Downloading taginfo DB (this is ~2.5 GB compressed)..."
    mkdir -p /Volumes/Seagate\ M3
    curl -L -o /tmp/taginfo-db.db.bz2 \
        https://taginfo.openstreetmap.org/download/taginfo-db.db.bz2
    bzip2 -d /tmp/taginfo-db.db.bz2
    mv /tmp/taginfo-db.db.sqlite /Volumes/Seagate\ M3/taginfo.sqlite
else
    echo "taginfo DB already on disk; skipping download"
fi

echo "=== [2/7] build filter-first cache ==="
.venv/bin/python -m src.cli --build-cache \
    --cache-path /Volumes/Seagate\ M3/tag_features.sqlite \
    --threshold "$THRESHOLD"

echo "=== [3/7] build standardize-first cache ==="
.venv/bin/python -m scripts.build_cache_standardize_first

echo "=== [4/7] run filter-first pipelines ==="
.venv/bin/python -m scripts.profile_clusters
.venv/bin/python -m scripts.profile_clusters_embeddings
.venv/bin/python -m scripts.save_base_key_families
.venv/bin/python -m scripts.compare_pipelines
.venv/bin/python -m scripts.write_env_agri_breakdown

echo "=== [5/7] run standardize-first pipelines ==="
.venv/bin/python -m scripts.profile_clusters_standardize_first
.venv/bin/python -m scripts.profile_clusters_embeddings_standardize_first
.venv/bin/python -m scripts.save_base_key_families_standardize_first
.venv/bin/python -m scripts.compare_pipelines_standardize_first

echo "=== [6/7] global summary stats ==="
.venv/bin/python -m src.cli --summary-only

echo "=== [7/7] verify against MANIFEST.md ==="
.venv/bin/python -m scripts.reproducibility.verify

echo ""
echo "Done. To push the new artifacts to the HF bucket, run:"
echo "  .venv/bin/hf buckets sync data/ hf://buckets/NoeFlandre/osm_stats"
