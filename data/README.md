# osm-stats — Hugging Face bucket

This bucket is the canonical, citable copy of every artifact produced
by the [osm-stats](https://github.com/NoeFlandre/osm-stats) project.
It is updated together with the GitHub repository, and it is the place
to look for a reproducible snapshot of the data referenced in the
companion blog post
[OSM data analysis for environment and agriculture](https://noeflandre.com/posts/osm-data-analysis).

The bucket lives at
`hf://buckets/NoeFlandre/osm_stats`. The web view is at
<https://huggingface.co/buckets/NoeFlandre/osm_stats>.

## What is in here

| Folder | Purpose | Approx. size |
|---|---|---|
| `caches/` | The two thresholded sqlite caches the pipelines run on | 51 MB |
| `outputs/filter_first/` | All artifacts from the **filter-first** path (TF-IDF + embeddings) | ~28 MB |
| `outputs/standardize_first/` | All artifacts from the **standardize-first** path (TF-IDF + embeddings) | ~28 MB |
| `outputs/stats/` | Global stats written by the CLI summary command | ~2 KB |

### Element-type analysis

`outputs/standardize_first/tfidf/element_type_stats.csv` carries the
element-type breakdown of the 44 manually-kept env/agri superclusters
in two views, side by side:

- `sc_*` columns: **per-supercluster** view (noise excluded). One
  row per supercluster = one row per base key in the XLSX.
  Computed by `src.core.db.supercluster_element_type_stats` from
  the cluster_memberships CSV, joined to the source DB for
  element-type split. This is the primary view for the env/agri
  study: the 44 kept base keys are superclusters, so the right
  unit of analysis is the cluster-membership rollup, not the
  global base-key rollup.
- `src_*` columns: **per-base-key** view (source-DB rollup). One
  row per base key, computed by `src.core.db.element_type_stats`
  from the source `taginfo.sqlite`. This is a flat pre-clustering
  view that includes source rows that ended up in noise.

The XLSX (`outputs/standardize_first/tfidf/base_key_families.xlsx`)
has the same `sc_*` columns appended after the user's manual
`keep` column. The `keep` column is preserved as-is.

#### Per-supercluster numbers (noise excluded, cluster-member rollup)

| | Polygon-friendly | Point-heavy | All 44 |
|---|---:|---:|---:|
| Base keys (superclusters) | **33 / 44 (75.0 %)** | 11 / 44 (25.0 %) | 44 |
| Occurrences | **126,656,240 (62.7 %)** | 75,443,140 (37.3 %) | 202,099,380 |
| Tags (cluster members) | **746 (38.5 %)** | 1,193 (61.5 %) | 1,939 |

`is_polygon_friendly` is `(count_ways + count_relations) / count_all >= 0.5`,
exposed as `POLYGON_FRIENDLY_THRESHOLD` in
`src/core/db/element_type_stats.py`. The 11 point-heavy
superclusters include the 9 obvious point-only base keys
(`tree`, `tumulus`, `species`, `taxon`, `seamark`, `place`,
`product`, `geobasenhn`) plus 2 that flipped from the source-DB
view: `natural` (lots of `natural=tree` nodes in the cluster
members) and `removed` (its cluster members are mostly nodes).
The CSV does **not** address polygon size; that requires a
PBF extract and a separate step.
| `scripts/` | The 12 pipeline scripts, copied verbatim from `scripts/` in the GitHub repo | ~30 KB |
| `reproducibility/` | A single shell script that rebuilds the bucket from a fresh checkout | ~1 KB |
| `MANIFEST.md` | Per-file inventory: every file in the bucket, its size and a sha256 prefix | — |
| `SOURCE.md` | Where the source data comes from, how to (re-)download it, what the caches are | — |

Total: **43 files, ~106 MB**.

## What is NOT in here (and why)

- **The source `taginfo.sqlite` (14 GB)**. This is the raw input to the
  pipeline. It is a public dataset, re-downloadable from
  [taginfo.openstreetmap.org](https://taginfo.openstreetmap.org/download/taginfo-db.db.bz2),
  and would dominate the bucket size. See `SOURCE.md` for the
  download and extraction recipe.
- **The `.venv/`, `.git/`, and `__pycache__/` directories** from the
  working tree.
- **The 192 M-row `tags` source table**. It is a verbatim copy of
  what is already in the public source DB.

## How to use this bucket

### Inspect a specific output

HF storage buckets are accessed via `hf://` paths (not via the
plain HTTP `huggingface.co/...` URLs of regular repos). Any
fsspec-compatible library can read them directly:

```python
import pandas as pd
df = pd.read_csv("hf://buckets/NoeFlandre/osm_stats/outputs/standardize_first/tfidf/base_key_families.csv")
print(df.head(20))
```

For binary files (XLSX, sqlite) the same path works through
`huggingface_hub.HfFileSystem`:

```python
from huggingface_hub import HfFileSystem
fs = HfFileSystem()
with fs.open("hf://buckets/NoeFlandre/osm_stats/outputs/standardize_first/tfidf/base_key_families.xlsx", "rb") as f:
    ...
```

DuckDB, pyarrow, polars and any other fsspec-aware tool are also
supported; see the
[HF access patterns docs](https://huggingface.co/docs/hub/storage-buckets-access)
for the full list.

### Re-run the pipeline from a fresh checkout

```bash
git clone https://github.com/NoeFlandre/osm-stats
cd osm-stats
# (follow SOURCE.md to get taginfo.sqlite and build the cache)
.venv/bin/python -m scripts.reproducibility.reproduce
```

The `reproducibility/reproduce.sh` script is a single command that:

1. Downloads `taginfo.sqlite` if not already on disk.
2. Builds the two caches.
3. Runs the four pipeline variants (TF-IDF and embeddings on each
   of the two caches).
4. Produces the base-key-family CSVs and XLSX files.
5. Verifies that the resulting artifact hashes match the bucket's
   `MANIFEST.md`.

### Reproduce just the standardize-first outputs

```bash
.venv/bin/python -m scripts.build_cache_standardize_first
.venv/bin/python -m scripts.profile_clusters_standardize_first
.venv/bin/python -m scripts.profile_clusters_embeddings_standardize_first
.venv/bin/python -m scripts.save_base_key_families_standardize_first
.venv/bin/python -m scripts.compare_pipelines_standardize_first
```

The total wall time on an M-class laptop is roughly **10 minutes for
the cache build + 4 minutes for TF-IDF + 40 minutes for the
embeddings pipeline**.

## Provenance and timestamps

| Step | Source of truth | Reproducible? |
|---|---|---|
| Source `taginfo.sqlite` | [taginfo DB](https://taginfo.openstreetmap.org/download/taginfo-db.db.bz2), `data_until = 2026-06-09 00:59:09` | yes, re-download |
| `tag_features.sqlite` (filter-first) | `python -m src.cli --build-cache --threshold 500` | yes |
| `tag_features_standardize_first.sqlite` | `python -m scripts.build_cache_standardize_first` | yes, but HDBSCAN is non-deterministic |
| All `outputs/*` CSVs / XLSX / MD | The `scripts/*.py` files in this bucket, in the order listed above | yes (with the same caveat) |

The two caches were built on a 2026-06-16 snapshot of the source
DB. Re-running the pipeline on a newer snapshot will produce a
slightly different cache and slightly different cluster assignments;
this is expected.

## Companion resources

- Project GitHub: [github.com/NoeFlandre/osm-stats](https://github.com/NoeFlandre/osm-stats)
- Blog post: [noeflandre.com/posts/osm-data-analysis](https://noeflandre.com/posts/osm-data-analysis)
- Taginfo source: [taginfo.openstreetmap.org/download](https://taginfo.openstreetmap.org/download/taginfo-db.db.bz2)
- Embedding model used: [`potion-base-8M`](https://huggingface.co/minishlab/potion-base-8M) by Minish Lab (via `model2vec`)

## License

The source data is from OpenStreetMap, distributed under the
[ODbL](https://www.openstreetmap.org/copyright). The code in this
bucket is MIT (see the GitHub repo's `LICENSE` file).
