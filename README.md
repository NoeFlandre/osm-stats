# osm-stats

A reproducible pipeline that takes the public
[OSM taginfo](https://taginfo.openstreetmap.org/) statistics database
and turns it into **a curated list of OSM base keys relevant to
environment and agriculture**, by clustering the 225 k most-used
tags into semantic groups with two complementary methods (TF-IDF on
character n-grams, and `potion-base-8M` semantic embeddings).

The full writeup is in the companion blog post:
**[OSM data analysis for environment and agriculture](https://noeflandre.com/posts/osm-data-analysis)**.

## What's in this repo

- `src/` — the library code (cache builder, TF-IDF, SVD, HDBSCAN, embeddings runner, profile / render / compare).
- `scripts/` — the 12 pipeline scripts (filter-first + standardize-first, × 2 pipelines). Run with `python -m scripts.<name>`.
- `tests/` — 227 unit tests.
- `data/` — local staging area for the [HF bucket](https://huggingface.co/buckets/NoeFlandre/osm_stats). The bucket is the canonical home of every output; `data/` is the working copy used to push to it. The bucket-only files (`MANIFEST.md`, the two caches, all `outputs/`) are gitignored.
- `output/` — working directory. The pipelines write here. Gitignored. Identical contents to `data/outputs/`.

## What's in the Hugging Face bucket

[`hf://buckets/NoeFlandre/osm_stats`](https://huggingface.co/buckets/NoeFlandre/osm_stats) — 47 files, ~106 MB.

| | |
|---|---|
| `caches/` | The two thresholded sqlite caches the pipelines run on (51 MB total) |
| `outputs/filter_first/` | TF-IDF + embeddings artifacts on the filter-first cache |
| `outputs/standardize_first/` | TF-IDF + embeddings artifacts on the standardize-first cache (the method retained in the blog post) |
| `outputs/stats/` | Global stats from the CLI summary command |
| `scripts/` | A copy of the 12 pipeline scripts, frozen at the version that produced the artifacts |
| `reproducibility/reproduce.sh` | A one-command reproducer that rebuilds the bucket from a fresh checkout |
| `MANIFEST.md` | Per-file inventory with sizes and sha256 prefixes |
| `SOURCE.md` | Where the source DB comes from, how to (re-)download it |

The `data/` directory in this repo mirrors the bucket contents
1:1. To push a new version of the artifacts to the bucket:

```bash
.venv/bin/hf buckets sync ./data hf://buckets/NoeFlandre/osm_stats \
    --exclude '*.DS_Store' --exclude '__pycache__'
```

## How to use the data without running the pipeline

The bucket is accessed via `hf://` paths through `HfFileSystem`
(buckets are not served as static HTTP files, unlike regular repos).
Any fsspec-compatible library works:

```python
import pandas as pd
df = pd.read_csv(
    "hf://buckets/NoeFlandre/osm_stats/"
    "outputs/standardize_first/tfidf/base_key_families.csv"
)
print(df.head(20))
```

Or open the sqlite cache directly:

```python
from huggingface_hub import HfFileSystem
import sqlite3, io
fs = HfFileSystem()
with fs.open("hf://buckets/NoeFlandre/osm_stats/caches/tag_features_standardize_first.sqlite", "rb") as f:
    con = sqlite3.connect(":memory:")
    con.deserialize(f.read())
```

The XLSX files in the bucket are the manual-classification sheets:
open them in Excel, fill the `keep` column with your env/agri
verdict, save the file. The two files (one per pipeline) are
`outputs/standardize_first/{tfidf,embeddings}/base_key_families.xlsx`.

## How to reproduce from a fresh checkout

```bash
git clone https://github.com/NoeFlandre/osm-stats
cd osm-stats
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# Download the 14 GB source DB (see data/SOURCE.md for details)
curl -L -o /tmp/taginfo-db.db.bz2 \
    https://taginfo.openstreetmap.org/download/taginfo-db.db.bz2
bzip2 -d /tmp/taginfo-db.db.bz2
mv /tmp/taginfo-db.db.sqlite /Volumes/Seagate\ M3/taginfo.sqlite

# Run the whole pipeline (~55 minutes wall time on M-class hardware)
.venv/bin/python -m scripts.reproducibility.reproduce
```

See `data/reproducibility/reproduce.sh` for the exact 7-step recipe.

## Provenance and dates

| Resource | Date | Source |
|---|---|---|
| `taginfo.sqlite` (input) | 2026-06-09 | [taginfo download](https://taginfo.openstreetmap.org/download/taginfo-db.db.bz2) |
| `tag_features.sqlite` (filter-first) | 2026-06-10 | `python -m src.cli --build-cache` |
| `tag_features_standardize_first.sqlite` | 2026-06-16 | `python -m scripts.build_cache_standardize_first` |
| All `outputs/*` | 2026-06-15 / 2026-06-16 | The `scripts/*.py` files in this repo |

## Element-type analysis

The XLSX (`output/standardize_first/{tfidf,embeddings}/base_key_families.xlsx`)
and CSV (`output/standardize_first/{tfidf,embeddings}/element_type_stats.csv`)
both carry an element-type breakdown for the manually-kept base keys
(157 for TF-IDF, 169 for embeddings). They are computed by two
complementary functions in `src/core/db/`:

- `element_type_stats` (per-base-key, source-DB rollup): for every
  (key, value) pair in the source `taginfo.sqlite` whose key's
  first colon prefix is X, sum the element-type split. This is a
  flat, pre-clustering view.
- `supercluster_element_type_stats` (per-supercluster, cluster
  rollup): for every cluster whose medoid's base key is X, take
  the union of its members; sum the element-type split over that
  union. Noise (cluster_id = -1) is excluded.

The per-supercluster view is the right unit of analysis for this
study because the labeled base keys are *superclusters* — one row
in the XLSX represents a group of clusters, not a global base-key
rollup. A supercluster can contain members with different own base
keys than the supercluster's own (e.g. a cluster with medoid
`tree|species:oak` may also contain `forest|species:oak`), and it
does not contain source rows that ended up in noise.

### TF-IDF pipeline — 157 kept out of 427 superclusters

The TF-IDF pipeline produced 8,832 real clusters + 78,270 noise
points from the 225,684 standardized tags in the cache. The
non-noise 147,414 tags cover 2,246,255,835 occurrences; the noise
78,270 tags cover another 1,122,085,693. Together the 225,684
tags in the cluster memberships file cover the 3,368,341,528
occurrences reported in the blog post.

| | Tags | Occurrences | Real clusters |
|---|---:|---:|---:|
| All cluster memberships (incl. noise) | **225,684** | 3,368,341,528 | 8,832 |
| &nbsp;&nbsp;Real clusters (noise excluded) | 147,414 | 2,246,255,835 | 8,832 |
| &nbsp;&nbsp;Noise (cluster_id = -1) | 78,270 | 1,122,085,693 | — |
| Real, by base-key label (157 yes kept) | **14,858** | **1,167,476,227** | **859** |
| Real, by base-key label (270 not-kept: 54 uncertain + 216 no) | 132,556 | 1,078,779,608 | 7,973 |

So the 157 "yes" labels cover 10.1 % of all real-cluster tags
(14,858 / 147,414) and 52.0 % of all real-cluster occurrences
(1,167,476,227 / 2,246,255,835), but only 9.7 % of the clusters
(859 / 8,832). The kept superclusters are the high-volume ones
(`building`, `highway`, `landuse`, `natural`, `tiger`, `area`,
`surface`, `water`, `wetland`, etc.) — the long tail of small,
specialized base keys was filtered out by the manual labeling.

### Embeddings pipeline — 169 kept out of 433 superclusters

The embeddings pipeline (potion-base-8M) produced 4,954 real
clusters + 106,498 noise points from the same 225,684 standardized
tags. The non-noise 119,186 tags cover 2,565,137,600 occurrences;
the noise 106,498 tags cover another 803,203,928. The embeddings
clusters are fewer and larger than TF-IDF's, and the noise fraction
is higher.

| | Tags | Occurrences | Real clusters |
|---|---:|---:|---:|
| All cluster memberships (incl. noise) | **225,684** | 3,368,341,528 | 4,954 |
| &nbsp;&nbsp;Real clusters (noise excluded) | 119,186 | 2,565,137,600 | 4,954 |
| &nbsp;&nbsp;Noise (cluster_id = -1) | 106,498 | 803,203,928 | — |
| Real, by base-key label (169 yes kept) | **17,612** | **978,614,046** | **511** |
| Real, by base-key label (264 not-kept: 57 uncertain + 207 no) | 101,574 | 1,586,523,554 | 4,443 |

The 169 "yes" labels cover 14.8 % of all real-cluster tags
(17,612 / 119,186) and 38.2 % of all real-cluster occurrences
(978,614,046 / 2,565,137,600), but only 10.3 % of the clusters
(511 / 4,954). The kept superclusters are similar to TF-IDF
(`building`, `highway`, `natural`, `tiger`, `landuse`, etc.) but
with semantic-cluster members: e.g. embeddings place `natural=water`,
`natural=wetland`, `natural=wood`, `natural=tree`, `natural=scrub`
together in one cluster because they describe environmental
features.

### Per-supercluster numbers (noise excluded, cluster-member rollup)

#### TF-IDF (157 kept)

| | Polygon-friendly | Point-heavy | All 157 |
|---|---:|---:|---:|
| Base keys (superclusters) | **110 / 157 (70.1 %)** | 47 / 157 (29.9 %) | 157 |
| Occurrences | **1,061,684,644 (90.9 %)** | 105,791,583 (9.1 %) | 1,167,476,227 |
| Tags (cluster members) | **12,504 (84.2 %)** | 2,354 (15.8 %) | 14,858 |
| Real clusters | **670 (78.0 %)** | 189 (22.0 %) | 859 |

#### Embeddings (169 kept)

| | Polygon-friendly | Point-heavy | All 169 |
|---|---:|---:|---:|
| Base keys (superclusters) | **118 / 169 (69.8 %)** | 51 / 169 (30.2 %) | 169 |
| Occurrences | **798,138,011 (81.6 %)** | 180,476,035 (18.4 %) | 978,614,046 |
| Tags (cluster members) | **15,683 (89.0 %)** | 1,929 (11.0 %) | 17,612 |
| Real clusters | **359 (70.3 %)** | 152 (29.7 %) | 511 |

`is_polygon_friendly` is `(count_ways + count_relations) / count_all >= 0.5`,
exposed as `POLYGON_FRIENDLY_THRESHOLD` in
`src/core/db/element_type_stats.py`. The CSVs do **not** address
polygon size; that requires a PBF extract and a separate step.

These numbers are independently verified by:

- `tests/core/db/test_supercluster_stats_audit.py` (TF-IDF pipeline)
- `tests/core/db/test_embeddings_supercluster_stats_audit.py` (embeddings pipeline)

Each suite re-computes every metric two ways from the raw input
files and asserts they match the function's output. The TF-IDF
audit caught three real bugs that earlier outputs had
(double-counting of source-DB element-type counts when the same
(key, value) appears in multiple clusters within a supercluster;
missing `TRIM()` in the source-DB aggregate; Cyrillic/German-umlaut
case-mismatch when re-lowercasing in Python with `str.lower()`).
The embeddings audit verifies the same function on the embeddings
artifacts. 45 / 45 tests pass (22 unit + 11 TF-IDF audit + 12
embeddings audit).

## Caveats

- **HDBSCAN is non-deterministic.** Re-running the pipeline on the
  same cache may produce very similar clusters but not bit-identical
  medoid/membership CSVs. The bucket's `MANIFEST.md` marks the
  non-deterministic files; `data/MANIFEST.md` is included in the
  bucket so the next maintainer can compare. `scripts/reproducibility/verify.py`
  does the bit-by-bit check on the deterministic files.
- **The 192 M-row `tags` source table is not in the bucket** — it's
  the public taginfo DB, re-downloadable from the URL above.
- **The taginfo DB is updated weekly.** The artifacts in the bucket
  are pinned to a 2026-06-09 snapshot. Re-running on a newer
  snapshot will reflect additional mapper activity since then.

## License

- Code: MIT (see `LICENSE`).
- Source data: OpenStreetMap contributors, [ODbL](https://www.openstreetmap.org/copyright).

## Companion resources

- Blog post: <https://noeflandre.com/posts/osm-data-analysis>
- HF bucket: <https://huggingface.co/buckets/NoeFlandre/osm_stats>
- Embedding model: [`potion-base-8M`](https://huggingface.co/minishlab/potion-base-8M) (Minish Lab, via `model2vec`)
- Taginfo source: <https://taginfo.openstreetmap.org/download>
