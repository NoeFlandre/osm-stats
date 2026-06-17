# SOURCE — where the input data comes from

This file documents the inputs to the pipeline. The buckets in
`hf://buckets/NoeFlandre/osm_stats` are derived data, not raw data.

## The source DB

Everything starts from the **taginfo statistics database**, a public
dataset produced by the
[taginfo project](https://taginfo.openstreetmap.org/). The DB
contains aggregate counts of all OSM keys and tags across the planet;
it does **not** contain coordinates, OSM IDs, or any geometry. It
is the authoritative source for "how many OSM objects carry this
key=value" numbers, broken down by element type (node, way,
relation).

### How to download

```bash
# ~2.5 GB compressed, ~14 GB extracted
curl -L -o taginfo-db.db.bz2 \
  https://taginfo.openstreetmap.org/download/taginfo-db.db.bz2
bzip2 -d taginfo-db.db.bz2
mv taginfo-db.db.sqlite /Volumes/Seagate\ M3/taginfo.sqlite
```

The path the project expects is `/Volumes/Seagate M3/taginfo.sqlite`
(override with the `OSM_DB_PATH` environment variable).

### What is in the DB (high level)

| Table | Rows | What it has |
|---|---|---|
| `keys` | ~110 k | One row per distinct OSM key with global and per-element-type counts |
| `tags` | ~192 M | One row per distinct `key=value` with global and per-element-type counts |
| `key_combinations` | millions | Most common co-occurring keys |
| `tag_distributions` | millions | Per-key, per-element-type histograms (PNG blobs) |
| `prevalent_values` | millions | Most common values for each key |
| `similar_keys` | thousands | Keys that often co-occur |
| `source` | 1 | Build metadata (date the DB was generated, etc.) |

This project reads from `tags` and from `keys` only. The exact
schema for `tags` is:

```sql
CREATE TABLE tags (
    key              VARCHAR,
    value            VARCHAR,
    count_all        INTEGER DEFAULT 0,
    count_nodes      INTEGER DEFAULT 0,
    count_ways       INTEGER DEFAULT 0,
    count_relations  INTEGER DEFAULT 0,
    in_wiki          INTEGER DEFAULT 0,
    in_wiki_en       INTEGER DEFAULT 0
);
```

## The thresholded caches

The raw `tags` table is 192 M rows. The pipeline operates on a
thresholded view (`count_all >= 500`), which collapses it to about
225 k rows. There are two thresholded caches, one for each
preprocessing strategy:

| Cache | Path | Rows | Occurrences | How it is built |
|---|---|---:|---:|---|
| `tag_features.sqlite` | `caches/tag_features.sqlite` | 224,123 | 3,350,015,993 | `python -m src.cli --build-cache --threshold 500` (filter-first) |
| `tag_features_standardize_first.sqlite` | `caches/tag_features_standardize_first.sqlite` | 225,684 | 3,368,341,528 | `python -m scripts.build_cache_standardize_first` (standardize-first) |

Both caches share the same `tag_features(key, value, count_all, feature)`
schema, where `feature = key + "|" + value` is the standardized
representation used for clustering. The two are interchangeable as
input to `read_cache_df` and the rest of the pipeline; only the
`count_all` aggregation step differs.

### Why two caches?

The two paths differ in **the order of standardization and
filtering**:

- **Filter-first** (the historical default in this project): the
  source DB is filtered to `count_all >= 500` first, then tags are
  standardized. This drops every tag below the threshold, including
  the ones that would pass the threshold if you first standardized
  them with their near-duplicates.
- **Standardize-first** (the variant retained in the blog post):
  the source DB is grouped by the standardized `(key, value)`
  expression in SQL, `count_all` is summed within each group, and
  then `count_all >= 500` is applied to the aggregated count. This
  rescues case/whitespace variants that individually fall below
  the threshold but together exceed it.

The deltas between the two caches are small (+1,561 rows, +18.3 M
occurrences) but real, and they change the cluster assignments
downstream.

## Caveats and known non-determinism

- **HDBSCAN is non-deterministic.** The cluster assignments depend
  on the order in which the rows are presented to the algorithm and
  on the random tie-breaking. Re-running the pipeline on the same
  cache may produce clusters with the same boundaries and very
  similar counts, but not bit-identical medoid files. The four
  `cluster_medoids*.csv` files in `outputs/` are pinned to the run
  used for the blog post.
- **Embedding lookups in `potion-base-8M` are deterministic.** Two
  runs of the embeddings pipeline on the same input always produce
  the same embeddings and the same clusters. The
  non-determinism in the embeddings path is purely from HDBSCAN.
- **The taginfo DB is updated weekly.** The version used here was
  generated on 2026-06-09. A re-run on a newer version will reflect
  the additional mapper activity between then and the new build
  date.

## Acknowledgements

- **OpenStreetMap** contributors, who produce the underlying data
  ([ODbL](https://www.openstreetmap.org/copyright)).
- **Taginfo** (by Jochen Topf), which makes the aggregate statistics
  available as a public DB.
- **Minish Lab**, for the `potion-base-8M` embedding model
  ([HF](https://huggingface.co/minishlab/potion-base-8M),
  [paper](https://arxiv.org/abs/2405.15324)).
