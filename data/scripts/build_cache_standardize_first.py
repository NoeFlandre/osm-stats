"""Build the **standardize-first** cache from the raw taginfo DB.

This is the second of the two cache build modes. It groups the
entire ``tags`` table by the standardized ``(key, value)`` expression
in SQL, sums ``count_all`` within each group, and keeps only the
groups whose merged count reaches the threshold.

It rescues rows that the filter-first path
(``scripts/build_cache_filter_first.py``, which is the default
``--build-cache`` mode of ``src.cli.cli``) would have dropped: e.g.
``landuse`` with 286 occurrences and ``Landuse`` with 450 occurrences
both individually below the threshold, but together (736) over it.

Run with:
    .venv/bin/python -m scripts.build_cache_standardize_first

The new cache is written to
``/Volumes/Seagate M3/tag_features_standardize_first.sqlite`` and the
filter-first cache is left untouched.
"""
import time
from pathlib import Path

from src.config import resolve_db_path
from src.core.storage.cache import (
    build_cache_db_standardize_first,
    read_cache_df,
)

SOURCE_DB = "/Volumes/Seagate M3/taginfo.sqlite"
OUTPUT_CACHE = Path("/Volumes/Seagate M3/tag_features_standardize_first.sqlite")
MIN_COUNT = 500

# Reference numbers from the filter-first path. We use them at the
# end of the build to print the row/occurrence deltas, so the
# maintainer can see at a glance what the new variant rescued.
FILTER_FIRST_ROWS = 224_123
FILTER_FIRST_OCCURRENCES = 3_350_015_993


def main() -> None:
    t0 = time.time()
    print(f"Building standardize-first cache from {SOURCE_DB}")
    print(f"Writing to {OUTPUT_CACHE}")
    print(f"min_count = {MIN_COUNT:,}")
    print()
    print("Aggregating the full tags table by the standardized "
          "(key, value) expression. This is one SQL pass over the "
          "192 M-row source, so it is materially slower than the "
          "filter-first streaming build.")
    print()

    def progress(n: int, _last: int | None) -> None:
        elapsed = time.time() - t0
        print(f"  wrote {n:,} rows so far  ({elapsed:.1f}s elapsed)")

    build_cache_db_standardize_first(
        SOURCE_DB,
        OUTPUT_CACHE,
        min_count=MIN_COUNT,
        progress=progress,
    )
    print(f"  -> done in {time.time() - t0:.1f}s")
    print()

    # Sanity-check the output and print deltas vs the filter-first cache.
    df = read_cache_df(OUTPUT_CACHE, min_count=MIN_COUNT)
    n_rows = len(df)
    n_occ = int(df["count_all"].sum())
    delta_rows = n_rows - FILTER_FIRST_ROWS
    delta_occ = n_occ - FILTER_FIRST_OCCURRENCES

    print(f"Filter-first rows:                {FILTER_FIRST_ROWS:,}")
    print(f"Standardize-first rows:           {n_rows:,}")
    print(f"Delta rows:                       {delta_rows:+,}")
    print()
    print(f"Filter-first occurrences:         {FILTER_FIRST_OCCURRENCES:,}")
    print(f"Standardize-first occurrences:    {n_occ:,}")
    print(f"Delta occurrences:                {delta_occ:+,}")
    print()
    print(f"Cache verified: {n_rows:,} rows, columns={list(df.columns)}")
    print(f"Path: {OUTPUT_CACHE}")


if __name__ == "__main__":
    # Allow the source DB path to be overridden by the env var, so the
    # script honors the same convention as the rest of the codebase.
    SOURCE_DB = str(resolve_db_path())
    main()
