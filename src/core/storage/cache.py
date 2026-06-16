"""Materialize a small SQLite cache of the thresholded + standardized tag set.

The full planet taginfo database is 14 GB and 192 M rows. Reading it is slow
and depends on the external drive. This module writes a compact cache file
that contains only the rows above the ``count_all`` threshold, with the
standardized ``feature`` column already computed. All downstream stages
(clustering, taxonomy, blog plots) read from this cache instead.

Two build modes are exposed:

* :func:`build_cache_db_streaming` — **filter-first**: pages through the
  source DB in fixed-size batches, dropping ``count_all < min_count`` rows
  before standardizing. This is the cheap, fast path.
* :func:`build_cache_db_standardize_first` — **standardize-first**:
  groups the entire ``tags`` table by the standardized
  ``(key, value)`` pair in SQL, sums ``count_all`` within each group, then
  keeps only the groups whose sum reaches ``min_count``. This rescues
  rows that would have been dropped by the filter-first path (e.g.
  ``landuse`` and ``Landuse`` separately under the threshold collapse to
  one group over the threshold). It is significantly slower on the
  192 M-row source because SQLite must aggregate the full table.

Both functions write the same ``CACHE_SCHEMA`` and are interchangeable
downstream: :func:`read_cache_df` and :func:`add_base_key_column` work
identically on the output of either.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable, Union

import pandas as pd

from src.core.db.queries import QueryBuilder
from src.core.features.base_key import parse_base_key
from src.core.features.standardize import standardize_dataframe

CACHE_SCHEMA = """
CREATE TABLE tag_features (
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    count_all  INTEGER NOT NULL,
    feature    TEXT NOT NULL,
    PRIMARY KEY (key, value)
);
"""


ProgressCb = Callable[[int, int | None], None]  # (rows_written_so_far, last_count_all)


# SQL expression that mirrors src.core.features.standardize._normalize_column
# for the case statement. Empty-after-trim strings become 'none' (the
# missing-value token). The two halves are joined with '|' as the delimiter.
# This expression is the GROUP BY key for the standardize-first path.
STANDARDIZED_KEY_EXPR = (
    "CASE WHEN LOWER(TRIM(key))   = '' THEN 'none' "
    "ELSE LOWER(TRIM(key)) END"
)
STANDARDIZED_VALUE_EXPR = (
    "CASE WHEN LOWER(TRIM(value)) = '' THEN 'none' "
    "ELSE LOWER(TRIM(value)) END"
)
STANDARDIZED_FEATURE_EXPR = (
    f"{STANDARDIZED_KEY_EXPR} || '|' || {STANDARDIZED_VALUE_EXPR}"
)


def build_cache_db_streaming(
    db,  # any DBProtocol: has execute_query(sql, params) -> DataFrame
    output_path: Union[str, Path],
    min_count: int = 500,
    batch_size: int = 50_000,
    progress: ProgressCb | None = None,
) -> Path:
    """Stream rows from *db* into the cache, in batches of *batch_size* rows.

    A keyset on ``count_all`` is used to advance the scan in O(batch) time
    per page. Each batch is committed independently so the build can be
    interrupted and re-run (replacing the cache file from scratch).
    Returns the cache path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    total_written = 0
    last_count: int | None = None
    first_page = True

    with sqlite3.connect(output_path) as conn:
        conn.executescript(CACHE_SCHEMA)
        conn.commit()

        while True:
            if first_page:
                sql, params = QueryBuilder.first_tags_by_min_count(
                    min_count=min_count, batch_size=batch_size
                )
                first_page = False
            else:
                assert last_count is not None  # set on the previous iteration
                sql, params = QueryBuilder.next_tags_by_min_count(
                    min_count=min_count,
                    after_count=last_count,
                    batch_size=batch_size,
                )
            page = db.execute_query(sql, params)
            if page.empty:
                break

            std = standardize_dataframe(page[["key", "value", "count_all"]])
            std = std[["key", "value", "count_all", "feature"]]
            conn.executemany(
                QueryBuilder.TAG_FEATURES_INSERT[0],
                std.itertuples(index=False, name=None),
            )
            conn.commit()

            total_written += len(std)
            last_count = int(std["count_all"].iloc[-1])
            if progress is not None:
                progress(total_written, last_count)

            if len(page) < batch_size:
                break

        # Build the index after the bulk load - faster than per-batch.
        conn.execute(QueryBuilder.TAG_FEATURES_INDEX[0])
        conn.commit()

    return output_path


def build_cache_db_standardize_first(
    source_db_path: Union[str, Path],
    output_path: Union[str, Path],
    min_count: int = 500,
    progress: ProgressCb | None = None,
) -> Path:
    """Build the cache with **standardize-first** ordering.

    The source DB must expose a ``tags`` table with columns
    ``(key TEXT, value TEXT, count_all INTEGER)`` (the canonical
    taginfo schema). The function:

    1. ``GROUP BY`` the standardized ``(key, value)`` expression
       (``LOWER(TRIM(...))`` with empty -> ``"none"``, joined with
       ``|``).
    2. ``SUM(count_all)`` within each group, so typos and case
       variants collapse to a single row with the merged volume.
    3. ``HAVING SUM(count_all) >= min_count`` to keep only the
       groups that reach the threshold once merged.
    4. ``INSERT`` each surviving group into the new ``tag_features``
       table.

    The whole aggregation is one ``INSERT ... SELECT ... GROUP BY``
    statement, so SQLite handles the work in a single pass over the
    source ``tags`` table. On the 192 M-row source this is materially
    slower than :func:`build_cache_db_streaming` (which streams and
    drops uninteresting rows before any aggregation), but it rescues
    rows whose standardized form would have cleared the threshold only
    after merging with their near-duplicates.

    Parameters
    ----------
    source_db_path:
        Path to the source SQLite database (the taginfo download).
    output_path:
        Path to write the new cache. Any existing file is replaced.
    min_count:
        Minimum merged ``count_all`` to keep a group (default 500).
    progress:
        Optional ``(rows_written, last_count_all)`` callback. Because
        the SQL aggregation is a single statement, only the final
        ``(n_rows, None)`` is reported (no incremental ticks). For
        fine-grained progress, time the call and report "done".

    Returns the path that was written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    if progress is not None:
        progress(0, None)

    with sqlite3.connect(output_path) as out_conn:
        out_conn.executescript(CACHE_SCHEMA)
        # ATTACH the source DB so the GROUP BY below can read its
        # ``tags`` table from the same connection. We qualify every
        # reference as ``src.tags`` to make the data flow explicit.
        # The DETACH is implicit: closing out_conn below releases the
        # attachment. Calling DETACH explicitly is unreliable on
        # in-memory or freshly-created source DBs (it can race with
        # SQLite's lock cleanup).
        out_conn.execute(f"ATTACH DATABASE ? AS src", (str(source_db_path),))
        out_conn.execute(
            f"""
            INSERT INTO tag_features (key, value, count_all, feature)
            SELECT
                {STANDARDIZED_KEY_EXPR}        AS std_key,
                {STANDARDIZED_VALUE_EXPR}      AS std_value,
                SUM(count_all)                 AS merged_count,
                {STANDARDIZED_FEATURE_EXPR}    AS std_feature
            FROM src.tags
            GROUP BY {STANDARDIZED_FEATURE_EXPR}
            HAVING SUM(count_all) >= ?
            """,
            (min_count,),
        )
        # Build the index after the bulk load - faster than per-row.
        out_conn.execute(QueryBuilder.TAG_FEATURES_INDEX[0])
        out_conn.commit()

        n_rows = out_conn.execute(
            "SELECT COUNT(*) FROM tag_features"
        ).fetchone()[0]

    if progress is not None:
        progress(int(n_rows), None)

    return output_path


def read_cache_df(
    cache_path: Union[str, Path],
    min_count: int | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Read the tag_features table as a DataFrame, optionally filtered by count and capped by *limit*."""
    cache_path = Path(cache_path)
    with sqlite3.connect(cache_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(tag_features)").fetchall()}
        with_base_key = "base_key" in cols
        if min_count is None:
            sql, params = QueryBuilder.tag_features_select_all(
                limit=limit, with_base_key=with_base_key
            )
        else:
            sql, params = QueryBuilder.tag_features_select(
                min_count=min_count, limit=limit, with_base_key=with_base_key
            )
        return pd.read_sql_query(sql, conn, params=params)


def add_base_key_column(cache_path: Union[str, Path]) -> None:
    """Add (or replace) a ``base_key`` column on the ``tag_features`` table.

    The base key is the OSM key namespace root (everything before the
    first colon) extracted from the ``feature`` column. The function is
    idempotent: a second call updates the column in place. The new column
    makes it trivial to whitelist or blacklist entire tag families
    (e.g. ``base_key IN ('landuse', 'natural')``) without re-running the
    pipeline.

    Implementation: a single SQL UPDATE that extracts the base key with
    a SQLite expression on the ``feature`` column. This keeps the
    operation O(n) with no Python-side per-row round-trips.
    """
    cache_path = Path(cache_path)
    with sqlite3.connect(cache_path) as conn:
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(tag_features)").fetchall()
        }
        if "base_key" not in cols:
            conn.execute("ALTER TABLE tag_features ADD COLUMN base_key TEXT")
        # SUBSTR(feature, 1, INSTR(feature, '|') - 1) gives us the key
        # portion; the first colon truncates it to the namespace root.
        # LOWER + TRIM normalize the value to match what parse_base_key
        # would produce in Python.
        conn.execute(
            """
            UPDATE tag_features
            SET base_key = LOWER(TRIM(
                CASE
                    WHEN INSTR(feature, ':') > 0
                         AND INSTR(feature, ':') < INSTR(feature, '|')
                    THEN SUBSTR(feature, 1, INSTR(feature, ':') - 1)
                    ELSE SUBSTR(feature, 1, INSTR(feature, '|') - 1)
                END
            ))
            WHERE base_key IS NULL
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_features_base_key "
            "ON tag_features(base_key)"
        )
        conn.commit()
