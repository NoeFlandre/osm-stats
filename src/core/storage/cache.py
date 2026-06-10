"""Materialize a small SQLite cache of the thresholded + standardized tag set.

The full planet taginfo database is 14 GB and 192 M rows. Reading it is slow
and depends on the external drive. This module writes a compact cache file
that contains only the rows above the ``count_all`` threshold, with the
standardized ``feature`` column already computed. All downstream stages
(clustering, taxonomy, blog plots) read from this cache instead.

:func:`build_cache_db_streaming` pages through the source DB in fixed-size
batches using a keyset on ``count_all``, committing after each batch. This
bounds memory usage and lets the build resume across disconnects on a 14 GB
source.
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
