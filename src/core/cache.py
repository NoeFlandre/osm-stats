"""Materialize a small SQLite cache of the thresholded + standardized tag set.

The full planet taginfo database is 14 GB and 192 M rows. Reading it is slow
and depends on the external drive. This module writes a compact cache file
that contains only the rows above the ``count_all`` threshold, with the
standardized ``feature`` column already computed. All downstream stages
(clustering, taxonomy, blog plots) read from this cache instead.

The streaming variant (``build_cache_db_streaming``) pages through the source
DB in fixed-size batches using a keyset on ``count_all``, committing after
each batch. This bounds memory usage and lets the build resume across
disconnects on a 14 GB source.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable, Iterable, Tuple, Union

import pandas as pd

from src.core.standardize import standardize_dataframe

CACHE_SCHEMA = """
CREATE TABLE tag_features (
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    count_all  INTEGER NOT NULL,
    feature    TEXT NOT NULL,
    PRIMARY KEY (key, value)
);
"""


def build_cache_db(df: pd.DataFrame, output_path: Union[str, Path]) -> Path:
    """Write a sqlite cache of *df* to *output_path* in one shot.

    Use :func:`build_cache_db_streaming` for large DataFrames that don't fit
    in memory or come from a slow source.
    """
    output_path = Path(output_path)
    required = {"key", "value", "count_all"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"build_cache_db: missing columns {sorted(missing)}")

    raw = df[["key", "value", "count_all"]].copy()
    std = standardize_dataframe(raw)
    out = std[["key", "value", "count_all", "feature"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    with sqlite3.connect(output_path) as conn:
        conn.executescript(CACHE_SCHEMA)
        conn.executemany(
            "INSERT INTO tag_features (key, value, count_all, feature) "
            "VALUES (?, ?, ?, ?)",
            out.itertuples(index=False, name=None),
        )
        conn.execute("CREATE INDEX idx_features_count ON tag_features(count_all DESC)")
        conn.commit()
    return output_path


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

    Returns the cache path. The schema is the same as
    :func:`build_cache_db`.
    """
    from src.core.queries import QueryBuilder

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    total_written = 0
    after: int | None = None
    last_count: int | None = None

    with sqlite3.connect(output_path) as conn:
        conn.executescript(CACHE_SCHEMA)
        conn.commit()

        while True:
            sql, params = QueryBuilder.iter_tags_by_min_count(
                min_count=min_count, batch_size=batch_size, after_count=after
            )
            page = db.execute_query(sql, params)
            if page.empty:
                break

            std = standardize_dataframe(page[["key", "value", "count_all"]])
            std = std[["key", "value", "count_all", "feature"]]
            conn.executemany(
                "INSERT OR IGNORE INTO tag_features (key, value, count_all, feature) "
                "VALUES (?, ?, ?, ?)",
                std.itertuples(index=False, name=None),
            )
            conn.commit()

            total_written += len(std)
            last_count = int(std["count_all"].iloc[-1])
            after = last_count  # next page: strictly less than the last seen
            if progress is not None:
                progress(total_written, last_count)

            if len(page) < batch_size:
                break

        # Build the index after the bulk load - faster than per-batch.
        conn.execute("CREATE INDEX idx_features_count ON tag_features(count_all DESC)")
        conn.commit()

    return output_path


def read_cache_df(
    cache_path: Union[str, Path],
    min_count: int | None = None,
) -> pd.DataFrame:
    """Read the tag_features table as a DataFrame, optionally filtered by count."""
    cache_path = Path(cache_path)
    sql = "SELECT key, value, count_all, feature FROM tag_features"
    params: tuple = ()
    if min_count is not None:
        sql += " WHERE count_all >= ?"
        params = (min_count,)
    sql += " ORDER BY count_all DESC"
    with sqlite3.connect(cache_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)
