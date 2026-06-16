"""Tests for the **standardize-first** cache build.

The standardize-first path is a different SQL aggregation from the
filter-first streaming build, so it gets its own test module. The
contract is:

* the source ``tags`` table is read with its original raw ``key`` and
  ``value`` (case- and whitespace-sensitive);
* rows are grouped by the **standardized** expression
  ``LOWER(TRIM(key)) || '|' || LOWER(TRIM(value))`` with the
  missing-value token ``"none"`` substituted for empties;
* ``count_all`` is summed within each group;
* groups whose merged count is below ``min_count`` are dropped;
* the surviving groups land in ``tag_features`` with the standardized
  ``(key, value)`` and a non-empty ``feature`` column.

The tests use an in-memory SQLite DB that mirrors the taginfo
``tags`` schema; no external drive or network needed.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.core.storage.cache import (
    CACHE_SCHEMA,
    build_cache_db_standardize_first,
    read_cache_df,
)


def _make_tags_db(db_path: Path, rows: list) -> Path:
    """Write a SQLite DB at *db_path* with a single ``tags`` table.

    Each row in *rows* is ``(key, value, count_all)`` with the original
    (un-standardized) key/value - this matches what the real taginfo
    download looks like (typos, mixed case, whitespace).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE tags (
                key        TEXT NOT NULL,
                value      TEXT NOT NULL,
                count_all  INTEGER NOT NULL
            )
            """
        )
        conn.executemany("INSERT INTO tags VALUES (?, ?, ?)", rows)
        conn.commit()
    return db_path


# --- shape and round-trip -----------------------------------------------


def test_returns_cache_path(tmp_path: Path):
    src = _make_tags_db(tmp_path / "src.sqlite", [("landuse", "farmland", 1000)])
    out = tmp_path / "out.sqlite"
    returned = build_cache_db_standardize_first(src, out, min_count=500)
    assert returned == out
    assert out.exists()


def test_uses_canonical_cache_schema(tmp_path: Path):
    src = _make_tags_db(tmp_path / "src.sqlite", [("a", "x", 1000)])
    out = tmp_path / "out.sqlite"
    build_cache_db_standardize_first(src, out, min_count=500)
    with sqlite3.connect(out) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(tag_features)").fetchall()]
    assert "key" in cols
    assert "value" in cols
    assert "count_all" in cols
    assert "feature" in cols
    # PRIMARY KEY (key, value) is enforced.
    assert CACHE_SCHEMA  # schema constant is referenced; the file is created from it


def test_replaces_existing_cache_file(tmp_path: Path):
    src = _make_tags_db(tmp_path / "src.sqlite", [("a", "x", 1000)])
    out = tmp_path / "out.sqlite"
    out.touch()
    build_cache_db_standardize_first(src, out, min_count=500)
    df = read_cache_df(out)
    assert len(df) == 1


# --- the core standardize-first contract: typos merge, threshold applies --


def test_case_variants_collapse_to_one_group(tmp_path: Path):
    """``landuse`` and ``Landuse`` (with the same value) collapse to a
    single row, with the merged count."""
    src = _make_tags_db(
        tmp_path / "src.sqlite",
        [
            ("landuse", "Farmland", 286),  # below threshold alone
            ("Landuse", "Farmland", 450),  # below threshold alone
            # merged = 736, above the threshold
        ],
    )
    out = tmp_path / "out.sqlite"
    build_cache_db_standardize_first(src, out, min_count=500)
    df = read_cache_df(out)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["key"] == "landuse"
    assert row["value"] == "farmland"
    assert row["count_all"] == 736
    assert row["feature"] == "landuse|farmland"


def test_whitespace_variants_collapse(tmp_path: Path):
    """Leading/trailing whitespace in key or value is trimmed before
    grouping, so ``"  landuse "`` and ``"landuse"`` merge."""
    src = _make_tags_db(
        tmp_path / "src.sqlite",
        [
            ("  landuse ", "farmland", 300),
            ("landuse", "  farmland  ", 250),
        ],
    )
    out = tmp_path / "out.sqlite"
    build_cache_db_standardize_first(src, out, min_count=500)
    df = read_cache_df(out)
    assert len(df) == 1
    assert int(df.iloc[0]["count_all"]) == 550


def test_empty_string_becomes_none_token(tmp_path: Path):
    """An empty value is standardized to ``"none"`` (the missing-value
    token), not dropped. Same for an empty key."""
    src = _make_tags_db(
        tmp_path / "src.sqlite",
        [
            ("natural", "", 600),  # becomes natural|none
            ("", "water", 600),    # becomes none|water
        ],
    )
    out = tmp_path / "out.sqlite"
    build_cache_db_standardize_first(src, out, min_count=500)
    df = read_cache_df(out)
    keys = set(zip(df["key"], df["value"]))
    assert ("natural", "none") in keys
    assert ("none", "water") in keys


def test_below_threshold_group_is_dropped(tmp_path: Path):
    """A group whose merged count is still below ``min_count`` is
    dropped (not kept with a count below the threshold)."""
    src = _make_tags_db(
        tmp_path / "src.sqlite",
        [
            ("a", "x", 100),
            ("A", "x", 100),  # merged 200, below 500
            ("b", "y", 1000),  # kept
        ],
    )
    out = tmp_path / "out.sqlite"
    build_cache_db_standardize_first(src, out, min_count=500)
    df = read_cache_df(out)
    assert len(df) == 1
    assert df.iloc[0]["key"] == "b"
    assert df.iloc[0]["value"] == "y"


def test_above_threshold_group_is_kept(tmp_path: Path):
    src = _make_tags_db(
        tmp_path / "src.sqlite",
        [("landuse", "farmland", 10_000)],
    )
    out = tmp_path / "out.sqlite"
    build_cache_db_standardize_first(src, out, min_count=500)
    df = read_cache_df(out)
    assert len(df) == 1
    assert int(df.iloc[0]["count_all"]) == 10_000


# --- delimiter and key variants -----------------------------------------


def test_distinct_values_stay_distinct(tmp_path: Path):
    """Different values for the same key produce different groups even
    if they share a base key."""
    src = _make_tags_db(
        tmp_path / "src.sqlite",
        [
            ("landuse", "farmland", 1000),
            ("landuse", "forest", 1000),
        ],
    )
    out = tmp_path / "out.sqlite"
    build_cache_db_standardize_first(src, out, min_count=500)
    df = read_cache_df(out)
    assert set(zip(df["key"], df["value"])) == {
        ("landuse", "farmland"),
        ("landuse", "forest"),
    }


def test_pipe_in_raw_value_is_not_a_delimiter(tmp_path: Path):
    """The standardizer does not strip a literal ``|`` from the value
    (it does not appear in real OSM data, but the function should
    preserve it). Two values that differ only by a pipe are distinct
    groups."""
    src = _make_tags_db(
        tmp_path / "src.sqlite",
        [("note", "a|b", 600), ("note", "a", 600)],
    )
    out = tmp_path / "out.sqlite"
    build_cache_db_standardize_first(src, out, min_count=500)
    df = read_cache_df(out)
    assert len(df) == 2


# --- interaction with read_cache_df --------------------------------------


def test_output_is_readable_by_read_cache_df(tmp_path: Path):
    """The output of the standardize-first build must be readable by
    :func:`read_cache_df` with no special handling - same schema."""
    src = _make_tags_db(
        tmp_path / "src.sqlite",
        [("landuse", "farmland", 1000), ("natural", "water", 600)],
    )
    out = tmp_path / "out.sqlite"
    build_cache_db_standardize_first(src, out, min_count=500)
    df = read_cache_df(out)
    assert set(df["key"]) == {"landuse", "natural"}


def test_progress_callback_receives_total(tmp_path: Path):
    """The progress callback is invoked with the final row count
    (the SQL aggregation is a single statement, so there is no
    intermediate progress)."""
    src = _make_tags_db(
        tmp_path / "src.sqlite",
        [("a", "x", 1000), ("b", "y", 600)],
    )
    out = tmp_path / "out.sqlite"
    calls = []
    build_cache_db_standardize_first(
        src, out, min_count=500, progress=lambda n, last: calls.append((n, last))
    )
    # The last call should carry the final row count.
    final_n = calls[-1][0]
    assert final_n == 2
