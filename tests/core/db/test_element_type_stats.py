"""Tests for ``src.core.db.element_type_stats``.

For the kept env/agri base keys, are the objects we will end up
fetching actually polygons? Or are most of them points
(e.g. ``monitoring:water_quality`` is always a node)?

The source ``taginfo.sqlite`` already carries the answer per
(key, value) in the form of ``count_nodes / count_ways /
count_relations``. We aggregate those per base key, so for a base
key like ``landuse`` we sum across ``landuse=farmland``,
``landuse=residential``, ... and report the element-type split.
The same applies to a colon-prefixed base key like ``addr``
(covers ``addr``, ``addr:city``, ``addr:street``, ...).

The test fixture is an in-memory SQLite with a hand-built ``tags``
table. We do not touch the 14 GB real DB from unit tests.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from src.core.db.element_type_stats import (
    ELEMENT_TYPE_STATS_COLUMNS,
    POLYGON_FRIENDLY_THRESHOLD,
    element_type_stats,
)


def _make_tags_db(rows: list[tuple]) -> sqlite3.Connection:
    """Build an in-memory SQLite with a ``tags`` table populated from *rows*.

    Each row is ``(key, value, count_all, count_nodes, count_ways,
    count_relations)``.
    """
    con = sqlite3.connect(":memory:")
    con.execute(
        """
        CREATE TABLE tags (
            key              VARCHAR,
            value            VARCHAR,
            count_all        INTEGER,
            count_nodes      INTEGER,
            count_ways       INTEGER,
            count_relations  INTEGER
        );
        """
    )
    con.executemany(
        "INSERT INTO tags VALUES (?, ?, ?, ?, ?, ?)", rows
    )
    con.commit()
    return con


def test_empty_input_returns_empty_dataframe_with_canonical_columns():
    """An empty base_keys list short-circuits to an empty DataFrame that
    still has the canonical schema (callers can rely on the columns)."""
    con = _make_tags_db([])
    out = element_type_stats([], con)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ELEMENT_TYPE_STATS_COLUMNS
    assert len(out) == 0


def test_polygon_heavy_base_key_is_flagged_polygon_friendly():
    """A base key whose occurrences are mostly on ways+relations
    (e.g. ``landuse`` -> almost entirely polygons) is flagged as
    polygon-friendly."""
    con = _make_tags_db(
        [
            # 100 occurrences of landuse=farmland, all on ways (polygons)
            ("landuse", "farmland", 100, 0, 95, 5),
            ("landuse", "forest", 200, 5, 180, 15),
        ]
    )
    out = element_type_stats(["landuse"], con)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["base_key"] == "landuse"
    assert row["count_all"] == 300
    assert row["count_nodes"] == 5
    assert row["count_ways"] == 275
    assert row["count_relations"] == 20
    assert row["ways_pct"] == pytest.approx(275 / 300)
    assert row["nodes_pct"] == pytest.approx(5 / 300)
    assert row["relations_pct"] == pytest.approx(20 / 300)
    assert bool(row["is_polygon_friendly"]) is True


def test_point_heavy_base_key_is_not_polygon_friendly():
    """A base key whose occurrences are mostly on nodes
    (e.g. ``monitoring:water_quality`` is always a node) is flagged
    as not polygon-friendly."""
    con = _make_tags_db(
        [
            # 200 occurrences of monitoring:water_quality=yes, all on nodes
            ("monitoring:water_quality", "yes", 200, 195, 5, 0),
        ]
    )
    out = element_type_stats(["monitoring"], con)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["base_key"] == "monitoring"
    assert row["count_all"] == 200
    assert row["count_nodes"] == 195
    assert row["ways_pct"] == pytest.approx(5 / 200)
    assert bool(row["is_polygon_friendly"]) is False


def test_colon_prefixed_base_key_aggregates_all_subkeys():
    """A base key of ``addr`` covers ``addr``, ``addr:city``,
    ``addr:street``, ``addr:postcode``, ... — all of them are rolled
    up under the same base key."""
    con = _make_tags_db(
        [
            ("addr", "foo", 10, 1, 8, 1),       # 1 node, 8 ways, 1 relation
            ("addr:city", "bar", 50, 5, 40, 5),
            ("addr:street", "baz", 30, 3, 25, 2),
            ("addr:postcode", "12345", 10, 1, 7, 2),
        ]
    )
    out = element_type_stats(["addr"], con)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["base_key"] == "addr"
    assert row["count_all"] == 100
    assert row["count_nodes"] == 10
    assert row["count_ways"] == 80
    assert row["count_relations"] == 10
    assert row["n_tags"] == 4


def test_unknown_base_key_returns_zero_row():
    """A base key with no matches in the source DB appears in the
    output with zeros rather than being silently dropped (callers
    can build a complete picture of which kept base keys have no
    data in the source)."""
    con = _make_tags_db([("landuse", "farmland", 100, 0, 95, 5)])
    out = element_type_stats(["nonexistent_base_key"], con)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["base_key"] == "nonexistent_base_key"
    assert row["count_all"] == 0
    assert row["n_tags"] == 0
    assert bool(row["is_polygon_friendly"]) is False


def test_multiple_base_keys_returned_in_input_order():
    """The output preserves the order of the input base_keys list."""
    con = _make_tags_db(
        [
            ("landuse", "farmland", 100, 0, 95, 5),
            ("highway", "residential", 200, 10, 190, 0),
            ("monitoring:water_quality", "yes", 50, 50, 0, 0),
        ]
    )
    out = element_type_stats(
        ["monitoring", "landuse", "highway"], con
    )
    assert list(out["base_key"]) == ["monitoring", "landuse", "highway"]


def test_polygon_friendly_threshold_is_exposed():
    """The threshold for ``is_polygon_friendly`` is exposed as a
    module constant so callers can read the cutoff without parsing
    the boolean column."""
    assert isinstance(POLYGON_FRIENDLY_THRESHOLD, float)
    assert 0.0 < POLYGON_FRIENDLY_THRESHOLD < 1.0


def test_threshold_value_default_is_half():
    """The threshold is the default 0.5 (i.e. at least half of the
    occurrences must be on ways+relations to be considered
    polygon-friendly)."""
    con = _make_tags_db(
        [
            # 100 on nodes, 100 on ways -> exactly 50% ways -> friendly
            ("x", "y", 200, 100, 100, 0),
        ]
    )
    out = element_type_stats(["x"], con)
    assert bool(out.iloc[0]["is_polygon_friendly"]) is True
    # Slightly under the threshold: not friendly
    con2 = _make_tags_db(
        [
            ("x", "y", 201, 101, 100, 0),
        ]
    )
    out2 = element_type_stats(["x"], con2)
    assert bool(out2.iloc[0]["is_polygon_friendly"]) is False


def test_accepts_path_in_addition_to_connection():
    """The function accepts either a ``sqlite3.Connection`` or a
    ``pathlib.Path`` to the source DB, so callers can pass a path
    directly without managing a connection."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "tags.sqlite"
        con = sqlite3.connect(db)
        con.execute(
            "CREATE TABLE tags (key VARCHAR, value VARCHAR, "
            "count_all INTEGER, count_nodes INTEGER, count_ways INTEGER, "
            "count_relations INTEGER)"
        )
        con.executemany(
            "INSERT INTO tags VALUES (?, ?, ?, ?, ?, ?)",
            [("landuse", "farmland", 100, 0, 95, 5)],
        )
        con.commit()
        con.close()

        out = element_type_stats(["landuse"], db)
        assert len(out) == 1
        assert out.iloc[0]["base_key"] == "landuse"
        assert out.iloc[0]["count_all"] == 100
