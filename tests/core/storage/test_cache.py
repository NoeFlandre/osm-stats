import sqlite3

import pandas as pd

from src.core.storage.cache import (
    CACHE_SCHEMA,
    build_cache_db_streaming,
    read_cache_df,
)


class FakeStreamingDB:
    """Returns thresholded rows in deterministic pages, in descending count_all."""

    def __init__(self, rows):
        self._rows = sorted(rows, key=lambda r: -r[2])  # sort by count_all desc
        self.calls = []

    def execute_query(self, query, params=None):
        self.calls.append((query, params))
        min_count, *rest = params if params else (None,)
        # Keyset pagination: first page has no upper bound on count_all.
        if len(rest) == 1:
            upper = None
            batch = rest[0]
        else:
            upper = rest[0]
            batch = rest[1]
        out = []
        for k, v, c in self._rows:
            if c < min_count:
                continue
            if upper is not None and c >= upper:
                continue
            out.append((k, v, c))
            if len(out) >= batch:
                break
        return pd.DataFrame(out, columns=["key", "value", "count_all"])


# --- read_cache_df --------------------------------------------------------


def test_read_cache_df_round_trip(tmp_path):
    out = tmp_path / "cache.sqlite"
    src = pd.DataFrame(
        {
            "key": ["landuse", "natural"],
            "value": ["farmland", "water"],
            "count_all": [10_000, 5_000],
        }
    )
    build_cache_db_streaming(FakeStreamingDB([("landuse", "farmland", 10_000),
                                              ("natural", "water", 5_000)]), out,
                             min_count=500, batch_size=10)
    df = read_cache_df(out)
    assert list(df.columns) == ["key", "value", "count_all", "feature"]
    assert list(zip(df["key"], df["value"])) == [
        ("landuse", "farmland"),
        ("natural", "water"),
    ]


def test_read_cache_df_supports_min_count_filter(tmp_path):
    out = tmp_path / "cache.sqlite"
    build_cache_db_streaming(
        FakeStreamingDB(
            [("a", "x", 1_000), ("b", "y", 500), ("c", "z", 100)]
        ),
        out,
        min_count=500,
        batch_size=10,
    )
    df = read_cache_df(out, min_count=500)
    assert len(df) == 2
    assert set(df["key"]) == {"a", "b"}


def test_read_cache_df_supports_limit(tmp_path):
    out = tmp_path / "cache.sqlite"
    build_cache_db_streaming(
        FakeStreamingDB([(f"k{i}", f"v{i}", 1_000 - i) for i in range(10)]),
        out,
        min_count=500,
        batch_size=10,
    )
    df = read_cache_df(out, limit=3)
    assert len(df) == 3
    assert list(df["value"]) == ["v0", "v1", "v2"]


def test_read_cache_df_min_count_and_limit_compose(tmp_path):
    out = tmp_path / "cache.sqlite"
    build_cache_db_streaming(
        FakeStreamingDB(
            [("a", "x", 2_000), ("b", "y", 1_500), ("c", "z", 800), ("d", "w", 600)]
        ),
        out,
        min_count=500,
        batch_size=10,
    )
    df = read_cache_df(out, min_count=1_000, limit=1)
    assert len(df) == 1
    assert df.iloc[0]["value"] == "x"


def test_cache_schema_constant_matches_table():
    assert "CREATE TABLE" in CACHE_SCHEMA
    assert "tag_features" in CACHE_SCHEMA


# --- build_cache_db_streaming --------------------------------------------


def test_streaming_writes_all_rows_above_threshold(tmp_path):
    rows = [
        ("landuse", "farmland", 10_000),
        ("landuse", "forest", 6_000),
        ("landuse", "residential", 3_000),
        ("building", "house", 4_000),
        ("building", "shed", 200),  # below threshold, must be skipped
    ]
    db = FakeStreamingDB(rows)
    out = tmp_path / "cache.sqlite"

    build_cache_db_streaming(db, out, min_count=500, batch_size=2)

    df = read_cache_df(out)
    assert set(zip(df["key"], df["value"])) == {
        ("landuse", "farmland"),
        ("landuse", "forest"),
        ("landuse", "residential"),
        ("building", "house"),
    }


def test_streaming_paginates_and_uses_keyset(tmp_path):
    rows = [(f"k{i}", f"v{i}", 1000 - i) for i in range(10)]
    db = FakeStreamingDB(rows)
    out = tmp_path / "cache.sqlite"

    build_cache_db_streaming(db, out, min_count=500, batch_size=3)

    # 4 pages: 3 + 3 + 3 + 1
    assert len(db.calls) == 4
    # The second call onward must carry a keyset on count_all
    for call in db.calls[1:]:
        params = call[1]
        assert len(params) == 3  # (min_count, after_count, batch_size)


def test_streaming_calls_progress(tmp_path):
    rows = [(f"k{i}", f"v{i}", 1000 - i) for i in range(5)]
    db = FakeStreamingDB(rows)
    progress_calls = []

    build_cache_db_streaming(
        db,
        tmp_path / "cache.sqlite",
        min_count=500,
        batch_size=2,
        progress=lambda n, last: progress_calls.append((n, last)),
    )

    assert progress_calls and all(n > 0 for n, _ in progress_calls)


def test_streaming_replaces_existing_file(tmp_path):
    out = tmp_path / "cache.sqlite"
    out.touch()
    db = FakeStreamingDB([("a", "b", 1000)])
    build_cache_db_streaming(db, out, min_count=500, batch_size=10)
    df = read_cache_df(out)
    assert len(df) == 1
