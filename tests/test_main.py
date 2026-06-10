"""Tests for the orchestration in ``src.core.pipeline`` (the ``run`` and
``build_feature_cache`` functions). CLI parsing tests live in
``tests/test_cli.py``."""
import sqlite3

from src.config import DEFAULT_OUTPUT_DIR, resolve_db_path
from src.core.storage.cache import read_cache_df
from src.core.db.database import OSMDatabase
from src.core.pipeline.pipeline import build_feature_cache, build_summary, run


def test_resolve_db_path_uses_env_in_main(monkeypatch, tmp_path):
    custom = tmp_path / "taginfo.sqlite"
    custom.touch()
    monkeypatch.setenv("OSM_DB_PATH", str(custom))
    assert resolve_db_path() == custom
    # And the default still exists for callers that need a sane fallback.
    assert DEFAULT_OUTPUT_DIR.name == "output"


def test_build_summary_uses_querybuilder(temp_sqlite):
    db_file = temp_sqlite(
        "CREATE TABLE keys (key TEXT, count_all INTEGER)",
        rows=[("a", 1), ("b", 2)],
    )
    # Use a single combined schema instead: build the DB manually.
    combined = db_file.parent / "combined.sqlite"
    conn = sqlite3.connect(combined)
    conn.executescript(
        "CREATE TABLE keys (key TEXT, count_all INTEGER);"
        "CREATE TABLE tags (key TEXT, value TEXT, count_all INTEGER);"
    )
    conn.executemany("INSERT INTO keys VALUES (?, ?)", [("a", 1), ("b", 2)])
    conn.executemany("INSERT INTO tags VALUES (?, ?, ?)", [("a", "x", 3)])
    conn.commit()
    conn.close()

    summary = build_summary(OSMDatabase(str(combined)))

    assert dict(zip(summary["metric"], summary["value"])) == {
        "total_distinct_keys": 2,
        "total_key_occurrences": 3,
        "total_distinct_tags": 1,
        "total_tag_occurrences": 3,
    }


def test_run_writes_csvs_to_output_dir(tmp_path):
    db_file = tmp_path / "taginfo.sqlite"
    conn = sqlite3.connect(db_file)
    conn.executescript(
        "CREATE TABLE keys (key TEXT, count_all INTEGER);"
        "CREATE TABLE tags (key TEXT, value TEXT, count_all INTEGER);"
    )
    conn.executemany(
        "INSERT INTO keys VALUES (?, ?)",
        [("building", 5), ("highway", 2)],
    )
    conn.executemany(
        "INSERT INTO tags VALUES (?, ?, ?)",
        [("building", "house", 4), ("building", "garage", 1)],
    )
    conn.commit()
    conn.close()

    out = tmp_path / "out"
    audit = run(OSMDatabase(str(db_file)), out)

    assert (out / "global_summary.csv").exists()
    assert (out / "audit_building.csv").exists()
    assert list(audit["value"]) == ["house", "garage"]


def test_run_summary_only_skips_audit(tmp_path):
    db_file = tmp_path / "taginfo.sqlite"
    conn = sqlite3.connect(db_file)
    conn.executescript(
        "CREATE TABLE keys (key TEXT, count_all INTEGER);"
        "CREATE TABLE tags (key TEXT, value TEXT, count_all INTEGER);"
    )
    conn.commit()
    conn.close()

    out = tmp_path / "out"
    run(OSMDatabase(str(db_file)), out, summary_only=True)

    assert (out / "global_summary.csv").exists()
    assert not (out / "audit_building.csv").exists()


def test_build_feature_cache_writes_and_reads_back(temp_sqlite, tmp_path):
    db_file = temp_sqlite(
        "CREATE TABLE tags (key TEXT, value TEXT, count_all INTEGER)",
        rows=[
            ("landuse", "farmland", 10_000),
            ("landuse", "forest", 600),
            ("building", "house", 400),
            ("highway", "track", 2_000),
        ],
    )
    cache_path = tmp_path / "cache.sqlite"
    out_path = build_feature_cache(OSMDatabase(str(db_file)), cache_path, min_count=500)

    assert out_path == cache_path
    assert cache_path.exists()

    df = read_cache_df(cache_path, min_count=500)
    assert set(zip(df["key"], df["value"])) == {
        ("landuse", "farmland"),
        ("landuse", "forest"),
        ("highway", "track"),
    }
    # The feature column is standardized.
    assert set(df["feature"]) == {
        "landuse|farmland",
        "landuse|forest",
        "highway|track",
    }
