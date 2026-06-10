import sqlite3

from src.core.storage.cache import add_base_key_column, read_cache_df


def test_add_base_key_column_writes_correct_values(tmp_path):
    cache = tmp_path / "cache.sqlite"
    with sqlite3.connect(cache) as conn:
        conn.executescript(
            """
            CREATE TABLE tag_features (
                key TEXT, value TEXT, count_all INTEGER, feature TEXT,
                PRIMARY KEY (key, value)
            );
            """
        )
        conn.executemany(
            "INSERT INTO tag_features VALUES (?, ?, ?, ?)",
            [
                ("landuse", "farmland", 10, "landuse|farmland"),
                ("addr:street", "hauptstraße", 5, "addr:street|hauptstraße"),
                ("natural", "water", 7, "natural|water"),
            ],
        )
        conn.commit()

    add_base_key_column(cache)

    with sqlite3.connect(cache) as conn:
        rows = conn.execute(
            "SELECT key, base_key FROM tag_features ORDER BY count_all DESC"
        ).fetchall()
    assert rows == [
        ("landuse", "landuse"),
        ("natural", "natural"),
        ("addr:street", "addr"),
    ]


def test_add_base_key_column_is_idempotent(tmp_path):
    cache = tmp_path / "cache.sqlite"
    with sqlite3.connect(cache) as conn:
        conn.executescript(
            """
            CREATE TABLE tag_features (
                key TEXT, value TEXT, count_all INTEGER, feature TEXT,
                PRIMARY KEY (key, value)
            );
            """
        )
        conn.executemany(
            "INSERT INTO tag_features VALUES (?, ?, ?, ?)",
            [("landuse", "farmland", 1, "landuse|farmland")],
        )
        conn.commit()

    add_base_key_column(cache)
    add_base_key_column(cache)  # must not crash

    df = read_cache_df(cache)
    assert df.iloc[0]["base_key"] == "landuse"


def test_add_base_key_column_appears_in_read_cache_df(tmp_path):
    cache = tmp_path / "cache.sqlite"
    with sqlite3.connect(cache) as conn:
        conn.executescript(
            """
            CREATE TABLE tag_features (
                key TEXT, value TEXT, count_all INTEGER, feature TEXT,
                PRIMARY KEY (key, value)
            );
            """
        )
        conn.executemany(
            "INSERT INTO tag_features VALUES (?, ?, ?, ?)",
            [("landuse", "farmland", 1, "landuse|farmland")],
        )
        conn.commit()

    add_base_key_column(cache)
    df = read_cache_df(cache)
    assert "base_key" in df.columns
    assert df.iloc[0]["base_key"] == "landuse"
