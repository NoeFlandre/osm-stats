import sqlite3
import pytest

from src.core.db.queries import QueryBuilder


def test_paramless_constants_are_sql_params_tuples():
    # All QueryBuilder surfaces - even the parameterless ones - must be
    # ``(sql, params)`` tuples so a single execute_query(sql, params)
    # call site works for every query.
    for name in ("TOP_KEYS", "TOP_TAGS", "METADATA",
                 "GLOBAL_KEY_AGGREGATES", "GLOBAL_TAG_AGGREGATES",
                 "TAG_FEATURES_INSERT", "TAG_FEATURES_INDEX"):
        sql, params = getattr(QueryBuilder, name)
        assert isinstance(sql, str)
        assert params == ()


def test_tag_features_select_query_filters_by_min_count():
    sql, params = QueryBuilder.tag_features_select(min_count=500)
    assert "tag_features" in sql
    assert "count_all >= ?" in sql
    assert "ORDER BY count_all DESC" in sql
    assert params == (500,)


def test_tag_features_index_uses_bare_create():
    # The cache builder always recreates the file from scratch, so the
    # index SQL must be the bare form (no IF NOT EXISTS needed).
    sql, _ = QueryBuilder.TAG_FEATURES_INDEX
    assert "CREATE INDEX" in sql
    assert "IF NOT EXISTS" not in sql
    assert "ON tag_features(count_all DESC)" in sql


def test_tag_features_select_with_limit():
    sql, params = QueryBuilder.tag_features_select(min_count=500, limit=10)
    assert "LIMIT ?" in sql
    assert params == (500, 10)


def test_tag_features_select_all_no_filter():
    sql, params = QueryBuilder.tag_features_select_all()
    assert "tag_features" in sql
    assert "count_all >=" not in sql
    assert params == ()


def test_tag_features_select_all_with_limit():
    sql, params = QueryBuilder.tag_features_select_all(limit=5)
    assert "LIMIT ?" in sql
    assert params == (5,)


def test_first_tags_by_min_count_no_upper_bound():
    sql, params = QueryBuilder.first_tags_by_min_count(500, 1000)
    assert "count_all >= ?" in sql
    assert "<" not in sql  # no upper bound on the first page
    assert "LIMIT ?" in sql
    assert params == (500, 1000)


def test_next_tags_by_min_count_strict_upper_bound():
    sql, params = QueryBuilder.next_tags_by_min_count(500, 1234, 1000)
    assert "count_all >= ?" in sql
    assert "count_all < ?" in sql
    assert "LIMIT ?" in sql
    assert params == (500, 1234, 1000)


# --- QueryBuilder ---------------------------------------------------------


def test_select_tag_values_uses_placeholder_not_interpolation():
    sql, params = QueryBuilder.select_tag_values("building", limit=10)
    assert "?" in sql
    assert "building" not in sql  # value must be passed as a parameter
    assert params == ("building", 10)


def test_select_tag_values_default_limit():
    sql, params = QueryBuilder.select_tag_values("building")
    assert params == ("building", 50)


def test_select_tag_values_runs_against_sqlite(temp_sqlite):
    db_file = temp_sqlite(
        "CREATE TABLE tags (key TEXT, value TEXT, count_all INTEGER)",
        rows=[
            ("building", "house", 100),
            ("building", "greenhouse", 10),
            ("natural", "wood", 50),
        ],
    )
    sql, params = QueryBuilder.select_tag_values("building", limit=2)
    with sqlite3.connect(db_file) as conn:
        rows = conn.execute(sql, params).fetchall()
    assert rows == [("house", 100), ("greenhouse", 10)]


def test_select_tag_values_does_not_inject(temp_sqlite):
    db_file = temp_sqlite(
        "CREATE TABLE tags (key TEXT, value TEXT, count_all INTEGER)",
        rows=[("building", "house", 1)],
    )
    # If the key were interpolated, the OR 1=1 would leak all rows.
    evil = "building' OR '1'='1"
    sql, params = QueryBuilder.select_tag_values(evil, limit=10)
    with sqlite3.connect(db_file) as conn:
        rows = conn.execute(sql, params).fetchall()
    assert rows == []  # safely returns zero rows for the non-existent key


def test_select_top_keys_orders_by_count_desc(temp_sqlite):
    db_file = temp_sqlite(
        "CREATE TABLE keys (key TEXT, count_all INTEGER)",
        rows=[("a", 1), ("b", 100), ("c", 50)],
    )
    sql, params = QueryBuilder.TOP_KEYS
    with sqlite3.connect(db_file) as conn:
        rows = conn.execute(sql, params).fetchall()
    assert rows[0][0] == "b"
    assert rows[-1][0] == "a"


def test_select_top_tags_orders_by_count_desc(temp_sqlite):
    db_file = temp_sqlite(
        "CREATE TABLE tags (key TEXT, value TEXT, count_all INTEGER)",
        rows=[("a", "x", 5), ("a", "y", 50), ("b", "z", 10)],
    )
    sql, params = QueryBuilder.TOP_TAGS
    with sqlite3.connect(db_file) as conn:
        rows = conn.execute(sql, params).fetchall()
    assert rows[0] == ("a", "y", 50)


def test_global_key_aggregates_returns_totals(temp_sqlite):
    db_file = temp_sqlite(
        "CREATE TABLE keys (key TEXT, count_all INTEGER)",
        rows=[("a", 10), ("b", 20), ("c", 5)],
    )
    sql, params = QueryBuilder.GLOBAL_KEY_AGGREGATES
    with sqlite3.connect(db_file) as conn:
        row = conn.execute(sql, params).fetchone()
    assert row[0] == 3  # distinct keys
    assert row[1] == 35  # total occurrences


def test_global_tag_aggregates_returns_totals(temp_sqlite):
    db_file = temp_sqlite(
        "CREATE TABLE tags (key TEXT, value TEXT, count_all INTEGER)",
        rows=[("a", "x", 10), ("a", "y", 20), ("b", "z", 5)],
    )
    sql, params = QueryBuilder.GLOBAL_TAG_AGGREGATES
    with sqlite3.connect(db_file) as conn:
        row = conn.execute(sql, params).fetchone()
    assert row[0] == 3
    assert row[1] == 35


def test_metadata_query_returns_source_rows(temp_sqlite):
    db_file = temp_sqlite(
        "CREATE TABLE source (id INTEGER, name TEXT)",
        rows=[(1, "taginfo"), (2, "extra")],
    )
    sql, params = QueryBuilder.METADATA
    with sqlite3.connect(db_file) as conn:
        rows = conn.execute(sql, params).fetchall()
    assert rows == [(1, "taginfo"), (2, "extra")]
