import sqlite3
import pytest

from src.core.queries import QueryBuilder


def test_paramless_constants_are_sql_params_tuples():
    # All QueryBuilder surfaces - even the parameterless ones - must be
    # ``(sql, params)`` tuples so a single execute_query(sql, params)
    # call site works for every query.
    for name in ("TOP_KEYS", "TOP_TAGS", "METADATA",
                 "GLOBAL_KEY_AGGREGATES", "GLOBAL_TAG_AGGREGATES"):
        sql, params = getattr(QueryBuilder, name)
        assert isinstance(sql, str)
        assert params == ()


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
