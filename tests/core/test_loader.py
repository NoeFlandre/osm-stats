import sqlite3

import pandas as pd
import pytest

from src.core.audit import OSMTagAuditor
from src.core.database import OSMDatabase
from src.core.loader import load_and_threshold


class FakeDB:
    def __init__(self, frame):
        self.frame = frame
        self.calls = []

    def execute_query(self, query, params=None):
        self.calls.append((query, params))
        return self.frame


# --- load_and_threshold ---------------------------------------------------


def test_load_and_threshold_delegates_to_db_with_params():
    frame = pd.DataFrame(
        {
            "key": ["landuse", "highway"],
            "value": ["farmland", "track"],
            "count_all": [10_000, 2_000],
        }
    )
    db = FakeDB(frame)

    df = load_and_threshold(db, min_count=500)

    # The function is a thin pass-through to the DB; the actual filtering
    # happens in SQL and is covered by the integration test below.
    assert db.calls[0][1] == (500,)
    pd.testing.assert_frame_equal(df, frame)


def test_load_and_threshold_default_min_count_is_500():
    db = FakeDB(pd.DataFrame({"key": [], "value": [], "count_all": []}))
    load_and_threshold(db)
    assert db.calls[0][1] == (500,)


def test_load_and_threshold_uses_parameterized_query():
    db = FakeDB(pd.DataFrame({"key": [], "value": [], "count_all": []}))
    load_and_threshold(db, min_count=123)

    sql, params = db.calls[0]
    assert "?" in sql
    assert "count_all >= ?" in sql
    assert params == (123,)


def test_load_and_threshold_with_limit_passes_param():
    db = FakeDB(pd.DataFrame({"key": [], "value": [], "count_all": []}))
    load_and_threshold(db, min_count=500, limit=1000)
    sql, params = db.calls[0]
    assert "LIMIT ?" in sql
    assert params == (500, 1000)


def test_load_and_threshold_without_limit_omits_limit_clause():
    db = FakeDB(pd.DataFrame({"key": [], "value": [], "count_all": []}))
    load_and_threshold(db, min_count=500)
    sql, params = db.calls[0]
    assert "LIMIT" not in sql
    assert params == (500,)


def test_load_and_threshold_returns_dataframe_unchanged():
    frame = pd.DataFrame(
        {
            "key": ["a"],
            "value": ["b"],
            "count_all": [10_000],
        }
    )
    db = FakeDB(frame)
    df = load_and_threshold(db, min_count=500)
    pd.testing.assert_frame_equal(df, frame)


def test_load_and_threshold_integration_with_real_db(temp_sqlite):
    db_file = temp_sqlite(
        "CREATE TABLE tags (key TEXT, value TEXT, count_all INTEGER)",
        rows=[
            ("landuse", "farmland", 10_000),
            ("landuse", "forest", 600),
            ("building", "house", 400),
            ("highway", "track", 2_000),
            ("highway", "footway", 50),
        ],
    )
    db = OSMDatabase(str(db_file))
    df = load_and_threshold(db, min_count=500)

    assert set(zip(df["key"], df["value"])) == {
        ("landuse", "farmland"),
        ("landuse", "forest"),
        ("highway", "track"),
    }
