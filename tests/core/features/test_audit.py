import pandas as pd
import pytest

from src.core.db.database import OSMDatabase
from src.core.features.audit import OSMTagAuditor


class FakeDB:
    def __init__(self, frame):
        self.frame = frame
        self.calls = []

    def execute_query(self, query, params=None):
        self.calls.append((query, params))
        return self.frame


def test_auditor_top_values_uses_parameterized_query():
    db = FakeDB(pd.DataFrame({"value": ["house"], "count_all": [42]}))
    auditor = OSMTagAuditor(db)

    df = auditor.top_values("building", limit=10)

    assert list(df["value"]) == ["house"]
    assert len(db.calls) == 1
    sql, params = db.calls[0]
    assert "?" in sql
    assert "building" not in sql
    assert params == ("building", 10)


def test_auditor_default_limit_is_50():
    db = FakeDB(pd.DataFrame({"value": [], "count_all": []}))
    auditor = OSMTagAuditor(db)

    auditor.top_values("building")

    assert db.calls[0][1] == ("building", 50)


def test_auditor_top_values_returns_dataframe_unchanged():
    frame = pd.DataFrame({"value": ["a", "b"], "count_all": [2, 1]})
    db = FakeDB(frame)
    auditor = OSMTagAuditor(db)

    df = auditor.top_values("building", limit=2)

    pd.testing.assert_frame_equal(df, frame)


def test_auditor_integration_with_real_db(temp_sqlite):
    db_file = temp_sqlite(
        "CREATE TABLE tags (key TEXT, value TEXT, count_all INTEGER)",
        rows=[
            ("building", "house", 100),
            ("building", "greenhouse", 10),
            ("natural", "wood", 50),
        ],
    )
    db = OSMDatabase(str(db_file))
    auditor = OSMTagAuditor(db)

    df = auditor.top_values("building", limit=2)

    assert list(df["value"]) == ["house", "greenhouse"]
    assert list(df["count_all"]) == [100, 10]
