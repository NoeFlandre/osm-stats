import pandas as pd
import pytest

from src.core.database import OSMDatabase
from src.io.exporter import DataExporter


class FakeDB:
    """In-memory stand-in for OSMDatabase used to test DataExporter in isolation."""

    def __init__(self, frames):
        self._frames = iter(frames)
        self.calls = []

    def execute_query(self, query, params=None):
        self.calls.append((query, params))
        return next(self._frames)


def test_data_exporter_uses_db_protocol_and_writes_csv(tmp_path):
    db = FakeDB([pd.DataFrame({"id": [1], "name": ["test"]})])
    output_dir = tmp_path / "output"

    exporter = DataExporter(db, output_dir)
    df = exporter.export_query("SELECT * FROM mock_table", "mock_output.csv")

    expected_file = output_dir / "mock_output.csv"
    assert expected_file.exists()
    assert db.calls == [("SELECT * FROM mock_table", None)]
    assert "id,name" in expected_file.read_text()
    assert "1,test" in expected_file.read_text()
    assert list(df.columns) == ["id", "name"]


def test_data_exporter_forwards_params_to_db(tmp_path):
    db = FakeDB([pd.DataFrame({"v": ["x"]})])
    exporter = DataExporter(db, tmp_path)

    exporter.export_query("SELECT v FROM t WHERE k = ?", "out.csv", params=("k1",))

    assert db.calls[-1] == ("SELECT v FROM t WHERE k = ?", ("k1",))


def test_data_exporter_creates_output_dir(tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    db = FakeDB([pd.DataFrame({"x": [1]})])

    DataExporter(db, nested).export_query("SELECT 1", "f.csv")

    assert (nested / "f.csv").exists()


def test_data_exporter_integration_with_real_db(temp_sqlite, tmp_path):
    db_file = temp_sqlite(
        "CREATE TABLE mock_table (id INTEGER, name TEXT)",
        rows=[(1, "test")],
    )
    db = OSMDatabase(str(db_file))
    output_dir = tmp_path / "output"

    DataExporter(db, output_dir).export_query("SELECT * FROM mock_table", "mock_output.csv")

    expected_file = output_dir / "mock_output.csv"
    assert expected_file.exists()
    content = expected_file.read_text()
    assert "id,name" in content
    assert "1,test" in content
