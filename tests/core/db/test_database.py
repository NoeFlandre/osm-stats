import pytest
from src.core.db.database import OSMDatabase


def test_database_connection_and_query(temp_sqlite):
    db_file = temp_sqlite(
        "CREATE TABLE keys (key TEXT, count_all INTEGER)",
        rows=[("building", 100), ("highway", 50)],
    )
    db = OSMDatabase(str(db_file))
    df = db.execute_query("SELECT * FROM keys ORDER BY count_all DESC")

    assert len(df) == 2
    assert df.iloc[0]["key"] == "building"
    assert df.iloc[1]["key"] == "highway"


def test_database_file_not_found():
    with pytest.raises(FileNotFoundError):
        OSMDatabase("non_existent_db.sqlite")
