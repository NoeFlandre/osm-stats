import pytest
import sqlite3
import pandas as pd
from pathlib import Path
from src.database import OSMDatabase

@pytest.fixture
def memory_db_path(tmp_path):
    # Create a temporary sqlite database
    db_file = tmp_path / "test.sqlite"
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE keys (key TEXT, count_all INTEGER)")
    cursor.execute("INSERT INTO keys VALUES ('building', 100), ('highway', 50)")
    conn.commit()
    conn.close()
    return db_file

def test_database_connection_and_query(memory_db_path):
    db = OSMDatabase(str(memory_db_path))
    df = db.execute_query("SELECT * FROM keys ORDER BY count_all DESC")
    
    assert len(df) == 2
    assert df.iloc[0]['key'] == 'building'
    assert df.iloc[1]['key'] == 'highway'

def test_database_file_not_found():
    with pytest.raises(FileNotFoundError):
        OSMDatabase("non_existent_db.sqlite")
