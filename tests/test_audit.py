import pytest
from src.database import OSMDatabase
from src.queries import get_tag_values_query
import sqlite3

@pytest.fixture
def memory_db_path(tmp_path):
    db_file = tmp_path / "test.sqlite"
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE tags (key TEXT, value TEXT, count_all INTEGER)")
    cursor.execute("INSERT INTO tags VALUES ('building', 'house', 100)")
    cursor.execute("INSERT INTO tags VALUES ('building', 'greenhouse', 10)")
    cursor.execute("INSERT INTO tags VALUES ('natural', 'wood', 50)")
    conn.commit()
    conn.close()
    return db_file

def test_get_tag_values_query_logic(memory_db_path):
    db = OSMDatabase(str(memory_db_path))
    query = get_tag_values_query('building', limit=2)
    df = db.execute_query(query)
    
    assert len(df) == 2
    assert df.iloc[0]['value'] == 'house'
    assert df.iloc[1]['value'] == 'greenhouse'
    assert df.iloc[1]['count_all'] == 10
