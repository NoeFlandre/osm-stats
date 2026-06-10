import pytest
from pathlib import Path
from src.exporter import DataExporter
from src.database import OSMDatabase
import sqlite3

@pytest.fixture
def memory_db_path(tmp_path):
    db_file = tmp_path / "test.sqlite"
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE mock_table (id INTEGER, name TEXT)")
    cursor.execute("INSERT INTO mock_table VALUES (1, 'test')")
    conn.commit()
    conn.close()
    return db_file

def test_data_exporter(memory_db_path, tmp_path):
    db = OSMDatabase(str(memory_db_path))
    output_dir = tmp_path / "output"
    
    exporter = DataExporter(db, output_dir)
    exporter.export_query("SELECT * FROM mock_table", "mock_output.csv")
    
    expected_file = output_dir / "mock_output.csv"
    assert expected_file.exists(), "Exporter should create the output CSV file"
    
    content = expected_file.read_text()
    assert "id,name" in content, "CSV should contain headers"
    assert "1,test" in content, "CSV should contain the data row"
