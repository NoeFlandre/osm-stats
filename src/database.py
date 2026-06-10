import sqlite3
import pandas as pd
from pathlib import Path

class OSMDatabase:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}")

    def execute_query(self, query: str) -> pd.DataFrame:
        """Executes a SQL query and returns a Pandas DataFrame."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)
