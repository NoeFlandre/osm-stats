"""SQLite-backed access to an OSM taginfo database."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


class OSMDatabase:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}")

    def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
    ) -> pd.DataFrame:
        """Execute *query* (optionally with bound *params*) and return a DataFrame."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
