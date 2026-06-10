"""Write query results to CSV files.

:class:`DataExporter` only depends on a duck-typed ``DB`` protocol (an object
exposing ``execute_query(query, params=None) -> pd.DataFrame``). It does not
import :class:`~src.database.OSMDatabase`, which keeps the exporter unit-testable
with a fake.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, Tuple, Union

import pandas as pd


class DB(Protocol):
    def execute_query(
        self, query: str, params: Optional[Tuple] = None
    ) -> pd.DataFrame: ...


class DataExporter:
    def __init__(self, db: DB, output_dir: Union[str, Path]):
        self.db = db
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_query(
        self,
        query: str,
        filename: str,
        params: Optional[Tuple] = None,
    ) -> pd.DataFrame:
        """Run *query* (optionally with *params*) and persist the result to CSV.

        Returns the DataFrame that was written.
        """
        df = self.db.execute_query(query, params)
        df.to_csv(self.output_dir / filename, index=False)
        return df
