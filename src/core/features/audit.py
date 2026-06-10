"""High-level audit operations on an OSM taginfo database.

This module is the place to add domain logic (e.g. 'top values for a key').
It depends on the small :class:`~src.core.types.DB` protocol, not on the
concrete :class:`~src.database.OSMDatabase`, which keeps it unit-testable
with a fake.
"""
from __future__ import annotations

import pandas as pd

from src.core.db.queries import QueryBuilder
from src.core.db.types import DB


class OSMTagAuditor:
    def __init__(self, db: DB):
        self.db = db

    def top_values(self, key: str, limit: int = 50) -> pd.DataFrame:
        """Return the top *limit* values for *key*, ordered by ``count_all`` DESC."""
        sql, params = QueryBuilder.select_tag_values(key, limit=limit)
        return self.db.execute_query(sql, params)
