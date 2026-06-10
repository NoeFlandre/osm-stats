"""High-level audit operations on an OSM taginfo database.

This module is the place to add domain logic (e.g. 'top values for a key').
It depends on the small :class:`~src.exporter.DB` protocol, not on the concrete
:class:`~src.database.OSMDatabase`, which keeps it unit-testable with a fake.
"""
from __future__ import annotations

from typing import Optional, Protocol, Tuple

import pandas as pd

from src.io.exporter import DB
from src.core.queries import QueryBuilder


class OSMTagAuditor:
    def __init__(self, db: DB):
        self.db = db

    def top_values(self, key: str, limit: int = 50) -> pd.DataFrame:
        """Return the top *limit* values for *key*, ordered by ``count_all`` DESC."""
        sql, params = QueryBuilder.select_tag_values(key, limit=limit)
        return self.db.execute_query(sql, params)
