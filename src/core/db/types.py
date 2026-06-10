"""Shared types used across the ``core`` and ``io`` packages.

The :class:`DB` protocol is the only thing the rest of the codebase needs to
talk to a database: a duck-typed object with ``execute_query(query, params)``.
Lives in ``core`` so the higher-level domain code can depend on it without
importing the I/O adapters in ``src.io``.
"""
from __future__ import annotations

from typing import Optional, Protocol, Tuple

import pandas as pd


class DB(Protocol):
    def execute_query(
        self, query: str, params: Optional[Tuple] = None
    ) -> pd.DataFrame: ...
