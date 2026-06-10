"""Compute and shape the global taginfo summary metrics.

Encapsulates the two aggregate queries (distinct keys, distinct tags, total
occurrences) behind a small typed object. The public surface is
:class:`SummaryBuilder` (a thin DB wrapper) and :class:`GlobalSummary` (the
result, with a tidy DataFrame serializer for the blog/CSVs).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from src.core.queries import QueryBuilder
from src.io.exporter import DB


@dataclass(frozen=True)
class GlobalSummary:
    total_distinct_keys: int
    total_key_occurrences: int
    total_distinct_tags: int
    total_tag_occurrences: int

    def to_dict(self) -> dict:
        return {
            "total_distinct_keys": self.total_distinct_keys,
            "total_key_occurrences": self.total_key_occurrences,
            "total_distinct_tags": self.total_distinct_tags,
            "total_tag_occurrences": self.total_tag_occurrences,
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "metric": [
                    "total_distinct_keys",
                    "total_key_occurrences",
                    "total_distinct_tags",
                    "total_tag_occurrences",
                ],
                "value": [
                    self.total_distinct_keys,
                    self.total_key_occurrences,
                    self.total_distinct_tags,
                    self.total_tag_occurrences,
                ],
            }
        )


class SummaryBuilder:
    def __init__(self, db: DB):
        self.db = db

    def build(self) -> GlobalSummary:
        keys = self.db.execute_query(*QueryBuilder.GLOBAL_KEY_AGGREGATES)
        tags = self.db.execute_query(*QueryBuilder.GLOBAL_TAG_AGGREGATES)
        if keys.empty or tags.empty:
            raise RuntimeError("SummaryBuilder: aggregate query returned no rows")

        def _scalar(frame: pd.DataFrame, col: str) -> int:
            value = frame[col].iloc[0]
            # SUM() over an empty table returns NULL -> NaN in pandas.
            # Coerce NaN to 0 so an empty source table is not a crash.
            return 0 if pd.isna(value) else int(value)

        return GlobalSummary(
            total_distinct_keys=_scalar(keys, "total_distinct_keys"),
            total_key_occurrences=_scalar(keys, "total_key_occurrences"),
            total_distinct_tags=_scalar(tags, "total_distinct_tags"),
            total_tag_occurrences=_scalar(tags, "total_tag_occurrences"),
        )
