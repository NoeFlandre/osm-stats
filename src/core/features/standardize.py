"""Standardize ``(key, value)`` tag pairs into a single feature string.

The downstream TF-IDF + clustering stages work on text features. This module
collapses each pair into one lowercase, whitespace-stripped string of the
form ``"key|value"``. ``|`` is chosen as the delimiter because it almost
never appears in real OSM tag values, which keeps the feature string
unambiguously parseable later.

The single public API is :func:`standardize_dataframe`. It normalizes the
``key`` and ``value`` columns in place (in a copy) and adds a new ``feature``
column. The vectorized normalizer :func:`_normalize_column` is the one and
only place that knows how to turn any input into a clean token.
"""
from __future__ import annotations

import pandas as pd

DELIMITER = "|"
MISSING_VALUE_TOKEN = "none"


def _normalize_column(series: pd.Series) -> pd.Series:
    """Lowercase, strip, and replace empty strings with the missing-value token."""
    out = series.astype("string").fillna("").str.strip().str.lower()
    return out.where(out != "", MISSING_VALUE_TOKEN)


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with ``key`` and ``value`` normalized and a new ``feature`` column.

    Normalization is lowercase + strip + missing-value token. The input
    DataFrame is not mutated.
    """
    out = df.copy()
    out["key"] = _normalize_column(out["key"])
    out["value"] = _normalize_column(out["value"])
    out["feature"] = out["key"] + DELIMITER + out["value"]
    return out
