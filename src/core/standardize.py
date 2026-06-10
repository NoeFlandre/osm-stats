"""Standardize ``(key, value)`` tag pairs into a single feature string.

The downstream TF-IDF + clustering stages work on text features. This module
collapses each pair into one lowercase, whitespace-stripped string of the
form ``"key|value"``. ``|`` is chosen as the delimiter because it almost
never appears in real OSM tag values, which keeps the feature string
unambiguously parseable later.
"""
from __future__ import annotations

from typing import Optional, Union

import pandas as pd

DELIMITER = "|"
MISSING_VALUE_TOKEN = "none"


def build_feature_string(
    key: Union[str, None],
    value: Union[str, None],
) -> str:
    """Return ``"<key>|<value>"`` normalized to lowercase and stripped.

    Empty or missing values become the ``"none"`` token, so the feature
    string always has exactly one delimiter and is safely splittable.
    """
    k = (str(key) if key is not None else "").strip().lower() or MISSING_VALUE_TOKEN
    v = (str(value) if value is not None else "").strip().lower() or MISSING_VALUE_TOKEN
    return f"{k}{DELIMITER}{v}"


def build_feature_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized version of :func:`build_feature_string` over a DataFrame.

    Expects columns ``"key"`` and ``"value"``; both are coerced to string,
    lowercased, and stripped before joining.
    """
    keys = df["key"].astype("string").fillna("").str.strip().str.lower()
    values = df["value"].astype("string").fillna("").str.strip().str.lower()
    keys = keys.where(keys != "", MISSING_VALUE_TOKEN)
    values = values.where(values != "", MISSING_VALUE_TOKEN)
    return keys + DELIMITER + values


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
