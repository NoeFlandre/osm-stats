import pandas as pd
import pytest

from src.core.features.standardize import (
    DELIMITER,
    MISSING_VALUE_TOKEN,
    _normalize_column,
    standardize_dataframe,
)


# --- constants ------------------------------------------------------------


def test_delimiter_is_pipe():
    assert DELIMITER == "|"


def test_missing_value_token_is_none():
    assert MISSING_VALUE_TOKEN == "none"


# --- _normalize_column ----------------------------------------------------


def test_normalize_lowercases_and_strips():
    s = pd.Series(["  Landuse ", "Natural", "\tWater\n"])
    out = _normalize_column(s)
    assert list(out) == ["landuse", "natural", "water"]


def test_normalize_handles_missing_and_empty():
    s = pd.Series([None, "", "  ", "ok"])
    out = _normalize_column(s)
    assert list(out) == ["none", "none", "none", "ok"]


# --- standardize_dataframe ------------------------------------------------


def test_standardize_adds_feature_column():
    df = pd.DataFrame(
        {
            "key": ["landuse", "natural"],
            "value": ["farmland", "water"],
            "count_all": [10_000, 5_000],
        }
    )
    out = standardize_dataframe(df)
    assert "feature" in out.columns
    assert list(out["feature"]) == ["landuse|farmland", "natural|water"]


def test_standardize_normalizes_key_and_value_columns():
    df = pd.DataFrame(
        {
            "key": ["  Landuse ", "Natural"],
            "value": [" FarmLand ", " Water "],
            "count_all": [1, 2],
        }
    )
    out = standardize_dataframe(df)
    assert list(out["key"]) == ["landuse", "natural"]
    assert list(out["value"]) == ["farmland", "water"]


def test_standardize_replaces_empty_with_none_token():
    df = pd.DataFrame({"key": ["natural"], "value": [""], "count_all": [1]})
    out = standardize_dataframe(df)
    assert out.iloc[0]["feature"] == "natural|none"


def test_standardize_preserves_other_columns():
    df = pd.DataFrame(
        {
            "key": ["landuse"],
            "value": ["farmland"],
            "count_all": [10_000],
        }
    )
    out = standardize_dataframe(df)
    assert out.iloc[0]["count_all"] == 10_000


def test_standardize_does_not_mutate_input():
    df = pd.DataFrame({"key": ["Landuse"], "value": ["Farmland"], "count_all": [1]})
    standardize_dataframe(df)
    assert df.iloc[0]["key"] == "Landuse"
    assert df.iloc[0]["value"] == "Farmland"


def test_standardize_handles_missing_key_or_value():
    df = pd.DataFrame(
        {"key": [None, "natural"], "value": ["farmland", None], "count_all": [1, 2]}
    )
    out = standardize_dataframe(df)
    assert list(out["feature"]) == ["none|farmland", "natural|none"]


def test_standardize_empty_dataframe():
    df = pd.DataFrame({"key": [], "value": [], "count_all": []})
    out = standardize_dataframe(df)
    assert list(out["feature"]) == []
    assert "feature" in out.columns
