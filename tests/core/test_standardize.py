import pandas as pd
import pytest

from src.core.standardize import (
    DELIMITER,
    build_feature_string,
    build_feature_series,
    standardize_dataframe,
)


# --- build_feature_string -------------------------------------------------


def test_build_feature_string_basic():
    assert build_feature_string("landuse", "farmland") == "landuse|farmland"


def test_build_feature_string_lowercases_key_and_value():
    assert build_feature_string("Landuse", "FarmLand") == "landuse|farmland"


def test_build_feature_string_strips_whitespace():
    assert build_feature_string("  landuse ", " farmland  ") == "landuse|farmland"


def test_build_feature_string_handles_none_value():
    # 'key' with no explicit value (rare in OSM) becomes key|none.
    assert build_feature_string("natural", None) == "natural|none"


def test_build_feature_string_drops_empty_value():
    assert build_feature_string("natural", "") == "natural|none"


def test_build_feature_string_preserves_internal_underscores():
    assert build_feature_string("crop", "sugar_cane") == "crop|sugar_cane"


def test_delimiter_is_pipe_and_appears_in_output():
    assert DELIMITER == "|"
    assert "|" in build_feature_string("landuse", "farmland")


# --- build_feature_series -------------------------------------------------


def test_build_feature_series_on_dataframe():
    df = pd.DataFrame({"key": ["Landuse", "natural"], "value": ["Farmland", "  Water  "]})
    series = build_feature_series(df)
    assert list(series) == ["landuse|farmland", "natural|water"]


def test_build_feature_series_empty():
    df = pd.DataFrame({"key": [], "value": []})
    assert list(build_feature_series(df)) == []


# --- standardize_dataframe -----------------------------------------------


def test_standardize_dataframe_adds_feature_column():
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
    # Original columns preserved
    assert list(out["count_all"]) == [10_000, 5_000]


def test_standardize_dataframe_does_not_mutate_input():
    df = pd.DataFrame({"key": ["Landuse"], "value": ["Farmland"], "count_all": [1]})
    standardize_dataframe(df)
    assert df.iloc[0]["key"] == "Landuse"
    assert df.iloc[0]["value"] == "Farmland"


def test_standardize_dataframe_missing_column_raises():
    df = pd.DataFrame({"foo": ["bar"]})
    with pytest.raises(KeyError):
        standardize_dataframe(df)
