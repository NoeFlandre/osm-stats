import pytest

from src.core.features.base_key import parse_base_key


# --- equal sign (most common) -------------------------------------------


def test_standard_key_value():
    assert parse_base_key("landuse|farmland") == "landuse"


def test_addr_subkey_collapsed_to_addr():
    # Per spec: everything before the FIRST colon is the base key.
    # 'addr:street' has its first colon between 'addr' and 'street',
    # so the base key is 'addr'.
    assert parse_base_key("addr:street|hauptstraße") == "addr"


def test_addr_subsubkey_collapsed_to_addr():
    assert parse_base_key("addr:city:simc|0918123") == "addr"


# --- colon-only (no equals because pipe is our delimiter) ----------------


def test_colon_split_falls_back_to_first_segment():
    # The pipe is our joiner so the only delimiter inside is nothing.
    # A value with internal colons is rare but possible.
    assert parse_base_key("abandoned:aeroway|runway") == "abandoned"


# --- key with no value (key only) ---------------------------------------


def test_key_only():
    # Some features may be just "highway" with no value after the pipe.
    assert parse_base_key("highway|") == "highway"


def test_none_value_token_treated_as_key_only():
    # We standardized empty -> "none"; this is a real key with no value.
    assert parse_base_key("natural|none") == "natural"


# --- edge cases ---------------------------------------------------------


def test_empty_string_raises():
    with pytest.raises(ValueError):
        parse_base_key("")


def test_input_without_pipe_raises():
    with pytest.raises(ValueError):
        parse_base_key("landuse_farmland")


def test_whitespace_only_raises():
    with pytest.raises(ValueError):
        parse_base_key("   ")


def test_returns_lowercase():
    # The pipeline pre-standardizes, but defensive lowercase doesn't hurt.
    assert parse_base_key("Landuse|Farmland") == "landuse"
