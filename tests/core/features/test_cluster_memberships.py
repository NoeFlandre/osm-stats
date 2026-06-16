"""Tests for the per-tag cluster membership persistence.

The medoid CSV is a *summary* (one row per cluster + a noise row).
The membership CSV is the complementary *detail* (one row per cache
tag with its cluster assignment). The two are joined on
``cluster_id``.

The membership file is the **raw, unfiltered** output of HDBSCAN.
These tests guard that contract: no LLM filtering, no whitelist, the
noise bucket stays in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.features.cluster_memberships import (
    MEMBERSHIPS_COLUMNS,
    save_cluster_memberships,
)


# --- schema ---------------------------------------------------------------


def test_memberships_columns_constant_is_canonical():
    assert MEMBERSHIPS_COLUMNS == [
        "cluster_id",
        "base_key",
        "key",
        "value",
        "feature",
        "count_all",
    ]


# --- shape / round-trip ---------------------------------------------------


def test_save_cluster_memberships_writes_expected_shape(tmp_path):
    df = pd.DataFrame(
        {
            "key": ["landuse", "natural"],
            "value": ["farmland", "water"],
            "feature": ["landuse|farmland", "natural|water"],
            "count_all": [1_000, 500],
        }
    )
    labels = np.array([0, -1], dtype=np.int64)
    out = tmp_path / "memberships.csv"

    save_cluster_memberships(labels, df, out)

    written = pd.read_csv(out)
    assert list(written.columns) == MEMBERSHIPS_COLUMNS
    assert len(written) == 2
    assert list(written["cluster_id"]) == [0, -1]
    assert list(written["base_key"]) == ["landuse", "natural"]
    assert list(written["feature"]) == ["landuse|farmland", "natural|water"]


def test_save_cluster_memberships_preserves_count_all(tmp_path):
    df = pd.DataFrame(
        {
            "key": ["addr:street", "addr:city"],
            "value": ["hauptstraße", "berlin"],
            "feature": ["addr:street|hauptstraße", "addr:city|berlin"],
            "count_all": [10_000, 20_000],
        }
    )
    labels = np.array([0, 0], dtype=np.int64)
    out = tmp_path / "memberships.csv"

    save_cluster_memberships(labels, df, out)

    written = pd.read_csv(out)
    assert list(written["count_all"]) == [10_000, 20_000]


# --- input contract -------------------------------------------------------


def test_save_cluster_memberships_raises_on_length_mismatch(tmp_path):
    df = pd.DataFrame(
        {
            "key": ["a", "b"],
            "value": ["x", "y"],
            "feature": ["a|x", "b|y"],
            "count_all": [1, 2],
        }
    )
    labels = np.array([0], dtype=np.int64)
    with pytest.raises(ValueError, match="length"):
        save_cluster_memberships(labels, df, tmp_path / "out.csv")


def test_save_cluster_memberships_preserves_row_order(tmp_path):
    df = pd.DataFrame(
        {
            "key": ["landuse", "natural", "highway"],
            "value": ["farmland", "water", "track"],
            "feature": ["landuse|farmland", "natural|water", "highway|track"],
            "count_all": [1, 2, 3],
        }
    )
    # Permute the labels to verify the i-th row keeps its i-th label.
    labels = np.array([5, -1, 5], dtype=np.int64)
    out = tmp_path / "memberships.csv"

    save_cluster_memberships(labels, df, out)

    written = pd.read_csv(out)
    assert list(written["cluster_id"]) == [5, -1, 5]
    assert list(written["feature"]) == [
        "landuse|farmland",
        "natural|water",
        "highway|track",
    ]


# --- base key derivation --------------------------------------------------


def test_save_cluster_memberships_derives_base_key_from_feature(tmp_path):
    df = pd.DataFrame(
        {
            "key": ["addr:street", "addr:city:simc"],
            "value": ["hauptstraße", "0918123"],
            "feature": ["addr:street|hauptstraße", "addr:city:simc|0918123"],
            "count_all": [10, 20],
        }
    )
    labels = np.array([0, 0], dtype=np.int64)
    out = tmp_path / "memberships.csv"

    save_cluster_memberships(labels, df, out)

    written = pd.read_csv(out)
    # Per spec, the base key is everything before the FIRST colon.
    assert list(written["base_key"]) == ["addr", "addr"]


def test_save_cluster_memberships_base_key_for_no_colon(tmp_path):
    df = pd.DataFrame(
        {
            "key": ["landuse"],
            "value": ["farmland"],
            "feature": ["landuse|farmland"],
            "count_all": [1],
        }
    )
    labels = np.array([0], dtype=np.int64)
    out = tmp_path / "memberships.csv"

    save_cluster_memberships(labels, df, out)

    written = pd.read_csv(out)
    assert written.iloc[0]["base_key"] == "landuse"


# --- noise bucket is preserved --------------------------------------------


def test_save_cluster_memberships_keeps_noise_bucket(tmp_path):
    """The membership file is the raw, unfiltered HDBSCAN output.
    Noise points (label ``-1``) MUST be in the file."""
    df = pd.DataFrame(
        {
            "key": ["a", "b", "c"],
            "value": ["x", "y", "z"],
            "feature": ["a|x", "b|y", "c|z"],
            "count_all": [1, 2, 3],
        }
    )
    labels = np.array([0, -1, 1], dtype=np.int64)
    out = tmp_path / "memberships.csv"

    save_cluster_memberships(labels, df, out)

    written = pd.read_csv(out)
    # All three rows must be present; nothing is filtered.
    assert len(written) == 3
    assert -1 in set(written["cluster_id"])


# --- output file path -----------------------------------------------------


def test_save_cluster_memberships_creates_parent_dirs(tmp_path):
    df = pd.DataFrame(
        {
            "key": ["a"],
            "value": ["x"],
            "feature": ["a|x"],
            "count_all": [1],
        }
    )
    out = tmp_path / "nested" / "deeper" / "memberships.csv"
    save_cluster_memberships(np.array([0], dtype=np.int64), df, out)
    assert out.exists()


def test_save_cluster_memberships_returns_output_path(tmp_path):
    df = pd.DataFrame(
        {
            "key": ["a"],
            "value": ["x"],
            "feature": ["a|x"],
            "count_all": [1],
        }
    )
    out = tmp_path / "memberships.csv"
    returned = save_cluster_memberships(np.array([0], dtype=np.int64), df, out)
    assert returned == out
