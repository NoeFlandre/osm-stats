"""The env/agri whitelist must be a function applied to the live
cluster profile, not a hardcoded list. The test enforces both the
algorithm's properties and the constraint that every selected key
appears in the cluster base-key families.
"""
from pathlib import Path
import re

from src.core.features.env_agri_whitelist import (
    ENVI_AGRI_BASE_KEYS,
    select_env_agri_keys,
)


def _cluster_base_keys() -> set[str]:
    md = Path("output/cluster_profile.md").read_text()
    placeholder = "\x00PIPE\x00"
    keys = set()
    for line in md.splitlines():
        if not line.startswith("| ") or "base_key" in line or line.startswith("| ---"):
            continue
        cleaned = line.replace("\\|", placeholder)
        cells = [c.strip() for c in cleaned.split("|")]
        cells = [c.replace(placeholder, "\\|") for c in cells]
        if len(cells) < 5:
            continue
        bk = cells[1]
        if bk and bk != "---":
            keys.add(bk)
    return keys


# --- algorithmic selection ---------------------------------------------


def test_select_returns_nonempty_frozenset():
    out = select_env_agri_keys()
    assert isinstance(out, frozenset)
    assert len(out) >= 10, f"expected at least 10 env/agri keys, got {len(out)}"


def test_select_is_deterministic():
    """Two calls in the same process must return identical results."""
    a = select_env_agri_keys()
    b = select_env_agri_keys()
    assert a == b


def test_select_is_subset_of_cluster_base_keys():
    """The function only returns base keys that appear in the cluster
    profile. If a candidate never became a cluster medoid, the
    function drops it."""
    cluster_keys = _cluster_base_keys()
    missing = select_env_agri_keys() - cluster_keys
    assert not missing, f"select produced keys not in the cluster profile: {sorted(missing)}"


def test_select_includes_core_environmental_keys():
    required = {"natural", "landuse", "waterway", "water", "wetland"}
    missing = required - select_env_agri_keys()
    assert not missing, f"missing core env keys: {missing}"


def test_select_includes_agricultural_keys():
    """The current cluster profile has 'crop' as a cluster family."""
    assert "crop" in select_env_agri_keys()


def test_select_excludes_noise_families():
    """The selection must drop the obvious infrastructure / noise
    families that the cluster profile surfaces."""
    forbidden = {"addr", "source", "highway", "building", "railway", "name"}
    overlap = forbidden & select_env_agri_keys()
    assert not overlap, f"whitelist accidentally includes noise families: {overlap}"


# --- module-level constant ----------------------------------------------


def test_module_constant_matches_select():
    """ENVI_AGRI_BASE_KEYS is the cached output of select_env_agri_keys()
    at import time. They must agree - if they drift, the test fails and
    forces a regeneration."""
    assert ENVI_AGRI_BASE_KEYS == select_env_agri_keys()


def test_module_constant_is_frozenset():
    assert isinstance(ENVI_AGRI_BASE_KEYS, frozenset)


def test_module_constant_size_is_documented():
    """The blog post claims a specific whitelist size. Pin it."""
    assert len(ENVI_AGRI_BASE_KEYS) == 26, (
        f"whitelist size drifted to {len(ENVI_AGRI_BASE_KEYS)}; "
        f"regenerate ENVI_AGRI_BASE_KEYS or update the blog post"
    )
