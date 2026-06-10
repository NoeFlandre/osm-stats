"""The environmental / agricultural whitelist must be a strict subset
of the OSM base-key families derived from the cluster profile.

This test guards against silently introducing a base key that does not
appear in the 426 cluster-derived families. If a new candidate is added
to the whitelist, it must first be observed as a cluster base key in
``output/cluster_profile.md``.
"""
from pathlib import Path

from src.core.features.env_agri_whitelist import ENVI_AGRI_BASE_KEYS


def _cluster_base_keys() -> set[str]:
    md = Path("output/cluster_profile.md").read_text()
    import re
    keys = re.findall(r"^\| (\S+) \|", md, flags=re.MULTILINE)
    return {k for k in keys if k and k not in ("base_key", "---")}


def test_whitelist_is_nonempty():
    assert len(ENVI_AGRI_BASE_KEYS) >= 10, (
        "whitelist should have at least 10 env/agri base keys"
    )


def test_whitelist_is_subset_of_cluster_base_keys():
    cluster_keys = _cluster_base_keys()
    missing = set(ENVI_AGRI_BASE_KEYS) - cluster_keys
    assert not missing, (
        f"whitelist contains base keys not in the 426 cluster families: "
        f"{sorted(missing)}"
    )


def test_whitelist_contains_core_environmental_keys():
    required = {"natural", "landuse", "waterway", "wetland", "water"}
    missing = required - set(ENVI_AGRI_BASE_KEYS)
    assert not missing, f"missing core env keys: {missing}"


def test_whitelist_contains_agricultural_keys():
    # The 426 cluster families include 'crop' but exclude 'farmland' /
    # 'farmyard' / 'agricultural' / 'produce' - they were not the medoid
    # of any cluster. The subset test above is the source of truth.
    assert "crop" in ENVI_AGRI_BASE_KEYS


def test_whitelist_excludes_building_and_addr():
    """Sanity check: the noise families we explicitly want to drop."""
    forbidden = {"building", "addr", "source", "highway", "railway"}
    overlap = forbidden & set(ENVI_AGRI_BASE_KEYS)
    assert not overlap, f"whitelist accidentally includes noise families: {overlap}"


def test_whitelist_entries_are_unique():
    assert len(ENVI_AGRI_BASE_KEYS) == len(set(ENVI_AGRI_BASE_KEYS))
