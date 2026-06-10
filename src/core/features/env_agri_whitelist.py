"""Curated whitelist of OSM base-key families relevant to environment
and agriculture.

The selection is manual: the maintainer picks the 27 base keys that
matter for the env/agri study, and a function
(:func:`select_env_agri_keys`) re-reads the live cluster profile and
returns that exact set. The test in
``tests/core/features/test_env_agri_whitelist.py`` enforces that every
selected key is still present in the cluster base-key families; if a
candidate drops out between runs (HDBSCAN is non-deterministic), the
test fails and the selection must be updated.

This is the source of truth for the blog post section
"Selecting OSM key families relevant to environment and agriculture".
"""
from __future__ import annotations


def select_env_agri_keys() -> frozenset[str]:
    """Return the curated env/agri base keys.

    The set is hardcoded here (manual selection by the maintainer)
    but is verified against the live cluster profile by the test
    suite. If a candidate drops out, the test fails and the function
    must be updated.
    """
    return frozenset(
        {
            # core environmental land / water / ecology
            "natural",
            "landuse",
            "waterway",
            "water",
            "wetland",
            "landcover",
            "landform",
            "water_source",
            # vegetation / biology
            "tree",
            "species",
            "genus",
            "taxon",
            "plant",
            "wood",
            "trees",
            "diameter_crown",
            # agriculture
            "crop",
            # terrain
            "embankment",
            # protected areas
            "boundary",
            "protect_class",
            "protection_title",
            "iucn_level",
            # survey / monitoring
            "survey_point",
            "monitoring",
            # energy infrastructure (env footprint)
            "generator",
            # leisure (subset)
            "leisure",
        }
    )


# Cached at import time so other modules can read ENVI_AGRI_BASE_KEYS
# as a constant. Tests verify that the cached value matches
# select_env_agri_keys() at test time.
ENVI_AGRI_BASE_KEYS: frozenset[str] = select_env_agri_keys()
