"""Curated whitelist of OSM base-key families relevant to environment
and agriculture.

Every key in this list must appear in the cluster base-key families
derived from the HDBSCAN cluster profile (see
``output/cluster_profile.md``). The test in
``tests/core/features/test_env_agri_whitelist.py`` enforces that
constraint at test time: any new candidate that is not yet a cluster
base key will fail the test and force a re-run of the pipeline before
the whitelist is updated.

Note: HDBSCAN is non-deterministic, so the exact set of cluster base
keys can shift slightly between runs. The whitelist should be
re-validated against the latest profile whenever the pipeline is
re-run.
"""
from __future__ import annotations

ENVI_AGRI_BASE_KEYS: frozenset[str] = frozenset(
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
