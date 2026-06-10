"""Curated whitelist of OSM base-key families relevant to environment
and agriculture.

Every key in this list must appear in the 426 base-key families derived
from the HDBSCAN cluster profile (see ``output/cluster_profile.md``).
The test in ``tests/core/features/test_env_agri_whitelist.py`` enforces
that constraint: any new candidate that is not yet a cluster base key
will fail the test and force a re-run of the pipeline before the
whitelist is updated.
"""
from __future__ import annotations

# Core environmental land / water / ecology
# ---------------------------------------------------------------------------
# Direct land/water/ecology descriptors
# natural, landuse, waterway, water, wetland, landcover, landform, water_source
# ---------------------------------------------------------------------------
# Vegetation / biology
# tree, species, genus, taxon, plant, wood, trees, diameter_crown
# ---------------------------------------------------------------------------
# Agriculture
# crop
# ---------------------------------------------------------------------------
# Terrain features
# earth_bank, embankment
# ---------------------------------------------------------------------------
# Protected areas
# boundary, protect_class, protection_title, iucn_level
# ---------------------------------------------------------------------------
# Survey / monitoring
# survey_point, monitoring
# ---------------------------------------------------------------------------
# Energy infrastructure (env footprint)
# generator
# ---------------------------------------------------------------------------
# Leisure (subset that includes nature_reserve, garden, etc.)
# leisure

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
        "earth_bank",
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
