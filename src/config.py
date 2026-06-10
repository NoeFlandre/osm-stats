"""Configuration helpers for the OSM stats CLI.

The path to the taginfo sqlite database can be overridden via the
``OSM_DB_PATH`` environment variable; otherwise :data:`DEFAULT_DB_PATH` is used.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Union

PathLike = Union[str, os.PathLike]

DEFAULT_DB_PATH: PathLike = "/Volumes/Seagate M3/taginfo.sqlite"
ENV_VAR = "OSM_DB_PATH"
DEFAULT_OUTPUT_DIR: Path = Path("output")
# The cache lives next to the source DB on the Seagate drive so we don't fill
# the laptop SSD with the 14 GB working set. Override via --cache-path.
DEFAULT_CACHE_PATH: Path = Path("/Volumes/Seagate M3/tag_features.sqlite")
DEFAULT_THRESHOLD: int = 500


def resolve_db_path(env: Mapping[str, str] | None = None) -> Path:
    """Return the database path, preferring ``OSM_DB_PATH`` from *env*.

    Falls back to :data:`DEFAULT_DB_PATH` if the env var is unset or empty.
    """
    source = env if env is not None else os.environ
    value = source.get(ENV_VAR)
    if value:
        return Path(value)
    return Path(DEFAULT_DB_PATH)
