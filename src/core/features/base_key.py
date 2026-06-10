"""Extract the base OSM key from a standardized tag string.

The pipeline stores each tag as ``"<key>|<value>"``. The base key is the
*root* of that key, taken to be everything before the first colon. This
groups variants like::

    addr:street            -> addr
    addr:city:simc         -> addr
    abandoned:aeroway      -> abandoned
    landuse                -> landuse
    highway                -> highway

We use the first colon as the boundary because that is how OSM namespaces
its keys (everything after a colon is a sub-key of the prefix).
"""
from __future__ import annotations


def parse_base_key(feature: str) -> str:
    """Return the base OSM key for a ``"<key>|<value>"`` string.

    The key portion is everything before the pipe; the base key is the
    key portion truncated at the first colon (if any). Lowercased and
    stripped.
    """
    if not feature or not feature.strip():
        raise ValueError("parse_base_key: empty or whitespace-only input")
    if "|" not in feature:
        raise ValueError(f"parse_base_key: missing '|' delimiter in {feature!r}")
    key = feature.split("|", 1)[0].strip().lower()
    if not key:
        raise ValueError(f"parse_base_key: empty key in {feature!r}")
    base = key.split(":", 1)[0]
    return base
