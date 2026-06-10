"""SQL query helpers for the OSM taginfo database.

This module exposes two layers:

* :class:`QueryBuilder` - parameterized query helpers that return ``(sql, params)``
  tuples. Use these in production code; they are SQL-injection safe.
* :func:`get_tag_values_query` - legacy f-string helper kept for backward
  compatibility. New code should call ``QueryBuilder.select_tag_values`` instead.
"""
from __future__ import annotations

from typing import Tuple


class QueryBuilder:
    """Static factory of parameterized SQL fragments.

    All methods return ``(sql, params)`` suitable for sqlite3 ``execute(sql, params)``.
    Values are never interpolated into the SQL string.
    """

    TOP_KEYS = """
    SELECT key, count_all
    FROM keys
    ORDER BY count_all DESC
    LIMIT 10;
    """

    TOP_TAGS = """
    SELECT key, value, count_all
    FROM tags
    ORDER BY count_all DESC
    LIMIT 10;
    """

    METADATA = "SELECT * FROM source;"

    GLOBAL_KEY_AGGREGATES = """
    SELECT
        COUNT(key) AS total_distinct_keys,
        SUM(count_all) AS total_key_occurrences
    FROM keys;
    """

    GLOBAL_TAG_AGGREGATES = """
    SELECT
        COUNT(*) AS total_distinct_tags,
        SUM(count_all) AS total_tag_occurrences
    FROM tags;
    """

    @staticmethod
    def select_tag_values(key: str, limit: int = 50) -> Tuple[str, tuple]:
        """Top values for a given ``key`` in the ``tags`` table."""
        sql = """
        SELECT value, count_all
        FROM tags
        WHERE key = ?
        ORDER BY count_all DESC
        LIMIT ?;
        """
        return sql, (key, limit)


def get_tag_values_query(key: str, limit: int = 50) -> str:
    """.. deprecated::
        Use :meth:`QueryBuilder.select_tag_values` instead. This helper
        builds SQL via f-string interpolation and is unsafe for untrusted input.
    """
    return f"""
    SELECT
        value,
        count_all
    FROM tags
    WHERE key = '{key}'
    ORDER BY count_all DESC
    LIMIT {limit};
    """
