"""SQL query helpers for the OSM taginfo database.

:class:`QueryBuilder` is the single source of truth. Static methods return
``(sql, params)`` tuples, ready to pass to ``conn.execute(sql, params)`` or
the project's :class:`~src.io.exporter.DB` protocol. Class-level constants
are also ``(sql, params)`` tuples (with empty params) so the same
``execute_query(sql, params)`` shape works for every query.
"""
from __future__ import annotations

from typing import Tuple


class QueryBuilder:
    """Static factory of parameterized SQL fragments.

    Every public surface returns ``(sql, params)`` so a single
    ``execute_query(sql, params)`` call works for all queries. For
    parameterless queries ``params`` is the empty tuple.
    Values are never interpolated into the SQL string.
    """

    TOP_KEYS = (
        """
        SELECT key, count_all
        FROM keys
        ORDER BY count_all DESC
        LIMIT 10;
        """,
        (),
    )

    TOP_TAGS = (
        """
        SELECT key, value, count_all
        FROM tags
        ORDER BY count_all DESC
        LIMIT 10;
        """,
        (),
    )

    METADATA = ("SELECT * FROM source;", ())

    GLOBAL_KEY_AGGREGATES = (
        """
        SELECT
            COUNT(key) AS total_distinct_keys,
            SUM(count_all) AS total_key_occurrences
        FROM keys;
        """,
        (),
    )

    GLOBAL_TAG_AGGREGATES = (
        """
        SELECT
            COUNT(*) AS total_distinct_tags,
            SUM(count_all) AS total_tag_occurrences
        FROM tags;
        """,
        (),
    )

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

    @staticmethod
    def tags_by_min_count(
        min_count: int = 500, limit: int | None = None
    ) -> Tuple[str, tuple]:
        """All (key, value) pairs with ``count_all >= min_count``, ordered DESC.

        Used as the long-tail filter at the start of the analysis pipeline.
        If *limit* is set, the result is capped to that many rows.
        """
        sql = """
        SELECT key, value, count_all
        FROM tags
        WHERE count_all >= ?
        ORDER BY count_all DESC
        """
        if limit is None:
            return sql + ";", (min_count,)
        return sql + "LIMIT ?;", (min_count, limit)

    @staticmethod
    def iter_tags_by_min_count(
        min_count: int, batch_size: int, after_count: int | None = None
    ) -> Tuple[str, tuple]:
        """Keyset-paginated iterator over the thresholded tags table.

        Yields ``(key, value, count_all)`` rows with ``count_all >= min_count``,
        strictly less than *after_count* on subsequent calls. The caller is
        responsible for advancing *after_count*; the first call passes
        ``None`` (or omits the keyset entirely).
        """
        if after_count is None:
            sql = """
            SELECT key, value, count_all
            FROM tags
            WHERE count_all >= ?
            ORDER BY count_all DESC
            LIMIT ?;
            """
            return sql, (min_count, batch_size)
        sql = """
        SELECT key, value, count_all
        FROM tags
        WHERE count_all >= ? AND count_all < ?
        ORDER BY count_all DESC
        LIMIT ?;
        """
        return sql, (min_count, after_count, batch_size)
