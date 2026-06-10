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

    TAG_FEATURES_INSERT = (
        "INSERT OR IGNORE INTO tag_features (key, value, count_all, feature) "
        "VALUES (?, ?, ?, ?);",
        (),
    )

    TAG_FEATURES_INDEX = (
        "CREATE INDEX idx_features_count ON tag_features(count_all DESC);",
        (),
    )

    @staticmethod
    def tag_features_select(
        min_count: int, limit: int | None = None, with_base_key: bool = False
    ) -> Tuple[str, tuple]:
        """Select tag_features rows with ``count_all >= min_count``, ordered DESC.

        If *limit* is set, the result is capped to that many rows. The
        ``base_key`` column is only included in the projection when
        *with_base_key* is True (the column is added by
        :func:`src.core.storage.cache.add_base_key_column`).
        """
        cols = "key, value, count_all, feature, base_key" if with_base_key else "key, value, count_all, feature"
        sql = (
            f"SELECT {cols} FROM tag_features "
            "WHERE count_all >= ? ORDER BY count_all DESC"
        )
        if limit is None:
            return sql + ";", (min_count,)
        return sql + " LIMIT ?;", (min_count, limit)

    @staticmethod
    def tag_features_select_all(limit: int | None = None, with_base_key: bool = False) -> Tuple[str, tuple]:
        """Select all tag_features rows, ordered DESC. Optional *limit* cap."""
        cols = "key, value, count_all, feature, base_key" if with_base_key else "key, value, count_all, feature"
        sql = f"SELECT {cols} FROM tag_features ORDER BY count_all DESC"
        if limit is None:
            return sql + ";", ()
        return sql + " LIMIT ?;", (limit,)

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
    def first_tags_by_min_count(min_count: int, batch_size: int) -> Tuple[str, tuple]:
        """First page of the thresholded tags scan (no upper bound)."""
        sql = """
        SELECT key, value, count_all
        FROM tags
        WHERE count_all >= ?
        ORDER BY count_all DESC
        LIMIT ?;
        """
        return sql, (min_count, batch_size)

    @staticmethod
    def next_tags_by_min_count(
        min_count: int, after_count: int, batch_size: int
    ) -> Tuple[str, tuple]:
        """Subsequent page of the thresholded tags scan, strictly below *after_count*."""
        sql = """
        SELECT key, value, count_all
        FROM tags
        WHERE count_all >= ? AND count_all < ?
        ORDER BY count_all DESC
        LIMIT ?;
        """
        return sql, (min_count, after_count, batch_size)
