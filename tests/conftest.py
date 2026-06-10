"""Shared pytest fixtures and factories for osm-stats tests.

Centralizes SQLite fixture creation so individual test files do not duplicate
schema setup. Each test should declare only the data it cares about.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

import pytest


def make_db(db_path: Path, schema: str, rows: Iterable[Sequence] = ()) -> Path:
    """Create a sqlite db at *db_path* with a single *schema* and optional *rows*.

    The schema must define a single CREATE TABLE statement. Rows are inserted
    in column order of that table.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    table_name = _first_table_name(schema)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(schema)
        if rows:
            first = next(iter(rows))
            placeholders = ", ".join(["?"] * len(first))
            conn.executemany(
                f"INSERT INTO {table_name} VALUES ({placeholders})", rows
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _first_table_name(schema: str) -> str:
    match = re.search(r"CREATE\s+TABLE\s+(\w+)", schema, flags=re.IGNORECASE)
    if not match:
        raise ValueError("schema must contain a CREATE TABLE statement")
    return match.group(1)


@pytest.fixture
def temp_sqlite(tmp_path: Path):
    """Factory fixture: ``temp_sqlite(schema, rows=()) -> Path``.

    Usage:
        def test_x(temp_sqlite):
            db_file = temp_sqlite(
                "CREATE TABLE t (id INTEGER, name TEXT)",
                rows=[(1, "a"), (2, "b")],
            )
    """
    def _factory(schema: str, rows: Iterable[Sequence] = ()) -> Path:
        return make_db(tmp_path / "test.sqlite", schema, rows)
    return _factory
