import os
from pathlib import Path

from src.config import DEFAULT_DB_PATH, resolve_db_path


def test_resolve_db_path_uses_env(monkeypatch, tmp_path):
    custom = tmp_path / "custom.sqlite"
    custom.touch()
    monkeypatch.setenv("OSM_DB_PATH", str(custom))

    assert resolve_db_path() == custom


def test_resolve_db_path_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("OSM_DB_PATH", raising=False)
    assert resolve_db_path() == Path(DEFAULT_DB_PATH)


def test_resolve_db_path_explicit_env_overrides(monkeypatch, tmp_path):
    explicit = tmp_path / "explicit.sqlite"
    explicit.touch()
    monkeypatch.setenv("OSM_DB_PATH", str(explicit))

    assert resolve_db_path(env=os.environ) == explicit
