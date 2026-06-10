"""Tests for the CLI argument parser of ``src.cli``."""
from pathlib import Path

from src.cli import parse_args


def test_parse_args_default():
    args = parse_args([])
    assert args.build_cache is False
    assert args.threshold == 500
    assert args.summary_only is False
    # Default cache path lives on the same drive as the source DB.
    assert "Seagate" in str(args.cache_path)
    assert str(args.cache_path).endswith("tag_features.sqlite")


def test_parse_args_build_cache_with_threshold():
    args = parse_args(
        ["--build-cache", "--threshold", "1000", "--cache-path", "/tmp/x.sqlite"]
    )
    assert args.build_cache is True
    assert args.threshold == 1000
    assert args.cache_path == Path("/tmp/x.sqlite")


def test_parse_args_batch_size():
    args = parse_args(["--build-cache", "--batch-size", "10000"])
    assert args.batch_size == 10_000


def test_parse_args_summary_only():
    args = parse_args(["--summary-only"])
    assert args.summary_only is True
    assert args.build_cache is False
