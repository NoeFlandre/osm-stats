"""Reproducibility helpers for the osm-stats project.

This package provides the ``reproduce`` entry point (a thin wrapper
around ``reproduce.sh``) and the ``verify`` command, which checks
that the local outputs match the sizes and content hashes recorded
in ``data/MANIFEST.md``.

Run with:

    .venv/bin/python -m scripts.reproducibility.reproduce
    .venv/bin/python -m scripts.reproducibility.verify
"""
