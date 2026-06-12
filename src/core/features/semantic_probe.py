"""Semantic embedding probe for OSM ``key|value`` tags.

This module is an EXPERIMENT that lives alongside the TF-IDF + clustering
pipeline. It does not replace anything. The probe embeds a small set of
OSM tags with a static vector model (``potion-base-8M`` by default) and
exposes a cosine-similarity view so we can decide whether a full
re-derivation of the base-key / clustering logic on top of semantic
embeddings is worth it.

The probe is intentionally thin: it loads a ``StaticModel`` lazily, and
the two public methods (:meth:`SemanticProbe.embed` and
:meth:`SemanticProbe.similarity_matrix`) are direct wrappers around the
model's ``encode`` plus a small cosine-similarity helper.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from model2vec import StaticModel


DEFAULT_MODEL_NAME = "minishlab/potion-base-8M"


def cosine_similarity_matrix(vecs: np.ndarray) -> np.ndarray:
    """Return the (n, n) cosine similarity matrix for *vecs*.

    Each row of *vecs* is treated as one vector. Rows are normalized to
    unit length before the dot product so the diagonal is 1.0 for any
    nonzero row.
    """
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    # Avoid division by zero: a zero row would otherwise produce NaNs.
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    unit = vecs / safe_norms
    return unit @ unit.T


def _load_default_embedder() -> StaticModel:
    """Lazily build the default ``potion-base-8M`` embedder.

    Called only from :class:`SemanticProbe` when no embedder is supplied,
    so the unit tests (which always pass a fake embedder) never trigger a
    network download.
    """
    return StaticModel.from_pretrained(DEFAULT_MODEL_NAME)


class SemanticProbe:
    """Thin wrapper around a static vector model for OSM tag probing.

    Parameters
    ----------
    embedder:
        Any object that exposes ``encode(tags: Sequence[str]) -> np.ndarray``.
        When ``None`` (the default), a real :class:`model2vec.StaticModel`
        for ``potion-base-8M`` is loaded lazily on first use.
    """

    def __init__(self, embedder: Optional[object] = None) -> None:
        if embedder is None:
            self._embedder = _load_default_embedder()
        else:
            self._embedder = embedder

    def embed(self, tags: Sequence[str]) -> np.ndarray:
        """Encode *tags* into a ``(len(tags), dim)`` float array."""
        return self._embedder.encode(list(tags))

    def similarity_matrix(self, tags: Sequence[str]) -> np.ndarray:
        """Return the pairwise cosine similarity matrix of *tags*."""
        vecs = self.embed(tags)
        return cosine_similarity_matrix(vecs)
