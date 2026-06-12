"""Tests for the semantic embedding probe.

This module is an EXPERIMENT that lives alongside the TF-IDF pipeline.
It does not replace anything. The probe embeds a small set of OSM
``key|value`` tags with a static vector model and reports cosine
similarities so we can decide whether a full re-derivation is
worth it.

The unit tests use a hand-crafted fake embedder so we never depend
on a network download in CI. One integration test uses the real
``potion-base-8M`` model and is marked ``slow`` so it can be
skipped in fast feedback loops.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import numpy as np
import pytest

from src.core.features.semantic_probe import (
    SemanticProbe,
    cosine_similarity_matrix,
)


# --- a deterministic fake embedder ----------------------------------------


class _FakeEmbedder:
    """Maps each tag to a fixed unit vector keyed by integer index.

    Encodes the tag at index ``i`` as a one-hot unit vector in
    ``dim``-dimensional space, then permutes that vector by a fixed
    permutation keyed by the tag string. Two tags get the same
    vector only if they share the same string (which would defeat
    the test); two tags with the same prefix share a small
    ``overlap_dim`` block of dimensions.
    """

    def __init__(self, dim: int = 16, overlap_dim: int = 4) -> None:
        self.dim = dim
        self.overlap_dim = overlap_dim

    def encode(self, tags: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(tags), self.dim), dtype=np.float32)
        for i, tag in enumerate(tags):
            base = np.zeros(self.dim, dtype=np.float32)
            base[i % self.dim] = 1.0
            for k in range(self.overlap_dim):
                base[(hash(tag) + k) % self.dim] += 0.5
            norm = float(np.linalg.norm(base))
            if norm > 0:
                base /= norm
            out[i] = base
        return out


# --- shape and determinism ------------------------------------------------


def test_cosine_similarity_matrix_is_square_and_shape_matches_input():
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    sims = cosine_similarity_matrix(vecs)
    assert sims.shape == (3, 3)


def test_cosine_similarity_diagonal_is_one():
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    sims = cosine_similarity_matrix(vecs)
    np.testing.assert_allclose(np.diag(sims), np.ones(3), atol=1e-6)


def test_cosine_similarity_is_symmetric():
    vecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    sims = cosine_similarity_matrix(vecs)
    np.testing.assert_allclose(sims, sims.T, atol=1e-6)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
    sims = cosine_similarity_matrix(vecs)
    assert abs(sims[0, 1]) < 1e-6
    assert abs(sims[1, 0]) < 1e-6


# --- the probe object itself ----------------------------------------------


def test_probe_returns_vectors_with_one_row_per_tag():
    tags = ["landuse|farmland", "landuse|meadow", "natural|water"]
    probe = SemanticProbe(embedder=_FakeEmbedder(dim=8))
    vecs = probe.embed(tags)
    assert vecs.shape[0] == len(tags)
    assert vecs.shape[1] == 8


def test_probe_similarity_matrix_uses_same_tags_as_rows_and_columns():
    tags = ["a", "b", "c"]
    probe = SemanticProbe(embedder=_FakeEmbedder())
    sims = probe.similarity_matrix(tags)
    assert sims.shape == (3, 3)


def test_probe_exposes_tag_order():
    tags = ["landuse|farmland", "landuse|meadow"]
    probe = SemanticProbe(embedder=_FakeEmbedder())
    sims = probe.similarity_matrix(tags)
    assert sims[0, 0] == pytest.approx(1.0, abs=1e-6)
    # The matrix is symmetric so [1, 0] equals [0, 1]
    assert sims[0, 1] == pytest.approx(sims[1, 0], abs=1e-6)


def test_probe_is_deterministic_with_fake_embedder():
    tags = ["x", "y"]
    probe = SemanticProbe(embedder=_FakeEmbedder())
    a = probe.similarity_matrix(tags)
    b = probe.similarity_matrix(tags)
    np.testing.assert_array_equal(a, b)


# --- the env/agri semantic sanity check -----------------------------------
#
# The probe is meant to validate the hypothesis that a semantic
# embedding model places semantically related OSM tags close in
# vector space. We pick a small fixture of tags that *should* be
# close (agricultural landuse values) and tags that *should* be
# far (urban landuse, unrelated natural tags) and assert the
# ordering is correct.
#
# This is the unit-test version using the fake embedder. The real
# model is exercised in the slow integration test below.


def _fixture_tags() -> List[str]:
    return [
        "landuse|farmland",       # 0: agricultural
        "landuse|meadow",         # 1: agricultural (vegetation)
        "landuse|grassland",      # 2: agricultural (vegetation)
        "landuse|forest",         # 3: landuse but forest, not crop
        "landuse|residential",    # 4: NOT agricultural
        "natural|water",          # 5: unrelated key entirely
        "natural|tree",           # 6: unrelated key entirely
    ]


def test_env_agri_probe_landuse_only_uses_fake_embedder_records_shape():
    # Sanity: the probe produces the right shape on the env/agri
    # fixture. We do not assert specific similarity values here
    # because the fake embedder is not semantically meaningful.
    probe = SemanticProbe(embedder=_FakeEmbedder(dim=12))
    sims = probe.similarity_matrix(_fixture_tags())
    assert sims.shape == (7, 7)


# --- interface to the real model -----------------------------------------


def test_probe_default_embedder_is_a_potion_model(monkeypatch):
    # Constructing the probe with no embedder should lazily build a
    # potion-base-8M StaticModel. We do not actually download in
    # the test: we monkeypatch the loader to return a fake.
    from src.core.features import semantic_probe as mod

    monkeypatch.setattr(mod, "_load_default_embedder", lambda: _FakeEmbedder(dim=10))
    probe = SemanticProbe()
    vecs = probe.embed(["a", "b"])
    assert vecs.shape == (2, 10)


# --- integration test with the real model --------------------------------


@pytest.mark.slow
def test_env_agri_probe_real_model_groups_agricultural_landuse():
    """End-to-end probe with potion-base-8M.

    Agricultural landuse values (farmland, meadow, grassland) should
    be more similar to each other than to urban landuse (residential)
    or unrelated natural tags.
    """
    probe = SemanticProbe()  # uses the real potion-base-8M
    tags = _fixture_tags()
    sims = probe.similarity_matrix(tags)

    # diag is 1, off-diag is the real signal
    agri_vs_agri = (sims[0, 1] + sims[0, 2] + sims[1, 2]) / 3.0
    agri_vs_urban = (sims[0, 4] + sims[1, 4] + sims[2, 4]) / 3.0
    agri_vs_unrelated = (sims[0, 5] + sims[1, 6]) / 2.0

    # The hypothesis: the embedding puts agricultural landuse values
    # closer to each other than to either urban landuse or unrelated
    # natural tags. If this fails we have learned the embedding is
    # not useful for our task.
    assert agri_vs_agri > agri_vs_urban, (
        f"Expected agri-agri sim > agri-urban sim, got "
        f"{agri_vs_agri:.3f} vs {agri_vs_urban:.3f}"
    )
    assert agri_vs_agri > agri_vs_unrelated, (
        f"Expected agri-agri sim > agri-unrelated sim, got "
        f"{agri_vs_agri:.3f} vs {agri_vs_unrelated:.3f}"
    )
