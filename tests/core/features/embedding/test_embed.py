"""Tests for the batched embedding module.

The :mod:`src.core.features.embedding.embed` module is the canonical
batched entry point for embedding OSM ``key|value`` tag strings with a
Model2Vec :class:`StaticModel`. The real model is 224k tags so we must
slice the input into chunks of ``batch_size`` to bound peak memory.

The unit tests use a hand-crafted fake model that implements the
``encode(list[str]) -> np.ndarray`` protocol. One integration test
hits the real ``potion-base-8M`` model and is marked ``slow`` so it
can be skipped in fast feedback loops.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pytest

from src.core.features.embedding.embed import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_NAME,
    embed_tags,
)


# --- a deterministic fake model -------------------------------------------


class _FakeModel:
    """A deterministic stand-in for a :class:`model2vec.StaticModel`.

    Records every batch it is called with (preserving the slice) and
    returns a fixed-shape float32 array. The vector for each tag is
    a one-hot unit vector keyed on ``hash(tag) % dim`` so two distinct
    tags get distinct vectors and order is verifiable through the
    hashing.

    Exposes a ``dim`` attribute (mirroring :class:`model2vec.StaticModel`'s
    internal ``embedding`` array) so the production code can discover
    the output width without ever calling :meth:`encode` on an empty
    list. This is the same contract ``StaticModel`` offers through
    ``model.embedding.shape[-1]``.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.calls: List[List[str]] = []

    def encode(self, batch: Sequence[str]) -> np.ndarray:
        self.calls.append(list(batch))
        out = np.zeros((len(batch), self.dim), dtype=np.float32)
        for i, tag in enumerate(batch):
            out[i, hash(tag) % self.dim] = 1.0
        return out


# --- shape with a fake model ----------------------------------------------


def test_embed_tags_returns_one_row_per_tag():
    tags = [f"key|value_{i}" for i in range(10)]
    fake = _FakeModel(dim=16)
    out = embed_tags(tags, model=fake)
    assert out.shape == (10, 16)


def test_embed_tags_shape_uses_model_dim():
    tags = ["a", "b", "c"]
    fake = _FakeModel(dim=7)
    out = embed_tags(tags, model=fake)
    assert out.shape == (3, 7)


# --- batching: number of .encode calls ------------------------------------


def test_embed_tags_batches_in_chunks_of_batch_size():
    tags = [f"t{i}" for i in range(100)]
    fake = _FakeModel(dim=8)
    embed_tags(tags, model=fake, batch_size=25)
    assert len(fake.calls) == 4
    # First three batches are full (25 tags each), last is a tail of 25.
    for i, call in enumerate(fake.calls):
        assert len(call) == 25, f"batch {i} had {len(call)} tags"
    # Batches are concatenated back in input order.
    assert sum(fake.calls, []) == tags


def test_embed_tags_single_call_when_input_fits_in_one_batch():
    tags = [f"t{i}" for i in range(100)]
    fake = _FakeModel(dim=8)
    embed_tags(tags, model=fake, batch_size=200)
    assert len(fake.calls) == 1
    assert fake.calls[0] == tags


def test_embed_tags_partial_last_batch():
    tags = [f"t{i}" for i in range(10)]
    fake = _FakeModel(dim=8)
    embed_tags(tags, model=fake, batch_size=3)
    # 3 + 3 + 3 + 1 = 4 calls
    assert len(fake.calls) == 4
    assert [len(c) for c in fake.calls] == [3, 3, 3, 1]


# --- order preservation ---------------------------------------------------


def test_embed_tags_preserves_input_order():
    # Use the one-hot keying of the fake: row i should have a 1 at
    # position hash(tags[i]) % dim, and 0 elsewhere.
    tags = [f"unique_tag_{i}" for i in range(8)]
    fake = _FakeModel(dim=64)
    out = embed_tags(tags, model=fake, batch_size=4)
    assert out.shape == (8, 64)
    for i, tag in enumerate(tags):
        expected_idx = hash(tag) % 64
        # Row i has a 1 only at expected_idx.
        assert out[i, expected_idx] == pytest.approx(1.0)
        row = out[i].copy()
        row[expected_idx] = 0.0
        assert np.all(row == 0.0)


# --- dtype ----------------------------------------------------------------


def test_embed_tags_output_is_float32():
    # Real StaticModel returns float32 by default, but a fake could
    # return float64; the function must coerce to float32.
    class _Float64Model:
        def __init__(self) -> None:
            self.calls: List[List[str]] = []

        def encode(self, batch: Sequence[str]) -> np.ndarray:
            self.calls.append(list(batch))
            return np.ones((len(batch), 4), dtype=np.float64)

    fake = _Float64Model()
    out = embed_tags(["a", "b"], model=fake)
    assert out.dtype == np.float32


# --- empty input ----------------------------------------------------------


def test_embed_tags_empty_input_returns_zero_row_array():
    # We choose: empty input returns a shape-(0, dim) array. The fake
    # model is never called.
    fake = _FakeModel(dim=12)
    out = embed_tags([], model=fake)
    assert out.shape == (0, 12)
    assert out.dtype == np.float32
    assert fake.calls == []


# --- constants & docstring ------------------------------------------------


def test_default_model_name_is_potion_base_8m():
    assert DEFAULT_MODEL_NAME == "minishlab/potion-base-8M"


def test_default_batch_size_is_a_positive_power_of_two():
    # 4096 is a comfortable default that fits comfortably in CPU RAM
    # for 384-dim float32 vectors (~6 MB per batch).
    assert DEFAULT_BATCH_SIZE > 0
    assert DEFAULT_BATCH_SIZE & (DEFAULT_BATCH_SIZE - 1) == 0


def test_embed_tags_docstring_mentions_model_name():
    assert embed_tags.__doc__ is not None
    assert DEFAULT_MODEL_NAME in embed_tags.__doc__


# --- error handling: no model and no defaults ----------------------------


def test_embed_tags_raises_with_clear_error_when_no_model_and_no_network(monkeypatch):
    # If the user does not pass a model and the default loader fails
    # (e.g. no network in CI), the error message must point at the
    # missing model. We simulate the failure by monkeypatching the
    # loader used inside the module.
    from src.core.features.embedding import embed as embed_mod

    def _boom(model_name: str):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(embed_mod, "_load_default_model", _boom)
    with pytest.raises(RuntimeError, match="simulated network failure"):
        embed_tags(["a"])


# --- integration test with the real model --------------------------------


@pytest.mark.slow
def test_real_model_integration():
    """End-to-end: real potion-base-8M on a small env/agri fixture.

    Asserts shape, dtype, and parity with the existing
    :class:`SemanticProbe` on the first row. If the actual
    dimensionality of ``potion-base-8M`` is not 384 we update the
    test and report the actual value here.
    """
    from model2vec import StaticModel

    from src.core.features.semantic_probe import SemanticProbe

    tags = [
        "landuse|farmland",
        "landuse|meadow",
        "landuse|grassland",
        "landuse|forest",
        "landuse|residential",
        "natural|water",
        "natural|tree",
    ]

    # Build the real model once and share it between both calls so
    # the test is hermetic and fast.
    real_model = StaticModel.from_pretrained(DEFAULT_MODEL_NAME)
    # Probe parity: the existing probe must produce the same first
    # row for tags[0] as our batched entry point.
    probe = SemanticProbe(embedder=real_model)
    expected_first_row = probe.embed([tags[0]])[0]

    out = embed_tags(tags, model=real_model, batch_size=4)

    # Shape: 7 tags, dim is whatever the model actually reports.
    actual_dim = int(real_model.embedding_dim) if hasattr(real_model, "embedding_dim") else out.shape[1]
    assert out.shape == (7, actual_dim)
    assert out.dtype == np.float32
    # Parity on the first row.
    np.testing.assert_array_equal(out[0], expected_first_row)
