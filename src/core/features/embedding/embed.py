"""Batched Model2Vec embedding for OSM ``key|value`` tag strings.

The TF-IDF + clustering pipeline that powers the blog stats works on
character n-grams. This module is the semantic counterpart: it embeds
each tag with a :class:`model2vec.StaticModel` (default
``minishlab/potion-base-8M``, 256-dim, float32) and returns a
``(len(tags), dim)`` matrix.

The full OSM database has ~224k unique tags. Encoding the whole list
in one shot works but inflates peak memory. :func:`embed_tags` slices
the input into chunks of :data:`DEFAULT_BATCH_SIZE` (4096) and
concatenates the per-chunk encodings into one contiguous float32
array. The chunk size is the single knob that trades memory for
latency.

The function takes an optional ``model`` argument (anything with an
``encode(list[str]) -> np.ndarray`` method) so unit tests can pass a
hand-crafted fake and avoid the network download. When ``model`` is
``None`` a real :class:`model2vec.StaticModel` for
:data:`DEFAULT_MODEL_NAME` is loaded lazily on first use via
:func:`_load_default_model` (which the unit tests monkeypatch to
simulate failures).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


DEFAULT_MODEL_NAME = "minishlab/potion-base-8M"
DEFAULT_BATCH_SIZE = 4096


def _load_default_model(model_name: str = DEFAULT_MODEL_NAME):
    """Lazily build the default :class:`model2vec.StaticModel`.

    Separated from :func:`embed_tags` so unit tests can monkeypatch
    it without importing ``model2vec`` at test-collection time. The
    first successful call to this function downloads the model into
    the local Hugging Face cache.
    """
    from model2vec import StaticModel

    return StaticModel.from_pretrained(model_name)


def _model_output_dim(model) -> int:
    """Return the output width of *model* without calling ``encode``.

    Real :class:`model2vec.StaticModel` exposes its vector table as
    ``model.embedding`` (a 2D ``np.ndarray``). For test doubles that
    do not mimic that, a ``dim`` attribute is honored as a fallback.
    Returns 0 only when no width information is available, which is
    treated by callers as "empty input" and produces a ``(0, 0)``
    array.
    """
    embedding = getattr(model, "embedding", None)
    if embedding is not None and hasattr(embedding, "shape"):
        return int(embedding.shape[-1])
    dim = getattr(model, "dim", None)
    if isinstance(dim, int):
        return dim
    return 0


def _encode_in_batches(
    tags: Sequence[str],
    model,
    batch_size: int,
) -> np.ndarray:
    """Encode *tags* by calling ``model.encode`` on slices of *batch_size*.

    The empty case is handled before any ``encode`` call is made so
    the model never has to deal with a zero-length input.
    """
    n = len(tags)
    # Use a list of chunks to avoid repeated concatenations in the
    # common case where n is small.
    chunks: List[np.ndarray] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = model.encode(list(tags[start:end]))
        chunks.append(np.asarray(chunk))
    if not chunks:
        # ``model`` was never called. We still need a (0, dim) array,
        # but we do not know ``dim`` until the model is asked, so
        # call it once on an empty list to learn the dim. This is
        # only reachable if the caller passed an empty list and the
        # implementation does not handle it before the batch loop;
        # in practice the function-level short-circuit in
        # :func:`embed_tags` prevents reaching this branch.
        empty = model.encode([])
        return np.asarray(empty, dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def embed_tags(
    tags: Sequence[str],
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model: Optional[object] = None,
) -> np.ndarray:
    """Embed *tags* with a Model2Vec StaticModel and return a (len(tags), dim) float32 array.

    The input is encoded in chunks of *batch_size* to bound peak
    memory; chunks are concatenated row-wise in input order. The
    output is always ``np.float32`` regardless of the model's
    internal dtype, so downstream cosine-similarity code can rely on
    a single dtype.

    Parameters
    ----------
    tags:
        A sequence of ``"key|value"`` strings to embed.
    model_name:
        Hugging Face model id used when *model* is ``None``. Defaults
        to :data:`DEFAULT_MODEL_NAME` (``"minishlab/potion-base-8M"``).
    batch_size:
        Maximum number of tags per ``model.encode`` call. Defaults to
        :data:`DEFAULT_BATCH_SIZE` (4096).
    model:
        Optional pre-built embedder. Anything with an
        ``encode(list[str]) -> np.ndarray`` method works. When
        ``None``, a real :class:`model2vec.StaticModel` is loaded
        lazily on first use.

    Returns
    -------
    np.ndarray
        A ``(len(tags), dim)`` ``float32`` array. The ``i``-th row
        corresponds to the ``i``-th element of *tags*. For an empty
        input the result is ``(0, dim)`` and the model is not
        called.

    Notes
    -----
    This is the canonical batched entry point for the embedding
    pipeline. It bypasses :class:`SemanticProbe.embed` because that
    method is a thin non-batching wrapper; going through
    ``model.encode`` directly lets us stream chunks of *batch_size*
    through the static model's native batched encoder.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if model is None:
        model = _load_default_model(model_name)

    if len(tags) == 0:
        # Discover the dim from the model's own metadata rather than
        # calling ``encode`` on an empty list. The real StaticModel
        # exposes ``model.embedding`` (a 2D ``np.ndarray``); fakes
        # expose ``dim``. The model is never called on the empty
        # path so callers can rely on a zero-side-effect fast path.
        dim = _model_output_dim(model)
        return np.zeros((0, dim), dtype=np.float32)

    out = _encode_in_batches(tags, model, batch_size)
    return np.asarray(out, dtype=np.float32)
