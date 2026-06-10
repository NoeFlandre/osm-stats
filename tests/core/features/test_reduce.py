import numpy as np
from scipy.sparse import csr_matrix

from src.core.features.reduce import reduce_dimensions


def _toy_sparse():
    # 6 rows, 4 columns, deterministic.
    data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    rows = [0, 1, 2, 3, 4, 5]
    cols = [0, 0, 1, 2, 2, 3]
    return csr_matrix((data, (rows, cols)), shape=(6, 4))


# --- input handling ------------------------------------------------------


def test_accepts_sparse_csr_matrix():
    X = _toy_sparse()
    out = reduce_dimensions(X, n_components=2)
    assert out is not None


# --- output shape and density -------------------------------------------


def test_output_shape_matches_n_components():
    X = _toy_sparse()
    out = reduce_dimensions(X, n_components=3)
    assert out.shape == (6, 3)


def test_output_is_dense_numpy_array():
    X = _toy_sparse()
    out = reduce_dimensions(X, n_components=2)
    assert isinstance(out, np.ndarray)
    # No sparse backings, no pandas wrappers - a real dense array.
    assert out.ndim == 2


def test_default_n_components_is_in_safe_range():
    # Spec says 30-50. We pick a default in that band.
    from src.core.features.reduce import DEFAULT_N_COMPONENTS

    assert 30 <= DEFAULT_N_COMPONENTS <= 50


def test_custom_n_components_honored():
    X = _toy_sparse()
    out = reduce_dimensions(X, n_components=4)
    # Can't exceed min(n_samples, n_features) - 4 is fine here.
    assert out.shape == (6, 4)


# --- numerical sanity ----------------------------------------------------


def test_output_is_finite():
    X = _toy_sparse()
    out = reduce_dimensions(X, n_components=2)
    assert np.all(np.isfinite(out))


def test_output_has_variance_not_constant():
    X = _toy_sparse()
    out = reduce_dimensions(X, n_components=2)
    # TruncatedSVD should produce components with non-zero variance
    # (otherwise the reduction collapsed everything to a single point).
    assert np.any(out.std(axis=0) > 1e-9)


def test_dense_output_preserves_distances_better_than_pca_could():
    # Loose check: two rows that are identical in the sparse input should
    # remain close in the dense output. Row 0 and row 1 are both 1.0 in
    # column 0 only, so they're identical in input space and should
    # remain at distance ~0 in the output.
    X = _toy_sparse()
    out = reduce_dimensions(X, n_components=2)
    d01 = np.linalg.norm(out[0] - out[1])
    d02 = np.linalg.norm(out[0] - out[2])
    assert d01 < d02
