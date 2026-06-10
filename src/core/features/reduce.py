"""Dimensionality reduction for sparse TF-IDF matrices.

The character n-gram TF-IDF matrix is 224,123 rows by ~400,000 columns,
mostly zeros. Clustering algorithms (HDBSCAN, k-means) compute pairwise
distances in this space, which is O(n * d) per query and O(n^2 * d) for
a full pairwise matrix. With d=400,000 and n=224,123, that is intractable.

Truncated SVD (also called Latent Semantic Analysis in the text domain)
projects the sparse matrix into a dense, low-dimensional space while
preserving the *geometric* relationships between rows. Two strings that
shared many character n-grams in the sparse space end up close in the
dense space; two that shared few end up far apart. The dimensionality
default (50) is the upper end of the spec band: low enough that HDBSCAN
is fast, high enough to keep the relevant variance.
"""
from __future__ import annotations

from typing import Union

import numpy as np
from scipy.sparse import csr_matrix, spmatrix
from sklearn.decomposition import TruncatedSVD


DEFAULT_N_COMPONENTS = 50


def reduce_dimensions(
    matrix: Union[csr_matrix, spmatrix],
    n_components: int = DEFAULT_N_COMPONENTS,
) -> np.ndarray:
    """Project *matrix* into a dense ``(n_samples, n_components)`` array.

    The input is expected to be a sparse TF-IDF matrix (CSR or any scipy
    sparse). The output is a dense ``np.ndarray`` of shape
    ``(n_samples, n_components)`` ready for HDBSCAN or k-means.
    """
    svd = TruncatedSVD(n_components=n_components)
    return svd.fit_transform(matrix)
