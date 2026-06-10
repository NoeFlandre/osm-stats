"""Character-level TF-IDF feature extraction for OSM tag strings.

The blog pipeline clusters OSM ``(key, value)`` pairs that mean the same
thing but were mapped slightly differently - typos, plurals, language
variants. Word-level TF-IDF treats ``landuse|farmland`` and ``lanuse|farmland``
as completely different documents because no whole token matches. Character
n-grams fix that: ``lan``, ``and``, ``ndu`` etc. are shared between the two.

This module is a thin wrapper around :class:`sklearn.feature_extraction.text.TfidfVectorizer`
configured for character n-grams. Defaults follow the blog spec:

* ``analyzer='char'``
* ``ngram_range=(3, 5)``
* ``min_df=2``

The function returns the sparse TF-IDF matrix plus the vocabulary mapping
(``ngram_string -> column_index``) so downstream clustering can inspect
which sub-strings drive each cluster.
"""
from __future__ import annotations

from typing import Dict, Hashable, Tuple, Union

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def build_char_tfidf_matrix(
    features: Union[pd.Series, list],
    ngram_range: Tuple[int, int] = (3, 5),
    min_df: int = 2,
) -> Tuple[csr_matrix, Dict[Hashable, int]]:
    """Fit and transform *features* into a sparse char-level TF-IDF matrix.

    Returns ``(matrix, vocabulary)`` where *vocabulary* maps each surviving
    n-gram to its column index in the matrix.
    """
    if isinstance(features, pd.Series):
        corpus = features.tolist()
    else:
        corpus = list(features)

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        min_df=min_df,
    )
    matrix = vectorizer.fit_transform(corpus)
    return matrix, vectorizer.vocabulary_
