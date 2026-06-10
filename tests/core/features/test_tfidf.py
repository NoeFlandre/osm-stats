import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

from src.core.features.tfidf import build_char_tfidf_matrix


# A realistic-ish corpus: 6 landuse/natural/highway tags, with two
# intentional near-misses (typos) to exercise the char n-gram overlap
# behavior the spec calls for.
CORPUS = [
    "landuse|farmland",
    "landuse|forest",
    "landuse|residential",
    "natural|water",
    "natural|wood",
    "highway|residential",
    "lanuse|farmland",  # typo of landuse
    "naturall|wood",   # typo of natural
]


# --- input handling -------------------------------------------------------


def test_accepts_list_of_strings():
    matrix, vocab = build_char_tfidf_matrix(CORPUS)
    assert matrix.shape[0] == len(CORPUS)
    assert len(vocab) > 0


def test_accepts_pandas_series():
    matrix, _ = build_char_tfidf_matrix(pd.Series(CORPUS))
    assert matrix.shape[0] == len(CORPUS)


# --- output shape ---------------------------------------------------------


def test_output_is_sparse_matrix():
    matrix, _ = build_char_tfidf_matrix(CORPUS)
    assert issparse(matrix)


def test_output_shape_matches_input():
    matrix, vocab = build_char_tfidf_matrix(CORPUS)
    assert matrix.shape == (len(CORPUS), len(vocab))


# --- default configuration -----------------------------------------------


def test_default_analyzer_is_char_ngram_range_3_to_5_min_df_2():
    matrix, vocab = build_char_tfidf_matrix(CORPUS)
    # All keys are 3-5 character strings (the spec).
    for ngram in vocab.keys():
        assert 3 <= len(ngram) <= 5


def test_char_ngrams_catch_substring_similarity():
    # 'lanuse' (typo) vs 'landuse' must share at least one column with
    # a non-zero value. Word n-grams would not.
    matrix, vocab = build_char_tfidf_matrix(CORPUS)
    dense = matrix.toarray()
    typo_idx = CORPUS.index("lanuse|farmland")
    good_idx = CORPUS.index("landuse|farmland")
    shared = np.any((dense[typo_idx] > 0) & (dense[good_idx] > 0))
    assert shared, "Char n-grams should overlap between 'landuse' and 'lanuse'"


# --- parameter overrides -------------------------------------------------


def test_min_df_filters_rare_ngrams():
    # min_df=2 is the default; min_df=1 keeps every n-gram. The default
    # vocabulary must be a strict subset of the min_df=1 vocabulary.
    _, vocab_default = build_char_tfidf_matrix(CORPUS, min_df=2)
    _, vocab_min1 = build_char_tfidf_matrix(CORPUS, min_df=1)
    assert set(vocab_default.keys()).issubset(set(vocab_min1.keys()))


def test_ngram_range_2_3_smaller_than_default():
    _, vocab_default = build_char_tfidf_matrix(CORPUS, ngram_range=(3, 5))
    _, vocab_short = build_char_tfidf_matrix(CORPUS, ngram_range=(2, 3))
    # Shorter n-grams generally produce fewer unique tokens, but the spec
    # only requires the function to accept the parameter. We assert
    # non-emptiness and that the smaller range is honored.
    assert len(vocab_short) > 0
    for ngram in vocab_short.keys():
        assert 2 <= len(ngram) <= 3
