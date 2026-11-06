"""Correctness tests for the shared analysis helpers.

    python -m pytest tests/ -q

``masked_corr`` is vectorised for speed, so it is checked against a plain
double loop over ``scipy.stats.pearsonr``. The closed form it uses is the one
that loses precision on near-constant profiles, which is exactly what KIBA has,
so the degenerate cases are pinned down here rather than discovered in a figure.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.stats import ConstantInputWarning, pearsonr

from src.eda import kmer_profiles, masked_corr

MIN_OVERLAP = 10


def brute_force(x: np.ndarray, min_overlap: int = MIN_OVERLAP) -> np.ndarray:
    """Reference implementation: one pair at a time, no algebra.

    A constant row's standard deviation is not exactly 0 in floating point, so
    ``pearsonr`` is reached and correctly warns that the correlation is
    undefined before returning NaN. That is the right answer, so the warning is
    suppressed rather than worked around - suppressing it here keeps the
    reference independent of the implementation under test.
    """
    n = len(x)
    out = np.full((n, n), np.nan)
    np.fill_diagonal(out, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        for i in range(n):
            for j in range(i + 1, n):
                both = ~np.isnan(x[i]) & ~np.isnan(x[j])
                a, b = x[i, both], x[j, both]
                if both.sum() < min_overlap or a.std() == 0 or b.std() == 0:
                    continue
                out[i, j] = out[j, i] = pearsonr(a, b).statistic
    return out


@pytest.fixture
def sparse_matrix() -> np.ndarray:
    rng = np.random.default_rng(0)
    x = rng.normal(11.5, 1.2, size=(40, 60))
    x[rng.random(x.shape) < 0.40] = np.nan
    x[7, :] = 11.2  # constant and fully observed: correlation is undefined
    x[8, :20] = np.nan  # too little overlap with some rows
    return x


def test_matches_brute_force(sparse_matrix):
    fast = masked_corr(sparse_matrix, min_overlap=MIN_OVERLAP)
    slow = brute_force(sparse_matrix)
    both_defined = np.isfinite(fast) & np.isfinite(slow)
    assert both_defined.sum() > 500, "test matrix produced too few usable pairs"
    np.testing.assert_allclose(fast[both_defined], slow[both_defined], atol=1e-9)


def test_agrees_on_which_pairs_are_undefined(sparse_matrix):
    fast = masked_corr(sparse_matrix, min_overlap=MIN_OVERLAP)
    slow = brute_force(sparse_matrix)
    np.testing.assert_array_equal(np.isnan(fast), np.isnan(slow))


def test_constant_profile_is_nan_not_a_number(sparse_matrix):
    """Row 7 is constant, so every correlation with it is undefined.

    Its variance does not cancel to exactly zero in floating point, so a
    ``var > 0`` guard would let a meaningless value through.
    """
    fast = masked_corr(sparse_matrix, min_overlap=MIN_OVERLAP)
    off_diagonal = [v for k, v in enumerate(fast[7]) if k != 7]
    assert np.isnan(off_diagonal).all()


def test_low_overlap_is_dropped():
    x = np.full((3, 20), np.nan)
    x[0, :15] = np.arange(15)
    x[1, :15] = np.arange(15) * 2.0  # 15 shared, perfectly correlated
    x[2, :4] = np.arange(4)          # only 4 shared with row 0
    fast = masked_corr(x, min_overlap=MIN_OVERLAP)
    assert fast[0, 1] == pytest.approx(1.0)
    assert np.isnan(fast[0, 2])


@pytest.mark.parametrize("scale", [0.02, 1.2, 100.0])
def test_stays_within_unit_range(scale):
    """Tight variance around a large mean is the cancellation-prone case."""
    rng = np.random.default_rng(1)
    x = rng.normal(11.2, scale, size=(30, 50))
    x[rng.random(x.shape) < 0.3] = np.nan
    fast = masked_corr(x, min_overlap=MIN_OVERLAP)
    finite = fast[np.isfinite(fast)]
    assert finite.min() >= -1.0
    assert finite.max() <= 1.0


def test_is_symmetric_with_unit_diagonal(sparse_matrix):
    fast = masked_corr(sparse_matrix, min_overlap=MIN_OVERLAP)
    np.testing.assert_array_equal(np.diag(fast), np.ones(len(fast)))
    np.testing.assert_allclose(fast, fast.T, equal_nan=True)


# --------------------------------------------------------------------------
# kmer_profiles: the protein representation Task 3 falls back to, so its
# column indexing and its handling of the X residue are pinned down here.
# --------------------------------------------------------------------------

def test_kmer_counts_land_in_the_right_columns():
    """``AAAAC`` holds AAA twice and AAC once, and nothing else.

    Rows come back L2-normalised, so the counts are checked as a ratio: the
    absolute scale is the normalisation's business, the columns are this
    function's.
    """
    out = kmer_profiles(["AAAAC"], k=3)
    aaa = 0  # A is index 0, so AAA is column 0
    aac = 0 * 400 + 0 * 20 + 1  # C is index 1
    assert np.count_nonzero(out[0]) == 2
    assert out[0, aaa] / out[0, aac] == pytest.approx(2.0)
    np.testing.assert_allclose(out[0, [aaa, aac]], np.array([2.0, 1.0]) / np.sqrt(5), atol=1e-6)


def test_windows_touching_an_unknown_residue_are_skipped():
    """DAVIS and KIBA contain X, which must not be mapped onto a real residue.

    ``AAXAA`` has three windows and every one of them contains the X, so the
    row is empty - not quietly counted as some other k-mer.
    """
    assert np.count_nonzero(kmer_profiles(["AAXAA"], k=3)) == 0
    # The flanking windows of a longer sequence still survive.
    kept = kmer_profiles(["AAAXCCC"], k=3)
    assert np.count_nonzero(kept) == 2  # AAA and CCC


def test_rows_are_unit_length_so_a_dot_product_is_a_cosine():
    seqs = ["ACDEFGHIKLMNPQRSTVWY" * 3, "AAACCCGGGTTTAAACCCGG", "MKVLAAGIVGLNLGGK"]
    out = kmer_profiles(seqs, k=3)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
    cosine = out @ out.T
    np.testing.assert_allclose(np.diag(cosine), 1.0, atol=1e-6)
    assert cosine.min() >= -1e-6


def test_identical_sequences_are_perfectly_similar_and_distinct_ones_are_not():
    seq = "MKVLAAGIVGLNLGGKMKVLAAGIVGLNLGGK"
    out = kmer_profiles([seq, seq, "WWWWWWWWWWWWWWWW"], k=3)
    assert out[0] @ out[1] == pytest.approx(1.0, abs=1e-6)
    assert out[0] @ out[2] == pytest.approx(0.0, abs=1e-6)


def test_sequence_shorter_than_k_is_an_empty_row_not_a_crash():
    out = kmer_profiles(["AA", "AAAA"], k=3)
    assert np.count_nonzero(out[0]) == 0
    assert np.count_nonzero(out[1]) == 1  # AAA twice over, so one column
    assert out.shape == (2, 20**3)
