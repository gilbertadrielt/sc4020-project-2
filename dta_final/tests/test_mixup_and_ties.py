"""Tests for the two things added in the merge: GBA-Mixup and exact tie handling."""
from __future__ import annotations

import numpy as np
import pytest

from dta.cold import gba_mixup
from dta.metrics import (_expected_hit_at_k, _expected_reciprocal_rank,
                         ranking_metrics)


# --------------------------------------------------------------------- mixup


def test_mixup_returns_requested_count_and_stays_in_hull():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8))
    y = rng.normal(size=60)
    groups = np.repeat(np.arange(12), 5)
    Xv, yv = gba_mixup(X, y, groups, n_virtual=40, seed=1)
    assert Xv.shape == (40, 8) and yv.shape == (40,)
    # An interpolation of two training points cannot leave their range.
    assert Xv.min() >= X.min() - 1e-6 and Xv.max() <= X.max() + 1e-6
    assert yv.min() >= y.min() - 1e-6 and yv.max() <= y.max() + 1e-6


def test_mixup_is_deterministic_given_a_seed():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 4)), rng.normal(size=40)
    groups = np.repeat(np.arange(8), 5)
    a = gba_mixup(X, y, groups, 20, seed=7)
    b = gba_mixup(X, y, groups, 20, seed=7)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def test_mixup_without_usable_neighbours_returns_empty():
    X, y = np.zeros((5, 3)), np.zeros(5)
    groups = np.arange(5)  # every group has one member, so no pair can be mixed
    Xv, yv = gba_mixup(X, y, groups, n_virtual=10)
    assert len(Xv) == 0 and len(yv) == 0


# ----------------------------------------------------------------------- ties


def test_expected_hit_at_k_matches_simulation():
    rng = np.random.default_rng(0)
    n_above, n_tied, n_tied_best = 3, 7, 2
    for k in (1, 4, 6, 12):
        draws = []
        for _ in range(20000):
            positions = rng.permutation(n_tied)[:n_tied_best]
            draws.append(1.0 if n_above + positions.min() + 1 <= k else 0.0)
        assert _expected_hit_at_k(n_above, n_tied, n_tied_best, k) == pytest.approx(
            float(np.mean(draws)), abs=0.015)


def test_expected_reciprocal_rank_matches_simulation():
    rng = np.random.default_rng(1)
    n_above, n_tied, n_tied_best = 2, 6, 1
    draws = [1.0 / (n_above + int(rng.permutation(n_tied)[:n_tied_best].min()) + 1)
             for _ in range(20000)]
    assert _expected_reciprocal_rank(n_above, n_tied, n_tied_best) == pytest.approx(
        float(np.mean(draws)), abs=0.01)


def test_constant_predictor_scores_exactly_the_random_row():
    """The point of the exact form: no variance, and equality rather than nearly."""
    groups = np.repeat(np.arange(120), 57)
    y = np.random.default_rng(0).normal(size=len(groups))
    flat = np.zeros(len(groups))
    first = ranking_metrics(groups, y, flat, seed=1)
    second = ranking_metrics(groups, y, flat, seed=999)
    assert first.hit_at_1 == pytest.approx(first.random_hit_at_1, abs=1e-12)
    assert first.mrr == pytest.approx(first.random_mrr, abs=1e-12)
    assert first.hit_at_1 == pytest.approx(1 / 57, abs=1e-12)
    assert first.hit_at_1 == second.hit_at_1  # identical across seeds


def test_perfect_and_inverted_rankings_are_unchanged_by_the_new_code():
    groups = np.repeat(np.arange(30), 10)
    y = np.random.default_rng(2).normal(size=len(groups))
    assert ranking_metrics(groups, y, y).hit_at_1 == pytest.approx(1.0)
    assert ranking_metrics(groups, y, -y).hit_at_1 == pytest.approx(0.0)
