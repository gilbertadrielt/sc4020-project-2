"""Metrics are verified against definitions, not against previous output.

The concordance index is the one most worth testing: it is computed through
Kendall's tau-b for speed, and that derivation is easy to get subtly wrong in
the presence of ties, which this data is full of.  Every test here compares
the fast path against the O(n^2) definition or against an exact closed form.
"""
import numpy as np
import pytest

from dta import metrics as M


# ---------------------------------------------------------------------------
# Concordance index
# ---------------------------------------------------------------------------


def test_ci_matches_bruteforce_on_random_data():
    rng = np.random.default_rng(0)
    for _ in range(25):
        n = int(rng.integers(5, 120))
        y = rng.normal(size=n)
        p = rng.normal(size=n)
        assert M.concordance_index(y, p) == pytest.approx(
            M.concordance_index_bruteforce(y, p), abs=1e-9
        )


def test_ci_matches_bruteforce_with_heavy_ties():
    """The realistic case: most labels share one censored value."""
    rng = np.random.default_rng(1)
    for _ in range(25):
        n = int(rng.integers(10, 150))
        y = np.where(rng.random(n) < 0.7, 5.0, rng.uniform(5, 10, n))
        p = np.round(rng.normal(size=n), 1)  # deliberate ties in prediction too
        assert M.concordance_index(y, p) == pytest.approx(
            M.concordance_index_bruteforce(y, p), abs=1e-9
        )


def test_ci_perfect_and_reversed():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert M.concordance_index(y, y) == pytest.approx(1.0)
    assert M.concordance_index(y, -y) == pytest.approx(0.0)


def test_ci_constant_prediction_is_exactly_half():
    """The cold-start failure mode. scipy gives NaN here; 0.5 is correct.

    A model that predicts one constant ties every comparable pair, so it earns
    half credit on all of them. Returning NaN would hide the whole finding.
    """
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert M.concordance_index(y, np.full(5, 7.0)) == 0.5


def test_ci_undefined_when_all_truth_tied():
    assert np.isnan(M.concordance_index(np.full(6, 3.0), np.arange(6.0)))


def test_ci_is_invariant_to_monotone_rescaling():
    rng = np.random.default_rng(2)
    y, p = rng.normal(size=60), rng.normal(size=60)
    base = M.concordance_index(y, p)
    assert M.concordance_index(y, 3.5 * p + 11.0) == pytest.approx(base, abs=1e-12)
    assert M.concordance_index(y, np.exp(p)) == pytest.approx(base, abs=1e-12)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def test_random_hit_at_k_closed_form_matches_simulation():
    rng = np.random.default_rng(3)
    for n, m, k in [(10, 1, 1), (10, 1, 5), (20, 3, 5), (7, 2, 3), (50, 1, 10)]:
        exact = M._random_hit_at_k(n, m, k)
        best = np.zeros(n, dtype=bool)
        best[:m] = True
        sim = np.mean([
            bool(M._rank_desc(rng.random(n), rng)[best].min() < k) for _ in range(6000)
        ])
        assert exact == pytest.approx(sim, abs=0.02)


def test_random_hit_at_k_edge_cases():
    assert M._random_hit_at_k(5, 1, 5) == 1.0   # k covers everything
    assert M._random_hit_at_k(5, 5, 1) == 1.0   # every item is best
    assert M._random_hit_at_k(100, 1, 1) == pytest.approx(0.01)


def test_perfect_ranker_scores_one():
    g = np.repeat([0, 1, 2], 8)
    y = np.tile(np.arange(8.0), 3)
    r = M.ranking_metrics(g, y, y, seed=0)
    assert r.hit_at_1 == pytest.approx(1.0)
    assert r.mrr == pytest.approx(1.0)


def test_constant_predictor_matches_its_own_random_baseline():
    """A model that cannot rank should score exactly like random, not worse.

    This is what the random tie-breaking convention buys, and it is why the
    drug-mean baseline reads as 'equal to random' instead of 'worse than'.
    """
    rng = np.random.default_rng(4)
    g = np.repeat(np.arange(300), 6)
    y = rng.normal(size=g.size)
    r = M.ranking_metrics(g, y, np.zeros_like(y), seed=7)
    assert r.hit_at_1 == pytest.approx(r.random_hit_at_1, abs=0.03)
    assert r.mrr == pytest.approx(r.random_mrr, abs=0.03)


def test_groups_with_all_tied_truth_are_skipped_not_scored():
    g = np.array([0, 0, 0, 1, 1, 1])
    y = np.array([5.0, 5.0, 5.0, 1.0, 2.0, 3.0])
    r = M.ranking_metrics(g, y, np.array([1.0, 2, 3, 1, 2, 3]), seed=0)
    assert r.n_groups_scored == 1
    assert r.n_groups_skipped == 1


def test_tied_best_counts_as_hit_if_any_is_in_top_k():
    # Two candidates share the best true score; the model puts one first.
    g = np.zeros(4, dtype=int)
    y = np.array([9.0, 9.0, 1.0, 2.0])
    p = np.array([0.0, 5.0, 1.0, 2.0])
    assert M.ranking_metrics(g, y, p, seed=0).hit_at_1 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Grouped CI
# ---------------------------------------------------------------------------


def test_grouped_ci_averages_per_group_not_pooled():
    g = np.array([0, 0, 0, 1, 1, 1])
    y = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 3.0, 3.0, 2.0, 1.0])  # group 0 perfect, group 1 reversed
    assert M.grouped_concordance_index(g, y, p)["mean_ci"] == pytest.approx(0.5)


def test_per_ligand_ci_is_exactly_half_for_a_target_blind_model():
    """The Task 3 result, as a unit test.

    A model of the form mu + b_drug gives every target of a given drug the
    same score, so per-ligand CI must be exactly 0.5 whatever the biases are.
    """
    rng = np.random.default_rng(5)
    n_drugs, n_targets = 40, 12
    g = np.repeat(np.arange(n_drugs), n_targets)
    y = rng.normal(size=g.size)
    drug_bias = rng.normal(size=n_drugs)
    pred = drug_bias[g]  # depends on the drug only, never on the target
    assert M.grouped_concordance_index(g, y, pred)["mean_ci"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Error and uncertainty
# ---------------------------------------------------------------------------


def test_masked_mse_subsets_correctly():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.0, 2.0, 0.0, 0.0])
    assert M.masked_mse(y, p, np.array([True, True, False, False])) == pytest.approx(0.0)
    assert M.masked_mse(y, p, np.array([False, False, True, True])) == pytest.approx(12.5)
    assert np.isnan(M.masked_mse(y, p, np.zeros(4, bool)))


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(6)
    g = np.repeat(np.arange(60), 5)
    y = rng.normal(size=g.size)
    p = y + rng.normal(scale=0.5, size=g.size)
    point = M.mse(y, p)
    lo, hi = M.bootstrap_interval(g, y, p, M.mse, n_boot=300, seed=0)
    assert lo < point < hi


def test_bootstrap_resamples_groups_not_rows():
    """Resampling rows would understate the interval; groups are the unit."""
    rng = np.random.default_rng(7)
    g = np.repeat(np.arange(25), 20)
    y = rng.normal(size=g.size)
    p = y + rng.normal(scale=0.5, size=g.size)
    lo, hi = M.bootstrap_interval(g, y, p, M.mse, n_boot=300, seed=0)
    assert hi - lo > 0
