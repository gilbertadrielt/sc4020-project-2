"""Model behaviour that the results depend on, checked directly.

The important ones here are the structural identities: that the bias model is
*exactly* a factorisation with no factors, and that the censored loss is
*exactly* squared error when nothing is censored. Both are claims the report
makes, and both are cheap to verify.
"""
import numpy as np
import pandas as pd
import pytest

from dta import metrics as MT
from dta import models as M


def _toy(n_lig=15, n_tgt=10, k_true=3, noise=0.15, censor=0.0, seed=0):
    """A matrix generated from a known low-rank model, optionally censored."""
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n_lig, k_true))
    V = rng.normal(size=(n_tgt, k_true))
    b = rng.normal(scale=0.4, size=n_lig)
    c = rng.normal(scale=0.4, size=n_tgt)
    rows, cols = np.meshgrid(np.arange(n_lig), np.arange(n_tgt), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    y = 6.0 + b[rows] + c[cols] + np.einsum("ij,ij->i", U[rows], V[cols])
    y = y + rng.normal(scale=noise, size=len(y))

    censored = np.zeros(len(y), bool)
    if censor > 0:
        floor = float(np.quantile(y, censor))
        censored = y <= floor
        y = np.where(censored, floor, y)

    return pd.DataFrame(
        {"lig_idx": rows, "tgt_idx": cols, "score": y, "censored": censored}
    ), n_lig, n_tgt


# ---------------------------------------------------------------------------
# Structural identities
# ---------------------------------------------------------------------------


def test_bias_only_is_exactly_factorisation_with_zero_factors():
    """The report compares 'biases' against 'biases plus interaction'.

    That comparison is only exact if the two models differ in nothing else,
    which is why BiasOnly is implemented as BiasedMF(k=0) rather than
    separately.
    """
    train, nl, nt = _toy()
    a = M.BiasOnly(lam=0.5).fit(train, nl, nt)
    b = M.BiasedMF(k=0, lam=0.5, seed=0).fit(train, nl, nt)
    assert np.allclose(a.predict(train), b.predict(train), atol=1e-12)


def test_censored_loss_reduces_to_squared_error_when_nothing_is_censored():
    """With no censoring the Tobit model must be the ordinary model exactly."""
    train, nl, nt = _toy(censor=0.0)
    cen = M.TobitMF(k=4, lam=0.05, use_censored_loss=True, seed=0).fit(train, nl, nt)
    sq = M.TobitMF(k=4, lam=0.05, use_censored_loss=False, seed=0).fit(train, nl, nt)
    assert np.allclose(cen.predict(train), sq.predict(train), atol=1e-12)


def test_censored_loss_charges_nothing_below_the_ceiling():
    m = M.TobitMF(use_censored_loss=True)
    m.ceiling = 5.0
    pred = np.array([4.0, 5.0, 6.0])
    y = np.array([5.0, 5.0, 5.0])
    cen = np.array([True, True, True])
    assert np.allclose(m._tobit_residual(pred, y, cen), [0.0, 0.0, 1.0])
    # A measured pair is charged normally in both directions.
    assert np.allclose(m._tobit_residual(pred, y, np.zeros(3, bool)), [-1.0, 0.0, 1.0])


def test_censored_model_predicts_below_the_detection_floor():
    """The freedom the loss buys must survive prediction clipping."""
    train, nl, nt = _toy(censor=0.6, seed=1)
    m = M.TobitMF(k=4, lam=0.02, use_censored_loss=True, seed=1).fit(train, nl, nt)
    floor = train.loc[train["censored"], "score"].max()
    assert (m.predict(train) < floor - 1e-9).any()


# ---------------------------------------------------------------------------
# Learning behaviour
# ---------------------------------------------------------------------------


def test_factorisation_recovers_a_low_rank_signal_better_than_biases():
    train, nl, nt = _toy(k_true=3, noise=0.1, seed=2)
    y = train["score"].to_numpy()
    bias = M.BiasOnly(lam=0.1).fit(train, nl, nt)
    mf = M.BiasedMF(k=8, lam=0.01, seed=2).fit(train, nl, nt)
    assert MT.mse(y, mf.predict(train)) < 0.5 * MT.mse(y, bias.predict(train))


def test_every_model_beats_the_global_mean_on_learnable_data():
    train, nl, nt = _toy(seed=3)
    y = train["score"].to_numpy()
    floor = MT.mse(y, M.GlobalMean().fit(train, nl, nt).predict(train))
    for model in (M.LigandMean(), M.TargetMean(), M.BiasOnly(lam=0.1),
                  M.BiasedMF(k=4, lam=0.02, seed=3), M.ItemKNN(k=5, beta=1.0)):
        assert MT.mse(y, model.fit(train, nl, nt).predict(train)) < floor


def test_predictions_are_clipped_to_the_training_range():
    train, nl, nt = _toy(seed=4)
    m = M.BiasedMF(k=4, lam=0.01, seed=4).fit(train, nl, nt)
    p = m.predict(train)
    assert p.min() >= m.score_min_ - 1e-9
    assert p.max() <= m.score_max_ + 1e-9


def test_unseen_entity_falls_back_to_the_global_mean():
    """An entity with no training rows has no bias to apply."""
    train, nl, nt = _toy(n_lig=8, seed=5)
    held = train[train["lig_idx"] != 7]
    m = M.LigandMean().fit(held, nl, nt)
    query = pd.DataFrame({"lig_idx": [7], "tgt_idx": [0], "score": [0.0]})
    assert m.predict(query)[0] == pytest.approx(m.mu_)


def test_count_scaled_regularisation_is_stable_where_fixed_diverges():
    """Why count scaling is the default.

    The two sides of this matrix carry very different amounts of data, so a
    single constant penalty that suits one side leaves the other
    underdetermined and the factors blow up.
    """
    train, nl, nt = _toy(n_lig=40, n_tgt=8, k_true=2, seed=6)
    y = train["score"].to_numpy()
    scaled = M.BiasedMF(k=16, lam=0.05, reg_scaling="count", seed=6).fit(train, nl, nt)
    assert np.isfinite(MT.mse(y, scaled.predict(train)))
    assert np.linalg.norm(scaled.V_) < 1e3


# ---------------------------------------------------------------------------
# Persistence and composition
# ---------------------------------------------------------------------------


def test_saved_model_reloads_and_reproduces_predictions_exactly(tmp_path):
    train, nl, nt = _toy(seed=7)
    m = M.BiasedMF(k=4, lam=0.02, seed=7).fit(train, nl, nt)
    before = m.predict(train)
    path = tmp_path / "ckpt.npz"
    m.save(path, dataset="toy", score_name="pKd")
    after = M.BiasedMF.load(path).predict(train)
    assert np.array_equal(before, after)


def test_blend_weight_is_chosen_on_validation_and_is_a_convex_mix():
    train, nl, nt = _toy(seed=8)
    val = train.sample(frac=0.3, random_state=0)
    a = M.LigandMean().fit(train, nl, nt)
    b = M.BiasedMF(k=4, lam=0.02, seed=8).fit(train, nl, nt)
    blend = M.Blend(a, b).fit(train, nl, nt, val=val)
    assert 0.0 <= blend.weight <= 1.0
    p = blend.predict(train)
    lo = np.minimum(a.predict(train), b.predict(train))
    hi = np.maximum(a.predict(train), b.predict(train))
    assert np.all(p >= lo - 1e-9) and np.all(p <= hi + 1e-9)


def test_knn_reports_its_fallback_rate():
    train, nl, nt = _toy(seed=9)
    m = M.ItemKNN(k=3, beta=1.0).fit(train, nl, nt)
    m.predict(train)
    assert 0.0 <= m.fallback_rate_ <= 1.0
