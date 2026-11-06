"""Correctness tests for the validation grid search.

The property that matters is the boring one: the configuration that wins is the
one that scored best on the fold it was given, and the model handed back is
that configuration's fitted model rather than whichever happened to be fitted
last. Those are the two ways a search quietly reports the wrong thing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import cf, tuning
from tests.test_cf import low_rank_frame


def test_grid_is_the_cartesian_product():
    out = tuning.grid(a=[1, 2], b=["x", "y", "z"])
    assert len(out) == 6
    assert {"a": 1, "b": "x"} in out
    assert {"a": 2, "b": "z"} in out


def test_grid_with_no_axes_is_one_empty_configuration():
    assert tuning.grid() == [{}]


@pytest.fixture
def folds():
    frame = low_rank_frame(n_drugs=25, n_targets=20, k=3)
    rng = np.random.default_rng(2)
    draw = rng.random(len(frame))
    return (frame[draw < 0.7].reset_index(drop=True),
            frame[draw >= 0.7].reset_index(drop=True))


def test_tune_returns_the_configuration_that_scored_best(folds):
    train, val = folds
    result = tuning.tune(
        lambda **p: cf.BiasedMF(n_epochs=20, **p),
        {"n_factors": [1, 3], "reg": [0.05, 2.0]},
        train, val, model_name="mf", dataset="synthetic",
        select_on="ci", verbose=False,
    )
    assert len(result.table) == 4
    # The table is sorted best-first, and best_params must agree with row 0.
    top = result.table.iloc[0]
    assert result.best_params["n_factors"] == top["n_factors"]
    assert result.best_params["reg"] == top["reg"]
    assert top["ci"] == result.table["ci"].max()


def test_the_returned_model_is_the_winner_not_the_last_one_fitted(folds):
    """The failure mode this guards: returning ``models[-1]`` after the loop."""
    train, val = folds
    result = tuning.tune(
        lambda **p: cf.BiasedMF(n_epochs=20, **p),
        {"n_factors": [3], "reg": [0.05, 50.0]},  # heavy reg is clearly worse
        train, val, model_name="mf", select_on="ci", verbose=False,
    )
    assert result.best.reg == result.best_params["reg"]
    # Re-scoring the returned model on val must reproduce the winning score.
    from src.metrics import concordance_index
    assert concordance_index(val["score"], result.best.predict(val)) == pytest.approx(
        result.best_score
    )


def test_lower_is_better_metrics_sort_the_other_way(folds):
    train, val = folds
    result = tuning.tune(
        lambda **p: cf.BiasedMF(n_epochs=20, **p),
        {"n_factors": [1, 3], "reg": [0.05, 20.0]},
        train, val, model_name="mf", select_on="mse", verbose=False,
    )
    assert result.table.iloc[0]["mse"] == result.table["mse"].min()
    assert result.best_score == result.table["mse"].min()


def test_sensitivity_reports_how_many_configs_were_effectively_tied(folds):
    train, val = folds
    result = tuning.tune(
        lambda **p: cf.BiasOnly(**p),
        {"reg": [1.0, 1.01, 1.02, 500.0]},
        train, val, model_name="bias", select_on="ci", verbose=False,
    )
    close = result.sensitivity(within=0.005)
    assert 1 <= len(close) <= 4
    assert len(result.sensitivity(within=0.0)) >= 1
    # A huge regularisation is a genuinely different model, so a tight band
    # must not sweep it in with the three near-identical ones.
    assert len(close) < 4


def test_leaderboard_has_one_row_per_search(folds):
    train, val = folds
    results = [
        tuning.tune(lambda **p: cf.BiasOnly(**p), {"reg": [1.0, 5.0]},
                    train, val, model_name="bias", dataset="d1",
                    select_on="ci", verbose=False),
        tuning.tune(lambda **p: cf.BiasedMF(n_epochs=10, **p), {"n_factors": [2]},
                    train, val, model_name="mf", dataset="d1",
                    select_on="ci", verbose=False),
    ]
    board = tuning.leaderboard(results)
    assert len(board) == 2
    assert board.index.names == ["dataset", "model"]
    assert board["configs_tried"].tolist() == [2, 1]


# --------------------------------------------------------------------------
# early stopping: the epoch count read off the curve instead of guessed
# --------------------------------------------------------------------------

def test_early_stopping_restores_the_best_state_not_the_last(folds):
    train, val = folds
    model = cf.BiasedMF(n_factors=3, reg=0.05, n_epochs=60, patience=4).fit(train, val=val)

    assert model.best_epoch is not None
    assert model.best_epoch <= len(model.history)
    # The restored state must score what the best epoch scored, not what the
    # final epoch did.
    assert model._monitor_score(val) == pytest.approx(min(model.history), abs=1e-9)


def test_early_stopping_halts_once_the_monitor_stops_improving(folds, monkeypatch):
    """The stopping rule, checked against a scripted validation curve.

    ALS with ridge regularisation converges monotonically, so a model fitted on
    well-behaved data can legitimately improve for every epoch of a long run -
    on the synthetic fixtures here it does, even with 20 factors fitted to
    rank-3 data at high noise. Feeding the rule a curve that turns upward tests
    the mechanism instead of hoping the data happens to overfit.
    """
    train, val = folds
    curve = iter([5.0, 4.0, 3.0, 3.5, 3.6, 3.7, 3.8, 3.9])
    model = cf.BiasedMF(n_factors=2, reg=0.5, n_epochs=50, patience=3)
    monkeypatch.setattr(model, "_monitor_score", lambda frame: next(curve))
    model.fit(train, val=val)

    assert model.best_epoch == 3, "the minimum of the curve is its third point"
    assert len(model.history) == 6, "three improving epochs, then three of patience"


def test_epochs_never_exceed_the_best_by_more_than_the_patience(folds):
    """The invariant that holds whether or not the curve ever turns."""
    for patience in (2, 5):
        model = cf.BiasedMF(n_factors=3, reg=0.05, n_epochs=40,
                            patience=patience).fit(*folds[:1], val=folds[1])
        assert len(model.history) <= model.best_epoch + patience


def test_without_a_validation_fold_it_runs_every_epoch(folds):
    train, _ = folds
    model = cf.BiasedMF(n_factors=3, reg=0.05, n_epochs=7).fit(train)
    assert model.history == []
    assert model.best_epoch is None


def test_monitor_must_be_a_known_metric(folds):
    train, val = folds
    with pytest.raises(ValueError, match="monitor must be"):
        cf.BiasedMF(n_factors=2, monitor="accuracy").fit(train, val=val)
