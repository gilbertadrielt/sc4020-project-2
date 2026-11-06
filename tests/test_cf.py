"""Correctness tests for the Task 2 models.

Three things are worth pinning down. That the factorisation actually recovers
structure, rather than fitting biases and calling it a day. That a saved model
reloads to identical predictions, which the brief requires and which is easy to
get subtly wrong with id-to-index maps. And that an unseen target degrades to
the per-drug mean rather than to a crash or a silent zero - that behaviour is
the baseline Task 3 is measured against, so it has to be exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import cf, metrics


def low_rank_frame(n_drugs=30, n_targets=24, k=3, noise=0.05, seed=0) -> pd.DataFrame:
    """A matrix that genuinely is ``mu + b + c + u.v``, so MF should recover it."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n_drugs, k))
    v = rng.normal(size=(n_targets, k))
    b = rng.normal(scale=0.5, size=n_drugs)
    c = rng.normal(scale=0.5, size=n_targets)
    rows = []
    for i in range(n_drugs):
        for j in range(n_targets):
            score = 7.0 + b[i] + c[j] + u[i] @ v[j] + rng.normal(scale=noise)
            rows.append({"drug_id": f"d{i}", "target_id": f"t{j}", "score": score})
    return pd.DataFrame(rows)


@pytest.fixture
def data():
    frame = low_rank_frame()
    rng = np.random.default_rng(1)
    held = rng.random(len(frame)) < 0.2
    return frame[~held].reset_index(drop=True), frame[held].reset_index(drop=True)


def test_factorisation_beats_bias_only_on_a_low_rank_matrix(data):
    """If the interaction term is real, the model using it must do better."""
    train, test = data
    bias = cf.BiasOnly().fit(train)
    mf = cf.BiasedMF(n_factors=3, reg=0.05, n_epochs=25).fit(train)

    bias_mse = metrics.mse(test["score"], bias.predict(test))
    mf_mse = metrics.mse(test["score"], mf.predict(test))
    assert mf_mse < bias_mse / 2, f"MF {mf_mse:.4f} did not clearly beat bias {bias_mse:.4f}"


def test_factorisation_converges(data):
    """Validation error must fall over epochs, not wander."""
    train, test = data
    mf = cf.BiasedMF(n_factors=3, reg=0.05, n_epochs=15).fit(train, val=test)
    assert mf.history[-1] < mf.history[0]
    assert mf.history[-1] == pytest.approx(min(mf.history), rel=0.2)


def test_baselines_are_ordered_as_expected(data):
    """Each baseline should use strictly more information than the last."""
    train, test = data
    errors = {}
    for model in (cf.GlobalMean(), cf.DrugMean(), cf.BiasOnly()):
        model.fit(train)
        errors[model.name] = metrics.mse(test["score"], model.predict(test))
    assert errors["global mean"] > errors["per-drug mean"] > errors["bias only"]


def test_a_model_is_named_before_it_is_fitted():
    """Results tables key on the name, so it cannot depend on fit() running."""
    for model in (cf.GlobalMean(), cf.DrugMean(), cf.TargetMean(), cf.BiasOnly(),
                  cf.BiasedMF(), cf.ItemKNN()):
        assert model.name and model.name != "model"


def test_global_mean_predicts_the_training_mean(data):
    train, test = data
    model = cf.GlobalMean().fit(train)
    assert np.allclose(model.predict(test), train["score"].mean())


# --------------------------------------------------------------------------
# unseen entities: the Task 3 starting line
# --------------------------------------------------------------------------

def test_an_unseen_target_falls_back_to_the_per_drug_prediction(data):
    """No c_j and no v_j, so the prediction is mu + b_i - for every new target.

    This is why Task 3 exists: the fallback gives one value per drug, so every
    cold target gets the same score and their ranking is arbitrary.
    """
    train, _ = data
    mf = cf.BiasedMF(n_factors=3, reg=0.05, n_epochs=10).fit(train)
    cold = pd.DataFrame({
        "drug_id": ["d0", "d0", "d1"],
        "target_id": ["brand_new", "also_new", "brand_new"],
        "score": [7.0, 8.0, 6.0],
    })
    pred = mf.predict(cold)
    assert pred[0] == pytest.approx(pred[1]), "two unseen targets must score alike"
    assert pred[0] != pytest.approx(pred[2]), "different drugs still differ"
    assert pred[0] == pytest.approx(mf.mu + mf.b[mf.drug_index["d0"]])


def test_an_unseen_drug_and_target_fall_back_to_the_global_mean(data):
    train, _ = data
    model = cf.BiasOnly().fit(train)
    unknown = pd.DataFrame({"drug_id": ["nobody"], "target_id": ["nothing"], "score": [7.0]})
    assert model.predict(unknown)[0] == pytest.approx(model.mu)


def test_knn_reports_how_often_it_had_no_neighbour(data):
    train, test = data
    knn = cf.ItemKNN(k=5).fit(train)
    knn.predict(test)
    assert 0.0 <= knn.fallback_rate <= 1.0


def test_knn_falls_back_rather_than_failing_on_an_unseen_target(data):
    train, _ = data
    knn = cf.ItemKNN(k=5).fit(train)
    cold = pd.DataFrame({"drug_id": ["d0"], "target_id": ["brand_new"], "score": [7.0]})
    assert np.isfinite(knn.predict(cold)).all()
    assert knn.fallback_rate == pytest.approx(1.0)


# --------------------------------------------------------------------------
# persistence: the brief requires a reloaded checkpoint to predict identically
# --------------------------------------------------------------------------

def test_saved_model_reloads_to_identical_predictions(data, tmp_path):
    train, test = data
    mf = cf.BiasedMF(n_factors=3, reg=0.05, n_epochs=10).fit(train)
    path = mf.save(tmp_path / "model.npz")

    reloaded = cf.BiasedMF.load(path)
    np.testing.assert_allclose(reloaded.predict(test), mf.predict(test), rtol=0, atol=0)
    assert reloaded.n_factors == mf.n_factors
    assert reloaded.mu == pytest.approx(mf.mu)


def test_reloaded_model_still_maps_ids_not_positions(data, tmp_path):
    """A row-order change must not change a prediction.

    The index maps are saved as id arrays, so a reloaded model has to look
    entities up by id. Shuffling the frame is the cheapest way to catch a
    positional shortcut.
    """
    train, test = data
    mf = cf.BiasedMF(n_factors=3, reg=0.05, n_epochs=10).fit(train)
    reloaded = cf.BiasedMF.load(mf.save(tmp_path / "model.npz"))

    shuffled = test.sample(frac=1.0, random_state=3)
    np.testing.assert_allclose(
        reloaded.predict(shuffled),
        mf.predict(test)[shuffled.index.to_numpy()],
    )
