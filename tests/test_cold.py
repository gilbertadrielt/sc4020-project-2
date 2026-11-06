"""Correctness tests for the Task 3 cold-target models.

The thing these models must do, and that Task 2's models cannot, is give two
different unseen targets two different predictions for the same drug. Several
tests below are variations on exactly that. The rest guard the leakage rule:
a cold target must never be scored from a training target that carries its own
sequence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import cf, cold
from src.features import kmer_profiles


def targets_frame() -> pd.DataFrame:
    """Three sequence families, plus a target unlike anything else.

    ``a*`` and ``b*`` are near-identical within family and dissimilar across
    it, so the neighbour rule has an obvious right answer. ``lonely`` shares
    almost no 3-mers with the rest, so it has to trigger the fallback.
    """
    family_a = "MKVLAAGIVGLNLGGK" * 6
    family_b = "PQRSTWYFHDECPQRS" * 6
    rows = [
        ("a1", family_a, "a1"),
        ("a2", family_a + "MKVL", "a2"),
        ("a3", "MKVL" + family_a, "a3"),
        ("b1", family_b, "b1"),
        ("b2", family_b + "PQRS", "b2"),
        ("b3", "PQRS" + family_b, "b3"),
        ("a1_mutant", family_a, "a1"),  # same seq_group as a1: a twin
        ("lonely", "WWWWCCCCWWWWCCCC" * 6, "lonely"),
    ]
    return pd.DataFrame(rows, columns=["target_id", "sequence", "seq_group"])


def pairs_frame(target_ids, n_drugs=8, seed=0) -> pd.DataFrame:
    """Scores where a target's family decides its profile."""
    rng = np.random.default_rng(seed)
    level = {"a": 9.0, "b": 5.5, "l": 7.0}
    rows = []
    for i in range(n_drugs):
        shift = rng.normal(scale=0.3)
        for t in target_ids:
            rows.append({"drug_id": f"d{i}", "target_id": t,
                         "score": level[t[0]] + shift + rng.normal(scale=0.1)})
    return pd.DataFrame(rows)


@pytest.fixture
def setup():
    targets = targets_frame()
    train_ids = ["a1", "a2", "b1", "b2"]
    cold_ids = ["a3", "b3", "lonely"]
    return targets, pairs_frame(train_ids), pairs_frame(cold_ids, seed=1)


def test_a_cold_target_gets_a_target_specific_prediction(setup):
    """The whole point: two unseen targets must not score alike for one drug.

    Task 2's models fail this by construction - they return mu + b_i for both.
    """
    targets, train, test = setup
    model = cold.SequenceNeighbour(targets, k=2, min_cosine=0.1).fit(train)
    pred = model.predict_frame(test)

    one_drug = pred[pred["drug_id"] == "d0"].set_index("target_id")["pred"]
    assert one_drug["a3"] != pytest.approx(one_drug["b3"], abs=1e-6)

    flat = cf.BiasedMF(n_factors=2, n_epochs=10).fit(train).predict_frame(test)
    flat_one = flat[flat["drug_id"] == "d0"].set_index("target_id")["pred"]
    assert flat_one["a3"] == pytest.approx(flat_one["b3"]), "Task 2 baseline changed"


def test_the_prediction_follows_the_right_sequence_family(setup):
    """``a3`` belongs with the high-scoring family, ``b3`` with the low one."""
    targets, train, test = setup
    model = cold.SequenceNeighbour(targets, k=2, min_cosine=0.1).fit(train)
    pred = model.predict_frame(test).groupby("target_id")["pred"].mean()
    assert pred["a3"] > pred["b3"] + 1.0


def test_it_beats_the_per_drug_mean_it_falls_back_to(setup):
    from src.metrics import mse

    targets, train, test = setup
    baseline = cf.DrugMean().fit(train)
    model = cold.SequenceNeighbour(targets, k=2, min_cosine=0.1).fit(train)
    assert mse(test["score"], model.predict(test)) < mse(test["score"], baseline.predict(test))


def test_a_sequence_identical_twin_is_never_borrowed_from(setup):
    """``a1_mutant`` carries a1's sequence, so a1 must not be its neighbour.

    The cold split already keeps whole sequence groups together, so this cannot
    arise there. The rule is enforced in the model anyway, because a split that
    forgot it would otherwise read as a spectacular result.
    """
    targets, train, _ = setup
    model = cold.SequenceNeighbour(targets, k=3, min_cosine=0.0).fit(train)
    neighbours, _ = model._neighbours("a1_mutant", k=3, min_cosine=0.0)
    borrowed = [model._train_ids[i] for i in neighbours]
    assert "a1" not in borrowed
    assert "a2" in borrowed, "it should still find the rest of the family"


def test_a_target_unlike_anything_in_training_falls_back(setup):
    targets, train, _ = setup
    model = cold.SequenceNeighbour(targets, k=3, min_cosine=0.5).fit(train)
    neighbours, _ = model._neighbours("lonely", k=3, min_cosine=0.5)
    assert len(neighbours) == 0

    lonely = pd.DataFrame({"drug_id": ["d0"], "target_id": ["lonely"], "score": [7.0]})
    pred = model.predict(lonely)
    assert pred[0] == pytest.approx(model.mu + model.b[model.drug_index["d0"]])


def test_the_threshold_controls_how_often_it_declines_to_guess(setup):
    """Within-family similarity here is 0.997-1.0 and across-family is 0.

    So a threshold above 0.997 cuts ``a3`` loose while ``b3`` keeps its exact
    match, and one above 1.0 cuts everything.
    """
    targets, train, test = setup
    rates = {}
    for threshold in (0.0, 0.999, 1.01):
        model = cold.SequenceNeighbour(targets, k=3, min_cosine=threshold).fit(train)
        model.predict(test)
        rates[threshold] = model.fallback_rate

    assert rates[0.0] == pytest.approx(1 / 3)    # only `lonely` has no neighbour
    assert rates[0.999] == pytest.approx(2 / 3)  # `a3` drops out too
    assert rates[1.01] == pytest.approx(1.0)     # nothing clears the bar


def test_a_zero_similarity_neighbour_is_not_evidence(setup):
    """Regression: ``min_cosine=0`` must not admit cosine-0 neighbours.

    ``lonely`` shares no 3-mer with any training target, so every similarity
    is exactly 0. Admitting those as neighbours made the weights sum to zero,
    and the weighted mean divided by it - putting NaN into the predictions at
    precisely the setting the threshold ablation uses.
    """
    targets, train, _ = setup
    model = cold.SequenceNeighbour(targets, k=3, min_cosine=0.0).fit(train)

    neighbours, weights = model._neighbours("lonely", k=3, min_cosine=0.0)
    assert len(neighbours) == 0, "a cosine-0 neighbour carries no information"

    frame = pd.DataFrame({"drug_id": ["d0", "d1"], "target_id": ["lonely"] * 2,
                          "score": [7.0, 7.0]})
    assert np.isfinite(model.predict(frame)).all()


def test_no_model_ever_returns_a_non_finite_prediction(setup):
    targets, train, test = setup
    for threshold in (0.0, 0.3, 0.999, 1.01):
        for model in (cold.SequenceNeighbour(targets, k=3, min_cosine=threshold),
                      cold.MFSequenceTransfer(targets, n_factors=2, n_epochs=10,
                                              k=3, min_cosine=threshold)):
            model.fit(train)
            assert np.isfinite(model.predict(test)).all(), f"{model.name} at {threshold}"


def test_warm_targets_are_untouched_by_the_cold_path(setup):
    """Scoring the training targets must match the plain bias model exactly."""
    targets, train, _ = setup
    model = cold.SequenceNeighbour(targets, k=2, min_cosine=0.1, reg=5.0).fit(train)
    bias = cf.BiasOnly(reg=5.0).fit(train)
    np.testing.assert_allclose(model.predict(train), bias.predict(train))


# --------------------------------------------------------------------------
# the parameter-transfer variant
# --------------------------------------------------------------------------

def test_mf_transfer_also_separates_two_cold_targets(setup):
    targets, train, test = setup
    model = cold.MFSequenceTransfer(targets, n_factors=2, n_epochs=20,
                                    k=2, min_cosine=0.1).fit(train)
    pred = model.predict_frame(test).groupby("target_id")["pred"].mean()
    assert pred["a3"] > pred["b3"] + 1.0


def test_mf_transfer_leaves_warm_predictions_to_the_underlying_model(setup):
    targets, train, _ = setup
    model = cold.MFSequenceTransfer(targets, n_factors=2, n_epochs=20,
                                    k=2, min_cosine=0.1).fit(train)
    np.testing.assert_allclose(model.predict(train), model.mf.predict(train))


def test_mf_transfer_needs_no_observed_pair_with_the_neighbour(setup):
    """Why it degrades more gracefully than borrowing scores.

    Score-transfer needs the drug to have been measured against a neighbour.
    Parameter-transfer only needs the neighbour's fitted vector, which exists
    whether or not that particular pair was ever assayed - so on a sparse
    matrix it falls back less often.
    """
    targets, train, test = setup
    sparse = train.sample(frac=0.35, random_state=0).reset_index(drop=True)

    scores = cold.SequenceNeighbour(targets, k=2, min_cosine=0.1).fit(sparse)
    params = cold.MFSequenceTransfer(targets, n_factors=2, n_epochs=20,
                                     k=2, min_cosine=0.1).fit(sparse)
    scores.predict(test)
    params.predict(test)
    assert params.fallback_rate <= scores.fallback_rate


def test_fallback_rate_is_a_proportion(setup):
    targets, train, test = setup
    for model in (cold.SequenceNeighbour(targets, k=2),
                  cold.MFSequenceTransfer(targets, n_factors=2, n_epochs=10, k=2)):
        model.fit(train)
        model.predict(test)
        assert 0.0 <= model.fallback_rate <= 1.0


def test_profiles_are_unit_length_so_similarity_is_a_cosine():
    profiles = kmer_profiles(targets_frame()["sequence"].to_numpy(), k=3)
    np.testing.assert_allclose(np.linalg.norm(profiles, axis=1), 1.0, atol=1e-6)


# --------------------------------------------------------------------------
# rung 3: learning the map from sequence to latent vector
# --------------------------------------------------------------------------

def test_latent_regression_separates_two_cold_targets(setup):
    targets, train, test = setup
    model = cold.LatentRegressor(targets, n_factors=2, n_epochs=20, ridge=0.1).fit(train)
    pred = model.predict_frame(test).groupby("target_id")["pred"].mean()
    assert pred["a3"] > pred["b3"] + 0.5


def test_latent_regression_has_no_threshold_but_still_degrades_gracefully(setup):
    """No rule declines a prediction, yet an orthogonal target still reverts.

    ``a3`` resembles the training families, so it gets a genuine prediction
    rather than the per-drug mean. ``lonely`` shares no 3-mer with anything in
    training, so it is orthogonal to the training span; the dual-form ridge
    then returns exactly zero for its latent vector and bias, and the
    prediction falls back to ``mu + b_i`` without any threshold being set.
    """
    targets, train, _ = setup
    model = cold.LatentRegressor(targets, n_factors=2, n_epochs=20, ridge=0.1).fit(train)
    floor = model.mu + model.b[model.drug_index["d0"]]

    assert model.fallback_rate == 0.0, "it never declines by rule"

    informative = pd.DataFrame({"drug_id": ["d0"], "target_id": ["a3"], "score": [9.0]})
    assert model.predict(informative)[0] != pytest.approx(floor)

    orthogonal = pd.DataFrame({"drug_id": ["d0"], "target_id": ["lonely"], "score": [7.0]})
    v_new, c_new = model._predicted_latents("lonely")
    assert np.allclose(v_new, 0.0) and c_new == pytest.approx(0.0)
    assert model.predict(orthogonal)[0] == pytest.approx(floor)


def test_latent_regression_leaves_warm_predictions_alone(setup):
    targets, train, _ = setup
    model = cold.LatentRegressor(targets, n_factors=2, n_epochs=20).fit(train)
    np.testing.assert_allclose(model.predict(train), model.mf.predict(train))


def test_stronger_ridge_shrinks_the_predicted_vectors(setup):
    targets, train, _ = setup
    loose = cold.LatentRegressor(targets, n_factors=2, n_epochs=20, ridge=0.01).fit(train)
    tight = cold.LatentRegressor(targets, n_factors=2, n_epochs=20, ridge=1000.0).fit(train)
    assert np.linalg.norm(tight.weights) < np.linalg.norm(loose.weights)


# --------------------------------------------------------------------------
# rung 2: no collaborative term at all
# --------------------------------------------------------------------------

def test_content_regression_predicts_from_structure_alone():
    """No collaborative term: a pair is a fingerprint next to a k-mer profile.

    Built larger than the other fixtures on purpose. Gradient boosting will not
    split a node below ``min_child_samples``, so on a handful of rows against
    9,024 feature columns it returns the training mean for everything and the
    test would be asserting nothing.
    """
    targets = targets_frame()
    train_ids, cold_ids = ["a1", "a2", "b1", "b2"], ["a3", "b3"]
    train = pairs_frame(train_ids, n_drugs=40)
    test = pairs_frame(cold_ids, n_drugs=40, seed=1)

    smiles = ["CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "CCBr", "CCF", "CCI"]
    ligands = pd.DataFrame({
        "drug_id": [f"d{i}" for i in range(40)],
        "smiles": [smiles[i % len(smiles)] for i in range(40)],
    })

    model = cold.ContentRegressor(targets, ligands, n_estimators=60).fit(train)
    pred = model.predict_frame(test).groupby("target_id")["pred"].mean()

    assert np.isfinite(pred.to_numpy()).all()
    assert pred["a3"] > pred["b3"], "the high-scoring family should rank higher"


def test_content_regression_refuses_an_id_it_has_no_structure_for(setup):
    targets, train, _ = setup
    ligands = pd.DataFrame({
        "drug_id": sorted(train["drug_id"].unique()),
        "smiles": ["CCO"] * 8,
    })
    model = cold.ContentRegressor(targets, ligands, n_estimators=20).fit(train)
    unknown = pd.DataFrame({"drug_id": ["ghost"], "target_id": ["a3"], "score": [7.0]})
    with pytest.raises(KeyError, match="no structure or sequence"):
        model.predict(unknown)
