"""Correctness tests for the Task 2 and Task 3 scoring.

The concordance index is the one worth testing hard. It is computed from
``kendalltau`` through an algebraic identity rather than by counting pairs, so
it is checked here against a double loop that counts them literally, including
the cases the identity has to get right: ties in the labels, ties in the
predictions, and the censored DAVIS shape where most labels are the same value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import concordance_index, evaluate, mse, ranking_metrics


def brute_force_ci(y_true, y_pred) -> float:
    """CI by its definition: every pair, counted one at a time."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    hits = usable = 0.0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] == y_true[j]:
                continue  # no true ordering, so the pair is not usable
            usable += 1
            hi, lo = (i, j) if y_true[i] > y_true[j] else (j, i)
            if y_pred[hi] > y_pred[lo]:
                hits += 1
            elif y_pred[hi] == y_pred[lo]:
                hits += 0.5
    return hits / usable if usable else float("nan")


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_ci_matches_brute_force_on_continuous_scores(seed):
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=60)
    pred = truth * 0.6 + rng.normal(scale=0.8, size=60)
    assert concordance_index(truth, pred) == pytest.approx(brute_force_ci(truth, pred))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ci_matches_brute_force_with_ties_on_both_sides(seed):
    """Both kinds of tie at once, which is what the tau-b correction is for."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 4, size=70).astype(float)
    pred = rng.integers(0, 3, size=70).astype(float)
    assert concordance_index(truth, pred) == pytest.approx(brute_force_ci(truth, pred))


def test_ci_matches_brute_force_on_a_censored_davis_shape():
    """70% of labels at the floor: the shape that makes CI the honest metric."""
    rng = np.random.default_rng(7)
    truth = np.where(rng.random(120) < 0.7, 5.0, rng.uniform(5, 10.8, 120))
    pred = truth + rng.normal(scale=0.5, size=120)
    assert concordance_index(truth, pred) == pytest.approx(brute_force_ci(truth, pred))


def test_perfect_and_reversed_orderings():
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    assert concordance_index(truth, truth) == pytest.approx(1.0)
    assert concordance_index(truth, -truth) == pytest.approx(0.0)


def test_constant_prediction_is_exactly_half():
    """Every usable pair is tied, so each counts half right."""
    truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert concordance_index(truth, np.full(5, 7.0)) == pytest.approx(0.5)


def test_constant_truth_is_undefined_not_chance():
    """No pair has a true ordering, so there is nothing to be right about."""
    assert np.isnan(concordance_index(np.full(6, 5.0), np.arange(6.0)))


def test_mse_is_in_score_units():
    assert mse([1.0, 2.0], [2.0, 4.0]) == pytest.approx((1 + 4) / 2)


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------

def frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["drug_id", "score", "pred"])


def test_hit_at_k_finds_the_truly_strongest_target():
    # Drug A: best true score (9.0) is predicted best, so Hit@1 = 1.
    # Drug B: best true score (9.0) is predicted third, so Hit@1 = 0, Hit@5 = 1.
    data = frame([
        ("A", 9.0, 8.0), ("A", 6.0, 5.0), ("A", 5.0, 4.0),
        ("B", 9.0, 1.0), ("B", 6.0, 3.0), ("B", 5.0, 2.0),
    ])
    out = ranking_metrics(data, ks=(1, 5))
    assert out["hit@1"] == pytest.approx(0.5)
    assert out["hit@5"] == pytest.approx(1.0)
    assert out["mrr"] == pytest.approx((1 / 1 + 1 / 3) / 2)
    assert out["drugs_ranked"] == 2


def test_a_tie_for_the_best_true_score_counts_if_either_is_found():
    data = frame([("A", 9.0, 1.0), ("A", 9.0, 5.0), ("A", 2.0, 3.0)])
    assert ranking_metrics(data, ks=(1,))["hit@1"] == pytest.approx(1.0)


def test_drugs_below_min_candidates_are_excluded():
    """A drug with 3 candidates cannot fail Hit@5, so it must not inflate it."""
    data = frame([
        ("A", 9.0, 1.0), ("A", 6.0, 2.0), ("A", 5.0, 3.0),
        ("B", 9.0, 9.0), ("B", 6.0, 6.0), ("B", 5.0, 5.0), ("B", 4.0, 4.0),
        ("B", 3.0, 3.0), ("B", 2.0, 2.0),
    ])
    assert ranking_metrics(data, ks=(5,), min_candidates=2)["drugs_ranked"] == 2
    kept = ranking_metrics(data, ks=(5,), min_candidates=6)
    assert kept["drugs_ranked"] == 1
    assert kept["hit@5"] == pytest.approx(1.0)  # drug B alone, correctly ranked


def test_a_constant_predictor_earns_chance_not_row_order():
    """The per-drug mean predicts one value for every candidate.

    Breaking those ties by row order would hand it whatever the dataset's id
    ordering happens to give. With random tie-breaking, Hit@1 over many drugs
    with 10 candidates each must land near 1/10.
    """
    rng = np.random.default_rng(0)
    rows = []
    for drug in range(400):
        for _ in range(10):
            rows.append((f"d{drug}", float(rng.normal()), 5.0))
    out = ranking_metrics(frame(rows), ks=(1, 5))
    assert out["hit@1"] == pytest.approx(0.10, abs=0.04)
    assert out["hit@5"] == pytest.approx(0.50, abs=0.07)


def test_evaluate_reports_error_over_strong_binders_separately():
    """On DAVIS the floor dominates MSE, so the strong subset is its own number."""
    data = frame([
        ("A", 5.0, 5.0), ("A", 5.0, 5.0), ("A", 5.0, 5.0),
        ("A", 9.0, 6.0), ("A", 8.0, 6.0),
    ])
    row = evaluate(data, label="floor-predictor", strong_threshold=7.0)
    assert row["n_strong"] == 2
    assert row["mse"] == pytest.approx((0 + 0 + 0 + 9 + 4) / 5)
    assert row["mse_strong"] == pytest.approx((9 + 4) / 2)
    assert row["mse_strong"] > row["mse"]
