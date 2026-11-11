import itertools
import numpy as np
import pandas as pd
import pytest
from src.metrics import concordance_index, evaluate, ranking_metrics, expected_reciprocal_rank, _hit_probability


def brute(y, p):
    values = [float(p[i] > p[j]) + .5 * float(p[i] == p[j])
              for i in range(len(y)) for j in range(len(y)) if y[i] > y[j]]
    return np.mean(values) if values else np.nan


@pytest.mark.parametrize('p,expected', [([1, 2, 3], 1.), ([3, 2, 1], 0.), ([2, 2, 2], .5)])
def test_ci_order(p, expected):
    assert concordance_index([1, 2, 3], p) == expected


def test_ci_ties_against_brute():
    rng = np.random.default_rng(5)
    for size in range(1, 50):
        y, p = rng.integers(0, 5, (2, size))
        actual, expected = concordance_index(y, p), brute(y, p)
        assert (np.isnan(actual) and np.isnan(expected)) or actual == pytest.approx(expected)
    assert np.isnan(concordance_index([], []))


def test_rank_tie_expectations_by_enumeration():
    for scores in [np.zeros(5), np.array([2, 2, 1, 1, 0])]:
        relevant = np.array([False, True, False, True, False])
        valid_orders = [order for order in itertools.permutations(range(5)) if all(scores[order[i]] >= scores[order[i+1]] for i in range(4))]
        first = np.array([next(i + 1 for i, j in enumerate(order) if relevant[j]) for order in valid_orders])
        assert expected_reciprocal_rank(scores, relevant) == pytest.approx(np.mean(1 / first))
        assert _hit_probability(scores, relevant, 1) == pytest.approx(np.mean(first <= 1))
        assert _hit_probability(scores, relevant, 3) == pytest.approx(np.mean(first <= 3))


def test_ranking_undefined_and_undersized_groups():
    frame = pd.DataFrame({'drug_id': [0] * 6 + [1] * 2 + [2], 'target_id': list(range(6)) + [0, 1, 0],
                          'affinity': [6, 5, 4, 3, 2, 1, 2, 2, 7]})
    perfect = evaluate(frame, frame.affinity)
    assert perfect['mse'] == 0 and perfect['hit1'] == perfect['hit5'] == perfect['mrr'] == 1
    assert perfect['hit10'] is None
    reverse = evaluate(frame, -frame.affinity.to_numpy())
    assert reverse['hit1'] == reverse['hit5_gt5'] == 0
    assert reverse['mrr'] == pytest.approx(1/6)
    tied, _ = ranking_metrics(frame, np.zeros(len(frame)))
    assert tied['hit1'] == pytest.approx(1/6)
    assert tied['hit5_gt5'] == pytest.approx(5/6)
    assert tied['ranking_singleton_drugs'] == tied['ranking_all_tied_drugs'] == tied['ranking_eligible_drugs'] == 1


def test_exact_mse_and_invalid_vectors():
    f = pd.DataFrame({'drug_id': [0, 0], 'target_id': [0, 1], 'affinity': [1., 3.]})
    assert evaluate(f, [2., 1.])['mse'] == 2.5
    with pytest.raises(ValueError):
        evaluate(f, [np.nan, 1])


def test_prediction_serialization_preserves_near_tie_rankings(tmp_path, monkeypatch):
    from src import plots
    from src.io import read_predictions
    (tmp_path / 'results/tables').mkdir(parents=True)
    (tmp_path / 'results/predictions').mkdir(parents=True)
    monkeypatch.setattr(plots, 'ROOT', tmp_path)
    p = np.array([1., np.nextafter(1., 2.), np.nextafter(np.nextafter(1., 2.), 2.)])
    frame = pd.DataFrame({'drug_id': [0, 0, 0], 'target_id': [0, 1, 2], 'affinity': [1., 2., 3.],
                          'original_drug_ids': ['d'] * 3, 'original_target_ids': ['a', 'b', 'c']})
    plots.export_predictions('test', 'test', frame, {'model': p})
    saved = read_predictions(tmp_path / 'results/predictions/test_test.csv')
    np.testing.assert_array_equal(saved.model, p)
    assert concordance_index(saved.affinity, saved.model) == 1
