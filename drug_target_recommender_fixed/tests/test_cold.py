import numpy as np
import pandas as pd
from src.cold import ColdTargetNeighbors


def test_cold_missing_observations_and_label_independence(tmp_path):
    train = pd.DataFrame({'drug_id': [0, 0, 1], 'target_id': [0, 1, 1],
                          'sequence': ['AAAA', 'CCCC', 'CCCC'], 'affinity': [2., 4., 6.]})
    query = pd.DataFrame({'drug_id': [0, 1, 2], 'target_id': [2, 2, 2],
                          'sequence': ['AAAC'] * 3, 'affinity': [999.] * 3})
    model = ColdTargetNeighbors(k=1).fit(train, 3)
    p, details = model.predict(query)
    np.testing.assert_allclose(p, [2., 6., 4.])
    np.testing.assert_array_equal(details['used_fallback'], [False, True, True])
    query['affinity'] = -999
    np.testing.assert_array_equal(model.predict(query)[0], p)
    path = tmp_path / 'cold.npz'
    model.save(path, {'task': 'test'})
    np.testing.assert_array_equal(ColdTargetNeighbors.load(path).predict(query)[0], p)


def test_learned_cold_model_label_independence_and_reload(tmp_path):
    from src.cold import ColdFactorRidge
    train = pd.DataFrame([(d, t, s, 2 + d + t / 3) for d in range(4) for t, s in enumerate(['AAAA', 'AAAC', 'CCCC', 'CCCD'])],
                         columns=['drug_id', 'target_id', 'sequence', 'affinity'])
    query = pd.DataFrame({'drug_id': [0, 1], 'target_id': [5, 5], 'sequence': ['ACCC'] * 2, 'affinity': [0., 0.]})
    model = ColdFactorRidge().fit(train, 4, 6, factors=2, epochs=10).set_alpha(1.)
    expected = model.predict(query)
    query.affinity = 9999
    np.testing.assert_array_equal(expected, model.predict(query))
    model.save(tmp_path / 'ridge.npz', {'task': 'test'})
    np.testing.assert_allclose(ColdFactorRidge.load(tmp_path / 'ridge.npz').predict(query), expected, atol=1e-12)
    import pytest
    with pytest.raises(AssertionError):
        model.predict(train.iloc[:2])
