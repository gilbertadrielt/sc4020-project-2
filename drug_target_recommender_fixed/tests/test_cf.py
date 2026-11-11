import numpy as np
import pandas as pd
from src.cf import Means, BiasOnly, MatrixFactorization


def fixture():
    return pd.DataFrame([(d, t, 2. + d * .3 + t * .5) for d in range(5) for t in range(6)],
                        columns=['drug_id', 'target_id', 'affinity'])


def test_means_fallback_and_bias_fit():
    train = fixture()
    means = Means(train, 6)
    assert means.predict(pd.DataFrame({'drug_id': [5]}))[0] == train.affinity.mean()
    model = BiasOnly(train, 5, 6, regularization=.01)
    assert np.mean((model.predict(train) - train.affinity) ** 2) < .001


def test_mf_reload_and_determinism(tmp_path):
    train = fixture()
    a = MatrixFactorization(5, 6, factors=3, max_epochs=100)
    b = MatrixFactorization(5, 6, factors=3, max_epochs=100)
    a.fit(train, train)
    b.fit(train, train)
    np.testing.assert_array_equal(a.predict(train), b.predict(train))
    assert np.mean((a.predict(train) - train.affinity) ** 2) < .02
    path = tmp_path / 'model.npz'
    a.save(path, {'task': 'test'}, list('abcde'), list('abcdef'))
    loaded = MatrixFactorization.load(path)
    np.testing.assert_array_equal(a.predict(train), loaded.predict(train))


def test_validation_labels_never_enter_gradients():
    train = fixture()
    validation = train.copy()
    validation['affinity'] = 999.
    a = MatrixFactorization(5, 6, factors=3, max_epochs=1)
    b = MatrixFactorization(5, 6, factors=3, max_epochs=1)
    a.fit(train, train)
    b.fit(train, validation)
    np.testing.assert_array_equal(a.U, b.U)
    np.testing.assert_array_equal(a.V, b.V)
    np.testing.assert_array_equal(a.drug_bias, b.drug_bias)


def test_knn_uses_only_observed_training_and_bias_fallback():
    from src.cf import TargetKNN
    train = fixture().iloc[:20].copy()
    model = TargetKNN(train, 6, 6, k=5, beta=10)
    query = pd.DataFrame({'drug_id': [0, 5], 'target_id': [5, 5], 'affinity': [-999, -999]})
    p, details = model.predict(query)
    assert details['used_fallback'][1]
    assert p[1] == model.bias.predict(query)[1]
    query.affinity = 999
    np.testing.assert_array_equal(model.predict(query)[0], p)
    assert (model.overlap >= 0).all()
    np.testing.assert_array_equal(model.overlap.diagonal(), np.bincount(train.target_id, minlength=6))
