"""Observed-pair biased matrix factorization with deterministic SGD."""
import json

import numpy as np
import pandas as pd
from numba import njit

from .common import SEED


def dimensions(frame):
    return int(frame.drug_id.max()) + 1, int(frame.target_id.max()) + 1


class Means:
    def __init__(self, train, n_drugs):
        self.mu = float(train.affinity.mean())
        self.drug_mean = np.full(n_drugs, self.mu)
        grouped = train.groupby('drug_id').affinity.mean()
        self.drug_mean[grouped.index.to_numpy()] = grouped.to_numpy()

    def predict(self, frame):
        return self.drug_mean[frame.drug_id.to_numpy()]


class BiasOnly:
    def __init__(self, train, n_drugs, n_targets, regularization=10., iterations=100):
        self.mu = float(train.affinity.mean())
        self.drug_bias, self.target_bias = np.zeros(n_drugs), np.zeros(n_targets)
        d, t, y = train.drug_id.to_numpy(), train.target_id.to_numpy(), train.affinity.to_numpy()
        dc, tc = np.bincount(d, minlength=n_drugs), np.bincount(t, minlength=n_targets)
        for _ in range(iterations):
            self.drug_bias = np.bincount(d, weights=y - self.mu - self.target_bias[t], minlength=n_drugs) / (dc + regularization)
            self.target_bias = np.bincount(t, weights=y - self.mu - self.drug_bias[d], minlength=n_targets) / (tc + regularization)

    def predict(self, frame):
        return self.mu + self.drug_bias[frame.drug_id.to_numpy()] + self.target_bias[frame.target_id.to_numpy()]


@njit(cache=True)
def sgd_epoch(d, t, y, order, mu, db, tb, u, v, lr, reg):
    for row in order:
        i, j = d[row], t[row]
        prediction = mu + db[i] + tb[j]
        for factor in range(u.shape[1]):
            prediction += u[i, factor] * v[j, factor]
        error = y[row] - prediction
        db[i] += lr * (error - reg * db[i])
        tb[j] += lr * (error - reg * tb[j])
        for factor in range(u.shape[1]):
            old_u, old_v = u[i, factor], v[j, factor]
            u[i, factor] += lr * (error * old_v - reg * old_u)
            v[j, factor] += lr * (error * old_u - reg * old_v)


class MatrixFactorization:
    def __init__(self, n_drugs, n_targets, factors=16, regularization=.05, learning_rate=.01,
                 max_epochs=200, patience=20, seed=SEED):
        self.params = dict(factors=factors, regularization=regularization, learning_rate=learning_rate,
                           max_epochs=max_epochs, patience=patience, seed=seed)
        rng = np.random.default_rng(seed)
        self.mu = 0.
        self.drug_bias, self.target_bias = np.zeros(n_drugs), np.zeros(n_targets)
        self.U = rng.normal(0, .05, (n_drugs, factors))
        self.V = rng.normal(0, .05, (n_targets, factors))

    def predict(self, frame):
        d, t = frame.drug_id.to_numpy(), frame.target_id.to_numpy()
        return self.mu + self.drug_bias[d] + self.target_bias[t] + np.einsum('ij,ij->i', self.U[d], self.V[t])

    def fit(self, train, validation):
        self.mu = float(train.affinity.mean())
        rng = np.random.default_rng(self.params['seed'])
        d = train.drug_id.to_numpy(dtype=np.int64)
        t = train.target_id.to_numpy(dtype=np.int64)
        y = train.affinity.to_numpy(dtype=float)
        best, best_state, stale, history = np.inf, None, 0, []
        for epoch in range(1, self.params['max_epochs'] + 1):
            sgd_epoch(d, t, y, rng.permutation(len(train)), self.mu, self.drug_bias, self.target_bias,
                      self.U, self.V, self.params['learning_rate'], self.params['regularization'])
            train_mse = float(np.mean((self.predict(train) - y) ** 2))
            validation_mse = float(np.mean((self.predict(validation) - validation.affinity.to_numpy()) ** 2))
            if not np.isfinite(validation_mse):
                raise ValueError('MF diverged')
            history.append({'epoch': epoch, 'train_mse': train_mse, 'validation_mse': validation_mse})
            if validation_mse < best - 1e-7:
                best, stale = validation_mse, 0
                self.best_epoch = epoch
                best_state = [a.copy() for a in [self.drug_bias, self.target_bias, self.U, self.V]]
            else:
                stale += 1
            if stale >= self.params['patience']:
                break
        self.drug_bias, self.target_bias, self.U, self.V = best_state
        self.validation_mse = best
        return pd.DataFrame(history)

    def save(self, path, metadata, drug_map, target_map):
        if len(drug_map) != len(self.U) or len(target_map) != len(self.V):
            raise ValueError('Checkpoint entity mappings must contain one row per processed entity')
        np.savez_compressed(path, mu=self.mu, drug_bias=self.drug_bias, target_bias=self.target_bias,
                            U=self.U, V=self.V, drug_ids=np.arange(len(self.U)), target_ids=np.arange(len(self.V)),
                            drug_map=np.asarray(drug_map, dtype=str), target_map=np.asarray(target_map, dtype=str),
                            metadata=json.dumps({**metadata, 'hyperparameters': self.params, 'best_epoch': self.best_epoch}))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(str(z['metadata']))
            model = cls(len(z['U']), len(z['V']), **metadata['hyperparameters'])
            model.mu = float(z['mu'])
            model.drug_bias, model.target_bias, model.U, model.V = [z[k].copy() for k in ['drug_bias', 'target_bias', 'U', 'V']]
            model.best_epoch, model.metadata = metadata['best_epoch'], metadata
        return model


class TargetKNN:
    """Training-only overlap-cosine of bias residuals, positive neighbours only."""
    def __init__(self, train, n_drugs, n_targets, k=10, beta=10., bias=None):
        self.k, self.beta = k, beta
        self.bias = BiasOnly(train, n_drugs, n_targets) if bias is None else bias
        self.observed = np.zeros((n_drugs, n_targets), dtype=bool)
        self.residual = np.zeros((n_drugs, n_targets))
        d, t = train.drug_id.to_numpy(), train.target_id.to_numpy()
        self.observed[d, t] = True
        self.residual[d, t] = train.affinity.to_numpy() - self.bias.predict(train)
        mask = self.observed.astype(float)
        self.overlap = mask.T @ mask
        # Norms are restricted to co-observing ligands for EACH target pair.
        norm = self.residual.T ** 2 @ mask
        denom = np.sqrt(norm * norm.T)
        self.similarity = np.divide(self.residual.T @ self.residual, denom,
                                    out=np.zeros_like(denom), where=denom > 1e-12)
        self.similarity = np.clip(self.similarity, -1, 1)
        np.fill_diagonal(self.similarity, 0.)

    def predict(self, frame):
        similarity = self.similarity * np.divide(self.overlap, self.overlap + self.beta,
                                                  out=np.zeros_like(self.overlap), where=self.overlap + self.beta > 0)
        n_drugs, n_targets = self.residual.shape
        adjustment = np.zeros_like(self.residual)
        support = np.zeros_like(self.residual, dtype=int)
        for target in np.unique(frame.target_id):
            order = np.argsort(-similarity[target], kind='stable')
            available = self.observed[:, order] & (similarity[target, order] > 0)
            chosen = available & (np.cumsum(available, axis=1) <= self.k)
            weights = chosen * similarity[target, order]
            denominator = weights.sum(axis=1)
            adjustment[:, target] = np.divide((weights * self.residual[:, order]).sum(axis=1), denominator,
                                               out=np.zeros(n_drugs), where=denominator > 0)
            support[:, target] = chosen.sum(axis=1)
        d, t = frame.drug_id.to_numpy(), frame.target_id.to_numpy()
        return self.bias.predict(frame) + adjustment[d, t], {'neighbor_support': support[d, t], 'used_fallback': support[d, t] == 0}
