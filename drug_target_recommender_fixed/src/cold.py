"""Sequence-neighbour transfer fitted exclusively to Task 3 training labels."""
import json

import numpy as np

from .cf import Means
from .features import FEATURE_POLICY, sequence_similarity


class ColdTargetNeighbors:
    def __init__(self, k=15, power=1., min_similarity=0., min_support=1):
        self.params = dict(k=k, power=power, min_similarity=min_similarity, min_support=min_support)

    def fit(self, train, n_drugs):
        targets = train[['target_id', 'sequence']].drop_duplicates().sort_values('target_id')
        assert targets.target_id.is_unique
        self.target_ids = targets.target_id.to_numpy(dtype=np.int64)
        self.sequences = targets.sequence.to_numpy(dtype=str)
        self.drug_means = Means(train, n_drugs).drug_mean
        self.mu = float(train.affinity.mean())
        self.observations = np.full((n_drugs, len(targets)), np.nan)
        indices = {int(target): j for j, target in enumerate(self.target_ids)}
        columns = train.target_id.map(indices).to_numpy()
        self.observations[train.drug_id.to_numpy(), columns] = train.affinity.to_numpy()
        self.drug_train_counts = np.bincount(train.drug_id.to_numpy(), minlength=n_drugs)
        return self

    def predict(self, frame):
        targets = frame[['target_id', 'sequence']].drop_duplicates().sort_values('target_id')
        assert targets.target_id.is_unique
        assert set(targets.target_id).isdisjoint(self.target_ids), 'Expected entirely cold targets'
        assert set(targets.sequence).isdisjoint(self.sequences), 'Identical sequence leakage'
        similarity = sequence_similarity(targets.sequence, self.sequences)
        weights = np.zeros_like(similarity)
        for i, row in enumerate(similarity):
            nearest = np.argsort(-row, kind='stable')[:self.params['k']]
            chosen = nearest[(row[nearest] >= self.params['min_similarity']) & (row[nearest] > 0)]
            weights[i, chosen] = row[chosen] ** self.params['power']
        observed = np.isfinite(self.observations)
        support = observed.astype(float) @ (weights > 0).T.astype(float)
        denominator = observed.astype(float) @ weights.T
        numerator = np.nan_to_num(self.observations) @ weights.T
        usable = (support >= self.params['min_support']) & (denominator > 0)
        output = np.broadcast_to(self.drug_means[:, None], denominator.shape).copy()
        output[usable] = numerator[usable] / denominator[usable]
        query_lookup = {int(target): j for j, target in enumerate(targets.target_id)}
        d, t = frame.drug_id.to_numpy(), frame.target_id.map(query_lookup).to_numpy()
        prediction = output[d, t]
        details = {'neighbor_support': support[d, t].astype(int), 'used_fallback': ~usable[d, t],
                   'max_train_similarity': similarity.max(axis=1)[t],
                   'unseen_drug': self.drug_train_counts[d] == 0}
        return prediction, details

    def save(self, path, metadata):
        np.savez_compressed(path, target_ids=self.target_ids, sequences=self.sequences,
                            observations=self.observations, drug_means=self.drug_means,
                            drug_train_counts=self.drug_train_counts, mu=self.mu,
                            metadata=json.dumps({**metadata, 'parameters': self.params, 'feature_policy': FEATURE_POLICY}))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(str(z['metadata']))
            model = cls(**metadata['parameters'])
            for name in ['target_ids', 'sequences', 'observations', 'drug_means', 'drug_train_counts']:
                setattr(model, name, z[name].copy())
            model.mu, model.metadata = float(z['mu']), metadata
        return model


class ColdFactorRidge:
    """C2: map full-sequence features to TRAINED target biases/latent factors.

    No target lookup/factor is used for a cold query. MF is fit only on cold-train.
    Fixed MF training budget is separate from validation-tuned ridge strength.
    """
    def fit(self, train, n_drugs, n_targets, factors=16, epochs=80):
        from .cf import MatrixFactorization
        from .features import protein_features
        self.params = {'factors': factors, 'epochs': epochs, 'regularization': .01, 'learning_rate': .02}
        targets = train[['target_id', 'sequence']].drop_duplicates().sort_values('target_id')
        self.target_ids = targets.target_id.to_numpy()
        self.sequences = targets.sequence.to_numpy(dtype=str)
        mf = MatrixFactorization(n_drugs, n_targets, factors=factors, max_epochs=epochs, patience=epochs,
                                 regularization=.01, learning_rate=.02)
        # Fixed-budget MF checkpoint by TRAIN error; no cold evaluation label access.
        self.history = mf.fit(train, train)
        self.mu, self.drug_bias, self.U = mf.mu, mf.drug_bias, mf.U
        self.drug_counts = np.bincount(train.drug_id, minlength=n_drugs)
        self.U[self.drug_counts == 0] = 0.
        self.target_outputs = np.column_stack([mf.target_bias[self.target_ids], mf.V[self.target_ids]])
        self.train_features = protein_features(self.sequences)
        self._kernel = (self.train_features @ self.train_features.T).toarray()
        return self

    def set_alpha(self, alpha):
        self.alpha = alpha
        # Ridge with intercept: center with statistics of training targets only.
        n = len(self.target_ids)
        kernel = self._kernel
        self.kernel_mean = kernel.mean(axis=0)
        self.kernel_grand_mean = float(kernel.mean())
        self.output_mean = self.target_outputs.mean(axis=0)
        centered = kernel - self.kernel_mean[None, :] - self.kernel_mean[:, None] + self.kernel_grand_mean
        self.dual = np.linalg.solve(centered + alpha * np.eye(n), self.target_outputs - self.output_mean)
        return self

    def predict(self, frame):
        from .features import protein_features
        targets = frame[['target_id', 'sequence']].drop_duplicates().sort_values('target_id')
        assert set(targets.target_id).isdisjoint(self.target_ids)
        assert set(targets.sequence).isdisjoint(self.sequences)
        cross = (protein_features(targets.sequence.tolist()) @ self.train_features.T).toarray()
        centered = cross - self.kernel_mean[None, :] - cross.mean(axis=1)[:, None] + self.kernel_grand_mean
        outputs = self.output_mean + centered @ self.dual
        lookup = {int(t): i for i, t in enumerate(targets.target_id)}
        d, t = frame.drug_id.to_numpy(), frame.target_id.map(lookup).to_numpy()
        return self.mu + self.drug_bias[d] + outputs[t, 0] + np.einsum('ij,ij->i', self.U[d], outputs[t, 1:])

    def save(self, path, metadata):
        np.savez_compressed(path, mu=self.mu, drug_bias=self.drug_bias, U=self.U, drug_counts=self.drug_counts,
                            target_ids=self.target_ids, sequences=self.sequences, target_outputs=self.target_outputs,
                            dual=self.dual, kernel_mean=self.kernel_mean, kernel_grand_mean=self.kernel_grand_mean,
                            output_mean=self.output_mean,
                            metadata=json.dumps({**metadata, 'parameters': self.params, 'alpha': self.alpha,
                                                 'feature_policy': 'fixed 3-mer + AAC + dipeptide frequencies; train-target-centered ridge'}))

    @classmethod
    def load(cls, path):
        from .features import protein_features
        model = cls()
        with np.load(path, allow_pickle=False) as z:
            for key in z.files:
                if key != 'metadata':
                    setattr(model, key, z[key].copy())
            model.metadata = json.loads(str(z['metadata']))
            model.params, model.alpha = model.metadata['parameters'], model.metadata['alpha']
        model.train_features = protein_features(model.sequences)
        return model
