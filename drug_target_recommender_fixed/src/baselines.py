"""Training-only mean and regularized additive bias estimators."""
import numpy as np
from .cf import Means, BiasOnly


class TargetMean:
    def __init__(self, train, n_targets):
        self.mu = float(train.affinity.mean())
        self.means = np.full(n_targets, self.mu)
        values = train.groupby('target_id').affinity.mean()
        self.means[values.index.to_numpy()] = values.to_numpy()

    def predict(self, frame):
        return self.means[frame.target_id.to_numpy()]
