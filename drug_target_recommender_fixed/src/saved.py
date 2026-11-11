"""Load selected warm auxiliary estimators without refitting any labels."""
import json
import numpy as np
from .common import ROOT
from .cf import BiasOnly, TargetKNN


def load_warm_aux(dataset):
    with np.load(ROOT / f'saved_models/{dataset}_baselines_knn.npz', allow_pickle=False) as z:
        metadata = json.loads(str(z['metadata']))
        bias = BiasOnly.__new__(BiasOnly)
        bias.mu = float(z['mu'])
        bias.drug_bias, bias.target_bias = z['drug_bias'].copy(), z['target_bias'].copy()
        knn = TargetKNN.__new__(TargetKNN)
        knn.bias, knn.k, knn.beta = bias, metadata['k'], metadata['beta']
        for key in ['residual', 'observed', 'similarity', 'overlap']:
            setattr(knn, key, z[key].copy())
        means = {key: z[key].copy() for key in ['drug_mean', 'target_mean']}
    return bias, knn, means

