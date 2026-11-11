"""Recreate prediction exports from selected saved models; never refit/reselect.

Use 17-digit floats and round-trip readers so serialization cannot create ranking
score ties. The original selected-model metric files remain the audit reference.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.config import ROOT, DATASETS, COLD_SEEDS
from src.cf import MatrixFactorization
from src.cold import ColdTargetNeighbors, ColdFactorRidge
from src.saved import load_warm_aux
from src.splits import load_splits
from src.plots import export_predictions
from src.common import log


def main():
    for dataset in DATASETS:
        _, parts, _ = load_splits(dataset, 'task2')
        model = MatrixFactorization.load(ROOT / f'saved_models/{dataset}_cf.npz')
        bias, knn, means = load_warm_aux(dataset)
        for split, part in parts.items():
            pk, details = knn.predict(part)
            predictions = {'global_mean': np.full(len(part), bias.mu), 'ligand_mean': means['drug_mean'][part.drug_id],
                           'target_mean': means['target_mean'][part.target_id], 'bias_only': bias.predict(part),
                           'target_knn': pk, 'MF': model.predict(part)}
            export_predictions(f'task2_{dataset}', split, part, predictions,
                               {'knn_support': details['neighbor_support'], 'knn_fallback': details['used_fallback']})
        for seed in COLD_SEEDS:
            _, parts, _ = load_splits(dataset, 'task3', seed)
            neighbors = ColdTargetNeighbors.load(ROOT / f'saved_models/{dataset}_neighbors_seed{seed}.npz')
            ridge = ColdFactorRidge.load(ROOT / f'saved_models/{dataset}_ridge_seed{seed}.npz')
            for split in ['validation', 'test']:
                part = parts[split]
                p, details = neighbors.predict(part)
                predictions = {'ligand_mean': neighbors.drug_means[part.drug_id],
                               'sequence_neighbors': p, 'factor_ridge': ridge.predict(part)}
                export_predictions(f'task3_{dataset}_seed{seed}', split, part, predictions, details)
    log('Phase 6 serialization', 'Re-exported selected checkpoint predictions with 17-digit round-trip precision. Audit found 15-digit CSV rounding could create near-ties in CI; no fitted state, hyperparameters or original model metrics changed.')
if __name__ == '__main__':
    main()
