"""Optional DAVIS train-only thinning with the original warm test held fixed."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.cf import MatrixFactorization, BiasOnly, dimensions
from src.common import ROOT, SEED, write_json, log
from src.splits import load_splits
from src.metrics import evaluate
from src.plots import save_figure


def main():
    path = ROOT / 'results/tables/task2_davis_thinning.csv'
    if path.exists():
        print('Using frozen thinning results')
        return
    frame, parts, manifest = load_splits('davis', 'task2')
    record = json.loads((ROOT / 'results/tables/task2_davis_run.json').read_text())
    params = {k: record['selected'][k] for k in ['factors', 'regularization', 'learning_rate', 'max_epochs', 'patience', 'seed']}
    nd, nt = dimensions(frame)
    train, val, test = parts['train'], parts['validation'], parts['test']
    order = np.random.default_rng(SEED).permutation(len(train))
    rows = []
    for fraction in [1., .5, .25, .1, .05]:
        subset = train.iloc[order[:max(1, int(len(train) * fraction))]]
        seen_d, seen_t = set(subset.drug_id), set(subset.target_id)
        unseen = ~test.drug_id.isin(seen_d) | ~test.target_id.isin(seen_t)
        if fraction == 1.:
            cached = pd.read_csv(ROOT / 'results/tables/task2_davis_metrics.csv')
            for name in ['bias_only', 'MF']:
                row = cached[(cached.split == 'test') & (cached.model == name)].iloc[0].to_dict()
                rows.append({**row, 'fraction': fraction, 'training_pairs': len(subset), 'unseen_test_pairs': 0})
            continue
        bias = BiasOnly(subset, nd, nt, regularization=record['bias_regularization'])
        mf = MatrixFactorization(nd, nt, **params)
        # Explicit zero factors for unobserved entities, including during early stopping.
        mf.U[[i for i in range(nd) if i not in seen_d]] = 0.
        mf.V[[i for i in range(nt) if i not in seen_t]] = 0.
        mf.fit(subset, val)
        for name, prediction in [('bias_only', bias.predict(test)), ('MF', mf.predict(test))]:
            rows.append({'model': name, 'fraction': fraction, 'training_pairs': len(subset),
                         'unseen_test_pairs': int(unseen.sum()), **evaluate(test, prediction)})
    table = pd.DataFrame(rows)
    table.to_csv(path, index=False)
    fig, ax = plt.subplots(figsize=(6, 3.7))
    for name, group in table.groupby('model'):
        ax.plot(group.fraction, group.mse, 'o-', label=name)
    ax.set(xscale='log', xlabel='Fraction of original training pairs retained', ylabel='Test MSE (pKd squared)', title='DAVIS fixed warm test; train-only thinning')
    ax.legend()
    save_figure(fig, 'task2_davis_thinning')
    write_json(ROOT / 'results/tables/task2_davis_thinning_run.json',
               {'split_sha256': manifest['split_sha256'], 'seed': SEED,
                'params_from': 'full-training validation selection; only epochs reselected on fixed validation',
                'unseen_policy': 'Unseen entity biases and factors zero; additive partial/global fallback',
                'test_used_for_selection': False})
    log('Phase 4 optional', 'DAVIS 100/50/25/10/5% nested train-only thinning completed; test unchanged; unseen entity counts and zero-factor fallback reported.')
if __name__ == '__main__':
    main()
