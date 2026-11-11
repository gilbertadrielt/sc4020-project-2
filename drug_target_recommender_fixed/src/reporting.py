"""Small Markdown/table helpers; no notebook-only analysis state."""
import numpy as np
import pandas as pd


def markdown_table(frame, digits=4):
    def fmt(value):
        if pd.isna(value):
            return '—'
        if isinstance(value, (float, np.floating)):
            return f'{value:.{digits}f}'
        return str(value).replace('|', ',').replace('\n', ' ')
    columns = list(frame.columns)
    return '\n'.join(['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |'] +
                     ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)])


def ligand_bootstrap_mse(frame, model, baseline, seed=2026, repetitions=1000):
    """Paired cluster bootstrap; keep each sampled ligand's whole candidate set."""
    work = frame[['drug_id', 'affinity']].copy()
    work['se_model'] = (frame.affinity - frame[model]) ** 2
    work['se_baseline'] = (frame.affinity - frame[baseline]) ** 2
    grouped = work.groupby('drug_id').agg(n=('affinity', 'size'), e=('se_model', 'sum'), b=('se_baseline', 'sum'))
    rng = np.random.default_rng(seed)
    index = rng.integers(len(grouped), size=(repetitions, len(grouped)))
    n, e, b = [grouped[k].to_numpy()[index].sum(axis=1) for k in ['n', 'e', 'b']]
    difference = (e - b) / n
    lo, hi = np.quantile(difference, [.025, .975])
    return {'model': model, 'baseline': baseline, 'ligands': len(grouped), 'replicates': repetitions,
            'mse_difference': float(work.se_model.mean() - work.se_baseline.mean()),
            'bootstrap_95_low': float(lo), 'bootstrap_95_high': float(hi), 'seed': seed}
