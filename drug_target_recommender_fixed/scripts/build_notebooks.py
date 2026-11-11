"""Create thin saved-artifact notebooks and optionally execute fresh kernels."""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nbformat as nbf
from src.common import ROOT, log
from src.config import CODE_ROOT

SETUP = '''from pathlib import Path
import sys
ROOT = Path.cwd()
if not (ROOT / 'src').is_dir():
    ROOT = ROOT.parent
assert (ROOT / 'src').is_dir(), 'Run from the project root or notebooks directory'
sys.path.insert(0, str(ROOT))
import json
import numpy as np
import pandas as pd
from IPython.display import display, Markdown, Image
from src.splits import load_splits
from src.cf import MatrixFactorization
from src.metrics import evaluate
from src.io import read_predictions
T = ROOT / 'results/tables'
'''


def make(name, title, description, cells):
    nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('# ' + title + '\n\n' + description), nbf.v4.new_code_cell(SETUP), *cells])
    nb.metadata['kernelspec'] = {'display_name': 'Python 3 (project .venv)', 'language': 'python', 'name': 'python3'}
    nb.metadata['language_info'] = {'name': 'python', 'version': '3.13'}
    path = ROOT / 'notebooks' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    m, c = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    paths = []
    paths.append(make('02_task2_cf.ipynb', 'Task 2: warm-pair collaborative filtering',
                      'Frozen seed-2026 pairs, selected checkpoints and saved predictions. No fitting occurs in this notebook.', [
        c("summary = pd.read_csv(T / 'task2_summary.csv')\ndisplay(summary[summary.split == 'test'][['dataset','model','mse','ci_global','ci_per_ligand','hit1','hit5','hit10','mrr']])"),
        m('Every held-out ligand and target is warm. Hash checks and the split validator run on load. Predictions below come from reloaded archives.'),
        c("for dataset in ['davis','kiba']:\n    frame, parts, manifest = load_splits(dataset, 'task2')\n    model = MatrixFactorization.load(ROOT / f'saved_models/{dataset}_cf.npz')\n    assert model.metadata['split_sha256'] == manifest['split_sha256']\n    saved = read_predictions(ROOT / f'results/predictions/task2_{dataset}_test.csv')\n    np.testing.assert_allclose(model.predict(parts['test']), saved.MF, atol=1e-12, rtol=1e-12)\n    display(Markdown(f'## {dataset.upper()}'))\n    display(pd.read_csv(T / f'task2_{dataset}_tuning.csv').sort_values('validation_mse').head(8))\n    display(Image(filename=str(ROOT / f'results/figures/task2_{dataset}_diagnostics.png')))"),
        c("display(pd.read_csv(T / 'paired_ligand_bootstrap.csv').query(\"task == 'task2'\"))\ndisplay(pd.read_csv(T / 'task2_davis_thinning.csv')[['fraction','model','mse','unseen_test_pairs']])"),
        m('Interpretation and metric conventions: [Task 2 report notes](../results/task2_report_notes.md). CI is concordance; bootstrap intervals are separate. Test data never selected settings.')]))
    paths.append(make('03_task2_case_study.ipynb', 'Task 2: held-out ligand case studies',
                      'Three cases per dataset selected for sufficient measured test candidates. Checkpoints are loaded, never refitted.', [
        c("for dataset in ['davis','kiba']:\n    _, parts, manifest = load_splits(dataset, 'task2')\n    model = MatrixFactorization.load(ROOT / f'saved_models/{dataset}_cf.npz')\n    cases = pd.read_csv(T / f'task2_{dataset}_case_summary.csv')\n    rankings = pd.read_csv(T / f'task2_{dataset}_case_rankings.csv')\n    display(Markdown(f'## {dataset.upper()}'))\n    display(cases[['original_drug_ids','selection_reason','test_candidates','true_best_predicted_rank','true_best_bias_rank','mse','ci_global','hit1','hit5','mrr']])\n    for drug in cases.drug_id:\n        query = parts['test'][parts['test'].drug_id == drug]\n        ranked = rankings[rankings.drug_id == drug]\n        lookup = dict(zip(query.target_id, model.predict(query)))\n        np.testing.assert_allclose(ranked.prediction, ranked.target_id.map(lookup), atol=1e-12)\n        assert ranked.target_id.isin(query.target_id).all()\n        display(ranked.head(10))"),
        m('Target IDs can have exact-sequence aliases. The CSV lists actual source IDs and processed keys. All true-best ties are relevant; expected Hit/MRR average prediction ties. No training pairs appear in these tables.')]))
    paths.append(make('04_task3_cold.ipynb', 'Task 3: unseen target sequences',
                      'Five frozen sequence-grouped splits per dataset; C0, sequence neighbours and learned factor ridge, trained separately on each cold-train partition.', [
        c("display(pd.read_csv(T / 'task3_aggregate.csv'))\ndisplay(pd.read_csv(T / 'task3_coverage_cost.csv'))"),
        m('Sample standard deviation is across target splits, not a confidence interval. Fixed 3-mer cosine is not alignment identity; similarity buckets are descriptive.'),
        c("from src.cold import ColdTargetNeighbors, ColdFactorRidge\nfor dataset in ['davis','kiba']:\n    for seed in range(2026,2031):\n        _, parts, manifest = load_splits(dataset,'task3',seed)\n        saved = read_predictions(ROOT / f'results/predictions/task3_{dataset}_seed{seed}_test.csv')\n        neighbors = ColdTargetNeighbors.load(ROOT / f'saved_models/{dataset}_neighbors_seed{seed}.npz')\n        ridge = ColdFactorRidge.load(ROOT / f'saved_models/{dataset}_ridge_seed{seed}.npz')\n        assert set(parts['test'].sequence).isdisjoint(parts['train'].sequence)\n        assert set(parts['test'].target_id).isdisjoint(neighbors.target_ids)\n        assert neighbors.metadata['split_sha256'] == manifest['split_sha256']\n        np.testing.assert_allclose(neighbors.predict(parts['test'])[0], saved.sequence_neighbors, atol=1e-12)\n        np.testing.assert_allclose(ridge.predict(parts['test']), saved.factor_ridge, atol=1e-12)\n    display(Markdown(f'## {dataset.upper()} · seed 2026'))\n    display(pd.read_csv(T / f'task3_{dataset}_seed2026_similarity_buckets.csv'))\n    display(Image(filename=str(ROOT / f'results/figures/task3_{dataset}_seed2026_similarity.png')))"),
        m('Method, failure modes, all per-split metrics and assistance attribution: [Task 3 report notes](../results/task3_report_notes.md).')]))
    if a.execute:
        from nbclient import NotebookClient
        # ipykernel in this process's venv is selected by PATH; no global kernel installation.
        previous = os.environ.get('PATH', '')
        os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + previous
        for path in paths:
            nb = nbf.read(path, as_version=4)
            client = NotebookClient(nb, timeout=240, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}})
            client.execute()
            nbf.write(nb, path)
            print('Executed', path.name, flush=True)
        log('Phase 6 notebooks', 'All three thin notebooks executed top-to-bottom in separate fresh kernels, loading frozen splits/checkpoints/results without fitting.')
if __name__ == '__main__':
    main()
