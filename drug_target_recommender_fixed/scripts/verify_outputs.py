"""Independent artifact reconciliation; no fitting or model selection occurs here."""
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.common import ROOT, sha256, write_json, log, provenance
from src.config import CODE_ROOT, DATASETS, COLD_SEEDS, SEED
from src.splits import load_splits
from src.data import load_raw
from src.metrics import evaluate, random_ranking_reference
from src.cf import MatrixFactorization, BiasOnly, TargetKNN
from src.cold import ColdTargetNeighbors, ColdFactorRidge
from src.saved import load_warm_aux
from src.io import read_predictions


def compare_metrics(part, predictions, metrics):
    for row in metrics.to_dict('records'):
        if row['model'] == 'random_ranking':
            actual = random_ranking_reference(part)
        else:
            actual = evaluate(part, predictions[row['model']].to_numpy())
        for key, value in actual.items():
            saved = row[key]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                assert pd.isna(saved), (key, value, saved)
            else:
                np.testing.assert_allclose(value, saved, rtol=1e-11, atol=1e-12, err_msg=key)




def verify():
    checks = []
    for dataset in DATASETS:
        _, _, matrix, source = load_raw(dataset)
        clean, parts, manifest = load_splits(dataset, 'task2')
        assert clean.n_merged.sum() == np.isfinite(matrix).sum(), 'Real source has no invalid observations; every measurement must be accounted for'
        assert not clean.duplicated(['smiles', 'sequence']).any()
        assert clean.score.equals(clean.affinity)
        counts = json.loads((ROOT / f'data/processed/{dataset}_cleaning_summary.json').read_text())
        assert counts['missing_nan'] == int(np.isnan(matrix).sum())
        assert counts['unique_pairs'] == len(clean)
        if dataset == 'davis':
            assert counts['input_scale'] == 'Kd_nM'
            assert clean.is_censored.any() and (clean.censored_fraction <= 1).all()
        else:
            assert counts['input_scale'] == 'DeepDTA_KIBA'
            assert matrix.size - np.isfinite(matrix).sum() > 0
            assert len(clean) < clean.drug_id.nunique() * clean.target_id.nunique()
        model = MatrixFactorization.load(ROOT / f'saved_models/{dataset}_cf.npz')
        assert model.metadata['split_sha256'] == manifest['split_sha256']
        np.testing.assert_allclose(model.mu, parts['train'].affinity.mean(), atol=1e-12)
        with np.load(ROOT / f'saved_models/{dataset}_cf.npz', allow_pickle=False) as z:
            for kind, key in [('ligands', 'drug_map'), ('targets', 'target_map')]:
                entity = pd.read_csv(ROOT / f'data/processed/{kind}_{dataset}.csv')
                field = 'original_drug_id' if kind == 'ligands' else 'original_target_id'
                np.testing.assert_array_equal(z[key], entity[field].astype(str))
            assert len(z['drug_ids']) == len(z['drug_map']) == len(z['U'])
            assert len(z['target_ids']) == len(z['target_map']) == len(z['V'])
        reload_record = json.loads((ROOT / f'results/tables/task2_{dataset}_reload.json').read_text())
        assert reload_record['fresh_process'] and reload_record['ranking_order_identical']
        assert reload_record['model_sha256'] == sha256(ROOT / f'saved_models/{dataset}_cf.npz')
        bias, knn, means = load_warm_aux(dataset)
        metrics = pd.read_csv(ROOT / f'results/tables/task2_{dataset}_metrics.csv')
        for split, part in parts.items():
            saved = read_predictions(ROOT / f'results/predictions/task2_{dataset}_{split}.csv')
            np.testing.assert_array_equal(saved.pair_index, part.index)
            np.testing.assert_allclose(saved.affinity, part.affinity, atol=1e-12)
            np.testing.assert_allclose(saved.MF, model.predict(part), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(saved.bias_only, bias.predict(part), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(saved.target_knn, knn.predict(part)[0], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(saved.ligand_mean, means['drug_mean'][part.drug_id], atol=1e-12)
            np.testing.assert_allclose(saved.target_mean, means['target_mean'][part.target_id], atol=1e-12)
            compare_metrics(part, saved, metrics[metrics.split == split])
        cases = pd.read_csv(ROOT / f'results/tables/task2_{dataset}_case_summary.csv')
        ranks = pd.read_csv(ROOT / f'results/tables/task2_{dataset}_case_rankings.csv')
        assert len(cases) == 3 and cases.top10_defined.all()
        for drug in cases.drug_id:
            query = parts['test'][parts['test'].drug_id == drug]
            saved = ranks[ranks.drug_id == drug]
            mapping = dict(zip(query.target_id, model.predict(query)))
            assert saved.target_id.isin(query.target_id).all()
            np.testing.assert_allclose(saved.prediction, saved.target_id.map(mapping), atol=1e-12)
        checks.append(dataset + ': raw checksums/clean counts/missingness, warm leakage, all saved predictions/metrics, cases, checkpoint entity mapping and fresh reload')
        for seed in COLD_SEEDS:
            _, cold_parts, cold_manifest = load_splits(dataset, 'task3', seed)
            run = json.loads((ROOT / f'results/tables/task3_{dataset}_seed{seed}_run.json').read_text())
            assert not run['test_used_for_selection'] and run['evaluation_target_training_observations'] == 0
            assert run['split_sha256'] == cold_manifest['split_sha256']
            neighbors = ColdTargetNeighbors.load(ROOT / f'saved_models/{dataset}_neighbors_seed{seed}.npz')
            ridge = ColdFactorRidge.load(ROOT / f'saved_models/{dataset}_ridge_seed{seed}.npz')
            for name, estimator in [('neighbors', neighbors), ('ridge', ridge)]:
                assert set(estimator.target_ids) == set(cold_parts['train'].target_id)
                assert set(estimator.sequences) == set(cold_parts['train'].sequence)
                np.testing.assert_allclose(estimator.mu, cold_parts['train'].affinity.mean(), atol=1e-12)
                reload = json.loads((ROOT / f'results/tables/task3_{dataset}_seed{seed}_{name}_reload.json').read_text())
                assert reload['fresh_process'] and reload['ranking_order_identical']
                assert reload['model_sha256'] == sha256(ROOT / f'saved_models/{dataset}_{name}_seed{seed}.npz')
            observations = neighbors.observations
            assert int(np.isfinite(observations).sum()) == len(cold_parts['train'])
            target_lookup = {t: i for i, t in enumerate(neighbors.target_ids)}
            tr = cold_parts['train']
            np.testing.assert_allclose(observations[tr.drug_id, tr.target_id.map(target_lookup)], tr.affinity, atol=1e-12)
            tag = f'task3_{dataset}_seed{seed}'
            metrics = pd.read_csv(ROOT / f'results/tables/{tag}_metrics.csv')
            selected = metrics[(metrics.split == 'validation') & (metrics.model != 'random_ranking')].sort_values(['mse', 'ci_global'], ascending=[True, False]).iloc[0].model
            assert run['selected_model'] == selected
            for split in ['validation', 'test']:
                part = cold_parts[split]
                saved = read_predictions(ROOT / f'results/predictions/{tag}_{split}.csv')
                np.testing.assert_array_equal(saved.pair_index, part.index)
                np.testing.assert_allclose(saved.affinity, part.affinity, atol=1e-12)
                np.testing.assert_allclose(saved.sequence_neighbors, neighbors.predict(part)[0], atol=1e-12)
                np.testing.assert_allclose(saved.factor_ridge, ridge.predict(part), atol=1e-12)
                compare_metrics(part, saved, metrics[metrics.split == split])
            buckets = pd.read_csv(ROOT / f'results/tables/{tag}_similarity_buckets.csv')
            assert (buckets.groupby('model').pairs.sum() == len(cold_parts['test'])).all()
            checks.append(f'{dataset}/seed={seed}: cold sequence leakage, training-only archive contents, selected validation model, reloaded predictions, metrics, bucket totals')
    for path in (ROOT / 'saved_models').glob('*.npz'):
        with np.load(path, allow_pickle=False) as z:
            assert all(z[key].dtype != object for key in z.files), path
    import nbformat
    for path in (ROOT / 'notebooks').glob('*.ipynb'):
        nb = nbformat.read(path, as_version=4)
        code = [c for c in nb.cells if c.cell_type == 'code']
        assert code and all(c.execution_count is not None for c in code)
        assert not any(o.output_type == 'error' for c in code for o in c.outputs)
    checks.append('All NPZ arrays load without pickle; all three notebooks executed without errors')
    aggregate = pd.read_csv(ROOT / 'results/tables/task3_aggregate.csv')
    assert len(aggregate) == 6 and (aggregate.mse_count == 5).all()
    for path in (ROOT / 'results').glob('task*_report_notes.md'):
        assert path.stat().st_size > 1000
    checks.append('Five-split aggregates and substantive generated report notes present')
    source_files = [p for folder in ['src', 'scripts', 'tests', 'notebooks'] for p in (CODE_ROOT / folder).glob('*') if p.is_file()]
    source_files += [CODE_ROOT / name for name in ['README.md', 'requirements.txt', 'requirements.lock.txt', 'agenda (1).md']]
    artifacts = [p for folder in ['data/raw', 'data/processed', 'saved_models', 'results/tables', 'results/predictions', 'results/figures', 'reports'] for p in (ROOT / folder).rglob('*') if p.is_file()]
    manifest = {**provenance(), 'platform': platform.platform(),
                'packages': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions())),
                'commands': ['.venv/bin/python scripts/run_pipeline.py --smoke', '.venv/bin/python scripts/run_pipeline.py'],
                'source_files': {str(p.relative_to(CODE_ROOT)): sha256(p) for p in source_files},
                'artifacts': {str(p.relative_to(ROOT)): sha256(p) for p in artifacts},
                'precision': 'Clean CSV labels frozen to 12 significant digits; prediction CSV 17 significant digits with round-trip parser; NPZ float64',
                'metric_audit_policy': 'Recomputing metrics from frozen scores verifies artifacts; no model refit or reselection from test',
                'workspace_policy': 'All experiment outputs belong to v1; archive or use another DTR_ROOT for a new version. Cached runs do not re-evaluate test.'}
    write_json(ROOT / 'results/reproducibility_manifest.json', manifest)
    write_json(ROOT / 'results/verification.json', {**provenance(), 'status': 'PASS', 'checks': checks, 'artifact_count': len(artifacts), 'source_count': len(source_files)})
    log('Phase 6 verification', f'PASS: {len(checks)} independent artifact reconciliation groups; {len(artifacts)} artifact hashes; fresh reload evidence and executed notebooks verified. No training/tuning occurred in this audit.')
    print('\n'.join(checks))
    print('VERIFICATION PASS', len(artifacts), 'artifact hashes')
if __name__ == '__main__':
    verify()
