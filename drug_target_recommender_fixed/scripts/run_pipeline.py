"""Reproducible phase runner; smoke outputs are isolated from frozen experiments."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CODE_ROOT, ROOT
from src.common import setup, log, write_json, provenance


def create_smoke_raw(directory):
    """Subset ACTUAL source data, never use synthetic final benchmark results."""
    import json
    import pickle
    import numpy as np
    from src.data import load_raw
    from src.common import sha256
    for dataset in ['davis', 'kiba']:
        drugs, targets, matrix, source = load_raw(dataset)
        # Select by label-free observed support, not affinity magnitude.
        rows = np.argsort(-np.isfinite(matrix).sum(axis=1), kind='stable')[:24]
        cols = np.argsort(-np.isfinite(matrix[rows]).sum(axis=0), kind='stable')[:30]
        dest = directory / 'data/raw' / dataset
        dest.mkdir(parents=True, exist_ok=True)
        values = {'ligands_can.txt': json.dumps({list(drugs)[i]: list(drugs.values())[i] for i in rows}).encode(),
                  'proteins.txt': json.dumps({list(targets)[i]: list(targets.values())[i] for i in cols}).encode(),
                  'Y': pickle.dumps(matrix[np.ix_(rows, cols)], protocol=4)}
        for name, content in values.items():
            path = dest / name
            if path.exists() and path.read_bytes() != content:
                raise RuntimeError('Existing smoke raw selection differs; use a new smoke directory')
            if not path.exists():
                path.write_bytes(content)
        smoke_source = {**source, 'smoke_subset': True, 'parent_files': source['files'],
                        'files': {name: {'url': 'local subset of pinned upstream', 'sha256': sha256(dest / name),
                                         'bytes': len(content)} for name, content in values.items()}}
        write_json(dest / 'source.json', smoke_source)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MPLBACKEND='Agg')
    if a.smoke:
        destination = CODE_ROOT / 'results/smoke/v1'
        create_smoke_raw(destination)
        env['DTR_ROOT'] = str(destination)
        phases = [('smoke cleaning', ['scripts/prepare_data.py']), ('smoke frozen splits', ['scripts/make_splits.py']),
                  ('smoke CF', ['scripts/train_cf.py', '--smoke']),
                  ('smoke cold', ['scripts/train_cold.py', '--smoke', '--seeds', '2026'])]
    else:
        destination = ROOT
        phases = [('Phase 0 source verification/download', ['scripts/download_data.py']),
                  ('Phase 0/1 tests', ['-m', 'pytest', 'tests/test_data.py', '-q']),
                  ('Phase 1 cleaning', ['scripts/prepare_data.py']),
                  ('Phase 2/3 tests', ['-m', 'pytest', 'tests/test_splits.py', 'tests/test_metrics.py', '-q']),
                  ('Phase 2 split freeze', ['scripts/make_splits.py']),
                  ('Phase 3/4 model tests', ['-m', 'pytest', 'tests/test_cf.py', 'tests/test_cold.py', '-q']),
                  ('Phase 4 Task 2', ['scripts/train_cf.py']), ('Phase 5 Task 3', ['scripts/train_cold.py']),
                  ('Phase 4 optional thinning', ['scripts/sparsity_experiment.py']),
                  ('Phase 6 exports', ['scripts/export_results.py']),
                  ('Phase 6 notebooks', ['scripts/build_notebooks.py', '--execute']),
                  ('Phase 6 tests', ['-m', 'pytest', '-q']), ('Phase 6 verification', ['scripts/verify_outputs.py'])]
    records = []
    for phase, args in phases:
        tick = time.perf_counter()
        print(phase, flush=True)
        result = subprocess.run([sys.executable, *args], cwd=CODE_ROOT, env=env)
        records.append({'phase': phase, 'command': [sys.executable, *args], 'exit_code': result.returncode,
                        'seconds': time.perf_counter() - tick})
        write_json(destination / 'results/pipeline_run.json', {'smoke': a.smoke, **provenance(), 'phases': records})
        if result.returncode:
            log(phase, f'FAILED with exit code {result.returncode}; inspect command in pipeline_run.json')
            raise SystemExit(result.returncode)
        log(phase, 'PASS' + (' (isolated real-data subset)' if a.smoke else ''))
    if a.smoke:
        log('Smoke', f'All stages PASS on actual-source subsets in {destination}; final frozen splits unchanged.')

if __name__ == '__main__':
    main()
