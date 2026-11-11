"""Frozen pair holdout and exact-sequence-grouped cold protocols."""
import json
import numpy as np
import pandas as pd
from .common import ROOT, SEED, sha256, write_immutable_csv, write_json, log
from .config import COLD_SEEDS

PARTS = ('train', 'validation', 'test')


def warm_split(frame, seed=SEED):
    rng = np.random.default_rng(seed)
    labels = np.full(len(frame), 'train', dtype=object)
    # Stratify by ligand without inspecting scores. Small groups may be train only.
    for positions in frame.groupby('drug_id', sort=True).indices.values():
        order = rng.permutation(positions)
        n = len(order)
        a = max(1, int(.7 * n))
        b = max(a, int(.85 * n))
        labels[order[a:b]], labels[order[b:]] = 'validation', 'test'
    changes = []
    train_targets = set(frame.iloc[np.flatnonzero(labels == 'train')].target_id)
    for target in sorted(set(frame.target_id) - train_targets):
        candidates = np.flatnonzero((frame.target_id.to_numpy() == target) & (labels != 'train'))
        i = int(rng.choice(candidates))
        changes.append({'pair_index': i, 'from': str(labels[i]), 'to': 'train', 'reason': 'target coverage'})
        labels[i] = 'train'
    return labels, changes


def cold_split(frame, seed=SEED):
    # Sequence grouping remains safe even for an uncleaned ID alias input.
    if 'sequence' not in frame:
        raise ValueError('Cold splitting requires exact protein sequences')
    sequences = np.random.default_rng(seed).permutation(sorted(frame.sequence.unique()))
    a, b = int(.7 * len(sequences)), int(.85 * len(sequences))
    if a == 0 or a == b or b == len(sequences):
        raise ValueError('Need enough unique sequences for three nonempty partitions')
    assignments = {s: p for p, group in zip(PARTS, [sequences[:a], sequences[a:b], sequences[b:]]) for s in group}
    return frame.sequence.map(assignments).to_numpy()


def validate_split(frame, labels, task):
    assert len(labels) == len(frame) and set(labels) == set(PARTS), 'All three partitions must be nonempty'
    assert not frame.duplicated(['drug_id', 'target_id']).any()
    parts = {p: frame.iloc[np.flatnonzero(labels == p)] for p in PARTS}
    keys = {p: set(zip(f.drug_id, f.target_id)) for p, f in parts.items()}
    for i, a in enumerate(PARTS):
        for b in PARTS[i + 1:]:
            assert keys[a].isdisjoint(keys[b]), 'Pair leakage'
            if task == 'task3':
                assert set(parts[a].sequence).isdisjoint(parts[b].sequence), 'Sequence leakage'
                assert set(parts[a].target_id).isdisjoint(parts[b].target_id), 'Target-ID crosses sequence partitions'
    if task == 'task2':
        for p in PARTS[1:]:
            assert set(parts[p].drug_id) <= set(parts['train'].drug_id)
            assert set(parts[p].target_id) <= set(parts['train'].target_id)
    elif task != 'task3':
        raise ValueError(task)
    return parts


def split_directory(dataset, task, seed=SEED):
    base = ROOT / 'data/processed/splits' / task / dataset
    return base if task == 'task2' else base / f'seed_{seed}'


def make_splits(dataset):
    source = ROOT / f'data/processed/pairs_{dataset}.csv'
    frame = pd.read_csv(source)
    for task, seeds in [('task2', [SEED]), ('task3', COLD_SEEDS)]:
        for seed in seeds:
            labels, moves = warm_split(frame, seed) if task == 'task2' else (cold_split(frame, seed), [])
            parts = validate_split(frame, labels, task)
            directory = split_directory(dataset, task, seed)
            table = frame[['drug_id', 'target_id']].copy()
            table.insert(0, 'pair_index', np.arange(len(frame)))
            table['split'] = labels
            write_immutable_csv(table, directory / 'split.csv')
            entity = frame[['target_id', 'sequence']].drop_duplicates().copy()
            entity['split'] = [labels[frame.index[frame.target_id == t][0]] if task == 'task3' else 'warm' for t in entity.target_id]
            write_immutable_csv(entity, directory / 'target_entities.csv')
            summary = {'dataset': dataset, 'task': task, 'seed': seed,
                       'clean_sha256': sha256(source), 'split_sha256': sha256(directory / 'split.csv'),
                       'entity_sha256': sha256(directory / 'target_entities.csv'), 'coverage_repair': moves,
                       'policy': 'per-ligand pair stratification + target coverage repair' if task == 'task2' else 'exact sequence group holdout; unseen ligands retained and explicitly flagged with global fallback',
                       'parts': {p: {'pairs': len(f), 'fraction': len(f) / len(frame), 'ligands': int(f.drug_id.nunique()),
                                      'targets': int(f.target_id.nunique()),
                                      'unseen_ligands': sorted(map(int, set(f.drug_id) - set(parts['train'].drug_id))),
                                      'unseen_ligand_pairs': int((~f.drug_id.isin(parts['train'].drug_id)).sum())}
                                 for p, f in parts.items()}}
            path = directory / 'manifest.json'
            if path.exists() and json.loads(path.read_text()) != summary:
                raise RuntimeError(f'Frozen manifest changed: {path}')
            write_json(path, summary)
            log('Phase 2', f'{dataset}/{task}/seed={seed}: leakage assertions PASS; sizes {[len(p) for p in parts.values()]}; {len(moves)} coverage moves; split SHA256 {summary["split_sha256"]}.')
            print(dataset, task, seed, summary['parts'], flush=True)


def load_splits(dataset, task, seed=SEED):
    directory = split_directory(dataset, task, seed)
    manifest = json.loads((directory / 'manifest.json').read_text())
    source = ROOT / f'data/processed/pairs_{dataset}.csv'
    for path, expected in [(source, manifest['clean_sha256']), (directory / 'split.csv', manifest['split_sha256']),
                           (directory / 'target_entities.csv', manifest['entity_sha256'])]:
        if sha256(path) != expected:
            raise ValueError(f'Frozen artifact changed: {path}')
    frame, saved = pd.read_csv(source), pd.read_csv(directory / 'split.csv')
    np.testing.assert_array_equal(saved[['drug_id', 'target_id']], frame[['drug_id', 'target_id']])
    np.testing.assert_array_equal(saved.pair_index, np.arange(len(frame)))
    return frame, validate_split(frame, saved.split.to_numpy(), task), manifest
