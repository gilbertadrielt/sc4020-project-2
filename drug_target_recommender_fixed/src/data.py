"""Checksummed DeepDTA source loading and observed canonical pair cleaning."""
import json
import pickle
from collections import OrderedDict
import numpy as np
import pandas as pd
from rdkit import Chem
from .common import ROOT, sha256, write_immutable_csv, write_json, log

AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWYBXZJUO')


def load_raw(dataset):
    directory = ROOT / 'data/raw' / dataset
    source = json.loads((directory / 'source.json').read_text())
    for name, record in source['files'].items():
        if sha256(directory / name) != record['sha256']:
            raise ValueError(f'Raw checksum mismatch: {dataset}/{name}')
    drugs = json.loads((directory / 'ligands_can.txt').read_text(), object_pairs_hook=OrderedDict)
    targets = json.loads((directory / 'proteins.txt').read_text(), object_pairs_hook=OrderedDict)
    # Only load the provenance-checked, pinned upstream pickle, never user pickles.
    with (directory / 'Y').open('rb') as handle:
        matrix = np.asarray(pickle.load(handle, encoding='latin1'), dtype=float)
    if matrix.shape != (len(drugs), len(targets)):
        raise ValueError('Ordered entity mappings do not match affinity matrix')
    return drugs, targets, matrix, source


def convert_scores(values, scale):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    result = np.full(values.shape, np.nan)
    if scale == 'Kd_nM':
        valid &= values > 0
        result[valid] = 9 - np.log10(values[valid])
    elif scale in ('pKd', 'DeepDTA_KIBA'):
        result[valid] = values[valid]
    else:
        raise ValueError(f'Unknown score scale: {scale}')
    return result, valid


def canonical_smiles(value):
    molecule = Chem.MolFromSmiles(str(value).strip())
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f'Invalid SMILES: {value!r}')
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def normalize_sequence(value):
    sequence = ''.join(str(value).split()).upper()
    if len(sequence) < 3 or not set(sequence) <= AMINO_ACIDS:
        raise ValueError('Invalid or shorter-than-3 protein sequence')
    return sequence


def aggregate_observations(measured):
    """Merge only actually measured source pairs, preserving per-pair provenance."""
    grouped = measured.groupby(['drug_id', 'target_id'], sort=True)
    clean = grouped.agg(affinity=('affinity', 'mean'), n_merged=('affinity', 'size'),
                        affinity_std=('affinity', 'std'), affinity_min=('affinity', 'min'),
                        affinity_max=('affinity', 'max'), censored_fraction=('censored', 'mean'),
                        suspicious_fraction=('suspicious', 'mean'),
                        original_drug_ids=('original_drug_id', lambda v: '|'.join(dict.fromkeys(v))),
                        original_target_ids=('original_target_id', lambda v: '|'.join(dict.fromkeys(v))),
                        source_cells=('source_cell', '|'.join)).reset_index()
    clean['affinity_std'] = clean.affinity_std.fillna(0.)
    clean['is_censored'] = clean.censored_fraction > 0
    clean['is_suspicious'] = clean.suspicious_fraction > 0
    return clean


def prepare(dataset):
    drugs, targets, matrix, source = load_raw(dataset)
    scores, valid_score = convert_scores(matrix, source['input_scale'])
    exclusions = []
    canonical, sequences = [], []
    for kind, mapping, output, fn in [('ligand', drugs, canonical, canonical_smiles),
                                       ('target', targets, sequences, normalize_sequence)]:
        for key, value in mapping.items():
            try:
                output.append(fn(value))
            except ValueError as exc:
                output.append(None)
                exclusions.append({'entity_type': kind, 'source_id': key, 'reason': str(exc)})
    drug_unique = list(dict.fromkeys(v for v in canonical if v is not None))
    target_unique = list(dict.fromkeys(v for v in sequences if v is not None))
    dl, tl = {s: i for i, s in enumerate(drug_unique)}, {s: i for i, s in enumerate(target_unique)}
    dm = pd.DataFrame({'original_drug_id': list(drugs), 'original_row': range(len(drugs)),
                       'input_smiles': list(drugs.values()), 'smiles': canonical,
                       'drug_id': [dl.get(s, -1) for s in canonical]})
    tm = pd.DataFrame({'original_target_id': list(targets), 'original_column': range(len(targets)),
                       'sequence': sequences, 'target_id': [tl.get(s, -1) for s in sequences]})
    valid_entity = (dm.drug_id.to_numpy()[:, None] >= 0) & (tm.target_id.to_numpy()[None, :] >= 0)
    valid = valid_score & valid_entity
    rows, cols = np.where(valid)
    measured = pd.DataFrame({'drug_id': dm.drug_id.to_numpy()[rows], 'target_id': tm.target_id.to_numpy()[cols],
                             'affinity': scores[rows, cols],
                             'original_drug_id': np.asarray(list(drugs), dtype=str)[rows],
                             'original_target_id': np.asarray(list(targets), dtype=str)[cols],
                             'source_cell': [f'{r}:{c}' for r, c in zip(rows, cols)],
                             'censored': np.isclose(scores[rows, cols], 5., rtol=0, atol=1e-12) if dataset == 'davis' else False,
                             'suspicious': (scores[rows, cols] <= 0) if dataset == 'kiba' else False})
    clean = aggregate_observations(measured)
    clean['smiles'] = clean.drug_id.map(dict(enumerate(drug_unique)))
    clean['sequence'] = clean.target_id.map(dict(enumerate(target_unique)))
    clean['score'] = clean.affinity
    clean['ligand_id'] = clean.drug_id
    clean['dataset'] = dataset
    clean['score_name'] = 'pKd' if dataset == 'davis' else 'DeepDTA_KIBA'
    clean['source_commit'] = source['commit']
    assert not clean.duplicated(['smiles', 'sequence']).any()
    assert np.isfinite(clean.affinity).all() and clean.n_merged.sum() == valid.sum()
    expected = set(zip(dm.drug_id.to_numpy()[rows], tm.target_id.to_numpy()[cols]))
    assert set(zip(clean.drug_id, clean.target_id)) == expected, 'Missing cells were filled or observations lost'
    assert clean.smiles.str.len().min() > 0 and clean.sequence.str.len().min() >= 3
    directory = ROOT / 'data/processed'
    write_immutable_csv(clean, directory / f'pairs_{dataset}.csv')
    write_immutable_csv(dm, directory / f'{dataset}_drug_mapping.csv')
    write_immutable_csv(tm, directory / f'{dataset}_target_mapping.csv')
    ligand_table = dm[dm.drug_id >= 0].groupby(['drug_id', 'smiles'], sort=True).original_drug_id.agg('|'.join).reset_index()
    target_table = tm[tm.target_id >= 0].groupby(['target_id', 'sequence'], sort=True).original_target_id.agg('|'.join).reset_index()
    write_immutable_csv(ligand_table, directory / f'ligands_{dataset}.csv')
    write_immutable_csv(target_table, directory / f'targets_{dataset}.csv')
    write_immutable_csv(clean[clean.n_merged > 1], directory / f'{dataset}_duplicate_audit.csv')
    write_immutable_csv(measured[measured.suspicious], directory / f'{dataset}_suspicious_audit.csv')
    write_immutable_csv(pd.DataFrame(exclusions, columns=['entity_type', 'source_id', 'reason']), directory / f'{dataset}_exclusions.csv')
    finite = matrix[np.isfinite(matrix)]
    summary = {'dataset': dataset, 'source': source, 'input_scale': source['input_scale'],
               'score_name': clean.score_name.iloc[0], 'higher_is_stronger': True,
               'raw_ligands': len(drugs), 'raw_targets': len(targets), 'raw_cells': int(matrix.size),
               'raw_finite': int(np.isfinite(matrix).sum()), 'missing_nan': int(np.isnan(matrix).sum()),
               'invalid_infinite': int(np.isinf(matrix).sum()), 'invalid_nonpositive_kd': int((np.isfinite(matrix) & (matrix <= 0)).sum()) if source['input_scale'] == 'Kd_nM' else 0,
               'invalid_entities': len(exclusions), 'entity_excluded_observations': int((valid_score & ~valid_entity).sum()),
               'valid_observations': int(valid.sum()), 'unique_ligands': int(clean.drug_id.nunique()),
               'unique_targets': int(clean.target_id.nunique()), 'unique_pairs': len(clean),
               'merged_groups': int((clean.n_merged > 1).sum()),
               'conflicting_groups': int((clean.affinity_min != clean.affinity_max).sum()),
               'missing_fraction': float(1 - len(clean) / (clean.drug_id.nunique() * clean.target_id.nunique())),
               'raw_min': float(finite.min()), 'raw_max': float(finite.max()),
               'score_quantiles': {str(k): float(v) for k, v in clean.affinity.quantile([0, .01, .25, .5, .75, .99, 1]).items()},
               'censored_raw_measurements': int(measured.censored.sum()), 'censored_clean_pairs': int(clean.is_censored.sum()),
               'suspicious_raw_measurements': int(measured.suspicious.sum()),
               'suspicious_policy': 'Finite nonpositive KIBA scores are flagged and retained as supplied; no assay-level evidence establishes invalidity. No outlier deletion.',
               'clean_sha256': sha256(directory / f'pairs_{dataset}.csv'),
               'identity_policy': 'canonical isomeric SMILES and whitespace-normalized uppercase exact sequence; no truncation',
               'checks_passed': ['unique_pairs', 'finite_labels', 'entity_validity', 'observed_key_set_equality', 'measurement_total']}
    write_json(directory / f'{dataset}_cleaning_summary.json', summary)
    log('Phase 1', f'{dataset}: {summary["raw_finite"]} raw finite observations -> {len(clean)} unique pairs; missingness and canonical identity assertions PASS.')
    return summary
