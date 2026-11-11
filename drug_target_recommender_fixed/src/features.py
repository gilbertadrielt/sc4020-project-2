"""Label-free full-sequence 3-mer frequencies, fixed A–Z vocabulary."""
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

FEATURE_POLICY = 'A-Z character 3-mer counts / number of 3-mers, L2 normalized; fixed 26^3 vocabulary'


def sequence_features(sequences):
    rows, cols, values = [], [], []
    for i, sequence in enumerate(sequences):
        sequence = str(sequence).upper()
        if len(sequence) < 3 or not sequence.isalpha() or not sequence.isascii():
            raise ValueError('Expected at least three A-Z amino-acid characters')
        counts = Counter((ord(sequence[j]) - 65) * 676 + (ord(sequence[j + 1]) - 65) * 26
                         + ord(sequence[j + 2]) - 65 for j in range(len(sequence) - 2))
        for key, count in counts.items():
            rows.append(i)
            cols.append(key)
            values.append(count / (len(sequence) - 2))
    matrix = csr_matrix((values, (rows, cols)), shape=(len(sequences), 26 ** 3), dtype=float)
    return normalize(matrix, norm='l2')


def sequence_similarity(left, right):
    return np.clip((sequence_features(left) @ sequence_features(right).T).toarray(), 0., 1.)


def protein_composition(sequences, order=1):
    """Fixed 20-AA composition; unknown residues contribute no counts."""
    from itertools import product
    vocabulary = {''.join(k): i for i, k in enumerate(product('ACDEFGHIKLMNPQRSTVWY', repeat=order))}
    output = np.zeros((len(sequences), len(vocabulary)))
    for i, sequence in enumerate(sequences):
        for j in range(max(0, len(sequence) - order + 1)):
            key = sequence[j:j + order]
            if key in vocabulary:
                output[i, vocabulary[key]] += 1
        output[i] /= max(1, len(sequence) - order + 1)
    return output


def protein_features(sequences):
    from scipy.sparse import hstack
    # All blocks use fixed vocabularies; no corpus or held-out label fitting.
    return hstack([sequence_features(sequences), csr_matrix(protein_composition(sequences)),
                   csr_matrix(protein_composition(sequences, 2))], format='csr')


def cache_features(dataset, frame):
    from scipy.sparse import save_npz
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from .common import ROOT, write_json, sha256
    targets = frame[['target_id', 'sequence']].drop_duplicates().sort_values('target_id')
    drugs = frame[['drug_id', 'smiles']].drop_duplicates().sort_values('drug_id')
    directory = ROOT / 'data/processed/features'
    directory.mkdir(parents=True, exist_ok=True)
    proteins = protein_features(targets.sequence.tolist())
    save_npz(directory / f'{dataset}_proteins.npz', proteins)
    np.save(directory / f'{dataset}_protein_ids.npy', targets.target_id.to_numpy(), allow_pickle=False)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = np.stack([generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in drugs.smiles])
    np.savez_compressed(directory / f'{dataset}_ligands.npz', drug_ids=drugs.drug_id.to_numpy(), morgan=fingerprints)
    write_json(directory / f'{dataset}_manifest.json', {
        'protein_features': FEATURE_POLICY + ' + 20-AA and 400-dipeptide frequencies; no learned corpus transformation',
        'drug_features': 'RDKit Morgan radius=2 2048-bit; cached for reuse, C2 uses cold-train MF drug factors',
        'protein_shape': list(proteins.shape), 'morgan_shape': list(fingerprints.shape),
        'protein_sha256': sha256(directory / f'{dataset}_proteins.npz'),
        'ligand_sha256': sha256(directory / f'{dataset}_ligands.npz')})
