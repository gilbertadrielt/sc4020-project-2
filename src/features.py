"""Drug and target representations for Task 3.

Task 2 never needs these: a warm model identifies a target by the column it
occupies. Task 3 has to score a protein it has never seen, so the only thing
left to go on is what the protein *is* - its amino-acid sequence - and what the
drug is, its SMILES string.

Nothing here touches an affinity. A sequence and a SMILES string are known for
any molecule before a single assay is run, so using them for a held-out target
is not leakage; using its measurements would be, and the cold split is what
prevents that.

``kmer_profiles`` lives here rather than in :mod:`src.eda` because Task 3 is its
main consumer, but Task 1 found the result that motivates it: protein k-mer
cosine predicts affinity-profile correlation as a **threshold, not a trend**.
Below ``KMER_USEFUL_COSINE`` the median profile correlation is flat at about
0.25; above it, it jumps to about 0.9. A neighbour method must therefore take
the few neighbours above the threshold and fall back otherwise, rather than
averaging a similarity-weighted vote over every target.
"""

from __future__ import annotations

import numpy as np

#: The 20 standard amino acids, in the order that fixes the k-mer column index.
#: DAVIS and KIBA also contain ``X`` for an unknown residue, which is excluded.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

#: Protein 3-mer cosine above which affinity-profile correlation jumps from
#: ~0.25 to ~0.9 (the elbow measured in ``eda.figure_sequence_vs_affinity``).
#: Below it, sequence similarity carries almost no information about binding.
KMER_USEFUL_COSINE = 0.3


def kmer_profiles(sequences, k: int = 3) -> np.ndarray:
    """L2-normalised k-mer composition vectors, one row per sequence.

    The feature space is every k-mer over the 20 standard amino acids
    (8,000 columns for k=3). This is the cheapest protein representation that
    still separates families, and the one an ESM ablation is measured against.

    DAVIS and KIBA sequences contain ``X`` (unknown residue), so any window
    touching a non-standard letter is skipped instead of being mapped onto a
    real amino acid. Rows are L2-normalised, which makes a dot product between
    two rows their cosine similarity.
    """
    code = np.full(256, -1, dtype=np.int64)
    for i, aa in enumerate(AMINO_ACIDS):
        code[ord(aa)] = i

    base = len(AMINO_ACIDS)
    dim = base**k
    sequences = list(sequences)
    out = np.zeros((len(sequences), dim), dtype=np.float32)

    for row, seq in enumerate(sequences):
        letters = np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)
        if len(letters) < k:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(code[letters], k)
        idx = np.zeros(len(windows), dtype=np.int64)
        for j in range(k):
            idx = idx * base + windows[:, j]
        # A window containing an unknown residue has a negative column, which
        # poisons the running index, so those windows are dropped by position.
        keep = (windows >= 0).all(axis=1)
        out[row] = np.bincount(idx[keep], minlength=dim)

    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, np.finfo(np.float32).tiny)


def morgan_fingerprints(smiles, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Binary Morgan (ECFP-like) fingerprints, one row per SMILES string.

    An unparseable SMILES gives an all-zero row rather than raising, so one bad
    string cannot take down a whole run; callers that care can spot them as
    rows summing to zero. Both datasets parse cleanly today (Task 1 checked:
    0 failures across 68 DAVIS and 2,111 KIBA ligands).
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    smiles = list(smiles)
    out = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for row, text in enumerate(smiles):
        mol = Chem.MolFromSmiles(text or "")
        if mol is not None:
            out[row] = generator.GetFingerprintAsNumPy(mol)
    return out


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two sets of vectors."""
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return np.clip(a @ b.T, -1.0, 1.0)


def tanimoto_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Tanimoto similarity between two sets of binary fingerprints."""
    intersection = a @ b.T
    size_a, size_b = a.sum(axis=1), b.sum(axis=1)
    union = size_a[:, None] + size_b[None, :] - intersection
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, intersection / union, 0.0)
