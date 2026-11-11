"""Loading and cleaning of the DAVIS / KIBA drug-target affinity benchmarks."""
import io, json, os, pickle
import numpy as np

DATA_ROOT = os.environ.get("DTA_DATA_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "deepdta"))

# The DeepDTA score matrices ship as pickles. Restrict unpickling to the plain
# numpy array constructors so a tampered file cannot execute arbitrary code.
_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("numpy", "float64"), ("numpy", "float32"),
}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def _load_matrix(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    return np.asarray(_SafeUnpickler(io.BytesIO(raw), encoding="latin1").load(), dtype=float)


class Dataset:
    """Observed ligand-target pairs plus the sequences needed for cold targets.

    rows / cols are index arrays into ligand_ids / target_ids; y holds the score
    on the modelling scale (pKd for DAVIS, supplied composite score for KIBA).
    """

    def __init__(self, name, ligand_ids, target_ids, smiles, sequences, rows, cols, y, strong_threshold):
        self.name = name
        self.ligand_ids = ligand_ids
        self.target_ids = target_ids
        self.smiles = smiles
        self.sequences = sequences
        self.rows = rows
        self.cols = cols
        self.y = y
        self.strong_threshold = strong_threshold

    @property
    def n_ligands(self):
        return len(self.ligand_ids)

    @property
    def n_targets(self):
        return len(self.target_ids)

    @property
    def n_pairs(self):
        return len(self.y)


def load(name, root=DATA_ROOT):
    """Load one benchmark and apply the cleaning the brief asks for."""
    name = name.lower()
    ligands = json.load(open(f"{root}/{name}/ligands_can.txt"))
    proteins = json.load(open(f"{root}/{name}/proteins.txt"))
    Y = _load_matrix(f"{root}/{name}/Y")

    ligand_ids = list(ligands)
    target_ids = list(proteins)
    smiles = [ligands[k] for k in ligand_ids]
    sequences = [proteins[k] for k in target_ids]
    assert Y.shape == (len(ligand_ids), len(target_ids)), Y.shape

    finite = np.isfinite(Y)
    if name == "davis":
        # Kd in nM. Require a strictly positive Kd before taking the log, then
        # pKd = 9 - log10(Kd). Kd = 10000 nM maps to the pKd = 5 screening floor.
        valid = finite & (Y > 0)
        rows, cols = np.nonzero(valid)
        y = 9.0 - np.log10(Y[rows, cols])
        strong = 7.0
    else:
        valid = finite
        rows, cols = np.nonzero(valid)
        y = Y[rows, cols].astype(float)
        strong = 12.1  # the usual KIBA active/inactive cut used by DeepDTA

    ds = Dataset(name.upper(), ligand_ids, target_ids, smiles, sequences,
                 rows.astype(np.int32), cols.astype(np.int32), y, strong)
    ds.raw_shape = Y.shape
    ds.raw_finite = int(finite.sum())
    ds.dropped_nonpositive = int(finite.sum() - valid.sum())
    return ds


def cleaning_report(ds):
    """Counts before/after dedup, both by ID pair and by (SMILES, sequence) pair."""
    n_cells = ds.raw_shape[0] * ds.raw_shape[1]
    id_pairs = len(ds.y)  # the matrix layout makes every ID pair unique by construction

    smiles = np.asarray(ds.smiles)
    seqs = np.asarray(ds.sequences)
    seq_keys = [f"{smiles[r]}||{seqs[c]}" for r, c in zip(ds.rows, ds.cols)]
    uniq_seq_pairs = len(set(seq_keys))

    return {
        "dataset": ds.name,
        "matrix": f"{ds.raw_shape[0]} x {ds.raw_shape[1]}",
        "cells": n_cells,
        "finite_entries": ds.raw_finite,
        "dropped_nonpositive_Kd": ds.dropped_nonpositive,
        "observed_pairs": id_pairs,
        "sparsity_missing_frac": 1.0 - ds.raw_finite / n_cells,
        "unique_ligand_ids": ds.n_ligands,
        "unique_target_ids": ds.n_targets,
        "unique_smiles": len(set(ds.smiles)),
        "unique_sequences": len(set(ds.sequences)),
        "unique_smiles_seq_pairs": uniq_seq_pairs,
        "duplicate_seq_pairs": id_pairs - uniq_seq_pairs,
    }
