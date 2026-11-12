"""Similarity matrices, treated as first-class objects rather than hidden inside a model.

Pulling kernels out into their own module is what makes the kernel methods in
this project cheap.  The same target kernel is used by the neighbourhood
model, by KronRLS and by the similarity-bucket analysis, so it is built and
checked once.

Three sources of similarity, with different availability:

``sequence``
    How alike two proteins are as amino-acid strings.  Available for a protein
    nobody has ever tested, which is what makes the cold-target task solvable
    at all.
``chemical``
    How alike two molecules are structurally.  Available for any molecule.
``profile``
    How alike two entities' *measured affinity profiles* are.  The
    collaborative signal: strong where data exists, and completely unavailable
    for an unseen entity.  It is computed from labels, so it must only ever
    see training rows.

A note on indexing.  The benchmark ships similarity matrices indexed by source
identifier (442 DAVIS targets), while this project keys entities by chemistry
(379 distinct sequences).  ``collapse_to_entities`` averages the rows and
columns of each merged group, and asserts that members of a group really were
near-identical, which is a useful check on the deduplication itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_RAW, FEATURES
from .data import SIM_FILES, load_raw

# ---------------------------------------------------------------------------
# Loading and index alignment
# ---------------------------------------------------------------------------


def load_raw_similarity(name: str, kind: str) -> np.ndarray:
    """Read a benchmark similarity file, indexed by source identifier."""
    path = DATA_RAW / name / SIM_FILES[name][kind]
    try:
        M = np.loadtxt(path)
    except ValueError:
        M = np.loadtxt(path, delimiter=",")
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"{path} is {M.shape}, expected a square matrix")
    return M.astype(float)


def normalise_similarity(S: np.ndarray) -> np.ndarray:
    """Cosine-style normalisation so self-similarity is 1 and values are comparable.

    Raw Smith-Waterman alignment scores grow with sequence length: DAVIS
    diagonals run from about 1,000 to 7,500, so an unnormalised score says as
    much about protein size as about relatedness.  Dividing by the geometric
    mean of the two self-scores removes that.
    """
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    out = S / np.outer(d, d)
    return np.clip(out, -1.0, 1.0)


def collapse_to_entities(S: np.ndarray, raw_ids: list, entity_ids: pd.Series) -> np.ndarray:
    """Map an identifier-indexed matrix onto chemistry-indexed entities.

    ``entity_ids`` holds the pipe-joined source identifiers merged into each
    entity.  Members of a group are the same molecule or sequence, so their
    similarity rows should already agree; they are averaged, and the
    disagreement is checked rather than assumed.
    """
    pos = {rid: i for i, rid in enumerate(raw_ids)}
    groups = [[pos[r] for r in str(ids).split("|") if r in pos] for ids in entity_ids]
    if any(len(g) == 0 for g in groups):
        missing = sum(len(g) == 0 for g in groups)
        raise KeyError(f"{missing} entities have no row in the similarity matrix")

    n = len(groups)
    rows = np.vstack([S[g, :].mean(axis=0) for g in groups])
    out = np.column_stack([rows[:, g].mean(axis=1) for g in groups])

    # Members of a merged group claim to be identical chemistry; if their
    # similarity profiles disagree badly, the merge was wrong.
    worst = 0.0
    for g in groups:
        if len(g) > 1:
            block = S[np.ix_(g, g)]
            worst = max(worst, float(np.ptp(block)))
    if worst > 0.25:
        raise AssertionError(
            f"merged entities disagree by up to {worst:.3f} in similarity; "
            "deduplication may have merged distinct chemistry"
        )
    assert out.shape == (n, n)
    return out


# ---------------------------------------------------------------------------
# Kernel hygiene
# ---------------------------------------------------------------------------


def make_psd(K: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, float]:
    """Project onto the nearest positive semi-definite matrix by clipping eigenvalues.

    Published similarity matrices are not guaranteed to be valid kernels.
    Kernel ridge regression assumes PSD, and a negative eigenvalue can make
    the regularised solve indefinite, so the matrix is symmetrised and its
    negative eigenvalues are clipped to zero.  The smallest original
    eigenvalue is returned so the report can state how far off it was.
    """
    K = 0.5 * (K + K.T)
    w, V = np.linalg.eigh(K)
    smallest = float(w.min())
    if smallest < -tol:
        K = (V * np.clip(w, 0.0, None)) @ V.T
        K = 0.5 * (K + K.T)
    return K, smallest


def centre_kernel(K: np.ndarray) -> np.ndarray:
    """Centre a kernel in feature space (needed for kernel alignment).

    ``H K H`` with ``H = I - 11'/n`` expands to subtracting the row means and
    the column means and adding back the grand mean.  Doing it that way is
    O(n^2) instead of the O(n^3) of two dense matrix products, which matters
    here: KIBA's drug kernel is 2068 x 2068 and the explicit form costs about
    twenty seconds every time it is called.
    """
    row = K.mean(axis=0, keepdims=True)
    col = K.mean(axis=1, keepdims=True)
    return K - row - col + K.mean()


# ---------------------------------------------------------------------------
# Sequence and chemical kernels built from scratch
# ---------------------------------------------------------------------------


def kmer_kernel(sequences, k: int = 3, max_features: int = 8192) -> np.ndarray:
    """Cosine similarity of TF-IDF weighted k-mer counts.

    A cheap, dependency-light stand-in for alignment: proteins are treated as
    documents and their length-k fragments as words.  Useful as a second
    kernel for the multiple-kernel experiment, and as a fallback when the
    benchmark's alignment matrix is unavailable.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        analyzer="char", ngram_range=(k, k), lowercase=False, max_features=max_features
    )
    X = vec.fit_transform(list(sequences))
    X = X.multiply(1.0 / np.sqrt(X.multiply(X).sum(axis=1).A1 + 1e-12)[:, None])
    return np.asarray((X @ X.T).todense(), dtype=float)


def morgan_fingerprints(smiles, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Binary structural fingerprints: one bit per local substructure."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdFingerprintGenerator

        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return _hashed_char_ngrams(smiles, n_bits)

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        out[i] = np.frombuffer(
            gen.GetFingerprintAsNumPy(mol).astype(np.uint8).tobytes(), dtype=np.uint8
        )
    return out


def _hashed_char_ngrams(smiles, n_bits: int) -> np.ndarray:
    """Fallback fingerprint when RDKit is unavailable."""
    out = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, s in enumerate(smiles):
        for n in (2, 3, 4):
            for j in range(len(s) - n + 1):
                out[i, hash(s[j : j + n]) % n_bits] = 1
    return out


def tanimoto_kernel(fp: np.ndarray) -> np.ndarray:
    """Tanimoto (Jaccard) similarity between binary fingerprints."""
    X = np.asarray(fp, dtype=float)
    inter = X @ X.T
    counts = X.sum(axis=1)
    union = counts[:, None] + counts[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def profile_kernel(
    pairs: pd.DataFrame,
    n_entities: int,
    axis: str = "tgt",
    train_mask=None,
    min_overlap: int = 3,
    shrinkage: float = 10.0,
) -> np.ndarray:
    """Similarity of measured affinity profiles, from training rows only.

    This is the collaborative signal, and it is the one kernel that can leak:
    it is built from labels.  ``train_mask`` is therefore required in any
    evaluation context.

    Two corrections matter on this data.  Profiles are centred per entity
    before correlating, because on DAVIS two inactive compounds are both a
    wall of the same censored value and would otherwise look perfectly
    similar.  And the correlation is shrunk towards zero by the number of
    co-measured partners, ``n / (n + shrinkage)``, because on KIBA two drugs
    share a median of only about seven targets and a correlation from three
    points is noise.
    """
    if train_mask is not None:
        pairs = pairs[np.asarray(train_mask, dtype=bool)]

    a_col = f"{axis}_idx"
    b_col = "lig_idx" if axis == "tgt" else "tgt_idx"
    n_b = int(pairs[b_col].max()) + 1 if len(pairs) else 1

    M = np.full((n_entities, n_b), np.nan)
    M[pairs[a_col].to_numpy(), pairs[b_col].to_numpy()] = pairs["score"].to_numpy()

    observed = np.isfinite(M)
    centred = np.where(observed, M - np.nanmean(M, axis=1, keepdims=True), 0.0)
    centred = np.nan_to_num(centred)

    overlap = observed.astype(float) @ observed.astype(float).T
    cov = centred @ centred.T
    norm = np.sqrt(np.outer(np.einsum("ij,ij->i", centred, centred),
                            np.einsum("ij,ij->i", centred, centred)))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(norm > 1e-12, cov / norm, 0.0)

    corr = np.where(overlap >= min_overlap, corr, 0.0)
    corr *= overlap / (overlap + shrinkage)
    np.fill_diagonal(corr, 1.0)
    return np.clip(np.nan_to_num(corr), -1.0, 1.0)


# ---------------------------------------------------------------------------
# Building and caching the standard set
# ---------------------------------------------------------------------------


def build_kernels(name: str, proc: dict, psd: bool = True, cache: bool = True) -> dict:
    """Every content kernel for one dataset, entity-indexed and PSD-projected.

    Profile kernels are deliberately NOT built here: they depend on labels and
    therefore on the split, so they are constructed per fold by the models
    that use them.
    """
    path = FEATURES / f"kernels_{name}.npz"
    if cache and path.exists():
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}

    ligands, targets = proc["ligands"], proc["targets"]
    raw = __import__("dta.data", fromlist=["load_raw"]).load_raw(name)

    out: dict[str, np.ndarray] = {}
    diag: dict[str, float] = {}

    # Target: the benchmark's own alignment scores, normalised and collapsed.
    S_t = normalise_similarity(load_raw_similarity(name, "target"))
    S_t = collapse_to_entities(S_t, raw.target_ids, targets["target_ids"])
    out["target_sequence"], diag["target_sequence"] = (make_psd(S_t) if psd else (S_t, 0.0))

    # Target: an independent k-mer view, for the multiple-kernel experiment.
    S_k = kmer_kernel(targets["sequence"].tolist())
    out["target_kmer"], diag["target_kmer"] = (make_psd(S_k) if psd else (S_k, 0.0))

    # Drug: the benchmark's structural similarity.
    S_d = normalise_similarity(load_raw_similarity(name, "drug"))
    S_d = collapse_to_entities(S_d, raw.ligand_ids, ligands["ligand_ids"])
    out["drug_structural"], diag["drug_structural"] = (make_psd(S_d) if psd else (S_d, 0.0))

    # Drug: Tanimoto on Morgan fingerprints, computed here rather than taken
    # from the benchmark, so the two drug kernels are genuinely independent.
    fp = morgan_fingerprints(ligands["smiles"].tolist())
    S_m = tanimoto_kernel(fp)
    out["drug_morgan"], diag["drug_morgan"] = (make_psd(S_m) if psd else (S_m, 0.0))
    out["drug_fingerprints"] = fp

    out["_min_eigenvalues"] = np.array(
        [(k, v) for k, v in diag.items()], dtype=object
    )
    if cache:
        np.savez_compressed(path, **out)
    return out


# ---------------------------------------------------------------------------
# Explicit feature vectors (for the models that are not kernel methods)
# ---------------------------------------------------------------------------

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def composition_features(sequences) -> np.ndarray:
    """Amino-acid (20) and dipeptide (400) composition: 420 numbers per protein.

    A standard, cheap protein descriptor.  It captures what a sequence is
    made of and a little of how it is ordered, needs no pretrained model, and
    is length-normalised so a 2,500-residue kinase is comparable with a
    300-residue one.
    """
    idx = {a: i for i, a in enumerate(AMINO_ACIDS)}
    n_aa = len(AMINO_ACIDS)
    out = np.zeros((len(sequences), n_aa + n_aa * n_aa), dtype=np.float32)
    for r, seq in enumerate(sequences):
        s = [idx[c] for c in str(seq) if c in idx]
        if not s:
            continue
        counts = np.bincount(s, minlength=n_aa).astype(np.float32)
        out[r, :n_aa] = counts / counts.sum()
        if len(s) > 1:
            a = np.asarray(s[:-1])
            b = np.asarray(s[1:])
            di = np.bincount(a * n_aa + b, minlength=n_aa * n_aa).astype(np.float32)
            out[r, n_aa:] = di / max(di.sum(), 1.0)
    return out


def kmer_svd_features(sequences, k: int = 3, n_components: int = 256,
                      seed: int = 2026) -> np.ndarray:
    """TF-IDF over k-mers, reduced by truncated SVD.

    There are 8,000 possible 3-mers against 229 to 379 proteins, so the matrix
    is far wider than it is tall and the SVD is doing real work rather than
    cosmetic compression.  Unlike ``composition_features`` this keeps some
    local sequence order, which is where the binding site lives.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [" ".join(str(s)[i:i + k] for i in range(max(len(str(s)) - k + 1, 0)))
            for s in sequences]
    sparse = TfidfVectorizer(analyzer="word", token_pattern=r"\S+",
                             min_df=2).fit_transform(docs)
    n_components = int(min(n_components, min(sparse.shape) - 1))
    reduced = TruncatedSVD(n_components=n_components,
                           random_state=seed).fit_transform(sparse)
    return np.asarray(reduced, dtype=np.float32)


def esm2_features(name: str, proc: dict, size: str = "650m") -> np.ndarray | None:
    """Mean-pooled ESM-2 embeddings, aligned to this project's entity indexing.

    The ``.npy`` files are produced on a GPU by ``scripts/colab_esm2.py`` and
    are indexed by source identifier in ``proteins.txt`` order, while entities
    here are distinct sequences.  Members of a merged group have a byte-identical
    sequence and therefore an identical embedding, so taking any one member's row
    is exact rather than an approximation.

    Returns ``None`` when the files are absent, so the project still runs
    end to end without a GPU step; the representation simply drops out of the
    ablation and the tables say so.
    """
    import json

    array_path = FEATURES / f"esm2_{size}_{name}.npy"
    order_path = FEATURES / "esm2_target_order.json"
    if not (array_path.exists() and order_path.exists()):
        return None

    embeddings = np.load(array_path).astype(np.float32)
    with open(order_path) as fh:
        source_ids = json.load(fh)[name]
    if len(source_ids) != embeddings.shape[0]:
        raise ValueError(
            f"ESM-2 for {name}: {embeddings.shape[0]} rows against "
            f"{len(source_ids)} saved identifiers")

    raw = load_raw(name)
    by_sequence = {seq: row for row, seq in enumerate(raw.sequences)}
    rows = []
    for sequence in proc["targets"]["sequence"]:
        row = by_sequence.get(sequence)
        if row is None:
            return None
        rows.append(row)
    return embeddings[np.asarray(rows)]


#: Protein representations compared in the Task 3 ablation, cheapest first.
#: The two ESM-2 blocks are present only when the embedding files are.
TARGET_BLOCKS = ("target", "target_kmer_svd", "target_sw_profile",
                 "target_esm2_35m", "target_esm2_650m")


def build_features(name: str, proc: dict, cache: bool = True) -> dict:
    """Explicit descriptors for the models that are not kernel methods.

    Five protein representations are built so that the cold-target models can
    be ablated over them rather than being tied to whichever one was written
    first.  Composition alone turned out to starve the feature-based models:
    they are the cheapest descriptors here and the weakest by a wide margin.
    """
    path = FEATURES / f"features_{name}.npz"
    if cache and path.exists():
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}

    ks = build_kernels(name, proc)
    sequences = proc["targets"]["sequence"].tolist()
    out = {
        "drug": np.asarray(ks["drug_fingerprints"], dtype=np.float32),
        "target": composition_features(sequences),
        "target_kmer_svd": kmer_svd_features(sequences),
        # Each protein described by its similarity to every other, which is the
        # KronRLS representation handed to a model that cannot take a kernel.
        "target_sw_profile": np.asarray(ks["target_sequence"], dtype=np.float32),
    }
    for size in ("35m", "650m"):
        block = esm2_features(name, proc, size=size)
        if block is not None:
            out[f"target_esm2_{size}"] = block
    if cache:
        np.savez_compressed(path, **out)
    return out


# ---------------------------------------------------------------------------
# Multiple kernels
# ---------------------------------------------------------------------------


def kernel_alignment(K: np.ndarray, Y: np.ndarray) -> float:
    """Centred kernel alignment between a kernel and an ideal target kernel.

    A closed-form, label-aware measure of how useful a kernel is, used to set
    multiple-kernel weights without an expensive search.
    """
    Kc, Yc = centre_kernel(K), centre_kernel(Y)
    num = float(np.sum(Kc * Yc))
    den = float(np.linalg.norm(Kc) * np.linalg.norm(Yc))
    return num / den if den > 1e-12 else 0.0


def combine_kernels(kernels: list, weights=None) -> np.ndarray:
    """Weighted sum of kernels, weights on the simplex (uniform by default)."""
    if weights is None:
        weights = np.full(len(kernels), 1.0 / len(kernels))
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 0.0, None)
    w = w / w.sum() if w.sum() > 0 else np.full(len(kernels), 1.0 / len(kernels))
    return sum(wi * Ki for wi, Ki in zip(w, kernels))


def alignment_weights(kernels: list, profile: np.ndarray) -> np.ndarray:
    """Weights proportional to each kernel's alignment with the measured profile.

    The weights are reportable in their own right: they say which source of
    similarity the data actually finds informative, which is a more
    interesting output than a single blended accuracy number.
    """
    a = np.array([max(kernel_alignment(K, profile), 0.0) for K in kernels])
    return a / a.sum() if a.sum() > 0 else np.full(len(kernels), 1.0 / len(kernels))


if __name__ == "__main__":
    import sys

    from . import data as D
    from .config import DATASETS

    for ds in [a for a in sys.argv[1:] if not a.startswith("--")] or list(DATASETS):
        proc = D.load_processed(ds)
        ks = build_kernels(ds, proc, cache="--no-cache" not in sys.argv)
        print(f"\n=== {ds.upper()}")
        for k, v in ks.items():
            if k.startswith("_"):
                continue
            arr = np.asarray(v)
            extra = ""
            if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                off = arr[~np.eye(len(arr), dtype=bool)]
                extra = f"  off-diag min/median/max {off.min():.3f} {np.median(off):.3f} {off.max():.3f}"
            print(f"  {k:20s} {str(arr.shape):16s}{extra}")
        for k, v in ks.get("_min_eigenvalues", []):
            print(f"  smallest eigenvalue before PSD projection: {k:20s} {float(v):+.3e}")
