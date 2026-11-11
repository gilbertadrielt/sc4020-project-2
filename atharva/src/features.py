"""Protein-sequence descriptors and similarity for cold-target prediction.

A cold target has no rows in training, so its ID carries no information. The
only thing left is its amino-acid sequence, which both of the mechanisms below
turn into something the model can use: a feature vector, and a similarity to
the targets that *were* seen in training.
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

AA = "ACDEFGHIKLMNPQRSTVWY"

# Coarse physicochemical groupings (the classic CTD partitions). Collapsing 20
# residues to 7 classes keeps the dipeptide space small enough to estimate from
# a few hundred training proteins.
GROUPS = {
    "aliphatic": "AGV", "small_polar": "STCNQ", "acidic": "DE", "basic": "KRH",
    "aromatic": "FYW", "hydrophobic": "ILMP", "other": "",
}
_G = {}
for gi, (_, residues) in enumerate(GROUPS.items()):
    for r in residues:
        _G[r] = gi
N_GROUPS = len(GROUPS)


def _clean(seq):
    return "".join(ch for ch in seq.upper() if ch in AA)


def descriptors(sequences):
    """Composition features: residues, group dipeptides, and length."""
    n = len(sequences)
    aa_idx = {a: i for i, a in enumerate(AA)}
    comp = np.zeros((n, len(AA)))
    grp_dip = np.zeros((n, N_GROUPS * N_GROUPS))
    lengths = np.zeros((n, 1))

    for i, raw in enumerate(sequences):
        s = _clean(raw)
        lengths[i, 0] = np.log1p(len(s))
        if not s:
            continue
        for ch in s:
            comp[i, aa_idx[ch]] += 1
        comp[i] /= len(s)
        g = np.fromiter((_G.get(ch, N_GROUPS - 1) for ch in s), int, len(s))
        if len(g) > 1:
            np.add.at(grp_dip[i], g[:-1] * N_GROUPS + g[1:], 1.0)
            grp_dip[i] /= (len(g) - 1)
    return np.hstack([comp, grp_dip, lengths])


def kmer_similarity(sequences, k=3):
    """Cosine similarity between L2-normalised k-mer count profiles.

    This stands in for a proper alignment identity: it is orders of magnitude
    cheaper, needs no external aligner, and preserves the ordering that matters
    here (are there close homologues of this target in training?).
    """
    docs = [_clean(s) for s in sequences]
    vec = CountVectorizer(analyzer="char", ngram_range=(k, k), lowercase=False, min_df=1)
    X = vec.fit_transform(docs).astype(np.float32)
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A.ravel()
    norms[norms == 0] = 1.0
    Xn = X.multiply(1.0 / norms[:, None]).tocsr()
    S = np.asarray((Xn @ Xn.T).todense(), dtype=float)
    np.clip(S, 0.0, 1.0, out=S)
    return S


def target_matrix(sequences, train_idx, n_components=48, seed=0):
    """Standardise descriptors and compress them, fitting only on training targets."""
    D = descriptors(sequences)
    scaler = StandardScaler().fit(D[train_idx])
    Z = scaler.transform(D)
    n_comp = int(min(n_components, len(train_idx) - 1, Z.shape[1]))
    pca = PCA(n_components=n_comp, random_state=seed).fit(Z[train_idx])
    return pca.transform(Z)
