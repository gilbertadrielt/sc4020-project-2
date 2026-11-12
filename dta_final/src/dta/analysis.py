"""Task 1 analytics: describing the data, and sizing up what is learnable.

Deliberately lightweight. The brief warns against fitting large models while
describing the data, so everything here is counting, alternating means, an
SVD and rank correlations.

Each function answers a question that a later task would otherwise have to
guess at:

``variance_decomposition``
    How much of the signal is "this drug binds a lot" and "this protein is
    easy to bind", and how much is genuine drug-by-protein interaction?  The
    interaction share is the ceiling on what any collaborative model can add
    over a bias model, so it turns "is a recommender necessary?" into
    arithmetic rather than opinion.

``svd_spectrum``
    Is the matrix close to low rank?  That is the assumption factorisation
    makes, and it is worth testing before relying on it.

``coverage_stats`` / ``overlap_stats``
    How much evidence exists per entity, and how much two entities share.
    Neighbourhood methods need overlap, so these predict in advance where
    they will fall back.

``missingness_bias``
    Are the measured cells a random sample?  If not, every held-out estimate
    inherits the bias, which belongs in the limitations rather than being
    quietly ignored.

``similarity_vs_profile``
    Do sequence-similar proteins bind similarly?  The cold-target task
    replaces a missing protein's history with its sequence, so this is the
    assumption that whole task stands on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ---------------------------------------------------------------------------
# Variance
# ---------------------------------------------------------------------------


def fit_additive(pairs, n_lig, n_tgt, n_iter=200, tol=1e-12) -> dict:
    """Unregularised mu + b_i + c_j by alternating means.

    Unregularised on purpose: Task 1 describes the data in hand rather than
    predicting unseen data, so the best additive *fit* is what is wanted, not
    the best additive *predictor*.
    """
    li = pairs["lig_idx"].to_numpy()
    ti = pairs["tgt_idx"].to_numpy()
    y = pairs["score"].to_numpy(dtype=float)

    mu = float(y.mean())
    b = np.zeros(n_lig)
    c = np.zeros(n_tgt)
    nl = np.maximum(np.bincount(li, minlength=n_lig), 1)
    nt = np.maximum(np.bincount(ti, minlength=n_tgt), 1)

    prev = np.inf
    for _ in range(n_iter):
        b = np.bincount(li, weights=y - mu - c[ti], minlength=n_lig) / nl
        c = np.bincount(ti, weights=y - mu - b[li], minlength=n_tgt) / nt
        resid = y - mu - b[li] - c[ti]
        sse = float(resid @ resid)
        if abs(prev - sse) < tol * max(sse, 1.0):
            break
        prev = sse
    return {"mu": mu, "b": b, "c": c, "fitted": mu + b[li] + c[ti], "resid": resid}


def variance_decomposition(pairs, n_lig, n_tgt) -> pd.DataFrame:
    """Share of score variance reachable by each kind of model."""
    y = pairs["score"].to_numpy(dtype=float)
    li = pairs["lig_idx"].to_numpy()
    ti = pairs["tgt_idx"].to_numpy()
    total = float(y.var())

    lig_mean = pd.Series(y).groupby(li).transform("mean").to_numpy()
    tgt_mean = pd.Series(y).groupby(ti).transform("mean").to_numpy()
    add = fit_additive(pairs, n_lig, n_tgt)

    def share(pred):
        return 1.0 - float(np.mean((y - pred) ** 2)) / total

    rows = [
        {"component": "ligand identity alone", "variance_explained": share(lig_mean)},
        {"component": "target identity alone", "variance_explained": share(tgt_mean)},
        {"component": "additive (mu + b + c)", "variance_explained": share(add["fitted"])},
    ]
    rows.append({
        "component": "interaction (CF headroom)",
        "variance_explained": 1.0 - rows[2]["variance_explained"],
    })
    out = pd.DataFrame(rows)
    out["total_variance"] = total
    return out


# ---------------------------------------------------------------------------
# Rank structure
# ---------------------------------------------------------------------------


def densest_block(matrix, n_rows=300, n_cols=150) -> np.ndarray:
    """The rows and columns with the most observations.

    KIBA is three quarters empty, so an SVD of the whole matrix would largely
    describe whatever filled the holes. Restricting to a dense corner keeps
    the claim honest, and the figure caption says which block it is.
    """
    obs = np.isfinite(matrix)
    rows = np.argsort(-obs.sum(axis=1))[:n_rows]
    sub = matrix[rows]
    cols = np.argsort(-np.isfinite(sub).sum(axis=0))[:n_cols]
    return sub[:, cols]


def svd_spectrum(matrix, n_components=20, centre=True) -> pd.DataFrame:
    """Energy per singular value, after the least-committal filling available.

    Missing cells are filled with the additive fit of the observed cells,
    which adds no interaction structure of its own, so any interaction the
    SVD then finds was present in the data rather than in the imputation.
    """
    M = np.array(matrix, dtype=float)
    mask = np.isfinite(M)
    if not mask.all():
        mu = np.nanmean(M)
        row = np.nan_to_num(np.nanmean(M, axis=1, keepdims=True), nan=mu)
        col = np.nan_to_num(np.nanmean(M, axis=0, keepdims=True), nan=mu)
        M = np.where(mask, M, row + col - mu)
    if centre:
        M = M - M.mean()

    s = np.linalg.svd(M, compute_uv=False)
    energy = s**2 / np.sum(s**2)
    n = min(n_components, len(s))
    return pd.DataFrame({
        "component": np.arange(1, n + 1),
        "singular_value": s[:n],
        "energy": energy[:n],
        "cumulative_energy": np.cumsum(energy)[:n],
        "imputed_fraction": float(1.0 - mask.mean()),
    })


# ---------------------------------------------------------------------------
# Coverage, overlap, missingness
# ---------------------------------------------------------------------------


def coverage_stats(pairs, n_lig, n_tgt) -> pd.DataFrame:
    rows = []
    for axis, col, n in (("ligand", "lig_idx", n_lig), ("target", "tgt_idx", n_tgt)):
        counts = np.bincount(pairs[col].to_numpy(), minlength=n)
        rows.append({
            "axis": axis, "n_entities": int(n),
            "min": int(counts.min()), "median": float(np.median(counts)),
            "max": int(counts.max()), "mean": float(counts.mean()),
            "n_with_fewer_than_5": int((counts < 5).sum()),
        })
    return pd.DataFrame(rows)


def overlap_stats(pairs, n_lig, n_tgt, max_entities=4000) -> pd.DataFrame:
    """How many partners two entities share: the evidence behind a similarity."""
    rows = []
    for axis, a_col, b_col, n_a, n_b in (
        ("ligand pairs", "lig_idx", "tgt_idx", n_lig, n_tgt),
        ("target pairs", "tgt_idx", "lig_idx", n_tgt, n_lig),
    ):
        if n_a > max_entities:
            rows.append({"axis": axis, "note": f"skipped: {n_a} entities"})
            continue
        F = np.zeros((n_a, n_b), dtype=np.int32)
        F[pairs[a_col].to_numpy(), pairs[b_col].to_numpy()] = 1
        O = F @ F.T
        iu = np.triu_indices(n_a, 1)
        o = O[iu]
        rows.append({
            "axis": axis, "n_pairs_compared": int(o.size),
            "median_overlap": float(np.median(o)), "mean_overlap": float(o.mean()),
            "frac_overlap_lt_5": float(np.mean(o < 5)),
            "frac_overlap_zero": float(np.mean(o == 0)), "note": "",
        })
    return pd.DataFrame(rows)


def missingness_bias(pairs, n_lig, n_tgt) -> pd.DataFrame:
    """Does how much an entity was measured relate to how it scored?"""
    rows = []
    for axis, col in (("ligand", "lig_idx"), ("target", "tgt_idx")):
        g = pairs.groupby(col)["score"]
        counts, means = g.size().to_numpy(), g.mean().to_numpy()
        if len(counts) < 3 or np.ptp(counts) == 0:
            rows.append({"axis": axis, "spearman_rho": np.nan, "p_value": np.nan,
                         "note": "constant observation count (complete matrix)"})
            continue
        rho, p = spearmanr(counts, means)
        rows.append({"axis": axis, "spearman_rho": float(rho), "p_value": float(p), "note": ""})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Abnormal values and duplicates
# ---------------------------------------------------------------------------


def censoring_stats(pairs) -> pd.DataFrame:
    if "censored" not in pairs or not pairs["censored"].any():
        return pd.DataFrame([{"metric": "censored pairs", "value": 0.0, "note": "none"}])
    cens = pairs["censored"].to_numpy()
    per_lig = pairs.groupby("lig_idx")["censored"].mean()
    return pd.DataFrame([
        {"metric": "censored pairs", "value": float(cens.sum()),
         "note": "label sits on the assay ceiling"},
        {"metric": "censored fraction", "value": float(cens.mean()), "note": ""},
        {"metric": "ligands >90% censored", "value": float((per_lig > 0.9).sum()),
         "note": "near-inactive compounds"},
        {"metric": "ligands <25% censored", "value": float((per_lig < 0.25).sum()),
         "note": "promiscuous compounds"},
        {"metric": "min per-ligand censored fraction", "value": float(per_lig.min()), "note": ""},
        {"metric": "max per-ligand censored fraction", "value": float(per_lig.max()), "note": ""},
    ])


def duplicate_conflicts(pairs, top=10) -> pd.DataFrame:
    """Merged duplicates ranked by how much their labels disagreed.

    The concrete argument for keying on chemistry rather than identifiers:
    the same molecule against the same sequence, reported twice, with answers
    that differ by orders of magnitude.
    """
    if "n_repeats" not in pairs:
        return pd.DataFrame()
    dup = pairs[pairs["n_repeats"] > 1]
    if dup.empty:
        return dup
    cols = [c for c in ["ligand_id", "target_id", "n_repeats", "score",
                        "score_min", "score_max", "spread"] if c in dup.columns]
    return dup.sort_values("spread", ascending=False)[cols].head(top)


def completion_requirement(n_lig, n_tgt, n_obs, ranks=(1, 2, 4, 8, 16, 32, 64)) -> pd.DataFrame:
    """Observations needed for matrix completion at each rank, against what exists.

    Recovering a rank-r matrix needs on the order of r(m+n)log(n) observed
    entries (Candes & Recht, 2009).  Comparing that with the training set
    says which ranks the data can support before anything is fitted.
    """
    m, n = n_lig, n_tgt
    need = [r * (m + n) * np.log(max(n, 2)) for r in ranks]
    return pd.DataFrame({
        "rank_r": list(ranks),
        "free_parameters": [r * (m + n) for r in ranks],
        "observations_needed": [float(x) for x in need],
        "observations_available": int(n_obs),
        "supported": [bool(n_obs >= x) for x in need],
        "observations_per_parameter": [n_obs / (r * (m + n)) for r in ranks],
    })


def similarity_vs_profile(S, matrix, min_overlap=5, max_pairs=200_000, seed=2026) -> dict:
    """Correlate sequence similarity with affinity-profile similarity."""
    M = np.array(matrix, dtype=float)
    n = M.shape[1]
    obs = np.isfinite(M)
    rng = np.random.default_rng(seed)

    iu = np.triu_indices(n, 1)
    idx = np.arange(iu[0].size)
    if idx.size > max_pairs:
        idx = rng.choice(idx, size=max_pairs, replace=False)

    seq, prof = [], []
    for t in idx:
        a, b = int(iu[0][t]), int(iu[1][t])
        both = obs[:, a] & obs[:, b]
        if both.sum() < min_overlap:
            continue
        x, y = M[both, a], M[both, b]
        if np.ptp(x) == 0 or np.ptp(y) == 0:
            continue
        seq.append(S[a, b])
        prof.append(float(np.corrcoef(x, y)[0, 1]))

    seq, prof = np.asarray(seq), np.asarray(prof)
    if seq.size < 10:
        return {"n_pairs": int(seq.size), "pearson_r": np.nan, "spearman_rho": np.nan,
                "p_value": np.nan, "seq_sim": seq, "profile_corr": prof}
    r, pr = pearsonr(seq, prof)
    rho, ps = spearmanr(seq, prof)
    return {"n_pairs": int(seq.size), "pearson_r": float(r), "pearson_p": float(pr),
            "spearman_rho": float(rho), "p_value": float(ps),
            "seq_sim": seq, "profile_corr": prof}


# ---------------------------------------------------------------------------
# Similarity buckets (used by the Task 3 limitation figure)
# ---------------------------------------------------------------------------

BUCKET_EDGES = (0.0, 0.10, 0.20, 0.35, 1.01)
BUCKET_LABELS = ("<0.10 (orphan)", "0.10-0.20", "0.20-0.35", ">0.35 (homologue)")


def similarity_buckets(test_pairs, max_sim, y_pred, dataset) -> pd.DataFrame:
    """Metrics split by how close the nearest training relative is.

    This is where a borrowing method stops working, and reporting it is the
    difference between a credible limitations section and a claim of general
    cold-start capability.  Shared bucket edges matter: two people using
    different cuts will reach different conclusions from the same model.
    """
    from .metrics import grouped_concordance_index, mse

    # pd.cut on an ndarray returns a Categorical, so wrap it to get a Series
    # whose comparisons give a boolean Series rather than a bare array.
    bucket = pd.Series(pd.cut(np.asarray(max_sim), bins=list(BUCKET_EDGES),
                              labels=list(BUCKET_LABELS), include_lowest=True))
    y_true = test_pairs["score"].to_numpy(dtype=float)
    lig = test_pairs["lig_idx"].to_numpy()
    y_pred = np.asarray(y_pred, dtype=float)

    rows = []
    for b in BUCKET_LABELS:
        m = (bucket == b).to_numpy()
        if m.sum() < 5:
            continue
        rows.append({
            "bucket": b, "n_pairs": int(m.sum()),
            "mse": mse(y_true[m], y_pred[m]),
            "ci_per_ligand": grouped_concordance_index(lig[m], y_true[m], y_pred[m])["mean_ci"],
            "mean_max_similarity": float(np.mean(np.asarray(max_sim)[m])),
        })
    return pd.DataFrame(rows)
