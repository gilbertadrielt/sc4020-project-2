"""Scoring: MSE, Concordance Index, and per-ligand ranking metrics."""
import numpy as np
from scipy.stats import kendalltau


def mse(y_true, y_pred):
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.mean(d * d))


def _tie_term(v):
    """Sum of t(t-1)/2 over groups of tied values."""
    _, counts = np.unique(v, return_counts=True)
    return float(np.sum(counts * (counts - 1) / 2.0))


def concordance_index(y_true, y_pred):
    """CI = (tau_b * sqrt(B/A) + 1) / 2, the form given in the project brief.

    A = pairs with different true scores, B = pairs with different predictions.
    Algebraically this is the standard (concordant + 0.5*ties)/comparable, but
    Kendall tau-b is O(n log n) so it scales to the full KIBA test fold.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    n = len(y_true)
    if n < 2:
        return float("nan")
    n0 = n * (n - 1) / 2.0
    A = n0 - _tie_term(y_true)
    B = n0 - _tie_term(y_pred)
    if A <= 0:
        return float("nan")
    if B <= 0:
        return 0.5  # predictions are all tied -> every comparable pair is a coin flip
    tau = kendalltau(y_true, y_pred, variant="b").statistic
    if not np.isfinite(tau):
        return float("nan")
    return float((tau * np.sqrt(B / A) + 1.0) / 2.0)


def _ci_bruteforce(y_true, y_pred):
    """O(n^2) reference implementation, used only to validate the fast path."""
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    dt = y_true[:, None] - y_true[None, :]
    dp = y_pred[:, None] - y_pred[None, :]
    iu = np.triu_indices(len(y_true), k=1)
    dt, dp = dt[iu], dp[iu]
    comparable = dt != 0
    if not comparable.any():
        return float("nan")
    dt, dp = dt[comparable], dp[comparable]
    conc = np.sum(np.sign(dt) == np.sign(dp))
    ties = np.sum(dp == 0)
    return float((conc + 0.5 * ties) / len(dt))


def ranking_metrics(ligand_idx, y_true, y_pred, ks=(1, 5, 10), min_candidates=2,
                    tie_seed=12345, exclude_degenerate=False):
    """Rank each ligand's held-out targets and score the retrieval of its best one.

    For every ligand we rank only the targets held out for that ligand. A hit at
    k means a target achieving the ligand's maximum true score is placed in the
    top k; ties for best count as a hit at the best-ranked tied target.

    Ties in the *predictions* are broken uniformly at random rather than by array
    order. Without this a constant predictor (global mean) inherits the dataset's
    ordering and scores well above chance for no reason.

    A ligand whose held-out targets all share one true score is "degenerate":
    every ranking is trivially correct. They are counted and can be excluded.
    """
    ligand_idx = np.asarray(ligand_idx)
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    rng = np.random.default_rng(tie_seed)

    hits = {k: [] for k in ks}
    rr, per_ligand_ci, n_cand = [], [], []
    n_degenerate = 0

    order = np.argsort(ligand_idx, kind="stable")
    li, yt, yp = ligand_idx[order], y_true[order], y_pred[order]
    bounds = np.flatnonzero(np.diff(li)) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(li)]):
        t, p = yt[lo:hi], yp[lo:hi]
        if len(t) < min_candidates:
            continue
        degenerate = t.max() == t.min()
        if degenerate:
            n_degenerate += 1
            if exclude_degenerate:
                continue
        n_cand.append(len(t))
        # rank 1 = highest predicted score; ties among predictions broken at random
        jitter = rng.random(len(p))
        ranks = np.empty(len(p), int)
        ranks[np.lexsort((jitter, -p))] = np.arange(1, len(p) + 1)
        best_rank = int(ranks[t == t.max()].min())
        for k in ks:
            hits[k].append(1.0 if best_rank <= k else 0.0)
        rr.append(1.0 / best_rank)
        ci = concordance_index(t, p)
        if np.isfinite(ci):
            per_ligand_ci.append(ci)

    out = {f"hit@{k}": float(np.mean(v)) if v else float("nan") for k, v in hits.items()}
    out["mrr"] = float(np.mean(rr)) if rr else float("nan")
    out["ci_per_ligand"] = float(np.mean(per_ligand_ci)) if per_ligand_ci else float("nan")
    out["n_ligands_ranked"] = len(rr)
    out["n_degenerate"] = n_degenerate
    out["mean_candidates"] = float(np.mean(n_cand)) if n_cand else float("nan")
    return out


def random_ranking_reference(ligand_idx, y_true, ks=(1, 5, 10), n_draws=20, seed=0):
    """Expected ranking metrics from shuffling predictions, averaged over draws."""
    rng = np.random.default_rng(seed)
    acc = []
    for d in range(n_draws):
        acc.append(ranking_metrics(ligand_idx, y_true, rng.random(len(y_true)),
                                   ks=ks, tie_seed=seed + d))
    keys = [f"hit@{k}" for k in ks] + ["mrr", "ci_per_ligand"]
    out = {k: float(np.mean([a[k] for a in acc])) for k in keys}
    out["n_ligands_ranked"] = acc[0]["n_ligands_ranked"]
    out["n_degenerate"] = acc[0]["n_degenerate"]
    out["mean_candidates"] = acc[0]["mean_candidates"]
    return out


def evaluate(ligand_idx, y_true, y_pred, strong_threshold, ks=(1, 5, 10)):
    """Full metric block for one model on one fold."""
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    strong = y_true >= strong_threshold
    res = {
        "mse": mse(y_true, y_pred),
        "mse_strong": mse(y_true[strong], y_pred[strong]) if strong.any() else float("nan"),
        "n_strong": int(strong.sum()),
        "ci_global": concordance_index(y_true, y_pred),
    }
    res.update(ranking_metrics(ligand_idx, y_true, y_pred, ks=ks))
    nd = ranking_metrics(ligand_idx, y_true, y_pred, ks=ks, exclude_degenerate=True)
    res["hit@1_nondegenerate"] = nd["hit@1"]
    return res
