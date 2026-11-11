"""Cold-target models: predicting for protein targets with no training rows."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor


class ColdContext:
    """Everything fitted on the cold-split training pairs, shared by all models."""

    def __init__(self, ds, sp, mf, feats, sim):
        self.ds, self.sp, self.mf, self.feats, self.sim = ds, sp, mf, feats, sim
        tr = sp["train"]
        r, c, y = ds.rows[tr], ds.cols[tr], ds.y[tr]
        self.mu = float(y.mean())
        s = np.bincount(r, weights=y, minlength=ds.n_ligands)
        n = np.bincount(r, minlength=ds.n_ligands)
        self.ligand_mean = np.where(n > 0, s / np.maximum(n, 1), self.mu)
        self.lo, self.hi = float(y.min()), float(y.max())

        # dense training score matrix restricted to the warm (training) targets
        self.train_targets = sp["train_targets"]
        pos = -np.ones(ds.n_targets, dtype=int)
        pos[self.train_targets] = np.arange(len(self.train_targets))
        keep = pos[c] >= 0
        self.R = np.zeros((ds.n_ligands, len(self.train_targets)))
        self.M = np.zeros((ds.n_ligands, len(self.train_targets)), dtype=bool)
        self.R[r[keep], pos[c[keep]]] = y[keep]
        self.M[r[keep], pos[c[keep]]] = True

    def clip(self, p):
        return np.clip(p, self.lo, self.hi)

    def max_similarity(self, targets):
        """Similarity of each given target to its closest training target."""
        return self.sim[np.ix_(np.atleast_1d(targets), self.train_targets)].max(axis=1)


def predict_ligand_mean(ctx, rows, cols):
    """C0. What warm CF collapses to when the target is unknown: a flat per-drug
    score that cannot tell two new targets apart."""
    return ctx.clip(ctx.ligand_mean[rows])


def predict_gba_knn(ctx, rows, cols, k=10, power=3.0, min_sim=0.0, return_fallback=False):
    """C1. Guilt by association: score a new target like its closest homologues.

    For cold target t and ligand i we average i's deviation-from-its-own-mean
    over the k training targets most similar to t that i was actually measured
    against, weighted by similarity^power. Neighbours below min_sim are refused,
    so a target with no credible homologue in training falls back to the ligand
    mean rather than averaging over unrelated proteins. That fallback is the
    failure mode C0 suffers for every target, and the rate is reported.
    """
    out = np.empty(len(rows))
    fallback = np.zeros(len(rows), dtype=bool)
    for t in np.unique(cols):
        m = cols == t
        lig = rows[m]
        s = ctx.sim[t, ctx.train_targets]
        order = np.argsort(-s)
        s_ord = s[order]
        obs = ctx.M[np.ix_(lig, order)] & (s_ord[None, :] >= min_sim)
        cum = np.cumsum(obs, axis=1)
        sel = obs & (cum <= k)
        w = (s_ord[None, :] ** power) * sel
        dev = (ctx.R[np.ix_(lig, order)] - ctx.ligand_mean[lig][:, None]) * sel
        tot = w.sum(axis=1)
        est = np.where(tot > 0, (w * dev).sum(axis=1) / np.maximum(tot, 1e-12), 0.0)
        out[m] = ctx.ligand_mean[lig] + est
        fallback[m] = tot <= 0
    out = ctx.clip(out)
    return (out, fallback) if return_fallback else out


def fit_factor_ridge(ctx, alpha=10.0):
    """C2 (fit). Learn sequence -> (target bias, target factors) from warm targets.

    The matrix factorisation already compressed every training target into a
    short vector. Regressing that vector from sequence descriptors lets a brand
    new sequence be placed in the same latent space and scored by the warm model.
    """
    tt = ctx.train_targets
    X = ctx.feats[tt]
    Y = np.column_stack([ctx.mf.c[tt], ctx.mf.V[tt]])
    model = Ridge(alpha=alpha).fit(X, Y)
    return model


def predict_factor_ridge(ctx, model, rows, cols):
    """C2 (predict)."""
    P = model.predict(ctx.feats[cols])
    c_hat, v_hat = P[:, 0], P[:, 1:]
    p = ctx.mf.mu + ctx.mf.b[rows] + c_hat + np.sum(ctx.mf.U[rows] * v_hat, axis=1)
    return ctx.clip(p)


def _content_features(ctx, rows, cols):
    return np.hstack([
        ctx.mf.U[rows],
        ctx.mf.b[rows][:, None],
        ctx.ligand_mean[rows][:, None],
        ctx.feats[cols],
    ])


def fit_content_regressor(ctx, max_iter=200, seed=0):
    """C3 (fit). Gradient boosting straight from (ligand factors, target
    descriptors) to the score, with no latent-space detour."""
    tr = ctx.sp["train"]
    X = _content_features(ctx, ctx.ds.rows[tr], ctx.ds.cols[tr])
    y = ctx.ds.y[tr]
    m = HistGradientBoostingRegressor(
        max_iter=max_iter, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=False, random_state=seed)
    return m.fit(X, y)


def predict_content(ctx, model, rows, cols):
    """C3 (predict)."""
    return ctx.clip(model.predict(_content_features(ctx, rows, cols)))


def gate_weights(ctx, cols, s0, tau):
    """Similarity gate: 1 means trust the neighbours, 0 means trust the regression."""
    s = ctx.sim[cols][:, ctx.train_targets].max(axis=1)
    return 1.0 / (1.0 + np.exp(-(s - s0) / max(tau, 1e-6)))


def predict_gated(ctx, p_knn, p_ridge, cols, s0, tau, fallback=None):
    """C4. Blend C1 and C2 with a weight that depends on how close the cold
    target's nearest training homologue is.

    Where a near neighbour exists, guilt-by-association is the sharper signal.
    Where the target is an orphan, the neighbour average is noise and the
    sequence regression should carry the prediction. A fixed blend weight cannot
    express that; this gate can. Pairs where C1 fell back to the ligand mean are
    handed entirely to C2.
    """
    w = gate_weights(ctx, cols, s0, tau)
    if fallback is not None:
        w = np.where(fallback, 0.0, w)
    return ctx.clip(w * p_knn + (1.0 - w) * p_ridge)
