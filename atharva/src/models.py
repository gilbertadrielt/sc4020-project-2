"""Collaborative filtering models for the warm-split task.

Users = ligands (drugs), items = protein targets, ratings = binding affinity.
Every model fits on training observations only.
"""
import numpy as np


class _Base:
    def fit(self, rows, cols, y, n_lig, n_tgt):
        raise NotImplementedError

    def predict(self, rows, cols):
        raise NotImplementedError

    def _clip(self, p):
        return np.clip(p, self.lo, self.hi)

    def _record_range(self, y):
        self.lo, self.hi = float(y.min()), float(y.max())


class GlobalMean(_Base):
    name = "B0 global mean"

    def fit(self, rows, cols, y, n_lig, n_tgt):
        self._record_range(y)
        self.mu = float(y.mean())
        return self

    def predict(self, rows, cols):
        return np.full(len(rows), self.mu)


class LigandMean(_Base):
    """Per-drug training mean. This is exactly what a warm CF model degrades to
    when a target carries no usable information, so it doubles as the Task 3
    fallback reference."""
    name = "B1 ligand mean"

    def fit(self, rows, cols, y, n_lig, n_tgt):
        self._record_range(y)
        self.mu = float(y.mean())
        s = np.bincount(rows, weights=y, minlength=n_lig)
        c = np.bincount(rows, minlength=n_lig)
        self.m = np.where(c > 0, s / np.maximum(c, 1), self.mu)
        return self

    def predict(self, rows, cols):
        return self.m[rows]


class TargetMean(_Base):
    name = "B2 target mean"

    def fit(self, rows, cols, y, n_lig, n_tgt):
        self._record_range(y)
        self.mu = float(y.mean())
        s = np.bincount(cols, weights=y, minlength=n_tgt)
        c = np.bincount(cols, minlength=n_tgt)
        self.m = np.where(c > 0, s / np.maximum(c, 1), self.mu)
        return self

    def predict(self, rows, cols):
        return self.m[cols]


class BiasOnly(_Base):
    """r_hat = mu + b_i + c_j, fitted by alternating ridge updates."""
    name = "B3 bias only"

    def __init__(self, lam=5.0, iters=25):
        self.lam, self.iters = lam, iters

    def fit(self, rows, cols, y, n_lig, n_tgt):
        self._record_range(y)
        self.mu = float(y.mean())
        b = np.zeros(n_lig); c = np.zeros(n_tgt)
        cnt_l = np.bincount(rows, minlength=n_lig)
        cnt_t = np.bincount(cols, minlength=n_tgt)
        for _ in range(self.iters):
            r = y - self.mu - c[cols]
            b = np.bincount(rows, weights=r, minlength=n_lig) / (cnt_l + self.lam)
            r = y - self.mu - b[rows]
            c = np.bincount(cols, weights=r, minlength=n_tgt) / (cnt_t + self.lam)
        self.b, self.c = b, c
        return self

    def predict(self, rows, cols):
        return self._clip(self.mu + self.b[rows] + self.c[cols])


class TargetKNN(_Base):
    """Item-based CF: targets are the items.

    Similarity is cosine over bias-corrected residuals, computed only on the
    ligands two targets share and shrunk by that overlap, so a 2-ligand overlap
    in sparse KIBA cannot masquerade as a confident neighbour.
    """
    name = "M1 target kNN"

    def __init__(self, k=20, beta=25.0, lam=5.0):
        self.k, self.beta, self.lam = k, beta, lam

    def fit(self, rows, cols, y, n_lig, n_tgt):
        self._record_range(y)
        self.base = BiasOnly(lam=self.lam).fit(rows, cols, y, n_lig, n_tgt)
        resid = y - self.base.predict(rows, cols)

        M = np.zeros((n_lig, n_tgt)); B = np.zeros((n_lig, n_tgt))
        M[rows, cols] = resid; B[rows, cols] = 1.0

        num = M.T @ M                      # sum of e_ij * e_ij' over shared ligands
        sq = (M * M).T @ B                 # sum of e_ij^2 restricted to shared support
        den = np.sqrt(np.maximum(sq, 0) * np.maximum(sq.T, 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(den > 0, num / den, 0.0)
        overlap = B.T @ B
        sim *= overlap / (overlap + self.beta)   # shrink by co-rating count
        np.fill_diagonal(sim, 0.0)
        self.sim = np.nan_to_num(sim)

        self._restore_support(rows, cols, resid)
        return self

    def _restore_support(self, rows, cols, resid):
        """Index the training residuals by ligand for neighbour lookup.

        Kept as flat arrays so the state round-trips through a .npz checkpoint.
        """
        rows = np.asarray(rows); cols = np.asarray(cols); resid = np.asarray(resid)
        self._fit_rows, self._fit_cols, self._fit_resid = rows, cols, resid
        order = np.argsort(rows, kind="stable")
        self._obs_t, self._obs_e = {}, {}
        r_s, c_s, e_s = rows[order], cols[order], resid[order]
        if len(r_s) == 0:
            return
        bounds = np.flatnonzero(np.diff(r_s)) + 1
        for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(r_s)]):
            self._obs_t[int(r_s[lo])] = c_s[lo:hi]
            self._obs_e[int(r_s[lo])] = e_s[lo:hi]

    def predict(self, rows, cols):
        out = np.zeros(len(rows))
        self.n_fallback = 0
        order = np.argsort(rows, kind="stable")
        r_s, c_s = rows[order], cols[order]
        bounds = np.flatnonzero(np.diff(r_s)) + 1
        for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(r_s)]):
            i = int(r_s[lo]); q = c_s[lo:hi]
            ot = self._obs_t.get(i)
            if ot is None or len(ot) == 0:
                self.n_fallback += hi - lo
                continue
            S = self.sim[np.ix_(q, ot)]            # (queries, candidate neighbours)
            oe = self._obs_e[i]
            k = min(self.k, S.shape[1])
            idx = np.argpartition(-np.abs(S), k - 1, axis=1)[:, :k]
            Sk = np.take_along_axis(S, idx, axis=1)
            Ek = oe[idx]
            w = np.abs(Sk).sum(axis=1)
            est = np.where(w > 0, (Sk * Ek).sum(axis=1) / np.maximum(w, 1e-12), 0.0)
            self.n_fallback += int((w <= 0).sum())
            out[order[lo:hi]] = est
        return self._clip(self.base.predict(rows, cols) + out)


class BiasedMF(_Base):
    """r_hat = mu + b_i + c_j + u_i . v_j, fitted by alternating least squares.

    Bias and factors are solved jointly per row/column, which is stable and
    deterministic given the seed (no learning rate to babysit).
    """
    name = "M2 biased MF"

    def __init__(self, k=16, lam=10.0, lam_bias=5.0, iters=25, seed=0):
        self.k, self.lam, self.lam_bias, self.iters, self.seed = k, lam, lam_bias, iters, seed

    @staticmethod
    def _group(idx, n):
        order = np.argsort(idx, kind="stable")
        starts = np.searchsorted(idx[order], np.arange(n), side="left")
        ends = np.searchsorted(idx[order], np.arange(n), side="right")
        return order, starts, ends

    def fit(self, rows, cols, y, n_lig, n_tgt):
        self._record_range(y)
        rng = np.random.default_rng(self.seed)
        self.mu = float(y.mean())
        k = self.k
        U = rng.normal(0, 0.01, (n_lig, k)); V = rng.normal(0, 0.01, (n_tgt, k))
        b = np.zeros(n_lig); c = np.zeros(n_tgt)
        reg = np.diag(np.r_[self.lam_bias, np.full(k, self.lam)])

        ord_l, sl, el = self._group(rows, n_lig)
        ord_t, st, et = self._group(cols, n_tgt)
        rl, cl, yl = rows[ord_l], cols[ord_l], y[ord_l]
        rt, ct, yt = rows[ord_t], cols[ord_t], y[ord_t]

        for _ in range(self.iters):
            for i in range(n_lig):                      # solve (b_i, u_i)
                a, z = sl[i], el[i]
                if a == z:
                    continue
                X = np.c_[np.ones(z - a), V[cl[a:z]]]
                t = yl[a:z] - self.mu - c[cl[a:z]]
                w = np.linalg.solve(X.T @ X + reg, X.T @ t)
                b[i], U[i] = w[0], w[1:]
            for j in range(n_tgt):                      # solve (c_j, v_j)
                a, z = st[j], et[j]
                if a == z:
                    continue
                X = np.c_[np.ones(z - a), U[rt[a:z]]]
                t = yt[a:z] - self.mu - b[rt[a:z]]
                w = np.linalg.solve(X.T @ X + reg, X.T @ t)
                c[j], V[j] = w[0], w[1:]
        self.U, self.V, self.b, self.c = U, V, b, c
        return self

    def predict(self, rows, cols):
        p = self.mu + self.b[rows] + self.c[cols] + np.sum(self.U[rows] * self.V[cols], axis=1)
        return self._clip(p)


class Blend(_Base):
    """Convex blend of two fitted models; the weight is chosen on validation."""
    name = "M3 kNN + MF blend"

    def __init__(self, a, b, w=0.5):
        self.a, self.b, self.w = a, b, w
        self.lo, self.hi = a.lo, a.hi

    def predict(self, rows, cols):
        return self._clip(self.w * self.a.predict(rows, cols)
                          + (1 - self.w) * self.b.predict(rows, cols))
