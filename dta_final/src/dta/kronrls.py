r"""Kronecker kernel ridge regression: one model for all four settings.

This is the model that ties the project together.  Collaborative filtering
answers "fill a gap for a drug and protein we know" and falls apart on an
unseen protein; content methods answer the cold question but throw away the
matrix.  Kernel ridge over the Kronecker product does both, and the only
thing that changes between settings is which kernel you hand it.

**The model.**  Score a pair by comparing it with every training pair, where
two pairs are similar when both their drugs and their targets are similar::

    K((d, t), (d', t')) = K_D[d, d'] * K_T[t, t']

Collecting the dual coefficients into a matrix ``A`` indexed by (drug,
target), the prediction surface over all pairs is simply::

    F = K_D @ A @ K_T

which is why predicting for a protein nobody has measured needs nothing new:
its column of ``K_T`` exists as soon as its sequence does.  That single fact
is what makes S3 and S4 solvable at all.

**Why not build the pair kernel.**  It has one row per observed pair, so for
KIBA it would be 117,657 x 117,657, about 110 GB.  Two solvers avoid it:

``cg``
    Conjugate gradients, where each matrix-vector product is evaluated as
    ``K_D @ A @ K_T`` and read off at the observed entries (the "vec trick").
    Works for any observation pattern.  Cost per iteration is
    O(n_d^2 n_t + n_d n_t^2) rather than O(|obs|^2).

``eigen``
    When every pair is observed, the eigendecompositions of the two small
    kernels give the solution in closed form, because the eigenvalues of a
    Kronecker product are the products of the eigenvalues.

DAVIS is a complete 68 x 379 grid after deduplication, so both solvers apply
there and must agree.  That agreement is asserted in the test suite, which
makes the iterative solver's correctness something the project demonstrates
rather than assumes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import BiasOnly, Model


class KronRLS(Model):
    r"""Kernel ridge regression over the Kronecker product of two kernels.

    Minimises, over observed training pairs::

        sum_{(d,t) in obs} (y_dt - F_dt)^2 + lam * <A, K_D A K_T>

    Parameters
    ----------
    lam
        Ridge strength.
    solver
        ``"auto"`` picks ``eigen`` when the training pairs form a complete
        grid and ``cg`` otherwise.
    center
        What to subtract before fitting, and add back when predicting.
        ``"bias"`` removes mu + b_drug + c_target first so the kernel model
        only has to explain the interaction, which is both easier and the
        part it is actually good at.  An entity absent from training gets a
        bias of zero, which is the honest choice: nothing was measured, so
        there is nothing to offset.
    """

    name = "kronrls"

    def __init__(
        self,
        lam: float = 1.0,
        solver: str = "auto",
        center: str = "bias",
        lam_bias: float = 1.0,
        cg_tol: float = 1e-6,
        cg_maxiter: int = 300,
        drug_kernel: str = "drug_structural",
        target_kernel: str = "target_sequence",
        seed: int = 2026,
    ):
        self.lam = float(lam)
        self.solver = solver
        self.center = center
        self.lam_bias = float(lam_bias)
        self.cg_tol = float(cg_tol)
        self.cg_maxiter = int(cg_maxiter)
        self.drug_kernel = drug_kernel
        self.target_kernel = target_kernel
        self.seed = int(seed)

    # -- kernel plumbing ----------------------------------------------------

    def _pick_kernels(self, kernels: dict) -> tuple[np.ndarray, np.ndarray]:
        if kernels is None:
            raise ValueError("KronRLS needs kernels; pass kernels=... to fit()")
        KD = kernels[self.drug_kernel] if isinstance(self.drug_kernel, str) else self.drug_kernel
        KT = kernels[self.target_kernel] if isinstance(self.target_kernel, str) else self.target_kernel
        return np.asarray(KD, dtype=float), np.asarray(KT, dtype=float)

    def _apply(self, A: np.ndarray) -> np.ndarray:
        """One Kronecker matrix-vector product, without forming the big matrix."""
        return self.KD_ @ A @ self.KT_

    # -- fitting ------------------------------------------------------------

    def fit(self, train, n_lig, n_tgt, val=None, kernels=None, **kw):
        self._remember_scale(train)
        self.n_lig_, self.n_tgt_ = int(n_lig), int(n_tgt)
        KD, KT = self._pick_kernels(kernels)
        if KD.shape[0] < n_lig or KT.shape[0] < n_tgt:
            raise ValueError(
                f"kernels are {KD.shape} and {KT.shape}, too small for "
                f"{n_lig} drugs and {n_tgt} targets"
            )
        self.KD_, self.KT_ = KD[:n_lig, :n_lig], KT[:n_tgt, :n_tgt]

        li, ti = self._idx(train)
        y = train["score"].to_numpy(dtype=float)

        # Remove what a bias model can already explain, so the kernel is left
        # with the interaction.
        if self.center == "bias":
            self.base_ = BiasOnly(lam=self.lam_bias).fit(train, n_lig, n_tgt)
            self.seen_lig_ = np.zeros(n_lig, bool)
            self.seen_tgt_ = np.zeros(n_tgt, bool)
            self.seen_lig_[np.unique(li)] = True
            self.seen_tgt_[np.unique(ti)] = True
            resid = y - self._baseline(li, ti)
        else:
            self.base_ = None
            self.mu_ = float(y.mean())
            resid = y - self.mu_

        complete = len(train) == n_lig * n_tgt
        solver = self.solver
        if solver == "auto":
            solver = "eigen" if complete else "cg"
        if solver == "eigen" and not complete:
            raise ValueError("the eigen solver needs every pair observed; use solver='cg'")

        Y = np.zeros((n_lig, n_tgt))
        Y[li, ti] = resid
        self.obs_ = np.zeros((n_lig, n_tgt), bool)
        self.obs_[li, ti] = True

        self.A_ = self._solve_eigen(Y) if solver == "eigen" else self._solve_cg(Y)
        self.solver_used_ = solver
        return self

    def _solve_eigen(self, Y: np.ndarray) -> np.ndarray:
        r"""Closed form for a complete grid.

        With ``K_D = Q_D L_D Q_D'`` and ``K_T = Q_T L_T Q_T'``, substituting
        ``A = Q_D B Q_T'`` into ``K_D A K_T + lam A = Y`` diagonalises it::

            B[i, j] = (Q_D' Y Q_T)[i, j] / (L_D[i] * L_T[j] + lam)
        """
        wd, Qd = np.linalg.eigh(self.KD_)
        wt, Qt = np.linalg.eigh(self.KT_)
        Ytil = Qd.T @ Y @ Qt
        denom = np.outer(wd, wt) + self.lam
        return Qd @ (Ytil / denom) @ Qt.T

    def _solve_cg(self, Y: np.ndarray) -> np.ndarray:
        """Conjugate gradients on the observed entries only.

        Solves ``(P K P + lam I) a = y`` where ``P`` restricts to observed
        pairs.  The operator is symmetric positive definite whenever the two
        kernels are PSD and ``lam > 0``, which is why the kernels are
        PSD-projected when they are built.
        """
        obs = self.obs_
        b = Y  # already zero off the observed set

        def op(A):
            return np.where(obs, self._apply(np.where(obs, A, 0.0)), 0.0) + self.lam * A

        A = np.zeros_like(b)
        r = b - op(A)
        p = r.copy()
        rs = float(np.sum(r * r))
        b_norm = float(np.sqrt(np.sum(b * b))) or 1.0
        self.cg_iters_ = 0
        for _ in range(self.cg_maxiter):
            if np.sqrt(rs) / b_norm < self.cg_tol:
                break
            Ap = op(p)
            denom = float(np.sum(p * Ap))
            if abs(denom) < 1e-300:
                break
            alpha = rs / denom
            A += alpha * p
            r -= alpha * Ap
            rs_new = float(np.sum(r * r))
            p = r + (rs_new / rs) * p
            rs = rs_new
            self.cg_iters_ += 1
        self.cg_residual_ = float(np.sqrt(rs) / b_norm)
        # Coefficients live only on observed pairs.
        return np.where(obs, A, 0.0)

    # -- prediction ---------------------------------------------------------

    def _baseline(self, li, ti) -> np.ndarray:
        """mu plus whichever biases are available for these entities."""
        if self.base_ is None:
            return np.full(len(li), self.mu_)
        b = np.where(self.seen_lig_[li], self.base_.b_[li], 0.0)
        c = np.where(self.seen_tgt_[ti], self.base_.c_[ti], 0.0)
        return self.base_.mu_ + b + c

    def predict(self, pairs):
        li, ti = self._idx(pairs)
        # Only the rows and columns actually queried are needed, which keeps
        # prediction cheap even when the full surface would be large.
        rows = np.unique(li)
        cols = np.unique(ti)
        row_pos = {int(r): i for i, r in enumerate(rows)}
        col_pos = {int(c): j for j, c in enumerate(cols)}
        F = self.KD_[np.ix_(rows, np.arange(self.n_lig_))] @ self.A_ @ \
            self.KT_[np.ix_(np.arange(self.n_tgt_), cols)]
        inter = F[[row_pos[int(i)] for i in li], [col_pos[int(j)] for j in ti]]
        return self._clip(self._baseline(li, ti) + inter)

    def predict_interaction(self, pairs) -> np.ndarray:
        li, ti = self._idx(pairs)
        return self.predict(pairs) - self._baseline(li, ti)


class KronRLSMKL(KronRLS):
    """KronRLS over a weighted combination of several kernels per side.

    Rather than choosing one notion of similarity, combine them and let the
    data say which matters.  The weights are themselves a result worth
    reporting: "sequence alignment gets 0.7 and k-mer composition 0.3" is a
    statement about the biology, not just a number in an accuracy table.

    Two ways to set the weights:

    ``alignment``
        Centred kernel alignment between each kernel and the observed
        affinity profile.  Closed form, one pass, no refitting.
    ``validation``
        Coordinate search on the simplex, scoring each candidate on the
        validation fold.  Slower, and it optimises the thing you actually
        care about.

    This is a simplified version of KronRLS-MKL (Nascimento et al. 2016),
    which alternates between the dual coefficients and the weights; the
    simplification is stated rather than hidden.
    """

    name = "kronrls_mkl"

    def __init__(
        self,
        lam: float = 1.0,
        drug_kernels=("drug_structural", "drug_morgan"),
        target_kernels=("target_sequence", "target_kmer"),
        weight_mode: str = "alignment",
        n_rounds: int = 2,
        grid=(0.0, 0.25, 0.5, 0.75, 1.0),
        **kw,
    ):
        super().__init__(lam=lam, **kw)
        self.drug_kernels = list(drug_kernels)
        self.target_kernels = list(target_kernels)
        self.weight_mode = weight_mode
        self.n_rounds = int(n_rounds)
        self.grid = tuple(grid)

    def fit(self, train, n_lig, n_tgt, val=None, kernels=None, **kw):
        from .kernels import alignment_weights, combine_kernels

        KDs = [np.asarray(kernels[k], dtype=float)[:n_lig, :n_lig] for k in self.drug_kernels]
        KTs = [np.asarray(kernels[k], dtype=float)[:n_tgt, :n_tgt] for k in self.target_kernels]

        if self.weight_mode == "alignment":
            li, ti = self._idx(train)
            y = train["score"].to_numpy(dtype=float)
            Y = np.zeros((n_lig, n_tgt))
            Y[li, ti] = y - y.mean()
            self.drug_weights_ = alignment_weights(KDs, Y @ Y.T)
            self.target_weights_ = alignment_weights(KTs, Y.T @ Y)
        else:
            self.drug_weights_, self.target_weights_ = self._search_weights(
                train, n_lig, n_tgt, val, KDs, KTs
            )

        merged = {
            "_drug": combine_kernels(KDs, self.drug_weights_),
            "_target": combine_kernels(KTs, self.target_weights_),
        }
        self.drug_kernel, self.target_kernel = "_drug", "_target"
        return super().fit(train, n_lig, n_tgt, val=val, kernels=merged, **kw)

    def _search_weights(self, train, n_lig, n_tgt, val, KDs, KTs):
        """Coordinate search over simplex weights, scored on validation."""
        from .kernels import combine_kernels
        from .metrics import mse

        wd = np.full(len(KDs), 1.0 / len(KDs))
        wt = np.full(len(KTs), 1.0 / len(KTs))
        if val is None or not len(val):
            return wd, wt

        def score(wd_, wt_):
            merged = {"_d": combine_kernels(KDs, wd_), "_t": combine_kernels(KTs, wt_)}
            m = KronRLS(lam=self.lam, solver=self.solver, center=self.center,
                        lam_bias=self.lam_bias, cg_tol=self.cg_tol,
                        cg_maxiter=self.cg_maxiter, drug_kernel="_d", target_kernel="_t")
            m.fit(train, n_lig, n_tgt, kernels=merged)
            return mse(val["score"].to_numpy(dtype=float), m.predict(val))

        best = score(wd, wt)
        for _ in range(self.n_rounds):
            for i in range(len(KDs)):
                for g in self.grid:
                    cand = wd.copy()
                    cand[i] = g
                    if cand.sum() <= 0:
                        continue
                    cand = cand / cand.sum()
                    s = score(cand, wt)
                    if s < best:
                        best, wd = s, cand
            for j in range(len(KTs)):
                for g in self.grid:
                    cand = wt.copy()
                    cand[j] = g
                    if cand.sum() <= 0:
                        continue
                    cand = cand / cand.sum()
                    s = score(wd, cand)
                    if s < best:
                        best, wt = s, cand
        return wd, wt
