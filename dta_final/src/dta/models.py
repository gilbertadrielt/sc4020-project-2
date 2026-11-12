"""Warm-start models, all behind one interface.

Every model in the project implements the same contract::

    fit(train, n_lig, n_tgt, val=None, kernels=None, dataset=None) -> self
    predict(pairs) -> np.ndarray

That is what makes a fair comparison mechanical instead of a negotiation:
swapping a model is one line, and no model gets to bring its own data
loading, its own split or its own metric along with it.

The ladder, each step fixing a specific weakness in the one before:

``GlobalMean``
    One number.  Defines what no skill looks like.
``LigandMean``
    Each drug's own average.  This is exactly what collaborative filtering
    falls back to when it has nothing else, so any claim that a recommender
    does real work is a claim that it beats this.
``TargetMean``
    Each protein's own average.
``BiasOnly``
    mu + b_i + c_j.  Everything explainable by *who* the drug and protein
    are, with no notion of chemistry between them.  Implemented as
    ``BiasedMF(k=0)`` so the comparison with the full model is exact by
    construction rather than approximately similar.
``BiasedMF``
    mu + b_i + c_j + u_i . v_j, fitted by alternating least squares.  The
    interaction term is the only part that can express "this drug happens to
    fit this protein", which is roughly 63% of the variance on DAVIS.
``TobitMF``
    The same model fitted with a censored loss.  Around 70% of DAVIS labels
    are a detection limit rather than a measurement, and squared error
    against them trains the model to reproduce an artefact.  See its
    docstring for the prediction this makes.
``ItemKNN``
    Neighbourhood collaborative filtering on bias residuals.
``Blend``
    Convex combination of two fitted models, weight chosen on validation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Model(ABC):
    """The interface every model obeys."""

    name: str = "model"

    @abstractmethod
    def fit(self, train: pd.DataFrame, n_lig: int, n_tgt: int, **kw) -> "Model":
        ...

    @abstractmethod
    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        ...

    # -- shared helpers -----------------------------------------------------

    def _remember_scale(self, train: pd.DataFrame) -> None:
        """Record the training score range, used to clip predictions.

        A model has no business predicting an affinity stronger than anything
        ever measured; clipping costs nothing and removes a class of silly
        errors from the tails.
        """
        y = train["score"].to_numpy(dtype=float)
        self.score_min_, self.score_max_ = float(y.min()), float(y.max())

    def _clip(self, pred: np.ndarray) -> np.ndarray:
        lo = getattr(self, "score_min_", None)
        if lo is None:
            return pred
        return np.clip(pred, self.score_min_, self.score_max_)

    @staticmethod
    def _idx(pairs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return pairs["lig_idx"].to_numpy(), pairs["tgt_idx"].to_numpy()


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class GlobalMean(Model):
    name = "global_mean"

    def fit(self, train, n_lig, n_tgt, **kw):
        self._remember_scale(train)
        self.mu_ = float(train["score"].mean())
        return self

    def predict(self, pairs):
        return np.full(len(pairs), self.mu_)


class _EntityMean(Model):
    """Shared implementation of the per-drug and per-protein averages."""

    col = "lig_idx"

    def fit(self, train, n_lig, n_tgt, **kw):
        self._remember_scale(train)
        n = n_lig if self.col == "lig_idx" else n_tgt
        self.mu_ = float(train["score"].mean())
        s = np.bincount(train[self.col].to_numpy(), weights=train["score"].to_numpy(), minlength=n)
        c = np.bincount(train[self.col].to_numpy(), minlength=n)
        # An entity with no training rows falls back to the global mean.
        self.means_ = np.where(c > 0, s / np.maximum(c, 1), self.mu_)
        return self

    def predict(self, pairs):
        return self._clip(self.means_[pairs[self.col].to_numpy()])


class LigandMean(_EntityMean):
    name = "ligand_mean"
    col = "lig_idx"


class TargetMean(_EntityMean):
    name = "target_mean"
    col = "tgt_idx"


# ---------------------------------------------------------------------------
# Biased matrix factorisation, alternating least squares
# ---------------------------------------------------------------------------


class BiasedMF(Model):
    r"""mu + b_i + c_j + u_i . v_j, fitted by ALS.

    Minimises, over observed training pairs only::

        sum (r_ij - mu - b_i - c_j - u_i.v_j)^2
            + lam * (||u_i||^2 + ||v_j||^2) + lam_bias * (b_i^2 + c_j^2)

    Each half-step has a closed-form solution, so ALS is deterministic and
    needs no learning rate.  Bias and factors are solved jointly per entity by
    augmenting the design matrix with a constant column, which keeps the
    k = 0 case exactly equal to the bias-only model instead of merely close.

    Missing cells contribute nothing.  They are not zeros, and a model that
    treated them as zeros would be fitting several hundred thousand
    measurements nobody made.
    """

    name = "biased_mf"

    def __init__(
        self,
        k: int = 32,
        lam: float = 0.1,
        lam_bias: float | None = None,
        n_iter: int = 60,
        patience: int = 8,
        tol: float = 1e-6,
        seed: int = 2026,
        reg_scaling: str = "count",
        verbose: bool = False,
    ):
        self.k = int(k)
        self.lam = float(lam)
        self.lam_bias = float(lam) if lam_bias is None else float(lam_bias)
        self.n_iter = int(n_iter)
        self.patience = int(patience)
        self.tol = float(tol)
        self.seed = int(seed)
        #: ``"count"`` multiplies the penalty by how many observations the
        #: entity has (ALS-WR, Zhou et al. 2008); ``"fixed"`` is the plain
        #: constant penalty.
        #:
        #: Count scaling is the default because the two sides of this matrix
        #: are wildly unbalanced: on DAVIS a drug has 265 training
        #: observations but a target has only 48, so a single constant lambda
        #: that regularises drugs sensibly leaves targets underdetermined at
        #: k = 32 and the factors diverge.  Scaling by count makes one lambda
        #: mean the same thing on both sides.
        self.reg_scaling = reg_scaling
        self.verbose = verbose

    # -- fitting ------------------------------------------------------------

    @staticmethod
    def _group(index: np.ndarray, n: int) -> list:
        """Row positions belonging to each entity."""
        order = np.argsort(index, kind="stable")
        sorted_idx = index[order]
        bounds = np.searchsorted(sorted_idx, np.arange(n + 1))
        return [order[bounds[e] : bounds[e + 1]] for e in range(n)]

    def _solve_side(self, groups, other_factors, other_bias, y, other_index, P, b):
        """One ALS half-step: update every entity on one side of the matrix."""
        k = self.k
        base = np.diag([self.lam_bias] + [self.lam] * k)
        for e, rows in enumerate(groups):
            if len(rows) == 0:
                P[e] = 0.0
                b[e] = 0.0
                continue
            j = other_index[rows]
            X = np.empty((len(rows), k + 1))
            X[:, 0] = 1.0
            if k:
                X[:, 1:] = other_factors[j]
            target = y[rows] - self.mu_ - other_bias[j]
            reg = base * len(rows) if self.reg_scaling == "count" else base
            A = X.T @ X + reg
            w = np.linalg.solve(A, X.T @ target)
            b[e] = w[0]
            if k:
                P[e] = w[1:]

    def fit(self, train, n_lig, n_tgt, val=None, **kw):
        self._remember_scale(train)
        rng = np.random.default_rng(self.seed)
        li, ti = self._idx(train)
        y = train["score"].to_numpy(dtype=float)

        self.n_lig_, self.n_tgt_ = int(n_lig), int(n_tgt)
        self.mu_ = float(y.mean())
        self.b_ = np.zeros(n_lig)
        self.c_ = np.zeros(n_tgt)
        scale = 0.1 / np.sqrt(max(self.k, 1))
        self.U_ = rng.normal(0.0, scale, size=(n_lig, self.k)) if self.k else np.zeros((n_lig, 0))
        self.V_ = rng.normal(0.0, scale, size=(n_tgt, self.k)) if self.k else np.zeros((n_tgt, 0))

        lig_groups = self._group(li, n_lig)
        tgt_groups = self._group(ti, n_tgt)

        best = (np.inf, None)
        bad = 0
        self.history_ = []
        for it in range(self.n_iter):
            self._solve_side(lig_groups, self.V_, self.c_, y, ti, self.U_, self.b_)
            self._solve_side(tgt_groups, self.U_, self.b_, y, li, self.V_, self.c_)

            train_mse = float(np.mean((y - self._raw(li, ti)) ** 2))
            score = train_mse
            if val is not None and len(val):
                vl, vt = self._idx(val)
                score = float(np.mean((val["score"].to_numpy(dtype=float)
                                       - self._clip(self._raw(vl, vt))) ** 2))
            self.history_.append({"iter": it, "train_mse": train_mse, "score": score})

            if score < best[0] - self.tol:
                best = (score, (self.b_.copy(), self.c_.copy(), self.U_.copy(), self.V_.copy()))
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best[1] is not None:  # restore the best iterate, not the last
            self.b_, self.c_, self.U_, self.V_ = best[1]
        self.n_iter_run_ = len(self.history_)
        return self

    # -- prediction ---------------------------------------------------------

    def _raw(self, li, ti) -> np.ndarray:
        out = self.mu_ + self.b_[li] + self.c_[ti]
        if self.k:
            out = out + np.einsum("ij,ij->i", self.U_[li], self.V_[ti])
        return out

    def predict(self, pairs):
        li, ti = self._idx(pairs)
        return self._clip(self._raw(li, ti))

    def predict_interaction(self, pairs) -> np.ndarray:
        """Only the u.v term, for the 'is this more than a lookup?' check."""
        if not self.k:
            return np.zeros(len(pairs))
        li, ti = self._idx(pairs)
        return np.einsum("ij,ij->i", self.U_[li], self.V_[ti])

    # -- persistence --------------------------------------------------------

    def save(self, path, **meta) -> None:
        """Write a self-contained checkpoint: no pickle, loads anywhere."""
        np.savez(
            path,
            model_name=self.name,
            mu=self.mu_, b=self.b_, c=self.c_, U=self.U_, V=self.V_,
            k=self.k, lam=self.lam,
            score_min=self.score_min_, score_max=self.score_max_,
            n_lig=self.n_lig_, n_tgt=self.n_tgt_,
            **meta,
        )

    @classmethod
    def load(cls, path) -> "BiasedMF":
        z = np.load(path, allow_pickle=False)
        m = cls(k=int(z["k"]), lam=float(z["lam"]))
        m.mu_ = float(z["mu"])
        m.b_, m.c_, m.U_, m.V_ = z["b"], z["c"], z["U"], z["V"]
        m.score_min_, m.score_max_ = float(z["score_min"]), float(z["score_max"])
        m.n_lig_, m.n_tgt_ = int(z["n_lig"]), int(z["n_tgt"])
        return m


class BiasOnly(BiasedMF):
    """mu + b_i + c_j. Literally ``BiasedMF`` with no factors."""

    name = "bias_only"

    def __init__(self, lam: float = 1.0, **kw):
        kw.pop("k", None)
        super().__init__(k=0, lam=lam, **kw)


# ---------------------------------------------------------------------------
# Censored matrix factorisation
# ---------------------------------------------------------------------------


class TobitMF(BiasedMF):
    r"""Biased MF trained with a loss that knows the assay had a floor.

    On DAVIS a label of pKd 5.0 does not mean "binds at 5.0".  It means the
    assay stopped at 10,000 nM and saw nothing, so the truth is *at most* 5.0.
    Around 70% of labels are like this.  Squared error against them teaches
    the model to reproduce a detection limit and spends most of its capacity
    doing so.

    The censored (Tobit) loss charges nothing for predicting below the floor::

        measured pair:  (pred - y)^2
        censored pair:  max(0, pred - ceiling)^2

    **How it is fitted.**  There is no closed form for that loss, but it can
    be written as a plain squared error against a moving target::

        max(0, pred - c)^2  ==  (pred - min(pred, c))^2

    So at the current parameters, substituting the pseudo-target
    ``y* = min(pred, c)`` for censored pairs turns the Tobit objective into an
    ordinary least-squares problem, which the exact ALS solver already
    handles.  Alternating "re-target, then re-fit" is a
    majorise-minimise scheme: each half decreases the true Tobit objective, so
    it converges rather than drifting, and it inherits ALS's speed and
    determinism instead of needing a learning rate.

    This is the classical iterative-imputation treatment of censored
    regression, and it is what makes the comparison below honest: with
    ``use_censored_loss=False`` the pseudo-target is just ``y`` and the model
    collapses *exactly* onto :class:`BiasedMF`.  The two variants then share
    an optimiser, an initialisation and a seed, so the only difference
    between them is the loss.

    **The prediction this makes**, stated before running it: MSE against the
    literal labels should get slightly *worse*, while ranking among strong
    binders should improve.  If so, the metric everyone reports is partly
    measuring agreement with an artefact.
    """

    name = "tobit_mf"

    def __init__(
        self,
        k: int = 32,
        lam: float = 0.05,
        ceiling: float | None = None,
        use_censored_loss: bool = True,
        n_outer: int = 30,
        inner_iter: int = 3,
        patience: int = 5,
        tol: float = 1e-7,
        seed: int = 2026,
        reg_scaling: str = "count",
        relax_lower_clip: float = 3.0,
    ):
        super().__init__(k=k, lam=lam, n_iter=inner_iter, patience=patience,
                         tol=tol, seed=seed, reg_scaling=reg_scaling)
        self.ceiling = ceiling
        self.use_censored_loss = bool(use_censored_loss)
        self.n_outer = int(n_outer)
        self.inner_iter = int(inner_iter)
        #: How far below the detection limit predictions are allowed to go.
        #: Every other model clips to the observed range, which is right when
        #: labels are measurements.  Here the lowest observed label IS the
        #: detection limit, so clipping there would force every weak
        #: prediction back onto the ceiling and discard exactly the freedom
        #: this loss exists to provide.  Three log units is a generous
        #: physical bound (Kd up to 10^7 nM).
        self.relax_lower_clip = float(relax_lower_clip)

    # -- the censored objective --------------------------------------------

    def _tobit_residual(self, pred, y, censored):
        """pred - y for measured pairs; the one-sided excess for censored ones."""
        r = pred - y
        if self.use_censored_loss and self.ceiling is not None and np.any(censored):
            r = np.where(censored, np.maximum(pred - self.ceiling, 0.0), r)
        return r

    def _objective(self, frame) -> float:
        """Model selection uses the same loss the model is trained on.

        Selecting on plain MSE would undo the point: it would prefer whichever
        iterate best reproduces the detection limit.
        """
        li, ti = self._idx(frame)
        y = frame["score"].to_numpy(dtype=float)
        cen = (frame["censored"].to_numpy(dtype=bool) if "censored" in frame
               else np.zeros(len(frame), bool))
        r = self._tobit_residual(self._raw(li, ti), y, cen)
        return float(np.mean(r ** 2))

    # -- fitting ------------------------------------------------------------

    def fit(self, train, n_lig, n_tgt, val=None, dataset=None, **kw):
        self._remember_scale(train)
        rng = np.random.default_rng(self.seed)
        li, ti = self._idx(train)
        y = train["score"].to_numpy(dtype=float)
        censored = (train["censored"].to_numpy(dtype=bool) if "censored" in train
                    else np.zeros(len(train), bool))

        if self.ceiling is None and censored.any():
            self.ceiling = float(y[censored].max())
        if censored.any() and self.use_censored_loss:
            self.score_min_ = self.score_min_ - self.relax_lower_clip

        self.n_lig_, self.n_tgt_ = int(n_lig), int(n_tgt)
        self.mu_ = float(y.mean())
        self.b_ = np.zeros(n_lig)
        self.c_ = np.zeros(n_tgt)
        scale = 0.1 / np.sqrt(max(self.k, 1))
        self.U_ = rng.normal(0.0, scale, (n_lig, self.k)) if self.k else np.zeros((n_lig, 0))
        self.V_ = rng.normal(0.0, scale, (n_tgt, self.k)) if self.k else np.zeros((n_tgt, 0))

        lig_groups = self._group(li, n_lig)
        tgt_groups = self._group(ti, n_tgt)

        best = (np.inf, None)
        bad = 0
        self.history_ = []
        for outer in range(self.n_outer):
            # Re-target: for a censored pair already predicted below the
            # ceiling the model is consistent with the evidence, so its
            # pseudo-target is its own prediction and it contributes nothing.
            if self.use_censored_loss and self.ceiling is not None and censored.any():
                pred = self._raw(li, ti)
                y_star = np.where(censored, np.minimum(pred, self.ceiling), y)
            else:
                y_star = y

            for _ in range(self.inner_iter):
                self._solve_side(lig_groups, self.V_, self.c_, y_star, ti, self.U_, self.b_)
                self._solve_side(tgt_groups, self.U_, self.b_, y_star, li, self.V_, self.c_)

            score = (self._objective(val) if (val is not None and len(val))
                     else self._objective(train))
            self.history_.append({"outer": outer, "score": score,
                                  "frac_below_ceiling": float(
                                      np.mean(self._raw(li, ti) < (self.ceiling or -np.inf)))})
            if score < best[0] - self.tol:
                best = (score, (self.b_.copy(), self.c_.copy(), self.U_.copy(), self.V_.copy()))
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best[1] is not None:
            self.b_, self.c_, self.U_, self.V_ = best[1]
        self.n_outer_run_ = len(self.history_)
        return self



# ---------------------------------------------------------------------------
# Neighbourhood collaborative filtering
# ---------------------------------------------------------------------------


class ItemKNN(Model):
    """Predict from a drug's scores on the most similar targets.

    Two corrections, each fixing a failure that shows up on this data:

    *Similarity on residuals, not raw scores.*  On raw DAVIS pKd two
    completely inactive compounds are both a wall of the same censored value
    and correlate perfectly, which is meaningless.  Removing the bias model
    first makes similarity describe the interaction rather than the baseline.

    *Shrinkage by overlap.*  Two targets sharing three measured drugs can show
    a correlation of 1.0 by luck.  Multiplying by ``n / (n + beta)`` discounts
    similarities estimated from thin evidence, which matters on KIBA where
    two entities share a median of about seven partners.

    When no neighbour of the query target has been measured against the query
    drug, the model falls back to the bias prediction, and the fallback rate
    is recorded because a method that falls back often is being judged mostly
    on its fallback.
    """

    name = "item_knn"

    def __init__(self, k: int = 20, beta: float = 50.0, lam_bias: float = 1.0,
                 min_overlap: int = 2, seed: int = 2026):
        self.k = int(k)
        self.beta = float(beta)
        self.lam_bias = float(lam_bias)
        self.min_overlap = int(min_overlap)
        self.seed = int(seed)

    def fit(self, train, n_lig, n_tgt, val=None, **kw):
        self._remember_scale(train)
        self.n_lig_, self.n_tgt_ = int(n_lig), int(n_tgt)
        self.bias_ = BiasOnly(lam=self.lam_bias).fit(train, n_lig, n_tgt)

        li, ti = self._idx(train)
        y = train["score"].to_numpy(dtype=float)
        resid = y - self.bias_._raw(li, ti)

        R = np.full((n_lig, n_tgt), np.nan)
        R[li, ti] = resid
        self.R_ = R
        obs = np.isfinite(R)
        self.obs_ = obs

        Z = np.where(obs, np.nan_to_num(R), 0.0)
        overlap = obs.T.astype(float) @ obs.astype(float)
        cov = Z.T @ Z
        sq = np.einsum("ij,ij->j", Z, Z)
        norm = np.sqrt(np.outer(sq, sq))
        with np.errstate(invalid="ignore", divide="ignore"):
            S = np.where(norm > 1e-12, cov / norm, 0.0)
        S = np.where(overlap >= self.min_overlap, S, 0.0)
        S *= overlap / (overlap + self.beta)
        np.fill_diagonal(S, 0.0)  # a target is not its own neighbour
        self.S_ = np.nan_to_num(S)
        return self

    def predict(self, pairs):
        li, ti = self._idx(pairs)
        out = self.bias_._raw(li, ti).astype(float)
        n_fallback = 0

        for drug in np.unique(li):
            rows = np.where(li == drug)[0]
            seen = np.where(self.obs_[drug])[0]
            if len(seen) == 0:
                n_fallback += len(rows)
                continue
            sims = self.S_[np.ix_(ti[rows], seen)]          # (queries, seen)
            vals = self.R_[drug, seen]                       # (seen,)

            kk = min(self.k, sims.shape[1])
            # Neighbours are chosen by |similarity|: a strongly anticorrelated
            # target is informative too, with its sign flipped.
            top = np.argpartition(-np.abs(sims), kk - 1, axis=1)[:, :kk]
            w = np.take_along_axis(sims, top, axis=1)
            v = vals[top]
            denom = np.abs(w).sum(axis=1)
            good = denom > 1e-12
            contrib = np.zeros(len(rows))
            contrib[good] = (w[good] * v[good]).sum(axis=1) / denom[good]
            out[rows] += contrib
            n_fallback += int((~good).sum())

        self.fallback_rate_ = n_fallback / max(len(pairs), 1)
        return self._clip(out)


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------


class Blend(Model):
    """Convex combination of two fitted models, weight chosen on validation.

    Included because the two collaborative models trade places depending on
    the metric, which is the signature of complementary errors: the
    neighbourhood model is usually better on squared error and the factor
    model better at putting the single best target first.  The weight is
    tuned on validation like any other hyperparameter, never on test.
    """

    name = "blend"

    def __init__(self, model_a: Model, model_b: Model, weight: float | None = None,
                 grid=np.linspace(0, 1, 21)):
        self.a, self.b = model_a, model_b
        self.weight = weight
        self.grid = np.asarray(grid, dtype=float)

    def fit(self, train, n_lig, n_tgt, val=None, **kw):
        # The components are expected to be fitted already; fit them if not.
        for m in (self.a, self.b):
            if not hasattr(m, "score_min_"):
                m.fit(train, n_lig, n_tgt, val=val, **kw)
        self.score_min_ = min(self.a.score_min_, self.b.score_min_)
        self.score_max_ = max(self.a.score_max_, self.b.score_max_)

        if self.weight is None:
            if val is None or not len(val):
                self.weight = 0.5
            else:
                yv = val["score"].to_numpy(dtype=float)
                pa, pb = self.a.predict(val), self.b.predict(val)
                errs = [np.mean((yv - (w * pa + (1 - w) * pb)) ** 2) for w in self.grid]
                self.weight = float(self.grid[int(np.argmin(errs))])
        return self

    def predict(self, pairs):
        w = 0.5 if self.weight is None else self.weight
        return self._clip(w * self.a.predict(pairs) + (1 - w) * self.b.predict(pairs))
