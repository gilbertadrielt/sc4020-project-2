"""Cold-start comparison models.

KronRLS handles unseen entities natively, so these exist to give it something
to be measured against, and to make the failure of the obvious approaches
visible rather than assumed.

``ColdFallback``
    What collaborative filtering degrades to when an entity is unseen:
    mu + b_drug.  Its prediction does not depend on the target at all, so
    every unseen target for a given drug receives the same score, every
    ranking among them is a tie, and per-ligand CI is exactly 0.5.  This is
    the number the whole cold-start section is built around.

``SequenceKNN``
    Guilt by association: borrow the drug's measurements on the most
    sequence-similar training targets.  Simple, and in practice very strong.

``FactorMap``
    Learn a map from protein features to the factor vector a matrix
    factorisation would have given the protein, then reuse the warm model.
    Conceptually appealing and empirically weak, which is worth reporting.

``ContentRegressor``
    Ignore the matrix: gradient boosting on drug features next to protein
    features.  Has to learn chemistry from scratch, and the data regime is
    too small for that to beat borrowing.

``SimilarityGatedBlend``
    Use the neighbourhood model where a close relative exists and the
    content model where it does not, with the cut chosen on validation.
    Removes the fallback problem that a pure neighbour method has when a
    query target has no usable neighbour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import BiasOnly, LigandMean, Model


class ColdFallback(LigandMean):
    """mu + b_drug. The baseline an unseen entity forces CF back onto."""

    name = "cold_fallback"


# ---------------------------------------------------------------------------
# Guilt by association
# ---------------------------------------------------------------------------


class SequenceKNN(Model):
    r"""Borrow a drug's scores on the training targets most similar to the query.

    .. math::

        \hat r_{d t^*} = \text{base}_{d t^*} +
            \frac{\sum_{t \in N_K} s(t^*, t)^\gamma \, (r_{dt} - \text{base}_{dt})}
                 {\sum_{t \in N_K} s(t^*, t)^\gamma}

    where :math:`N_K` is the K training targets most similar to the unseen
    target that this drug was actually measured against.

    The exponent sharpens the weighting: with gamma large, only genuinely
    close relatives contribute.  Residuals rather than raw scores are
    averaged so the borrowed quantity is the interaction, not the baseline.

    ``min_similarity`` refuses to borrow from a neighbour that is not
    actually similar.  Such a query falls back to the baseline, and the
    fallback rate is recorded, because a method that falls back on a third of
    its queries is being judged mostly on its fallback.
    """

    name = "sequence_knn"

    def __init__(
        self,
        k: int = 10,
        gamma: float = 3.0,
        min_similarity: float = 0.0,
        lam_bias: float = 1.0,
        axis: str = "target",
        target_kernel: str = "target_sequence",
        drug_kernel: str = "drug_structural",
        seed: int = 2026,
    ):
        self.k = int(k)
        self.gamma = float(gamma)
        self.min_similarity = float(min_similarity)
        self.lam_bias = float(lam_bias)
        #: ``"target"`` borrows across proteins (for S3), ``"drug"`` across
        #: molecules (for S2).  S4 has neither side warm, so this model is
        #: not applicable there and returns its baseline.
        self.axis = axis
        self.target_kernel = target_kernel
        self.drug_kernel = drug_kernel
        self.seed = int(seed)

    def fit(self, train, n_lig, n_tgt, val=None, kernels=None, **kw):
        self._remember_scale(train)
        self.n_lig_, self.n_tgt_ = int(n_lig), int(n_tgt)
        self.base_ = BiasOnly(lam=self.lam_bias).fit(train, n_lig, n_tgt)

        key = self.target_kernel if self.axis == "target" else self.drug_kernel
        size = n_tgt if self.axis == "target" else n_lig
        self.S_ = np.asarray(kernels[key], dtype=float)[:size, :size]

        li, ti = self._idx(train)
        self.seen_lig_ = np.zeros(n_lig, bool)
        self.seen_tgt_ = np.zeros(n_tgt, bool)
        self.seen_lig_[np.unique(li)] = True
        self.seen_tgt_[np.unique(ti)] = True

        resid = train["score"].to_numpy(dtype=float) - self._baseline(li, ti)
        self.R_ = np.full((n_lig, n_tgt), np.nan)
        self.R_[li, ti] = resid
        self.obs_ = np.isfinite(self.R_)
        return self

    def _baseline(self, li, ti) -> np.ndarray:
        b = np.where(self.seen_lig_[li], self.base_.b_[li], 0.0)
        c = np.where(self.seen_tgt_[ti], self.base_.c_[ti], 0.0)
        return self.base_.mu_ + b + c

    def predict(self, pairs):
        li, ti = self._idx(pairs)
        out = self._baseline(li, ti).astype(float)
        n_fallback = 0

        if self.axis == "target":
            query_of, other_of = ti, li
            observed_for = lambda d: np.where(self.obs_[d])[0]
            value_for = lambda d, cols: self.R_[d, cols]
        else:
            query_of, other_of = li, ti
            observed_for = lambda t: np.where(self.obs_[:, t])[0]
            value_for = lambda t, rows: self.R_[rows, t]

        for other in np.unique(other_of):
            rows = np.where(other_of == other)[0]
            seen = observed_for(other)
            if len(seen) == 0:
                n_fallback += len(rows)
                continue
            sims = self.S_[np.ix_(query_of[rows], seen)]
            vals = value_for(other, seen)

            kk = min(self.k, sims.shape[1])
            top = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
            w = np.take_along_axis(sims, top, axis=1)
            w = np.where(w >= self.min_similarity, np.clip(w, 0.0, None) ** self.gamma, 0.0)
            v = vals[top]
            denom = w.sum(axis=1)
            good = denom > 1e-12
            contrib = np.zeros(len(rows))
            contrib[good] = (w[good] * v[good]).sum(axis=1) / denom[good]
            out[rows] += contrib
            n_fallback += int((~good).sum())

        self.fallback_rate_ = n_fallback / max(len(pairs), 1)
        return self._clip(out)

    def max_similarity_to_train(self, pairs) -> np.ndarray:
        """How close the nearest usable training neighbour is, per query pair.

        Drives the similarity-bucket analysis and the gate below: it is the
        variable that decides whether borrowing can work at all.
        """
        li, ti = self._idx(pairs)
        out = np.zeros(len(pairs))
        if self.axis == "target":
            for d in np.unique(li):
                rows = np.where(li == d)[0]
                seen = np.where(self.obs_[d])[0]
                if len(seen):
                    out[rows] = self.S_[np.ix_(ti[rows], seen)].max(axis=1)
        else:
            for t in np.unique(ti):
                rows = np.where(ti == t)[0]
                seen = np.where(self.obs_[:, t])[0]
                if len(seen):
                    out[rows] = self.S_[np.ix_(li[rows], seen)].max(axis=1)
        return out


# ---------------------------------------------------------------------------
# Transferring factors
# ---------------------------------------------------------------------------


class FactorMap(Model):
    """Learn protein features -> matrix-factorisation factors, then reuse the warm model.

    Fit a factorisation on the training pairs, then a ridge regression from
    each training target's features to its learned factor vector and bias.
    An unseen target gets a predicted factor vector, which drops straight
    into the warm model.

    The appeal is obvious and the result is mediocre, which is the point of
    including it: the factors are fitted to whatever explains the training
    matrix and carry no obligation to be a smooth function of sequence, so
    regressing onto them discards most of the signal.
    """

    name = "factor_map"

    def __init__(self, k: int = 16, lam: float = 0.05, ridge_alpha: float = 1.0,
                 n_components: int = 64, target_block: str = "target_sw_profile",
                 seed: int = 2026):
        self.k = int(k)
        self.lam = float(lam)
        self.ridge_alpha = float(ridge_alpha)
        self.n_components = int(n_components)
        #: Which protein representation to regress from. Composition descriptors
        #: were the original choice and are the weakest of the five available;
        #: see the ablation in ``run_analysis.representation_ablation``.
        self.target_block = str(target_block)
        self.seed = int(seed)

    def fit(self, train, n_lig, n_tgt, val=None, kernels=None, features=None, **kw):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.linear_model import Ridge

        from .models import BiasedMF

        self._remember_scale(train)
        self.n_lig_, self.n_tgt_ = int(n_lig), int(n_tgt)
        self.mf_ = BiasedMF(k=self.k, lam=self.lam, seed=self.seed).fit(
            train, n_lig, n_tgt, val=val
        )

        X = np.asarray(features[self.target_block], dtype=float)[:n_tgt]
        seen = np.unique(train["tgt_idx"].to_numpy())
        self.seen_tgt_ = np.zeros(n_tgt, bool)
        self.seen_tgt_[seen] = True

        # The projection is fitted on TRAINING targets only. It uses no
        # labels, but fitting it on the held-out sequences too would let the
        # representation adapt to proteins the model must not have seen.
        n_comp = int(min(self.n_components, X.shape[1] - 1, max(len(seen) - 1, 1)))
        self.svd_ = TruncatedSVD(n_components=max(n_comp, 1), random_state=self.seed).fit(X[seen])
        Z = self.svd_.transform(X)

        Y = np.column_stack([self.mf_.c_[seen].reshape(-1, 1), self.mf_.V_[seen]])
        self.ridge_ = Ridge(alpha=self.ridge_alpha).fit(Z[seen], Y)
        self.Zall_ = Z
        return self

    def predict(self, pairs):
        li, ti = self._idx(pairs)
        pred_cv = self.ridge_.predict(self.Zall_[ti])
        c_hat = np.where(self.seen_tgt_[ti], self.mf_.c_[ti], pred_cv[:, 0])
        V_hat = np.where(self.seen_tgt_[ti][:, None], self.mf_.V_[ti], pred_cv[:, 1:])
        out = self.mf_.mu_ + self.mf_.b_[li] + c_hat
        if self.k:
            out = out + np.einsum("ij,ij->i", self.mf_.U_[li], V_hat)
        return self._clip(out)


# ---------------------------------------------------------------------------
# Content regression
# ---------------------------------------------------------------------------


def gba_mixup(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_virtual: int,
              alpha: float = 0.4, label_sigma: float = 0.5,
              seed: int = 2026) -> tuple[np.ndarray, np.ndarray]:
    r"""Guilt-by-association mixup: virtual training pairs between neighbours.

    From MixingDTA (Kim et al., *Bioinformatics* 41:i105-i114, 2025), which the
    project brief cites.  Pairs sharing an entity are treated as neighbours in a
    graph, a partner is sampled with a preference for a similar label, and both
    the feature vector and the label are interpolated:

    .. math::

        x_{\text{mix}} = \lambda x_p + (1 - \lambda) x_q, \qquad
        y_{\text{mix}} = \lambda y_p + (1 - \lambda) y_q, \qquad
        \lambda \sim \text{Beta}(\alpha, \alpha)

    ``groups`` decides who counts as a neighbour: drug indices give the
    drug-sharing variant, target indices the target-sharing one.  The published
    system pairs this with fine-tuned MolFormer-XL and ESM3 encoders, which is
    far beyond this project; the augmentation itself is the transferable part
    and is what is implemented here.

    Interpolating features only makes sense where the representation is locally
    linear in affinity.  It is not always, and the ablation reports where it
    helps and where it hurts rather than assuming.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    edges = np.r_[0, np.flatnonzero(np.diff(sorted_groups)) + 1, len(groups)]
    buckets = [order[edges[i]:edges[i + 1]] for i in range(len(edges) - 1)
               if edges[i + 1] - edges[i] >= 2]
    if not buckets or n_virtual <= 0:
        return np.empty((0, X.shape[1]), dtype=X.dtype), np.empty(0)

    sizes = np.array([len(b) for b in buckets], dtype=float)
    chosen = rng.choice(len(buckets), size=n_virtual, p=sizes / sizes.sum())
    Xv = np.empty((n_virtual, X.shape[1]), dtype=np.float32)
    yv = np.empty(n_virtual, dtype=float)

    for row, bucket_id in enumerate(chosen):
        bucket = buckets[bucket_id]
        p = bucket[rng.integers(len(bucket))]
        w = np.exp(-((y[bucket] - y[p]) ** 2) / (2.0 * label_sigma ** 2))
        w[bucket == p] = 0.0
        total = w.sum()
        q = bucket[rng.integers(len(bucket))] if total <= 0 else \
            bucket[rng.choice(len(bucket), p=w / total)]
        lam = rng.beta(alpha, alpha)
        Xv[row] = lam * X[p] + (1.0 - lam) * X[q]
        yv[row] = lam * y[p] + (1.0 - lam) * y[q]
    return Xv, yv


class ContentRegressor(Model):
    """Gradient boosting on [drug features, protein features]; no matrix at all.

    Features are projected to a few dense components first.  A 2,048-bit
    fingerprint beside a 420-dimensional composition vector gives roughly
    1,450 mostly-empty columns, and boosting then spends its time scanning
    split candidates that carry no signal.  As with ``FactorMap``, the
    projection for proteins is fitted on training targets only.
    """

    name = "content_regression"

    def __init__(self, n_estimators: int = 400, learning_rate: float = 0.06,
                 num_leaves: int = 63, n_drug_comp: int = 96, n_prot_comp: int = 96,
                 target_block: str = "target_sw_profile", mixup: str | None = None,
                 mixup_ratio: float = 0.5, mixup_alpha: float = 0.4,
                 n_jobs: int = 2, seed: int = 2026):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.num_leaves = int(num_leaves)
        self.n_drug_comp = int(n_drug_comp)
        self.n_prot_comp = int(n_prot_comp)
        #: Which protein representation to use; see ``kernels.TARGET_BLOCKS``.
        self.target_block = str(target_block)
        #: ``"drug"`` or ``"target"`` turns on GBA-Mixup over that axis, ``None``
        #: off.  Virtual rows are appended to the training matrix before fitting.
        self.mixup = mixup
        self.mixup_ratio = float(mixup_ratio)
        self.mixup_alpha = float(mixup_alpha)
        self.n_jobs = int(n_jobs)
        self.seed = int(seed)

    @staticmethod
    def _project(X, n_comp, fit_rows, seed):
        from sklearn.decomposition import TruncatedSVD

        X = np.asarray(X, dtype=np.float32)
        n_comp = int(min(n_comp, X.shape[1] - 1, max(len(fit_rows) - 1, 1)))
        if n_comp <= 0 or n_comp >= X.shape[1]:
            return X
        svd = TruncatedSVD(n_components=n_comp, random_state=seed).fit(X[fit_rows])
        return svd.transform(X).astype(np.float32)

    def fit(self, train, n_lig, n_tgt, val=None, features=None, **kw):
        self._remember_scale(train)
        li, ti = self._idx(train)
        self.Dr_ = self._project(features["drug"][:n_lig], self.n_drug_comp,
                                 np.unique(li), self.seed)
        self.Tr_ = self._project(features[self.target_block][:n_tgt], self.n_prot_comp,
                                 np.unique(ti), self.seed)

        X = np.hstack([self.Dr_[li], self.Tr_[ti]])
        y = train["score"].to_numpy(dtype=float)

        self.n_virtual_ = 0
        if self.mixup:
            groups = li if self.mixup == "drug" else ti
            Xv, yv = gba_mixup(X, y, groups,
                               n_virtual=int(self.mixup_ratio * len(y)),
                               alpha=self.mixup_alpha, seed=self.seed)
            self.n_virtual_ = len(yv)
            X = np.vstack([X, Xv])
            y = np.concatenate([y, yv])

        try:
            import lightgbm as lgb

            self.model_ = lgb.LGBMRegressor(
                n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                num_leaves=self.num_leaves, random_state=self.seed,
                n_jobs=self.n_jobs, verbose=-1,
            )
            if val is not None and len(val):
                vl, vt = self._idx(val)
                Xv = np.hstack([self.Dr_[vl], self.Tr_[vt]])
                yv = val["score"].to_numpy(dtype=float)
                stop = [lgb.early_stopping(40, verbose=False)]
                try:  # LightGBM >= 4.7 renamed eval_set to eval_X/eval_y
                    self.model_.fit(X, y, eval_X=Xv, eval_y=yv, callbacks=stop)
                except TypeError:
                    self.model_.fit(X, y, eval_set=[(Xv, yv)], callbacks=stop)
            else:
                self.model_.fit(X, y)
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor

            self.model_ = HistGradientBoostingRegressor(
                max_iter=self.n_estimators, learning_rate=self.learning_rate,
                random_state=self.seed,
            ).fit(X, y)
        return self

    def predict(self, pairs):
        li, ti = self._idx(pairs)
        X = np.hstack([self.Dr_[li], self.Tr_[ti]])
        return self._clip(np.asarray(self.model_.predict(X), dtype=float))


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------


class SimilarityGatedBlend(Model):
    """Neighbour model where a close relative exists, content model elsewhere.

    A pure neighbourhood method has nothing to say about a query with no
    similar training entity, and a content model is weak everywhere but
    degrades gracefully.  Switching between them on the similarity to the
    nearest training entity removes the fallback and keeps the strong method
    where it works.  The cut is a hyperparameter, chosen on validation.
    """

    name = "gated_blend"

    def __init__(self, knn: SequenceKNN, other: Model, threshold: float | None = None,
                 grid=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5), soft: bool = True):
        self.knn = knn
        self.other = other
        self.threshold = threshold
        self.grid = tuple(grid)
        #: A hard switch makes predictions discontinuous at the cut; the soft
        #: version ramps between the two models over a small band around it.
        self.soft = bool(soft)

    def _mix(self, pairs, threshold):
        s = self.knn.max_similarity_to_train(pairs)
        a = self.knn.predict(pairs)
        b = self.other.predict(pairs)
        if self.soft:
            width = max(threshold * 0.5, 1e-3)
            w = np.clip((s - threshold) / width + 0.5, 0.0, 1.0)
        else:
            w = (s >= threshold).astype(float)
        return w * a + (1 - w) * b

    def fit(self, train, n_lig, n_tgt, val=None, **kw):
        for m in (self.knn, self.other):
            if not hasattr(m, "score_min_"):
                m.fit(train, n_lig, n_tgt, val=val, **kw)
        self.score_min_ = min(self.knn.score_min_, self.other.score_min_)
        self.score_max_ = max(self.knn.score_max_, self.other.score_max_)

        if self.threshold is None:
            if val is None or not len(val):
                self.threshold = 0.1
            else:
                yv = val["score"].to_numpy(dtype=float)
                errs = [np.mean((yv - self._mix(val, t)) ** 2) for t in self.grid]
                self.threshold = float(self.grid[int(np.argmin(errs))])
        return self

    def predict(self, pairs):
        return self._clip(self._mix(pairs, self.threshold))
