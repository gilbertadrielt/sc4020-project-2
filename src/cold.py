"""Task 3: scoring a protein target with no training measurements.

On the cold-target split every model in :mod:`src.cf` returns the same numbers,
because none of them has anything target-specific to say. With no ``c_j`` and no
``v_j`` a prediction collapses to ``mu + b_i`` - one value per drug - so every
held-out target scores alike and their ranking is arbitrary. That identical row
is the baseline here, and it is what these models have to beat.

The way out is that a protein we have never assayed is not a protein we know
nothing about: we have its sequence. Task 1 measured how much that is worth,
and the answer shapes both models below. Protein 3-mer cosine predicts
affinity-profile correlation as a **threshold, not a trend** - Spearman rho is
only 0.06 on DAVIS, yet the median profile correlation jumps from 0.25 to 0.9
once cosine passes about 0.3, and the nearest cross-group neighbour clears that
bar for 71% of DAVIS targets and 72% of KIBA's.

So both models take the *few* neighbours above the threshold rather than a
similarity-weighted vote over every training target, which the same measurement
says would be drowned by the uninformative majority. Below the threshold they
decline to guess and fall back, and ``fallback_rate`` reports how often.

``SequenceNeighbour``
    Borrow the neighbours' residuals for this drug directly.
``MFSequenceTransfer``
    Give the cold target a latent vector - the similarity-weighted average of
    its neighbours' ``v_j`` and ``c_j`` - and reuse the fitted Task 2 model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cf import BiasedMF, Model
from .features import KMER_USEFUL_COSINE, kmer_profiles


class _SequenceSimilarity:
    """k-mer profiles for every target, and the neighbour rule shared by both models.

    Sequences are features, not labels: they are known for any protein before
    it is assayed, so using a held-out target's sequence is legitimate where
    using its measurements would not be.
    """

    def _build_profiles(self, targets: pd.DataFrame, train_target_ids, kmer: int) -> None:
        self._sequence = targets.set_index("target_id")["sequence"]
        self._group = targets.set_index("target_id")["seq_group"]
        order = list(self._sequence.index)
        self._row_of = {tid: i for i, tid in enumerate(order)}
        self._profiles = kmer_profiles(self._sequence.to_numpy(), k=kmer)

        self._train_ids = list(train_target_ids)
        rows = [self._row_of[t] for t in self._train_ids]
        self._train_profiles = self._profiles[rows]
        self._train_groups = self._group.loc[self._train_ids].to_numpy()

    def _neighbours(self, target_id: str, k: int, min_cosine: float):
        """Indices into the training targets, and their similarities.

        A training target sharing the cold target's ``seq_group`` is excluded.
        The cold split already guarantees whole groups stay on one side, so this
        never fires there - it is here so the model stays correct if it is ever
        handed a split that does not make that promise.
        """
        if target_id not in self._row_of:
            return np.empty(0, dtype=int), np.empty(0)

        sims = self._train_profiles @ self._profiles[self._row_of[target_id]]
        sims = np.where(self._train_groups == self._group[target_id], -1.0, sims)

        # Strictly positive, as well as above the threshold. Two kinase
        # sequences can share no 3-mer at all, giving cosine exactly 0, and a
        # zero-weight neighbour is not evidence - it is a neighbour the
        # weighted mean would divide by zero over. That matters at
        # ``min_cosine=0``, which is the ablation that asks whether the
        # threshold earns its place.
        top = np.argsort(-sims)[:k]
        top = top[(sims[top] >= min_cosine) & (sims[top] > 0.0)]
        return top, sims[top]


class SequenceNeighbour(Model, _SequenceSimilarity):
    """Predict a cold target from the training targets most like it in sequence.

    For drug *i* and cold target *j*, the prediction is ``mu + b_i``, plus a
    similarity-weighted average of ``c`` over the neighbours, plus a
    similarity-weighted average of the residuals those neighbours carry **for
    that same drug**. A neighbour the drug was never tested against contributes
    nothing, so the weights are renormalised per drug rather than per target.
    """

    def __init__(self, targets: pd.DataFrame, k: int = 5,
                 min_cosine: float = KMER_USEFUL_COSINE, kmer: int = 3,
                 reg: float = 5.0, name: str | None = None):
        super().__init__(name=name or f"sequence kNN (k={k})")
        self.targets = targets
        self.k = k
        self.min_cosine = min_cosine
        self.kmer = kmer
        self.reg = reg
        self.residual = np.zeros((0, 0))
        self.fallback_rate = float("nan")

    def fit(self, train: pd.DataFrame):
        self._fit_biases(train, reg=self.reg)
        rows = train["drug_id"].map(self.drug_index).to_numpy()
        cols = train["target_id"].map(self.target_index).to_numpy()

        self.residual = np.full((len(self.drug_index), len(self.target_index)), np.nan)
        self.residual[rows, cols] = (
            train["score"].to_numpy(float) - self.mu - self.b[rows] - self.c[cols]
        )
        self._build_profiles(self.targets, self.target_index, self.kmer)
        self.params = {"k": self.k, "min_cosine": self.min_cosine,
                       "kmer": self.kmer, "reg": self.reg}
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        rows, cols = self._positions(frame)
        out = self._bias_prediction(rows, cols)
        n_fallback = 0

        for target_id, idx in frame.groupby("target_id", sort=False).indices.items():
            if cols[idx[0]] >= 0:
                continue  # a warm target: the bias term already covers it

            neighbours, weights = self._neighbours(target_id, self.k, self.min_cosine)
            drug_rows = rows[idx]
            known_drug = drug_rows >= 0
            if len(neighbours) == 0 or not known_drug.any():
                n_fallback += len(idx)
                continue

            # The cold target's own bias, borrowed from the neighbours.
            c_new = float(weights @ self.c[neighbours] / weights.sum())

            block = self.residual[np.ix_(np.where(known_drug, drug_rows, 0), neighbours)]
            observed = np.isfinite(block) & known_drug[:, None]
            numerator = np.where(observed, block * weights[None, :], 0.0).sum(axis=1)
            denominator = np.where(observed, weights[None, :], 0.0).sum(axis=1)

            borrowed = np.where(denominator > 0, numerator / np.maximum(denominator, 1e-12), 0.0)
            out[idx] += c_new + borrowed
            n_fallback += int((denominator <= 0).sum())

        self.fallback_rate = n_fallback / max(len(frame), 1)
        return out


class LatentRegressor(Model, _SequenceSimilarity):
    """Learn the map from a protein's sequence to its latent vector.

    The most direct answer to what Task 3 asks. Matrix factorisation fails on a
    cold target because it has no ``v_j`` and no ``c_j``; this fits a function
    that produces them from the sequence, so the Task 2 formula applies
    unchanged to a protein that was never assayed.

    Ridge regression from the k-mer profile to each coordinate of ``v_j`` and
    to ``c_j``, trained on the warm targets where both sides are known. Ridge
    rather than anything larger because there are only a few hundred warm
    targets against 8,000 input columns - the problem is wildly
    under-determined, and the closed-form solution with a strong penalty is
    both the honest choice and instant to fit.

    There is no similarity threshold here, so ``fallback_rate`` stays 0 and
    every cold target gets ranked rather than left tied at the per-drug mean.
    It still degrades gracefully at the extreme, though not by any rule we
    wrote: in the dual form the prediction is the training similarities times
    the fitted coefficients, so a target sharing no k-mer with anything in
    training is orthogonal to the whole training span, receives exactly zero
    for ``v_new`` and ``c_new``, and falls back to ``mu + b_i`` on its own.

    Whether the absence of a threshold is a virtue is what the comparison is
    for. Task 1 found sequence similarity carries almost nothing below cosine
    0.3, so this model extrapolates through a region where the neighbour models
    deliberately decline - buying ranking coverage at the risk of being
    confidently wrong.
    """

    def __init__(self, targets: pd.DataFrame, n_factors: int = 40, reg: float = 5.0,
                 n_epochs: int = 400, patience: int = 12, monitor: str = "ci",
                 ridge: float = 1.0, kmer: int = 3, mf: BiasedMF | None = None,
                 name: str | None = None):
        super().__init__(name=name or f"latent regression (ridge={ridge})")
        self.targets = targets
        self.ridge = ridge
        self.kmer = kmer
        self.mf = mf or BiasedMF(n_factors=n_factors, reg=reg, n_epochs=n_epochs,
                                 patience=patience, monitor=monitor)
        self._mf_is_prefitted = mf is not None
        self.weights = np.zeros((0, 0))
        self.fallback_rate = 0.0

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None):
        if not self._mf_is_prefitted:
            self.mf.fit(train, val=val)
        self.mu = self.mf.mu
        self.b, self.c = self.mf.b, self.mf.c
        self.drug_index, self.target_index = self.mf.drug_index, self.mf.target_index
        self._build_profiles(self.targets, self.target_index, self.kmer)

        # Targets of the regression: each warm target's factors, then its bias.
        k = self.mf.n_factors
        y = np.column_stack([self.mf.V[:, :k], self.c])
        x = self._train_profiles.astype(np.float64)

        # Ridge in the n x n dual form. There are ~200-400 warm targets and
        # 8,000 features, so solving the 8000 x 8000 primal would be both
        # slower and no better determined.
        gram = x @ x.T
        alpha = np.linalg.solve(gram + self.ridge * np.eye(len(gram)), y)
        self.weights = x.T @ alpha

        self.params = dict(self.mf.params, ridge=self.ridge, kmer=self.kmer)
        return self

    def _predicted_latents(self, target_id: str):
        """``(v_new, c_new)`` for a target the factorisation never saw."""
        profile = self._profiles[self._row_of[target_id]].astype(np.float64)
        predicted = profile @ self.weights
        return predicted[:-1], float(predicted[-1])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        out = self.mf.predict(frame)
        rows, cols = self._positions(frame)
        n_factors = self.mf.n_factors

        for target_id, idx in frame.groupby("target_id", sort=False).indices.items():
            if cols[idx[0]] >= 0 or target_id not in self._row_of:
                continue
            v_new, c_new = self._predicted_latents(target_id)
            drug_rows = rows[idx]
            known = drug_rows >= 0
            interaction = np.zeros(len(idx))
            interaction[known] = self.mf.U[drug_rows[known], :n_factors] @ v_new
            out[idx] += c_new + interaction
        return out


class ContentRegressor(Model):
    """Predict affinity straight from drug structure and protein sequence.

    No collaborative term at all: a pair is a Morgan fingerprint concatenated
    with a protein k-mer profile, and gradient boosting maps that to a score.
    It therefore handles a cold target for the same reason it handles anything
    else - it never needed the target to have a column.

    This is the comparison that tells the report whether Task 3 needs the
    recommender at all. If content regression matches the neighbour models, the
    collaborative structure is doing no work once a target is unseen; if it
    falls short, the training targets' measurements carry information that
    structure alone does not.
    """

    def __init__(self, targets: pd.DataFrame, ligands: pd.DataFrame,
                 kmer: int = 3, n_bits: int = 1024, radius: int = 2,
                 n_components: int = 128,
                 n_estimators: int = 400, learning_rate: float = 0.05,
                 num_leaves: int = 63, seed: int = 0, name: str | None = None):
        super().__init__(name=name or "content regression (LightGBM)")
        self.targets = targets
        self.ligands = ligands
        self.kmer = kmer
        self.n_bits = n_bits
        self.radius = radius
        self.n_components = n_components
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.seed = seed
        self.model = None

    def _features(self, frame: pd.DataFrame) -> np.ndarray:
        drug_rows = frame["drug_id"].map(self._drug_row)
        target_rows = frame["target_id"].map(self._target_row)
        # Every id must resolve: unlike the collaborative models there is no
        # fallback here, so a missing SMILES or sequence is a broken input
        # rather than something to predict around.
        if drug_rows.isna().any() or target_rows.isna().any():
            missing = set(frame.loc[drug_rows.isna(), "drug_id"]) | set(
                frame.loc[target_rows.isna(), "target_id"])
            raise KeyError(f"no structure or sequence for {sorted(missing)[:5]}")
        return np.hstack([self._drug_features[drug_rows.to_numpy(int)],
                          self._target_features[target_rows.to_numpy(int)]])

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None):
        from lightgbm import LGBMRegressor

        from .features import kmer_profiles, morgan_fingerprints

        self._drug_row = {d: i for i, d in enumerate(self.ligands["drug_id"])}
        self._target_row = {t: i for i, t in enumerate(self.targets["target_id"])}
        self._drug_features = morgan_fingerprints(
            self.ligands["smiles"], radius=self.radius, n_bits=self.n_bits
        )
        profiles = kmer_profiles(self.targets["sequence"].to_numpy(), k=self.kmer)

        # 8,000 raw k-mer counts is the wrong input for a tree ensemble: each
        # split tests one column, and one k-mer's count says almost nothing on
        # its own, so the model pays for 8,000 histograms per node to learn
        # very little. The counts are also enormously redundant across kinases.
        # Projecting onto the leading components of the *training* targets'
        # profiles keeps the structure that separates families while making the
        # fit two orders of magnitude cheaper. Fitted on training targets only,
        # so no held-out sequence influences the basis.
        if self.n_components and self.n_components < profiles.shape[1]:
            train_rows = [self._target_row[t] for t in pd.unique(train["target_id"])]
            basis = np.linalg.svd(profiles[train_rows].astype(np.float64),
                                  full_matrices=False)[2][: self.n_components]
            self._basis = basis
            # Back to float32: the pair matrix is rows x (bits + components),
            # and float64 doubles it for no benefit to a tree ensemble.
            profiles = (profiles.astype(np.float64) @ basis.T).astype(np.float32)
        self._target_features = profiles

        self.model = LGBMRegressor(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            num_leaves=self.num_leaves, random_state=self.seed, verbose=-1,
        )
        self.model.fit(self._features(train), train["score"].to_numpy(float))
        self.mu = float(train["score"].mean())
        self.params = {"n_estimators": self.n_estimators, "kmer": self.kmer,
                       "n_bits": self.n_bits, "n_components": self.n_components,
                       "learning_rate": self.learning_rate,
                       "num_leaves": self.num_leaves}
        self.fallback_rate = 0.0
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._features(frame))


class MFSequenceTransfer(Model, _SequenceSimilarity):
    """Give the cold target a latent vector, then reuse the Task 2 model.

    This is the other half of the brief's first option: rather than averaging
    the neighbours' *scores*, average their *parameters*. ``v_new`` and
    ``c_new`` are similarity-weighted means of the neighbours' fitted vectors
    and biases, and the prediction is then the ordinary matrix-factorisation
    formula ``mu + b_i + c_new + u_i . v_new``.

    It inherits whatever the Task 2 search chose, so the comparison against
    :class:`SequenceNeighbour` isolates one question: is a cold target better
    described by its neighbours' raw measurements, or by their learned factors?
    """

    def __init__(self, targets: pd.DataFrame, n_factors: int = 40, reg: float = 5.0,
                 n_epochs: int = 400, patience: int = 12, monitor: str = "ci",
                 k: int = 5, min_cosine: float = KMER_USEFUL_COSINE, kmer: int = 3,
                 mf: BiasedMF | None = None, name: str | None = None):
        super().__init__(name=name or f"MF + sequence transfer (k={k})")
        self.targets = targets
        self.k = k
        self.min_cosine = min_cosine
        self.kmer = kmer
        # An already-fitted model can be passed in. Task 2 chose the
        # factorisation; Task 3 only chooses how to extend it to a target that
        # has no column, so a grid over the transfer rule must not refit the
        # factorisation once per configuration - that is both wasteful and a
        # different experiment from the one being run.
        self.mf = mf or BiasedMF(n_factors=n_factors, reg=reg, n_epochs=n_epochs,
                                 patience=patience, monitor=monitor)
        self._mf_is_prefitted = mf is not None
        self.fallback_rate = float("nan")

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None):
        if not self._mf_is_prefitted:
            self.mf.fit(train, val=val)
        # Share the fitted model's bookkeeping so the inherited helpers work.
        self.mu = self.mf.mu
        self.b, self.c = self.mf.b, self.mf.c
        self.drug_index, self.target_index = self.mf.drug_index, self.mf.target_index
        self._build_profiles(self.targets, self.target_index, self.kmer)
        self.params = dict(self.mf.params, k=self.k, min_cosine=self.min_cosine)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        out = self.mf.predict(frame)  # correct already for any warm target
        rows, cols = self._positions(frame)
        n_fallback = 0
        n_factors = self.mf.n_factors

        for target_id, idx in frame.groupby("target_id", sort=False).indices.items():
            if cols[idx[0]] >= 0:
                continue

            neighbours, weights = self._neighbours(target_id, self.k, self.min_cosine)
            drug_rows = rows[idx]
            known_drug = drug_rows >= 0
            if len(neighbours) == 0 or not known_drug.any():
                n_fallback += len(idx)
                continue

            share = weights / weights.sum()
            v_new = share @ self.mf.V[neighbours, :n_factors]
            c_new = float(share @ self.c[neighbours])

            interaction = np.zeros(len(idx))
            interaction[known_drug] = (
                self.mf.U[drug_rows[known_drug], :n_factors] @ v_new
            )
            out[idx] += c_new + interaction
            n_fallback += int((~known_drug).sum())

        self.fallback_rate = n_fallback / max(len(frame), 1)
        return out
