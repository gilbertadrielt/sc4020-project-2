"""Collaborative filtering for Task 2, plus the baselines it has to beat.

Every model here has the same shape: ``fit(train)`` then ``predict(frame)``,
both taking the ``drug_id``/``target_id``/``score`` frames that :mod:`src.split`
produces. That makes the results table a loop rather than five special cases,
and it means the Task 3 models can be dropped in beside these without changing
the evaluation code.

The models, weakest first:

``GlobalMean``, ``DrugMean``, ``TargetMean``
    The lookups. ``DrugMean`` is the one that matters: it is what CF itself
    falls back to when it has never seen a target, so it is the line Task 3 has
    to clear.
``BiasOnly``
    ``mu + b_i + c_j``. Often most of what a recommender achieves on a dense
    matrix, which is exactly why it is here - "CF beats the global mean" is not
    an interesting claim if bias-only beats CF.
``BiasedMF``
    ``mu + b_i + c_j + u_i . v_j``, fitted by alternating least squares.
``ItemKNN``
    Borrow from the targets most correlated with this one, over the drugs they
    share. The second model the brief asks for, and the one whose fallback rate
    tells you how thin KIBA's neighbourhoods really are.

**Unseen entities.** A drug or target absent from training has no bias and no
latent vector. Every model here predicts such a pair from whatever it does
know - for an unseen target that is ``mu + b_i``, the per-drug mean. This is not
a safety net bolted on; it is the behaviour Task 3 exists to improve on, so it
is implemented once, here, and measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .eda import masked_corr

SAVED_MODELS = Path(__file__).resolve().parent.parent / "saved_models"


def _index(values) -> dict:
    """Stable id -> position map."""
    return {v: i for i, v in enumerate(pd.unique(pd.Series(values)))}


@dataclass
class Model:
    """Shared fitting bookkeeping: the id maps, the global mean, the biases."""

    name: str = "model"
    mu: float = 0.0
    drug_index: dict = field(default_factory=dict)
    target_index: dict = field(default_factory=dict)
    b: np.ndarray = field(default_factory=lambda: np.zeros(0))
    c: np.ndarray = field(default_factory=lambda: np.zeros(0))
    params: dict = field(default_factory=dict)

    # -- helpers ----------------------------------------------------------
    def _positions(self, frame: pd.DataFrame):
        """Row and column positions, with -1 marking an entity never trained on."""
        rows = frame["drug_id"].map(self.drug_index).to_numpy()
        cols = frame["target_id"].map(self.target_index).to_numpy()
        rows = np.where(pd.isna(rows), -1, rows).astype(int)
        cols = np.where(pd.isna(cols), -1, cols).astype(int)
        return rows, cols

    def _bias_prediction(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """mu + b_i + c_j, with a missing entity contributing nothing.

        An unseen target therefore scores ``mu + b_i``: the per-drug mean, which
        is identical for every unseen target and so ranks them at random. That
        degeneracy is the whole point of Task 3.
        """
        out = np.full(len(rows), self.mu)
        known_row, known_col = rows >= 0, cols >= 0
        out[known_row] += self.b[rows[known_row]]
        out[known_col] += self.c[cols[known_col]]
        return out

    def _fit_biases(self, train: pd.DataFrame, reg: float = 5.0, passes: int = 10) -> None:
        """Damped per-entity means, alternated to convergence.

        The damping (``reg``) shrinks an entity's bias towards zero in
        proportion to how few observations support it, which is what keeps a
        KIBA drug with 11 measurements from getting the same confidence as one
        with 166.
        """
        self.drug_index = _index(train["drug_id"])
        self.target_index = _index(train["target_id"])
        rows = train["drug_id"].map(self.drug_index).to_numpy()
        cols = train["target_id"].map(self.target_index).to_numpy()
        scores = train["score"].to_numpy(float)

        self.mu = float(scores.mean())
        self.b = np.zeros(len(self.drug_index))
        self.c = np.zeros(len(self.target_index))
        n_drug = np.bincount(rows, minlength=len(self.b))
        n_target = np.bincount(cols, minlength=len(self.c))

        for _ in range(passes):
            resid = scores - self.mu - self.c[cols]
            self.b = np.bincount(rows, weights=resid, minlength=len(self.b)) / (reg + n_drug)
            resid = scores - self.mu - self.b[rows]
            self.c = np.bincount(cols, weights=resid, minlength=len(self.c)) / (reg + n_target)

    # -- interface --------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "Model":
        raise NotImplementedError

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """``frame`` with a ``pred`` column, ready for :func:`src.metrics.evaluate`."""
        return frame.assign(pred=self.predict(frame))


class GlobalMean(Model):
    # Names are set here, not in fit(), so a model is identifiable before it is
    # fitted - results tables key on the name.
    def __init__(self, name: str = "global mean"):
        super().__init__(name=name)

    def fit(self, train):
        self._fit_biases(train)
        self.b[:] = 0.0
        self.c[:] = 0.0
        return self

    def predict(self, frame):
        return np.full(len(frame), self.mu)


class RandomRanking(Model):
    """Noise drawn around the training distribution: the floor for every metric.

    The brief asks for it alongside the lookups, and it earns its place by
    pinning down what chance looks like on *these* folds. CI must land at 0.5
    and Hit@k at roughly k over the candidate count; anything a model gains is
    measured from here. It also catches a whole class of evaluation bug - if a
    scoring change ever makes random look good, the metric is broken.
    """

    def __init__(self, seed: int = 0, name: str = "random"):
        super().__init__(name=name)
        self.seed = seed
        self.sigma = 1.0

    def fit(self, train):
        self._fit_biases(train)
        self.b[:] = 0.0
        self.c[:] = 0.0
        self.sigma = float(train["score"].std())
        self.params = {"seed": self.seed}
        return self

    def predict(self, frame):
        # Seeded per call so a given frame always scores identically, which
        # keeps the results table reproducible.
        rng = np.random.default_rng(self.seed)
        return rng.normal(self.mu, self.sigma, len(frame))


class DrugMean(Model):
    """Each drug's training mean - what CF falls back to on an unseen target."""

    def __init__(self, name: str = "per-drug mean"):
        super().__init__(name=name)

    def fit(self, train):
        self._fit_biases(train)
        self.c[:] = 0.0
        return self

    def predict(self, frame):
        rows, cols = self._positions(frame)
        return self._bias_prediction(rows, np.full_like(cols, -1))


class TargetMean(Model):
    def __init__(self, name: str = "per-target mean"):
        super().__init__(name=name)

    def fit(self, train):
        self._fit_biases(train)
        self.b[:] = 0.0
        return self

    def predict(self, frame):
        rows, cols = self._positions(frame)
        return self._bias_prediction(np.full_like(rows, -1), cols)


class BiasOnly(Model):
    """mu + b_i + c_j. No interaction term, so no notion of compatibility."""

    def __init__(self, reg: float = 5.0, name: str = "bias only"):
        super().__init__(name=name)
        self.reg = reg

    def fit(self, train):
        self._fit_biases(train, reg=self.reg)
        self.params = {"reg": self.reg}
        return self

    def predict(self, frame):
        return self._bias_prediction(*self._positions(frame))


class BiasedMF(Model):
    """``mu + b_i + c_j + u_i . v_j``, fitted by alternating least squares.

    ALS rather than SGD because the matrices here are small on one side - 68
    drugs for DAVIS, 229 targets for KIBA - so each half-step is a few hundred
    ``k x k`` solves in numpy, where an SGD pass would be millions of Python
    iterations over single ratings.

    Biases are folded into the factor matrices rather than fitted separately:
    with ``u~ = [u, b, 1]`` and ``v~ = [v, 1, c]``, the dot product ``u~ . v~``
    expands to ``u . v + b + c``. Each half-step then solves for one side's free
    coefficients while the constant 1 on the other side carries that side's
    bias into the target vector.
    """

    def __init__(self, n_factors: int = 10, reg: float = 1.0, n_epochs: int = 30,
                 seed: int = 0, patience: int = 8, monitor: str = "mse",
                 name: str | None = None):
        super().__init__(name=name or f"biased MF (k={n_factors})")
        self.n_factors = n_factors
        self.reg = reg
        self.n_epochs = n_epochs
        self.seed = seed
        self.patience = patience
        self.monitor = monitor
        self.U = np.zeros((0, 0))
        self.V = np.zeros((0, 0))
        self.history: list[float] = []
        self.best_epoch: int | None = None

    # Layout of the augmented factor matrices.
    @property
    def _bias_col(self) -> int:
        return self.n_factors

    @property
    def _ones_col(self) -> int:
        return self.n_factors + 1

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None):
        self._fit_biases(train, reg=self.reg)
        rows = train["drug_id"].map(self.drug_index).to_numpy()
        cols = train["target_id"].map(self.target_index).to_numpy()
        resid = train["score"].to_numpy(float) - self.mu

        rng = np.random.default_rng(self.seed)
        k = self.n_factors
        n_drugs, n_targets = len(self.drug_index), len(self.target_index)

        # U = [u | b | 1], V = [v | 1 | c]
        self.U = np.zeros((n_drugs, k + 2))
        self.V = np.zeros((n_targets, k + 2))
        self.U[:, :k] = rng.normal(scale=0.05, size=(n_drugs, k))
        self.V[:, :k] = rng.normal(scale=0.05, size=(n_targets, k))
        self.U[:, self._bias_col] = self.b
        self.U[:, self._ones_col] = 1.0
        self.V[:, self._bias_col] = 1.0
        self.V[:, self._ones_col] = self.c

        by_drug = _group_positions(rows, n_drugs)
        by_target = _group_positions(cols, n_targets)

        # Early stopping. ``n_epochs`` is a ceiling, not a setting: given a
        # validation fold, the epoch count is read off the curve instead of
        # being guessed, and the best state is restored at the end so the
        # returned model is the one that scored best rather than the last one.
        best_score, best_state, since_best = np.inf, None, 0

        for epoch in range(self.n_epochs):
            # Solve for [u_i, b_i], holding V fixed; the 1 in U's last column
            # means V's own bias column is a known offset, not an unknown.
            self._half_step(self.U, self.V, by_drug, cols, resid,
                            free=slice(0, k + 1), fixed=self._ones_col)
            # Then for [v_j, c_j], where the free block is the factors plus
            # V's last column, so the slice is not contiguous with the bias.
            self._half_step(self.V, self.U, by_target, rows, resid,
                            free=np.r_[0:k, k + 1], fixed=self._bias_col)
            # ALS learns the biases inside the augmented matrices, so copy them
            # back out: _bias_prediction reads self.b and self.c, and on a cold
            # target it is the *learned* b_i that has to carry the prediction.
            self.b = self.U[:, self._bias_col].copy()
            self.c = self.V[:, self._ones_col].copy()

            if val is None:
                continue
            score = self._monitor_score(val)
            self.history.append(score)
            if score < best_score - 1e-9:
                best_score, since_best = score, 0
                best_state = (self.U.copy(), self.V.copy(), self.b.copy(), self.c.copy())
                self.best_epoch = epoch + 1
            else:
                since_best += 1
                if since_best >= self.patience:
                    break

        if best_state is not None:
            self.U, self.V, self.b, self.c = best_state

        self.params = {"n_factors": k, "reg": self.reg, "n_epochs": self.n_epochs,
                       "seed": self.seed, "monitor": self.monitor,
                       "best_epoch": self.best_epoch}
        return self

    def _monitor_score(self, val: pd.DataFrame) -> float:
        """Validation score to minimise. CI is negated so lower is always better."""
        from .metrics import concordance_index

        truth = val["score"].to_numpy(float)
        pred = self.predict(val)
        if self.monitor == "mse":
            return float(np.mean((truth - pred) ** 2))
        if self.monitor == "ci":
            return -float(concordance_index(truth, pred))
        raise ValueError(f"monitor must be 'mse' or 'ci', got {self.monitor!r}")

    def _half_step(self, this: np.ndarray, other: np.ndarray, groups, other_pos,
                   resid: np.ndarray, free, fixed: int) -> None:
        """Ridge-solve each row's free coefficients against its observed pairs."""
        free_idx = np.arange(this.shape[1])[free] if isinstance(free, slice) else np.asarray(free)
        eye = self.reg * np.eye(len(free_idx))
        for i, positions in enumerate(groups):
            if len(positions) == 0:
                continue
            partner = other[other_pos[positions]]
            a = partner[:, free_idx]
            # The partner's fixed column is a known contribution, so it moves
            # out of the unknowns and into the target vector.
            y = resid[positions] - partner[:, fixed] * this[i, fixed]
            this[i, free_idx] = np.linalg.solve(a.T @ a + eye, a.T @ y)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        rows, cols = self._positions(frame)
        out = self._bias_prediction(rows, cols)
        both = (rows >= 0) & (cols >= 0)
        if both.any():
            k = self.n_factors
            interaction = np.einsum("ij,ij->i", self.U[rows[both], :k], self.V[cols[both], :k])
            out[both] += interaction
        return out

    def save(self, path: Path | str) -> Path:
        """Everything needed to reproduce a prediction, in one .npz."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mu=self.mu, b=self.b, c=self.c, U=self.U, V=self.V,
            drug_ids=np.array(list(self.drug_index), dtype=object),
            target_ids=np.array(list(self.target_index), dtype=object),
            n_factors=self.n_factors, reg=self.reg, n_epochs=self.n_epochs,
            seed=self.seed, name=self.name,
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> "BiasedMF":
        z = np.load(Path(path), allow_pickle=True)
        model = cls(n_factors=int(z["n_factors"]), reg=float(z["reg"]),
                    n_epochs=int(z["n_epochs"]), seed=int(z["seed"]),
                    name=str(z["name"]))
        model.mu = float(z["mu"])
        model.b, model.c, model.U, model.V = z["b"], z["c"], z["U"], z["V"]
        model.drug_index = {d: i for i, d in enumerate(z["drug_ids"])}
        model.target_index = {t: i for i, t in enumerate(z["target_ids"])}
        return model


def _group_positions(codes: np.ndarray, n_groups: int) -> list[np.ndarray]:
    """Row positions belonging to each entity, computed once per fit."""
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(n_groups + 1))
    return [order[bounds[i]: bounds[i + 1]] for i in range(n_groups)]


class ItemKNN(Model):
    """Borrow from the most correlated targets the drug has already been tested on.

    Similarity is Pearson correlation between two targets' affinity profiles
    over the drugs they share, which is :func:`src.eda.masked_corr` - the same
    function Task 1 used to find the target families, so the figure and the
    model agree by construction.

    ``fallback_rate`` records how often a prediction had no usable neighbour and
    fell back to bias-only. On KIBA this is the number worth reporting: 35% of
    target pairs share fewer than 10 drugs, so thin neighbourhoods are a
    property of the data, not a bug in the model.
    """

    def __init__(self, k: int = 20, min_overlap: int = 10, min_sim: float = 0.0,
                 reg: float = 5.0, name: str | None = None):
        super().__init__(name=name or f"item-kNN (k={k})")
        self.k = k
        self.min_overlap = min_overlap
        self.min_sim = min_sim
        self.reg = reg
        self.sim = np.zeros((0, 0))
        self.matrix = np.zeros((0, 0))
        self.fallback_rate = float("nan")

    def fit(self, train: pd.DataFrame):
        self._fit_biases(train, reg=self.reg)
        rows = train["drug_id"].map(self.drug_index).to_numpy()
        cols = train["target_id"].map(self.target_index).to_numpy()

        # Residual matrix: what each rating is after the biases explain what
        # they can. Neighbours then vote on the part that is left.
        self.matrix = np.full((len(self.drug_index), len(self.target_index)), np.nan)
        self.matrix[rows, cols] = (
            train["score"].to_numpy(float) - self.mu - self.b[rows] - self.c[cols]
        )

        self.sim = masked_corr(self.matrix.T, min_overlap=self.min_overlap)
        np.fill_diagonal(self.sim, 0.0)  # a target is not its own neighbour
        self.sim[~np.isfinite(self.sim)] = 0.0
        self.sim[self.sim < self.min_sim] = 0.0
        self.params = {"k": self.k, "min_overlap": self.min_overlap,
                       "min_sim": self.min_sim, "reg": self.reg}
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        rows, cols = self._positions(frame)
        out = self._bias_prediction(rows, cols)
        n_fallback = 0

        for n, (i, j) in enumerate(zip(rows, cols)):
            if i < 0 or j < 0:
                n_fallback += 1
                continue
            observed = np.flatnonzero(np.isfinite(self.matrix[i]))
            weights = self.sim[j, observed]
            usable = np.flatnonzero(weights != 0.0)
            if len(usable) == 0:
                n_fallback += 1
                continue
            if len(usable) > self.k:
                usable = usable[np.argsort(-np.abs(weights[usable]))[: self.k]]
            w = weights[usable]
            out[n] += float(w @ self.matrix[i, observed[usable]] / np.abs(w).sum())

        self.fallback_rate = n_fallback / max(len(frame), 1)
        return out
