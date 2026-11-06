"""Grid search on the validation fold.

The models in :mod:`src.cf` have two kinds of number in them. ``u``, ``v``, ``b``
and ``c`` are learned by ``fit``. The rest - how many factors, how much
regularisation, how many neighbours - are chosen, and this module is where they
are chosen: fit every combination on train, score it on **val**, keep the
winner, and leave test alone until the one final run.

Picking the winner by test score would make the reported number optimistic by
an amount nobody can measure afterwards, which is why the grid never sees test
and :func:`tune` has no way to reach it.

**The selection metric is CI by default**, not MSE. 69.6% of DAVIS labels sit at
the pKd 5 censoring floor, so MSE rewards a model for predicting "weak"
everywhere; selecting on it would tune for the assay's detection limit. CI asks
whether the model orders a drug's targets correctly, which is what a
recommender is for. Every config's MSE and Hit@k are kept in the result table
regardless, so the report can show both and say which one chose.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import pandas as pd

from .cf import Model
from .metrics import evaluate

#: Lower is better for these; everything else is treated as higher-is-better.
_LOWER_IS_BETTER = {"mse", "rmse", "mse_strong"}


def grid(**axes) -> list[dict]:
    """Every combination of the given hyperparameter axes.

    ``grid(n_factors=[5, 10], reg=[0.1, 1])`` gives the four dicts that
    :func:`tune` passes to the model factory.
    """
    if not axes:
        return [{}]
    names = list(axes)
    return [dict(zip(names, values)) for values in itertools.product(*(axes[n] for n in names))]


@dataclass
class TuningResult:
    """Everything the report needs about one model's search."""

    model_name: str
    dataset: str
    select_on: str
    #: One row per configuration, with its validation metrics. Sorted best first.
    table: pd.DataFrame
    best_params: dict
    best: Model
    seconds: float = 0.0
    history: list[float] = field(default_factory=list)

    @property
    def best_score(self) -> float:
        return float(self.table.iloc[0][self.select_on])

    def sensitivity(self, within: float = 0.005) -> pd.DataFrame:
        """Configurations scoring within ``within`` of the winner.

        A wide band means the choice barely mattered and the report can say so
        instead of implying the grid found something precise.
        """
        best = self.best_score
        column = self.table[self.select_on]
        close = (best - column).abs() <= within
        return self.table[close]

    def __repr__(self) -> str:
        return (
            f"TuningResult({self.dataset}/{self.model_name}, "
            f"{len(self.table)} configs, best {self.select_on}="
            f"{self.best_score:.4f} at {self.best_params}, {self.seconds:.1f}s)"
        )


def tune(
    factory,
    axes: dict,
    train: pd.DataFrame,
    val: pd.DataFrame,
    model_name: str,
    dataset: str = "",
    select_on: str = "ci",
    strong_threshold: float | None = None,
    min_candidates: int = 2,
    fit_kwargs: dict | None = None,
    verbose: bool = True,
) -> TuningResult:
    """Fit every configuration on ``train``, score on ``val``, return the best.

    ``factory`` is called with one configuration's keyword arguments and must
    return an unfitted model. ``fit_kwargs`` is passed through to ``fit``, which
    is how :class:`~src.cf.BiasedMF` receives the validation fold it early-stops
    against - the epoch count is read off the curve rather than being another
    axis of the grid.
    """
    configs = grid(**axes)
    fit_kwargs = fit_kwargs or {}
    rows, models = [], []
    started = time.perf_counter()

    for i, params in enumerate(configs, 1):
        model = factory(**params)
        model.fit(train, **fit_kwargs)
        row = evaluate(
            model.predict_frame(val), label=model.name,
            strong_threshold=strong_threshold, min_candidates=min_candidates,
        )
        row |= params
        if getattr(model, "best_epoch", None) is not None:
            row["best_epoch"] = model.best_epoch
        if getattr(model, "fallback_rate", None) is not None:
            row["fallback_rate"] = model.fallback_rate
        rows.append(row)
        models.append(model)
        if verbose:
            print(f"  [{i:>3}/{len(configs)}] {params} -> "
                  f"{select_on}={row[select_on]:.4f}", flush=True)

    table = pd.DataFrame(rows)
    table = table.sort_values(select_on, ascending=select_on in _LOWER_IS_BETTER)
    best_position = table.index[0]

    return TuningResult(
        model_name=model_name, dataset=dataset, select_on=select_on,
        table=table.reset_index(drop=True),
        best_params=configs[best_position],
        best=models[best_position],
        seconds=time.perf_counter() - started,
        history=list(getattr(models[best_position], "history", [])),
    )


def leaderboard(results: list[TuningResult]) -> pd.DataFrame:
    """The winning configuration per model per dataset, for the report."""
    rows = []
    for r in results:
        best = r.table.iloc[0]
        rows.append({
            "dataset": r.dataset.upper(), "model": r.model_name,
            "configs_tried": len(r.table),
            "val_ci": best.get("ci"), "val_mse": best.get("mse"),
            "val_hit@1": best.get("hit@1"), "val_hit@5": best.get("hit@5"),
            "best_params": ", ".join(f"{k}={v}" for k, v in r.best_params.items()),
            "configs_within_0.005": len(r.sensitivity(0.005)),
            "seconds": round(r.seconds, 1),
        })
    return pd.DataFrame(rows).set_index(["dataset", "model"])
