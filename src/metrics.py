"""Scoring for Tasks 2 and 3: error, ordering agreement and ranking quality.

Both tasks are graded on the same numbers, so they live in one module and are
computed one way. Every function takes true and predicted scores for a set of
pairs and returns a float, except :func:`evaluate`, which returns the whole row
a results table needs.

Two things in here are easy to get wrong, so they are pinned down by tests:

* **CI is not Kendall's tau.** The concordance index is the share of *usable*
  pairs - those whose true scores differ - that the model orders correctly,
  counting a predicted tie as half right. It is recovered from ``tau-b`` with
  the identity in :func:`concordance_index`, which is O(n log n); the direct
  double loop is O(n^2) and does not finish on KIBA's test fold.
* **Hit@k is a ranking over a drug's own candidates**, so it depends on how many
  candidates that drug has. A drug with 3 held-out targets scores Hit@5 = 1
  whatever the model says. :func:`evaluate` therefore reports the candidate
  distribution beside the metric, and ``min_candidates`` filters the drugs that
  cannot fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

#: Ranking metrics need a tie-break among equal predictions. A constant
#: predictor (the per-drug mean, or CF's cold-target fallback) predicts one
#: value for every candidate, and breaking those ties by row order would score
#: it on the dataset's id ordering rather than on the model. Ties are broken
#: randomly from this seed instead, so a constant predictor earns chance.
TIE_BREAK_SEED = 20261115


def mse(y_true, y_pred) -> float:
    """Mean squared error in the dataset's own score units."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error, which is readable as pKd or KIBA units."""
    return float(np.sqrt(mse(y_true, y_pred)))


def concordance_index(y_true, y_pred) -> float:
    """Share of correctly ordered pairs; 0.5 is chance, 1.0 is perfect.

    Over every pair whose *true* scores differ, CI counts a pair as 1 when the
    predictions order it the same way, 0.5 when the predictions tie, and 0
    otherwise. Writing A for those usable pairs and B for the pairs whose
    predictions differ, ``tau-b`` gives ``C - D = tau * sqrt(A * B)`` and the
    definition rearranges to::

        CI = (tau * sqrt(B / A) + 1) / 2

    which is why this is a one-line call rather than a loop over 160 million
    pairs. Undefined when every true score is identical: there is no pair to
    order, and that returns NaN rather than a misleading 0.5.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if len(y_true) < 2:
        return float("nan")

    tau_result = kendalltau(y_true, y_pred, variant="b", nan_policy="omit")
    n_pairs = len(y_true) * (len(y_true) - 1) / 2
    a = n_pairs - _tied_pairs(y_true)  # pairs with different true scores
    b = n_pairs - _tied_pairs(y_pred)  # pairs with different predictions
    if a <= 0:
        return float("nan")
    if b <= 0:
        return 0.5  # every prediction tied: each usable pair is half right
    return float((tau_result.statistic * np.sqrt(b / a) + 1) / 2)


def _tied_pairs(values: np.ndarray) -> float:
    """Number of pairs sharing a value, summed over groups of equal values."""
    counts = np.unique(values, return_counts=True)[1]
    return float(np.sum(counts * (counts - 1) / 2))


def _ranked_candidates(frame: pd.DataFrame, rng: np.random.Generator):
    """Yield ``(hit_ranks, n_candidates)`` per drug, best-predicted first.

    ``hit_ranks`` are the 1-based ranks of the drug's truly strongest held-out
    targets. There can be several when the top true score is tied, and any of
    them counts, so the caller takes the smallest.
    """
    for _, group in frame.groupby("drug_id", sort=False):
        truth = group["score"].to_numpy(float)
        pred = group["pred"].to_numpy(float)
        if len(truth) < 2:
            continue  # nothing to rank: one candidate is always rank 1

        # Random tie-break, then sort descending by prediction.
        order = np.lexsort((rng.random(len(pred)), -pred))
        rank_of = np.empty(len(pred), int)
        rank_of[order] = np.arange(1, len(pred) + 1)

        best = np.flatnonzero(truth == truth.max())
        yield rank_of[best].min(), len(truth)


def ranking_metrics(
    frame: pd.DataFrame,
    ks=(1, 5),
    min_candidates: int = 2,
    seed: int = TIE_BREAK_SEED,
) -> dict:
    """Hit@k and MRR over each drug's own held-out targets.

    ``frame`` needs ``drug_id``, ``score`` and ``pred``. Drugs with fewer than
    ``min_candidates`` held-out targets are excluded: with k or fewer
    candidates Hit@k is 1 by construction and measures the split, not the
    model. The number kept and the candidate distribution come back in the
    result so the report can state what the metric was computed over.
    """
    rng = np.random.default_rng(seed)
    ranks, sizes = [], []
    for rank, n in _ranked_candidates(frame, rng):
        if n >= min_candidates:
            ranks.append(rank)
            sizes.append(n)

    if not ranks:
        return {f"hit@{k}": float("nan") for k in ks} | {
            "mrr": float("nan"), "drugs_ranked": 0,
            "median_candidates": float("nan"), "min_candidates_used": min_candidates,
        }

    ranks_arr = np.asarray(ranks)
    out = {f"hit@{k}": float(np.mean(ranks_arr <= k)) for k in ks}
    out["mrr"] = float(np.mean(1.0 / ranks_arr))
    out["drugs_ranked"] = len(ranks_arr)
    out["median_candidates"] = float(np.median(sizes))
    out["min_candidates_used"] = min_candidates
    return out


def evaluate(
    frame: pd.DataFrame,
    label: str = "",
    ks=(1, 5),
    min_candidates: int = 2,
    strong_threshold: float | None = None,
) -> dict:
    """Every metric for one set of predictions, as one row of a results table.

    ``frame`` needs ``drug_id``, ``score`` and ``pred``. ``strong_threshold``
    adds the error over the strong binders alone, which is the number that
    matters on DAVIS: 69.6% of its labels sit at the pKd 5 censoring floor, so
    overall MSE rewards a model for predicting "weak" everywhere and hides
    whether it got the binders right.
    """
    truth, pred = frame["score"].to_numpy(float), frame["pred"].to_numpy(float)
    row = {
        "model": label,
        "n_pairs": len(frame),
        "mse": mse(truth, pred),
        "rmse": rmse(truth, pred),
        "ci": concordance_index(truth, pred),
    }
    if strong_threshold is not None:
        strong = truth >= strong_threshold
        row["n_strong"] = int(strong.sum())
        row["mse_strong"] = mse(truth[strong], pred[strong]) if strong.any() else float("nan")
    row.update(ranking_metrics(frame, ks=ks, min_candidates=min_candidates))
    return row


def results_table(rows: list[dict]) -> pd.DataFrame:
    """Stack evaluate() rows into the table the report prints."""
    return pd.DataFrame(rows).set_index("model")
