"""Task 3 end to end, on the cold-target split.

    python scripts/make_splits.py    # first, if data/processed/ is empty
    python scripts/run_task3.py

Everything here uses ``<dataset>_cold_target.csv``, never the warm split. The
held-out folds contain whole sequence groups, so a validation or test target
has no training measurements and no sequence-identical twin in training - the
situation plain collaborative filtering cannot address.

Same protocol as Task 2, and the same reason for it: fit on train, choose on
**validation CI**, read test exactly once, at the end. The validation fold here
is the *cold* one, so hyperparameters are chosen under the conditions they will
be judged on. Tuning these models on the warm fold would select for warm
behaviour and tell us nothing about cold targets.

The baseline is not an arbitrary choice. On this split every Task 2 model -
bias-only, matrix factorisation, item-kNN - returns identical predictions,
because none has a ``c_j`` or ``v_j`` for an unseen target and all collapse to
``mu + b_i``. The per-drug mean *is* collaborative filtering here, and beating
it is the whole task.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import cf, cold, data, metrics, split, tuning, viz
from src.eda import STRONG_BINDING

SELECT_ON = "ci"
MIN_CANDIDATES = 5

#: The factorisation Task 2 selected. Task 3 reuses it rather than re-tuning:
#: the question here is how to extend a fitted model to a target with no
#: column, not which factorisation to fit.
MF_PARAMS = {"n_factors": 40, "reg": 5.0, "n_epochs": 400, "patience": 12,
             "monitor": SELECT_ON}

#: ``min_cosine=0`` is the ablation that asks whether Task 1's threshold finding
#: survives contact with a model: it takes the k nearest neighbours whatever
#: their similarity. If the threshold earns nothing, this wins.
AXES = {"k": [1, 3, 5, 10, 20], "min_cosine": [0.0, 0.2, 0.3, 0.4, 0.5]}


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


def warm_holdout(frame: pd.DataFrame, seed: int, frac: float = 0.10):
    """Carve a warm validation subset out of the cold *training* fold.

    The cold validation fold cannot early-stop a factorisation. It contains
    only unseen targets, so the model scores ``mu + b_i`` there however well
    its factors are fitted - and measured on KIBA, its CI *falls* monotonically
    from epoch 1 (0.7048, 0.6981, 0.6967, ...), because ``b_i`` is specialising
    to work alongside a ``c_j`` the cold fold never supplies. Early stopping
    against it therefore picks epoch 1 and hands the transfer model latent
    vectors that were never trained.

    The factorisation is a Task 2 object and needs a Task 2 fold to select on,
    so one is carved from inside cold-train: a random 10% of pairs, with the
    first pair of every drug and every target reserved to keep the rest warm.
    The cold validation fold is then spent only on what it can actually judge -
    the transfer rule.
    """
    rng = np.random.default_rng(seed)
    shuffled = frame.iloc[rng.permutation(len(frame))]
    reserved = (~shuffled.duplicated("drug_id") | ~shuffled.duplicated("target_id")).to_numpy()

    hold = np.zeros(len(frame), dtype=bool)
    free = np.flatnonzero(~reserved)
    hold[free[: int(round(frac * len(frame)))]] = True
    return shuffled[~hold].reset_index(drop=True), shuffled[hold].reset_index(drop=True)


def fit_shared_mf(train: pd.DataFrame, seed: int) -> cf.BiasedMF:
    """Choose the epoch count on a warm holdout, then refit on all of cold-train."""
    inner_train, inner_val = warm_holdout(train, seed=seed)
    probe = cf.BiasedMF(**MF_PARAMS).fit(inner_train, val=inner_val)
    settled = dict(MF_PARAMS, n_epochs=max(probe.best_epoch or 1, 1))
    print(f"shared factorisation: {probe.best_epoch} epochs chosen on a warm "
          f"holdout of {len(inner_val):,} pairs, refitted on all "
          f"{len(train):,}", flush=True)
    return cf.BiasedMF(**settled).fit(train)


def main() -> int:
    pd.set_option("display.width", 220, "display.max_columns", 60,
                  "display.float_format", lambda v: f"{v:,.4f}")

    test_rows, boards = [], []
    started = time.perf_counter()

    for dataset in ("davis", "kiba"):
        ds = data.load(dataset)
        s = split.Split.load(dataset, "cold_target")
        train, val, test = s.train, s.val, s.test
        strong = STRONG_BINDING["pKd" if dataset == "davis" else "KIBA"]

        banner(f"{dataset.upper()} cold-target: {s.report['groups_test']} of "
               f"{s.report['n_seq_groups']} sequence groups held out for test "
               f"(train {len(train):,}  val {len(val):,}  test {len(test):,})")

        # Fitted once and shared across the grid.
        shared_mf = fit_shared_mf(train, seed=s.seed)

        # Each rung of the brief's ladder, with the axes that rung actually has.
        # The two neighbour models share AXES; the regressions have their own.
        searches = {
            "sequence kNN": (
                lambda **p: cold.SequenceNeighbour(ds.targets, **p), AXES),
            "MF + transfer": (
                lambda **p: cold.MFSequenceTransfer(ds.targets, mf=shared_mf, **p), AXES),
            "latent regression": (
                lambda **p: cold.LatentRegressor(ds.targets, mf=shared_mf, **p),
                {"ridge": [0.01, 0.1, 1.0, 10.0, 100.0]}),
            # A deliberately small grid. One fit costs ~50 s on DAVIS and
            # ~110 s on KIBA even with the protein features reduced to 128
            # components, and this rung is the ablation rather than the
            # candidate - it exists to show what is lost when the
            # collaborative term is dropped entirely.
            "content regression": (
                lambda **p: cold.ContentRegressor(ds.targets, ds.ligands, **p),
                {"n_estimators": [300], "num_leaves": [31, 127],
                 "learning_rate": [0.05, 0.1]}),
        }

        tuned = {}
        for name, (factory, axes) in searches.items():
            print(f"\n{name}:", flush=True)
            result = tuning.tune(
                factory, axes, train, val, model_name=name, dataset=dataset,
                select_on=SELECT_ON, strong_threshold=strong,
                min_candidates=MIN_CANDIDATES,
            )
            print(f"  -> best {result.best_params}, val {SELECT_ON}="
                  f"{result.best_score:.4f}")
            tuned[name] = result
            viz.save_table(result.table, f"task3_grid_{dataset}_{name.replace(' ', '_')}")
        boards.append(tuning.leaderboard(list(tuned.values())))

        # ------------------------------------------------------------------
        # Test, once. Every choice above was made on the cold validation fold.
        # ------------------------------------------------------------------
        banner(f"{dataset.upper()} - cold-target test fold, scored once")
        finalists = [
            cf.DrugMean().fit(train),          # what all of Task 2 collapses to
            shared_mf,                          # shown collapsing, for the report
            *[tuned[n].best for n in searches],
        ]
        for model in finalists:
            row = metrics.evaluate(
                model.predict_frame(test), label=model.name,
                strong_threshold=strong, min_candidates=MIN_CANDIDATES,
            )
            row["dataset"] = dataset.upper()
            row["fallback"] = getattr(model, "fallback_rate", float("nan"))
            test_rows.append(row)

    banner("validation leaderboard (chosen on the cold val fold)")
    board = pd.concat(boards)
    print(board.to_string())
    viz.save_table(board, "task3_leaderboard")

    banner("cold-target test results")
    results = pd.DataFrame(test_rows)
    columns = ["dataset", "model", "n_pairs", "mse", "mse_strong", "ci",
               "hit@1", "hit@5", "mrr", "fallback", "drugs_ranked"]
    results = results[[c for c in columns if c in results]].set_index(["dataset", "model"])
    print(results.to_string())
    viz.save_table(results, "task3_test_results")

    print(f"\ntotal {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
