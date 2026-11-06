"""Task 2 end to end: tune on validation, then score the test fold once.

    python scripts/make_splits.py     # first, if data/processed/ is empty
    python scripts/run_task2.py

The protocol, which the report should state in these words:

1. Fit every hyperparameter configuration on **train**.
2. Pick the winner by **validation CI** - not MSE. 69.6% of DAVIS labels sit at
   the pKd 5 censoring floor, so MSE rewards predicting "weak" everywhere, and
   selecting on it would tune for the assay's detection limit rather than for
   the ranking the task asks about.
3. Score the winners on **test**, once, at the end of this script.

Test is read in exactly one place below, after every choice has been made.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import casestudy, cf, metrics, split, tuning, viz
from src.eda import STRONG_BINDING

SELECT_ON = "ci"

#: A drug needs enough held-out targets for Hit@5 to be able to fail. KIBA's
#: median test drug has 8 candidates, so without this the metric would partly
#: measure the split.
MIN_CANDIDATES = 5

#: Factor counts, per dataset. A first grid stopping at 40 had its winner sit
#: on the edge for both datasets, which means the grid was too small to claim
#: an optimum, so the ceiling was probed directly at reg=5:
#:
#:     DAVIS  k=20 .8858  k=40 .8914  k=55 .8914  k=68 .8914   saturated at 40
#:     KIBA   k=40 .8486  k=60 .8523  k=90 .8538  k=120 .8539  saturated near 90
#:
#: DAVIS has only 68 drugs, so 68 is its hard ceiling and the curve is flat well
#: before it. KIBA keeps gaining to about 90. Both grids now bracket their
#: plateau rather than ending at it.
#: k=120 is deliberately absent from KIBA's grid. The probe measured it at
#: 0.8539 against k=90's 0.8538 - nothing - while costing more per epoch than
#: every other configuration combined, and at low reg it failed to converge
#: inside a 400-epoch ceiling at all. The plateau is established by the probe;
#: the grid does not need to pay for it again.
FACTORS = {"davis": [5, 10, 20, 40, 55], "kiba": [10, 20, 40, 60, 90]}

#: 400, not 100. The first run stopped KIBA's k=40 fit at the 100-epoch ceiling
#: while it was still improving (its best epoch is 170). Larger k converges
#: faster - k=120 peaks at epoch 25 - so the ceiling costs little in practice,
#: but it has to sit above the slowest configuration in the grid.
EPOCH_CEILING = 400


def search_space(dataset: str) -> dict:
    """Model name -> (factory, hyperparameter axes, fit kwargs)."""
    return {
        "bias only": (
            lambda **p: cf.BiasOnly(**p),
            {"reg": [0.1, 1.0, 5.0, 10.0, 25.0]},
            {},
        ),
        "biased MF": (
            lambda **p: cf.BiasedMF(n_epochs=EPOCH_CEILING, patience=12,
                                    monitor=SELECT_ON, **p),
            # reg below 1 was uniformly bad in the first sweep at every k above
            # 5 (DAVIS k=40 scored 0.744 at reg=0.1 against 0.891 at reg=5),
            # and it is also the slowest region to converge. The surviving grid
            # brackets the optimum on both sides.
            {"n_factors": FACTORS[dataset], "reg": [1.0, 5.0, 10.0, 20.0]},
            "needs_val",
        ),
        "item-kNN": (
            lambda **p: cf.ItemKNN(**p),
            {"k": [5, 10, 20, 50, 100], "min_overlap": [5, 10, 20], "min_sim": [0.0, 0.1]},
            {},
        ),
    }


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


def main() -> int:
    pd.set_option("display.width", 220, "display.max_columns", 60,
                  "display.float_format", lambda v: f"{v:,.4f}")

    all_results, test_rows, leaderboards = [], [], []
    started = time.perf_counter()

    for dataset in ("davis", "kiba"):
        s = split.Split.load(dataset, "warm")
        train, val, test = s.train, s.val, s.test
        strong = STRONG_BINDING["pKd" if dataset == "davis" else "KIBA"]

        banner(f"{dataset.upper()} - tuning on validation "
               f"(train {len(train):,}  val {len(val):,}  test {len(test):,})")

        search = search_space(dataset)
        tuned = {}
        for name, (factory, axes, fit_kwargs) in search.items():
            print(f"\n{name}:", flush=True)
            result = tuning.tune(
                factory, axes, train, val, model_name=name, dataset=dataset,
                select_on=SELECT_ON, strong_threshold=strong,
                min_candidates=MIN_CANDIDATES,
                fit_kwargs={"val": val} if fit_kwargs == "needs_val" else {},
            )
            print(f"  -> best {result.best_params}, val {SELECT_ON}="
                  f"{result.best_score:.4f} "
                  f"({len(result.sensitivity(0.005))} configs within 0.005)")
            tuned[name] = result
            all_results.append(result)
            viz.save_table(result.table, f"task2_grid_{dataset}_{name.replace(' ', '_')}")

        leaderboards.append(tuning.leaderboard([tuned[n] for n in search]))

        # ------------------------------------------------------------------
        # The one and only look at test. Every choice above was made on val.
        # ------------------------------------------------------------------
        banner(f"{dataset.upper()} - test fold, scored once")
        finalists = [cf.RandomRanking().fit(train), cf.GlobalMean().fit(train),
                     cf.DrugMean().fit(train)]
        finalists += [tuned[n].best for n in search]

        # Every fold, because the brief asks for train, validation and test.
        # The train row is what shows whether a model memorised: a big gap
        # between train and test is the diagnosis, and it cannot be read off
        # the test column alone.
        for model in finalists:
            for fold_name, fold in (("train", train), ("val", val), ("test", test)):
                row = metrics.evaluate(
                    model.predict_frame(fold), label=model.name,
                    strong_threshold=strong, min_candidates=MIN_CANDIDATES,
                )
                row["dataset"] = dataset.upper()
                row["fold"] = fold_name
                row["params"] = ", ".join(f"{k}={v}" for k, v in model.params.items())
                test_rows.append(row)

        # The checkpoint the brief asks for, plus proof it reloads identically.
        best_mf = tuned["biased MF"].best
        path = best_mf.save(cf.SAVED_MODELS / f"{dataset}_cf.npz")
        reloaded = cf.BiasedMF.load(path)
        np.testing.assert_allclose(reloaded.predict(test), best_mf.predict(test))
        print(f"\nsaved {path.name} and verified reload predicts identically "
              f"(np.allclose over {len(test):,} test pairs)")

        # Driven from `reloaded`, not from the model still in memory, so the
        # case study doubles as evidence that the saved checkpoint is the thing
        # that produced the numbers.
        banner(f"{dataset.upper()} - case studies, from the reloaded checkpoint")
        drugs, how, details, summary = casestudy.report(reloaded, test, train, dataset)
        print(f"drugs chosen: {how}\n")
        for drug_id in drugs:
            print(f"drug {drug_id} - top {len(details[drug_id])} recommended targets")
            print(details[drug_id].to_string(index=False))
            print()
        print(summary.to_string())
        viz.save_table(summary, f"task2_case_study_{dataset}")
        viz.save_table(pd.concat(details, names=["drug_id", "row"]),
                       f"task2_case_study_{dataset}_detail")

    banner("leaderboard: winning configuration per model, chosen on validation")
    board = pd.concat(leaderboards)
    print(board.to_string())
    viz.save_table(board, "task2_leaderboard")

    banner("test results")
    results = pd.DataFrame(test_rows)
    columns = ["dataset", "fold", "model", "n_pairs", "mse", "mse_strong", "ci",
               "hit@1", "hit@5", "mrr", "drugs_ranked", "median_candidates", "params"]
    results = results[[c for c in columns if c in results]].set_index(["dataset", "model", "fold"])
    print(results.to_string())
    viz.save_table(results, "task2_test_results")

    print(f"\ntotal {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
