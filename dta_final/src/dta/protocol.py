"""The experiment runner: tune on validation, evaluate once on test, repeat over seeds.

One place decides which models run in which regime and how they are tuned, so
every number in the results table was produced by the same procedure.

**Tuning.**  Hyperparameters are chosen by grid search on the validation fold
of the first seed, then held fixed across the remaining seeds.  Re-tuning per
seed would be defensible too, but it costs five times as much and makes the
spread across seeds partly a spread in hyperparameters rather than in data.
Test is never consulted.

**Repetition.**  Every setting runs over five seeds and reports mean and
standard deviation.  Single-split protocols are the norm for warm-start work
in this field, but with 68 DAVIS drugs a single draw is fragile, so the warm
setting gets the same treatment as the cold ones.

**Which models run where.**  A model only enters a regime it can actually
answer.  Collaborative filtering is listed in the cold regimes through the
fallback it degrades to, because that degradation is the finding, not an
oversight.
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import cold as C
from . import data as D
from . import kernels as K
from . import metrics as MT
from . import models as M
from . import splits as S
from .config import DATASETS, N_BOOTSTRAP, SPLIT_SEEDS, TABLES
from .kronrls import KronRLS, KronRLSMKL


@dataclass
class Spec:
    """A model, how to build it, and what to search over."""

    name: str
    factory: callable
    grid: dict = field(default_factory=dict)
    needs: tuple = ()

    def candidates(self):
        if not self.grid:
            return [{}]
        keys = list(self.grid)
        return [dict(zip(keys, vals)) for vals in itertools.product(*(self.grid[k] for k in keys))]


# ---------------------------------------------------------------------------
# The zoo
# ---------------------------------------------------------------------------

MF_GRID = {"k": [8, 16, 32, 64], "lam": [0.01, 0.02, 0.05, 0.1]}
KNN_GRID = {"k": [5, 10, 20, 50], "beta": [10.0, 50.0]}
KRON_GRID = {"lam": [0.1, 0.3, 1.0, 3.0]}
SEQKNN_GRID = {"k": [5, 10, 20], "gamma": [1.0, 3.0, 5.0]}
#: Which protein representation the feature-based models use. Tuned on
#: validation like any other hyperparameter rather than fixed by hand, because
#: the ablation showed it matters more than anything else in these models.
BLOCK_GRID = {"target_block": ["target", "target_kmer_svd", "target_sw_profile",
                               "target_esm2_35m", "target_esm2_650m"]}
MIXUP_GRID = {"mixup": ["drug", "target"]}


def warm_specs():
    return [
        Spec("B0_global_mean", lambda **p: M.GlobalMean()),
        Spec("B1_ligand_mean", lambda **p: M.LigandMean()),
        Spec("B2_target_mean", lambda **p: M.TargetMean()),
        Spec("B3_bias_only", lambda **p: M.BiasOnly(**p), {"lam": [0.1, 1.0, 5.0]}),
        Spec("M1_item_knn", lambda **p: M.ItemKNN(**p), KNN_GRID),
        Spec("M2_biased_mf", lambda **p: M.BiasedMF(**p), MF_GRID),
        Spec("M4_tobit_mf", lambda **p: M.TobitMF(**p),
             {"k": [16, 32], "lam": [0.02, 0.05]}),
        Spec("M4b_tobit_squared", lambda **p: M.TobitMF(use_censored_loss=False, **p),
             {"k": [16, 32], "lam": [0.02, 0.05]}),
        Spec("M5_kronrls", lambda **p: KronRLS(**p), KRON_GRID, needs=("kernels",)),
        Spec("M6_kronrls_mkl", lambda **p: KronRLSMKL(**p), KRON_GRID, needs=("kernels",)),
    ]


def cold_target_specs():
    return [
        Spec("C0_fallback", lambda **p: C.ColdFallback()),
        Spec("C1_sequence_knn", lambda **p: C.SequenceKNN(axis="target", **p),
             SEQKNN_GRID, needs=("kernels",)),
        Spec("C2_factor_map", lambda **p: C.FactorMap(**p),
             {"k": [8, 16, 32], **BLOCK_GRID}, needs=("features",)),
        Spec("C3_content_reg", lambda **p: C.ContentRegressor(**p), BLOCK_GRID,
             needs=("features",)),
        Spec("C5_content_mixup", lambda **p: C.ContentRegressor(**p),
             {**BLOCK_GRID, **MIXUP_GRID}, needs=("features",)),
        Spec("M5_kronrls", lambda **p: KronRLS(**p), KRON_GRID, needs=("kernels",)),
        Spec("M6_kronrls_mkl", lambda **p: KronRLSMKL(**p), KRON_GRID, needs=("kernels",)),
    ]


def cold_drug_specs():
    return [
        Spec("C0_fallback", lambda **p: M.TargetMean()),
        Spec("C1_structure_knn", lambda **p: C.SequenceKNN(axis="drug", **p),
             SEQKNN_GRID, needs=("kernels",)),
        Spec("C3_content_reg", lambda **p: C.ContentRegressor(**p), {}, needs=("features",)),
        Spec("M5_kronrls", lambda **p: KronRLS(**p), KRON_GRID, needs=("kernels",)),
    ]


def both_new_specs():
    return [
        Spec("C0_fallback", lambda **p: M.GlobalMean()),
        Spec("C3_content_reg", lambda **p: C.ContentRegressor(**p), {}, needs=("features",)),
        Spec("M5_kronrls", lambda **p: KronRLS(**p), KRON_GRID, needs=("kernels",)),
    ]


SPECS = {
    "S1_pair": warm_specs,
    "S2_new_drug": cold_drug_specs,
    "S3_new_target": cold_target_specs,
    "S4_both_new": both_new_specs,
}


# ---------------------------------------------------------------------------
# Fit, tune, evaluate
# ---------------------------------------------------------------------------


def _fit(spec: Spec, params: dict, tr, nl, nt, va, ctx):
    model = spec.factory(**params)
    return model.fit(tr, nl, nt, val=va, kernels=ctx.get("kernels"),
                     features=ctx.get("features"), dataset=ctx.get("dataset"))


def tune(spec: Spec, tr, nl, nt, va, ctx, verbose=False) -> tuple[dict, pd.DataFrame]:
    """Grid search on validation MSE. Test is never touched here."""
    rows = []
    best = (np.inf, {})
    for params in spec.candidates():
        try:
            t0 = time.time()
            model = _fit(spec, params, tr, nl, nt, va, ctx)
            score = MT.mse(va["score"].to_numpy(dtype=float), model.predict(va))
            rows.append({**params, "val_mse": score, "seconds": time.time() - t0})
            if np.isfinite(score) and score < best[0]:
                best = (score, params)
        except Exception as exc:  # a bad corner of the grid must not kill the sweep
            rows.append({**params, "val_mse": np.nan, "error": str(exc)[:160]})
    return best[1], pd.DataFrame(rows)


def run_setting(
    dataset: str,
    setting: str,
    seeds=SPLIT_SEEDS,
    bootstrap_first_seed: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tune once, then evaluate every model on every seed of one regime."""
    proc = D.load_processed(dataset)
    pairs, nl, nt = proc["pairs"], proc["n_lig"], proc["n_tgt"]
    ctx = {
        "kernels": K.build_kernels(dataset, proc),
        "features": K.build_features(dataset, proc),
        "dataset": dataset,
    }
    specs = SPECS[setting]()

    # --- tune on the first seed -------------------------------------------
    a0 = S.load_split(dataset, setting, seeds[0])
    tr0, va0 = pairs[a0 == "train"], pairs[a0 == "val"]
    chosen, grids = {}, []
    for spec in specs:
        params, grid = tune(spec, tr0, nl, nt, va0, ctx)
        chosen[spec.name] = params
        grid.insert(0, "model", spec.name)
        grid.insert(0, "setting", setting)
        grid.insert(0, "dataset", dataset)
        grids.append(grid)
        if verbose:
            print(f"    tuned {spec.name:20s} -> {params}", flush=True)

    # --- evaluate across seeds --------------------------------------------
    rows = []
    for seed in seeds:
        assign = S.load_split(dataset, setting, seed)
        tr, va, te = pairs[assign == "train"], pairs[assign == "val"], pairs[assign == "test"]
        if not len(te):
            continue
        fitted = {}
        for spec in specs:
            t0 = time.time()
            try:
                model = _fit(spec, chosen[spec.name], tr, nl, nt, va, ctx)
                pred = model.predict(te)
                res = MT.evaluate(
                    te, pred, dataset, seed=seed,
                    bootstrap=bootstrap_first_seed and seed == seeds[0],
                    n_boot=N_BOOTSTRAP,
                )
                fitted[spec.name] = model
            except Exception as exc:
                print(f"    [warn] {spec.name} seed {seed}: {exc}", flush=True)
                continue
            res.update({
                "dataset": dataset, "setting": setting, "seed": seed,
                "model": spec.name, "params": json.dumps(chosen[spec.name]),
                "fallback_rate": getattr(model, "fallback_rate_", np.nan),
                "seconds": time.time() - t0,
            })
            rows.append(res)

        # Composite models reuse the components already fitted for this seed.
        for name, built in _composites(setting, fitted).items():
            try:
                built.fit(tr, nl, nt, val=va, kernels=ctx["kernels"],
                          features=ctx["features"], dataset=dataset)
                res = MT.evaluate(te, built.predict(te), dataset, seed=seed)
                res.update({
                    "dataset": dataset, "setting": setting, "seed": seed, "model": name,
                    "params": json.dumps({k: float(v) for k, v in
                                          [("weight", getattr(built, "weight", np.nan)),
                                           ("threshold", getattr(built, "threshold", np.nan))]
                                          if v is not None and np.isfinite(float(v))}),
                    "fallback_rate": np.nan, "seconds": np.nan,
                })
                rows.append(res)
            except Exception as exc:
                print(f"    [warn] {name} seed {seed}: {exc}", flush=True)

        if verbose:
            print(f"    seed {seed} done", flush=True)

    return pd.DataFrame(rows), pd.concat(grids, ignore_index=True)


def _composites(setting: str, fitted: dict) -> dict:
    """Models built out of two already-fitted models."""
    out = {}
    if setting == "S1_pair" and {"M1_item_knn", "M2_biased_mf"} <= set(fitted):
        out["M3_knn_mf_blend"] = M.Blend(fitted["M1_item_knn"], fitted["M2_biased_mf"])
    if setting == "S3_new_target" and {"C1_sequence_knn", "C3_content_reg"} <= set(fitted):
        out["C4_gated_blend"] = C.SimilarityGatedBlend(
            fitted["C1_sequence_knn"], fitted["C3_content_reg"]
        )
    return out


# ---------------------------------------------------------------------------
# Summarising
# ---------------------------------------------------------------------------

SUMMARY_METRICS = [
    "mse", "mse_strong", "mse_measured", "ci", "ci_measured",
    "ci_per_ligand", "ci_per_target", "hit_at_1", "hit_at_5", "mrr",
    "random_hit_at_1", "random_hit_at_5", "random_mrr", "fallback_rate",
]


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation across seeds."""
    have = [m for m in SUMMARY_METRICS if m in results.columns]
    g = results.groupby(["dataset", "setting", "model"], sort=False)[have]
    out = g.agg(["mean", "std"])
    out.columns = ["_".join(c) for c in out.columns]
    out["n_seeds"] = g.size().values
    return out.reset_index()


def main(datasets=None, settings=None, seeds=SPLIT_SEEDS, verbose=True) -> pd.DataFrame:
    datasets = datasets or list(DATASETS)
    settings = settings or list(SPECS)
    all_rows, all_grids = [], []
    for ds in datasets:
        for setting in settings:
            if verbose:
                print(f"\n=== {ds.upper()} / {setting}", flush=True)
            t0 = time.time()
            rows, grid = run_setting(ds, setting, seeds=seeds, verbose=verbose)
            all_rows.append(rows)
            all_grids.append(grid)
            rows.to_csv(TABLES / f"results_{ds}_{setting}.csv", index=False)
            grid.to_csv(TABLES / f"tuning_{ds}_{setting}.csv", index=False)
            if verbose:
                print(f"    [{time.time() - t0:.0f}s]", flush=True)

    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv(TABLES / "results_all.csv", index=False)
    grids = pd.concat(all_grids, ignore_index=True)
    grids.to_csv(TABLES / "tuning_all.csv", index=False)

    summary = summarise(results)
    summary.to_csv(TABLES / "summary_all.csv", index=False)
    if verbose:
        cols = ["dataset", "setting", "model", "mse_mean", "ci_mean",
                "ci_per_ligand_mean", "hit_at_1_mean"]
        print("\n" + summary[[c for c in cols if c in summary.columns]].to_string(index=False))
    return summary


if __name__ == "__main__":
    import sys

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ds = [a for a in args if a in DATASETS] or None
    st = [a for a in args if a in SPECS] or None
    quick = "--quick" in sys.argv
    main(datasets=ds, settings=st, seeds=SPLIT_SEEDS[:2] if quick else SPLIT_SEEDS)
