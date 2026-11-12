"""Task 1 end to end: clean, count, visualise and explain both datasets.

Writes every Task 1 table to ``results/tables`` and every figure to
``results/figures``, plus a JSON summary per dataset, so the report never
quotes a number that someone typed.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import analysis as A
from . import data as D
from . import kernels as K
from . import plots as P
from .config import DATASETS, RESULTS, SCORE_NAME, STRONG_THRESHOLD, TABLES


def format_examples(proc: dict, name: str, n: int = 3) -> pd.DataFrame:
    """A few rows exactly as stored, with the long strings joined back on."""
    pairs = D.attach_entities(proc["pairs"].head(n), proc)
    out = pairs[["ligand_id", "target_id"]].copy()
    out["SMILES"] = pairs["smiles"].str.slice(0, 40) + "..."
    out["sequence"] = pairs["sequence"].str.slice(0, 26) + "..."
    out["seq_len"] = pairs["sequence"].str.len()
    out[f"score ({SCORE_NAME[name]})"] = pairs["score"].round(3)
    out["n_repeats"] = pairs["n_repeats"]
    out["censored"] = pairs["censored"]
    return out


def run(name: str) -> dict:
    print(f"\n{'=' * 70}\nTASK 1  |  {name.upper()}\n{'=' * 70}")
    proc = D.load_processed(name)
    pairs, n_lig, n_tgt = proc["pairs"], proc["n_lig"], proc["n_tgt"]
    M = D.to_matrix(pairs, n_lig, n_tgt)

    format_examples(proc, name).to_csv(TABLES / f"task1_examples_{name}.csv", index=False)
    proc["counts"].to_csv(TABLES / f"task1_counts_{name}.csv", index=False)
    proc["cleaning_log"].to_csv(TABLES / f"task1_cleaning_{name}.csv", index=False)

    missing = float(np.mean(~np.isfinite(M)))
    print(f"  matrix {M.shape}, {missing:.1%} never measured")

    var = A.variance_decomposition(pairs, n_lig, n_tgt)
    var.to_csv(TABLES / f"task1_variance_{name}.csv", index=False)
    interaction = float(var.loc[var.component.str.startswith("interaction"),
                                "variance_explained"].iloc[0])
    print(f"  interaction share of variance (the CF ceiling): {interaction:.1%}")

    block = M if missing < 0.05 else A.densest_block(M)
    spectrum = A.svd_spectrum(block)
    spectrum.to_csv(TABLES / f"task1_svd_{name}.csv", index=False)
    cum5 = float(spectrum.loc[spectrum.component == 5, "cumulative_energy"].iloc[0])
    print(f"  top 5 SVD components hold {cum5:.1%} of the energy")

    for fn, fname in (
        (A.coverage_stats(pairs, n_lig, n_tgt), "coverage"),
        (A.overlap_stats(pairs, n_lig, n_tgt), "overlap"),
        (A.missingness_bias(pairs, n_lig, n_tgt), "missingness"),
        (A.censoring_stats(pairs), "censoring"),
        (A.completion_requirement(n_lig, n_tgt, int(0.7 * len(pairs))), "completion"),
    ):
        fn.to_csv(TABLES / f"task1_{fname}_{name}.csv", index=False)

    conflicts = A.duplicate_conflicts(pairs)
    if len(conflicts):
        conflicts.to_csv(TABLES / f"task1_duplicates_{name}.csv", index=False)
        print(f"  worst disagreement between merged duplicates: {conflicts.spread.max():.2f}")

    ks = K.build_kernels(name, proc)
    simprof = A.similarity_vs_profile(ks["target_sequence"], M)
    print(f"  sequence similarity vs affinity profile: rho = {simprof['spearman_rho']:.3f} "
          f"over {simprof['n_pairs']:,} target pairs")

    report = {
        "dataset": name,
        "score_name": SCORE_NAME[name],
        "shape": {"ligands": n_lig, "targets": n_tgt, "observed_pairs": int(len(pairs))},
        "sparsity": {
            "missing_cells": int(np.sum(~np.isfinite(M))),
            "total_cells": int(M.size),
            "missing_fraction": missing,
            "note": "a missing cell is an unmeasured pair, not a non-binding one",
        },
        "score_range": [float(pairs.score.min()), float(pairs.score.max())],
        "censored_fraction": float(pairs.censored.mean()),
        "strong_threshold": STRONG_THRESHOLD[name],
        "variance_explained": {r.component: float(r.variance_explained)
                               for r in var.itertuples()},
        "interaction_share": interaction,
        "svd_cumulative_energy_top5": cum5,
        "svd_block_shape": list(block.shape),
        "similarity_vs_profile": {k: v for k, v in simprof.items()
                                  if k not in ("seq_sim", "profile_corr")},
        "duplicate_max_spread": float(conflicts.spread.max()) if len(conflicts) else 0.0,
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(RESULTS / f"task1_{name}.json", "w") as fh:
        json.dump(report, fh, indent=2, default=float)

    return {"report": report, "matrix": M, "pairs": pairs, "variance": var,
            "spectrum": spectrum, "simprof": simprof}


def main(which=None) -> None:
    which = which or list(DATASETS)
    out = {ds: run(ds) for ds in which}

    print(f"\n{'=' * 70}\nFIGURES\n{'=' * 70}")
    P.use_style()
    paths = [
        P.heatmaps({ds: out[ds]["matrix"] for ds in which}),
        P.distributions(
            {ds: out[ds]["pairs"]["score"].to_numpy() for ds in which},
            {ds: out[ds]["pairs"]["censored"].to_numpy() for ds in which},
        ),
        P.structure({ds: out[ds]["variance"] for ds in which},
                    {ds: out[ds]["spectrum"] for ds in which}),
    ]
    for p in paths:
        print(f"  wrote {p.relative_to(RESULTS.parent)}")

    rows = []
    for ds in which:
        r = out[ds]["report"]
        rows.append({
            "dataset": ds.upper(),
            "score": r["score_name"],
            "drugs": r["shape"]["ligands"],
            "targets": r["shape"]["targets"],
            "observed pairs": r["shape"]["observed_pairs"],
            "never measured": f"{r['sparsity']['missing_fraction']:.1%}",
            "score range": f"{r['score_range'][0]:.2f} to {r['score_range'][1]:.2f}",
            "censored": f"{r['censored_fraction']:.1%}",
            "additive variance": f"{r['variance_explained']['additive (mu + b + c)']:.1%}",
            "interaction headroom": f"{r['interaction_share']:.1%}",
            "top-5 SVD energy": f"{r['svd_cumulative_energy_top5']:.1%}",
            "sequence vs profile rho": f"{r['similarity_vs_profile']['spearman_rho']:.3f}",
        })
    comp = pd.DataFrame(rows).set_index("dataset").T
    comp.to_csv(TABLES / "task1_comparison.csv")
    print("\nDAVIS vs KIBA\n" + comp.to_string())


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")] or None)
