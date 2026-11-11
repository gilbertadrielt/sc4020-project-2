"""Does guilt-by-association hold up when a cold target has no close homologue?

Each held-out target is scored on its own: across all ligands measured against
it, how well does each model order and predict them. Targets are then grouped
by their similarity to the nearest training target.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import metrics

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
MODELS = {"p_C0": "C0 ligand-mean fallback", "p_C1": "C1 guilt-by-association kNN",
          "p_C2": "C2 sequence -> MF factors", "p_C3": "C3 content regression",
          "p_C4": "C4 similarity-gated blend"}
LABELS = ["orphan (lowest third)", "moderate (middle third)", "near homologue (top third)"]


def per_target(detail):
    """One row per (seed, cold target): MSE and drug-ranking CI for each model."""
    out = []
    for (ds, seed, tgt), g in detail.groupby(["dataset", "seed", "target"]):
        if len(g) < 5:
            continue
        rec = {"dataset": ds, "seed": seed, "target": tgt,
               "n_ligands": len(g), "max_sim": float(g["max_sim"].iloc[0])}
        for col, name in MODELS.items():
            rec[f"mse::{name}"] = metrics.mse(g["y"].values, g[col].values)
            rec[f"ci::{name}"] = metrics.concordance_index(g["y"].values, g[col].values)
        out.append(rec)
    return pd.DataFrame(out)


def bucketise(pt):
    rows = []
    for ds, g in pt.groupby("dataset"):
        edges = np.quantile(g["max_sim"], [1 / 3, 2 / 3])
        idx = np.digitize(g["max_sim"], edges)
        for b in (0, 1, 2):
            sel = g[idx == b]
            if sel.empty:
                continue
            rec = {"dataset": ds, "bucket": LABELS[b],
                   "sim_range": f"{sel['max_sim'].min():.2f}-{sel['max_sim'].max():.2f}",
                   "n_targets": len(sel), "mean_ligands": round(sel["n_ligands"].mean(), 1)}
            for name in MODELS.values():
                rec[f"MSE {name.split()[0]}"] = sel[f"mse::{name}"].mean()
                rec[f"CI {name.split()[0]}"] = sel[f"ci::{name}"].mean()
            rec["CI gain C1 over C0"] = rec["CI C1"] - rec["CI C0"]
            rec["MSE gain C4 over C0"] = rec["MSE C0"] - rec["MSE C4"]
            rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    detail = pd.read_csv(f"{OUT}/task3_pair_detail.csv")
    pt = per_target(detail)
    pt.to_csv(f"{OUT}/task3_per_target.csv", index=False)
    bk = bucketise(pt)
    bk.to_csv(f"{OUT}/task3_similarity_buckets.csv", index=False)

    show = ["dataset", "bucket", "sim_range", "n_targets",
            "CI C0", "CI C1", "CI C2", "CI C4", "CI gain C1 over C0",
            "MSE C0", "MSE C1", "MSE C4"]
    print("Per-target drug ranking, grouped by similarity to the nearest training target")
    print("(CI here = ordering agreement across the ligands measured against that target)\n")
    print(bk[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nCorrelation between a cold target's nearest-homologue similarity and C1 quality:")
    for ds, g in pt.groupby("dataset"):
        from scipy.stats import spearmanr
        r_ci = spearmanr(g["max_sim"], g["ci::C1 guilt-by-association kNN"]).statistic
        r_mse = spearmanr(g["max_sim"], g["mse::C1 guilt-by-association kNN"]).statistic
        print(f"  {ds}: spearman(max_sim, CI) = {r_ci:+.3f}   spearman(max_sim, MSE) = {r_mse:+.3f}"
              f"   [{len(g)} cold targets]")
