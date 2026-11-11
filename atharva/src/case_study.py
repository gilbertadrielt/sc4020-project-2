"""Task 2 case study: three fixed test ligands per dataset.

Every prediction here comes from the checkpoint on disk, never from a model
left in memory, which is what the brief asks for.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import dtadata, splits, metrics, checkpoint

SEED = 42
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
CASES = {
    "davis": ["10127622", "11338033", "44259"],
    "kiba": ["CHEMBL1967538", "CHEMBL2004544", "CHEMBL565612"],
}


def run(name, top_n=5):
    ds = dtadata.load(name)
    sp = splits.warm_split(ds, seed=SEED)
    cf = checkpoint.LoadedCF(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models", f"{name}_cf.npz"))

    te = sp["test"]
    rte, cte, yte = ds.rows[te], ds.cols[te], ds.y[te]
    pte = cf.predict(rte, cte)

    id_to_idx = {str(v): i for i, v in enumerate(ds.ligand_ids)}
    summary, tables = [], []

    for lig_id in CASES[name]:
        if lig_id not in id_to_idx:
            summary.append({"dataset": ds.name, "ligand": lig_id, "status": "not in dataset"})
            continue
        i = id_to_idx[lig_id]
        m = rte == i
        if m.sum() < 2:
            summary.append({"dataset": ds.name, "ligand": lig_id,
                            "status": f"only {int(m.sum())} held-out targets"})
            continue

        t_true, t_pred, t_cols = yte[m], pte[m], cte[m]
        order = np.argsort(-t_pred, kind="stable")
        ranks = np.empty(len(t_pred), int)
        ranks[order] = np.arange(1, len(t_pred) + 1)
        best_mask = t_true == t_true.max()
        best_rank = int(ranks[best_mask].min())

        for pos in order[:top_n]:
            tables.append({
                "dataset": ds.name, "ligand": lig_id, "rank": int(ranks[pos]),
                "target": ds.target_ids[t_cols[pos]],
                "pred": round(float(t_pred[pos]), 3),
                "true": round(float(t_true[pos]), 3),
                "verdict": "correct" if best_mask[pos] else "wrong",
            })
        # also show where the true best target actually landed, if outside the top N
        if best_rank > top_n:
            pos = int(np.flatnonzero(best_mask & (ranks == best_rank))[0])
            tables.append({
                "dataset": ds.name, "ligand": lig_id, "rank": best_rank,
                "target": ds.target_ids[t_cols[pos]],
                "pred": round(float(t_pred[pos]), 3),
                "true": round(float(t_true[pos]), 3),
                "verdict": "correct (true best)",
            })

        rk = metrics.ranking_metrics(np.zeros(m.sum(), int), t_true, t_pred)
        summary.append({
            "dataset": ds.name, "ligand": lig_id, "status": "ok",
            "held_out_targets": int(m.sum()),
            "true_score_range": f"{t_true.min():.2f}-{t_true.max():.2f}",
            "mse": round(metrics.mse(t_true, t_pred), 4),
            "ci": round(metrics.concordance_index(t_true, t_pred), 4),
            "rank_of_true_best": best_rank,
            "n_tied_best": int(best_mask.sum()),
            "hit@1": rk["hit@1"], "hit@5": rk["hit@5"], "mrr": round(rk["mrr"], 4),
        })
    return pd.DataFrame(summary), pd.DataFrame(tables)


if __name__ == "__main__":
    S, T = [], []
    for name in ("davis", "kiba"):
        s, t = run(name)
        S.append(s); T.append(t)
    S = pd.concat(S, ignore_index=True); T = pd.concat(T, ignore_index=True)
    S.to_csv(f"{OUT}/task2_case_summary.csv", index=False)
    T.to_csv(f"{OUT}/task2_case_recommendations.csv", index=False)
    print("--- case summary (predictions loaded from saved_models/*.npz) ---")
    print(S.to_string(index=False))
    print("\n--- top recommendations ---")
    print(T.to_string(index=False))
