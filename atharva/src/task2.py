"""Task 2: warm-split collaborative filtering on DAVIS and KIBA.

Hyperparameters are chosen on validation by MSE; the test fold is scored once.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import dtadata, splits, models, metrics, checkpoint

SEED = 42
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models")


def _fold(ds, idx):
    return ds.rows[idx], ds.cols[idx], ds.y[idx]


def tune(ds, sp, log):
    tr, va = sp["train"], sp["valid"]
    rtr, ctr, ytr = _fold(ds, tr)
    rva, cva, yva = _fold(ds, va)
    nl, nt = ds.n_ligands, ds.n_targets
    trials = []

    best_bias = None
    for lam in (1.0, 2.0, 5.0, 10.0, 20.0):
        m = models.BiasOnly(lam=lam).fit(rtr, ctr, ytr, nl, nt)
        s = metrics.mse(yva, m.predict(rva, cva))
        trials.append({"model": "B3 bias only", "params": {"lam": lam}, "val_mse": s})
        if best_bias is None or s < best_bias[0]:
            best_bias = (s, lam, m)
    log(f"  bias-only  lam={best_bias[1]}  val MSE={best_bias[0]:.4f}")

    best_knn = None
    for k in (5, 10, 20, 40, 80):
        for beta in (5.0, 25.0, 100.0):
            m = models.TargetKNN(k=k, beta=beta, lam=best_bias[1]).fit(rtr, ctr, ytr, nl, nt)
            s = metrics.mse(yva, m.predict(rva, cva))
            trials.append({"model": "M1 target kNN", "params": {"k": k, "beta": beta}, "val_mse": s})
            if best_knn is None or s < best_knn[0]:
                best_knn = (s, {"k": k, "beta": beta, "lam": best_bias[1]}, m)
    log(f"  target kNN {best_knn[1]}  val MSE={best_knn[0]:.4f}")

    best_mf = None
    for k in (4, 8, 16, 32, 64):
        for lam in (1.0, 5.0, 10.0, 20.0, 50.0):
            m = models.BiasedMF(k=k, lam=lam, lam_bias=best_bias[1], iters=20, seed=SEED)
            m.fit(rtr, ctr, ytr, nl, nt)
            s = metrics.mse(yva, m.predict(rva, cva))
            trials.append({"model": "M2 biased MF", "params": {"k": k, "lam": lam}, "val_mse": s})
            if best_mf is None or s < best_mf[0]:
                best_mf = (s, {"k": k, "lam": lam, "lam_bias": best_bias[1]}, m)
    log(f"  biased MF  {best_mf[1]}  val MSE={best_mf[0]:.4f}")

    pk = best_knn[2].predict(rva, cva)
    pm = best_mf[2].predict(rva, cva)
    ws = np.arange(0, 1.001, 0.05)
    scores = [metrics.mse(yva, w * pk + (1 - w) * pm) for w in ws]
    w_best = float(ws[int(np.argmin(scores))])
    for w, s in zip(ws, scores):
        trials.append({"model": "M3 blend", "params": {"w": round(float(w), 2)}, "val_mse": s})
    log(f"  blend      w={w_best:.2f}  val MSE={min(scores):.4f}")

    return {
        "bias": best_bias, "knn": best_knn, "mf": best_mf,
        "blend_w": w_best, "val_mse_blend": float(min(scores)), "trials": trials,
    }


def run(name, log=print):
    ds = dtadata.load(name)
    sp = splits.warm_split(ds, seed=SEED)
    log(f"\n=== {ds.name} warm split (seed {SEED}) ===")
    log(f"  pairs train/valid/test = {len(sp['train'])}/{len(sp['valid'])}/{len(sp['test'])}"
        f"  promoted-to-train = {sp['promoted_to_train']}")

    t0 = time.time()
    sel = tune(ds, sp, log)
    log(f"  tuning finished in {time.time()-t0:.1f}s")

    tr, te = sp["train"], sp["test"]
    rtr, ctr, ytr = _fold(ds, tr)
    rte, cte, yte = _fold(ds, te)
    nl, nt = ds.n_ligands, ds.n_targets

    fitted = {
        "B0 global mean": models.GlobalMean().fit(rtr, ctr, ytr, nl, nt),
        "B1 ligand mean": models.LigandMean().fit(rtr, ctr, ytr, nl, nt),
        "B2 target mean": models.TargetMean().fit(rtr, ctr, ytr, nl, nt),
        "B3 bias only": sel["bias"][2],
        "M1 target kNN": sel["knn"][2],
        "M2 biased MF": sel["mf"][2],
    }
    fitted["M3 kNN + MF blend"] = models.Blend(sel["knn"][2], sel["mf"][2], sel["blend_w"])

    rows = []
    for mname, m in fitted.items():
        p_te = m.predict(rte, cte)
        rec = {"dataset": ds.name, "model": mname}
        rec.update(metrics.evaluate(rte, yte, p_te, ds.strong_threshold))
        rec["train_mse"] = metrics.mse(ytr, m.predict(rtr, ctr))
        rec["valid_mse"] = metrics.mse(ds.y[sp["valid"]],
                                       m.predict(ds.rows[sp["valid"]], ds.cols[sp["valid"]]))
        rows.append(rec)

    rnd = metrics.random_ranking_reference(rte, yte, seed=SEED)
    rows.append({"dataset": ds.name, "model": "random ranking", "mse": np.nan,
                 "mse_strong": np.nan, "ci_global": 0.5, "train_mse": np.nan,
                 "valid_mse": np.nan, **rnd})

    df = pd.DataFrame(rows)

    # checkpoint + reload equality check
    path = f"{CKPT}/{name}_cf.npz"
    checkpoint.save_cf(path, ds, sel["mf"][2], sel["knn"][2], sel["blend_w"], SEED,
                       {"knn": sel["knn"][1], "mf": sel["mf"][1], "blend_w": sel["blend_w"]})
    loaded = checkpoint.LoadedCF(path)
    p_mem = fitted["M3 kNN + MF blend"].predict(rte, cte)
    p_disk = loaded.predict(rte, cte)
    max_dev = float(np.abs(p_mem - p_disk).max())
    log(f"  checkpoint {path}: max |memory - reloaded| = {max_dev:.3e} "
        f"({'exact' if max_dev == 0 else 'MISMATCH'})")

    meta = {
        "dataset": ds.name, "seed": SEED,
        "n_train": len(tr), "n_valid": len(sp["valid"]), "n_test": len(te),
        "promoted_to_train": sp["promoted_to_train"],
        "best_knn": sel["knn"][1], "best_mf": sel["mf"][1], "blend_w": sel["blend_w"],
        "reload_max_deviation": max_dev,
        "strong_threshold": ds.strong_threshold,
        "cleaning": dtadata.cleaning_report(ds),
    }
    pd.DataFrame(sel["trials"]).to_csv(f"{OUT}/task2_{name}_tuning.csv", index=False)
    return df, meta, ds, sp, loaded


if __name__ == "__main__":
    all_df, all_meta = [], {}
    for name in ("davis", "kiba"):
        df, meta, *_ = run(name)
        all_df.append(df); all_meta[name] = meta
    out = pd.concat(all_df, ignore_index=True)
    out.to_csv(f"{OUT}/task2_test_results.csv", index=False)
    json.dump(all_meta, open(f"{OUT}/task2_meta.json", "w"), indent=2, default=float)
    cols = ["dataset", "model", "mse", "mse_strong", "ci_global", "ci_per_ligand",
            "hit@1", "hit@5", "hit@10", "mrr"]
    print("\n" + out[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
