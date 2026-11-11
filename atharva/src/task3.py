"""Task 3: cold-target generalisation on DAVIS and KIBA.

Whole target sequence-groups are held out, so test targets have no training
rows at all and their IDs are uninformative. Hyperparameters are tuned on a
separate set of *cold* validation targets; the test targets are scored once per
seed and results are averaged over 5 seeds.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import dtadata, splits, models, metrics, features, cold_models as cm

SEEDS = (0, 1, 2, 3, 4)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
MODEL_ORDER = ["C0 ligand-mean fallback", "C1 guilt-by-association kNN",
               "C2 sequence -> MF factors", "C3 content regression",
               "C4 similarity-gated blend"]


def _build_ctx(ds, sp, rank, sim, seed):
    tr = sp["train"]
    feats = features.target_matrix(ds.sequences, sp["train_targets"], seed=seed)
    mf = models.BiasedMF(k=rank, lam=5.0, lam_bias=1.0, iters=20, seed=seed)
    mf.fit(ds.rows[tr], ds.cols[tr], ds.y[tr], ds.n_ligands, ds.n_targets)
    return cm.ColdContext(ds, sp, mf, feats, sim)


def run_seed(ds, sim, seed, log):
    sp = splits.cold_target_split(ds, seed=seed)
    va, te = sp["valid"], sp["test"]
    rv, cv, yv = ds.rows[va], ds.cols[va], ds.y[va]
    rt, ct, yt = ds.rows[te], ds.cols[te], ds.y[te]

    # --- rank of the warm factorisation, chosen by how predictable it is from sequence
    best = None
    for rank in (4, 8, 16):
        ctx = _build_ctx(ds, sp, rank, sim, seed)
        for alpha in (1.0, 10.0, 100.0, 1000.0):
            rg = cm.fit_factor_ridge(ctx, alpha)
            s = metrics.mse(yv, cm.predict_factor_ridge(ctx, rg, rv, cv))
            if best is None or s < best[0]:
                best = (s, rank, alpha)
    _, rank, alpha = best
    ctx = _build_ctx(ds, sp, rank, sim, seed)
    ridge = cm.fit_factor_ridge(ctx, alpha)

    # --- C1 neighbourhood
    best_knn = None
    for k in (3, 5, 10, 20, 40):
        for power in (1.0, 3.0, 6.0):
            for ms in (0.0, 0.15, 0.25):
                s = metrics.mse(yv, cm.predict_gba_knn(ctx, rv, cv, k, power, ms))
                if best_knn is None or s < best_knn[0]:
                    best_knn = (s, {"k": k, "power": power, "min_sim": ms})
    kp = best_knn[1]

    # --- C3 boosting depth
    best_gb = None
    for mi in (100, 250):
        gb = cm.fit_content_regressor(ctx, mi, seed)
        s = metrics.mse(yv, cm.predict_content(ctx, gb, rv, cv))
        if best_gb is None or s < best_gb[0]:
            best_gb = (s, mi, gb)
    gb = best_gb[2]

    # --- C4 gate, blending C1 with whichever content model validates better
    pv_knn, fb_v = cm.predict_gba_knn(ctx, rv, cv, **kp, return_fallback=True)
    pv_ridge = cm.predict_factor_ridge(ctx, ridge, rv, cv)
    pv_gb = cm.predict_content(ctx, gb, rv, cv)
    partner_name, pv_partner = ("C2", pv_ridge) if best[0] <= best_gb[0] else ("C3", pv_gb)

    smax_v = ctx.sim[cv][:, ctx.train_targets].max(axis=1)
    best_gate = None
    for s0 in np.quantile(smax_v, [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]):
        for tau in (0.01, 0.03, 0.08):
            p = cm.predict_gated(ctx, pv_knn, pv_partner, cv, s0, tau, fb_v)
            s = metrics.mse(yv, p)
            if best_gate is None or s < best_gate[0]:
                best_gate = (s, float(s0), float(tau))
    _, s0, tau = best_gate

    # --- score the cold test targets once
    p_knn, fb_t = cm.predict_gba_knn(ctx, rt, ct, **kp, return_fallback=True)
    p_ridge = cm.predict_factor_ridge(ctx, ridge, rt, ct)
    p_gb = cm.predict_content(ctx, gb, rt, ct)
    p_partner = p_ridge if partner_name == "C2" else p_gb
    preds = {
        "C0 ligand-mean fallback": cm.predict_ligand_mean(ctx, rt, ct),
        "C1 guilt-by-association kNN": p_knn,
        "C2 sequence -> MF factors": p_ridge,
        "C3 content regression": p_gb,
        "C4 similarity-gated blend": cm.predict_gated(ctx, p_knn, p_partner, ct, s0, tau, fb_t),
    }
    fallbacks = {"C0 ligand-mean fallback": 1.0, "C1 guilt-by-association kNN": float(fb_t.mean()),
                 "C2 sequence -> MF factors": 0.0, "C3 content regression": 0.0,
                 "C4 similarity-gated blend": 0.0}

    rows = []
    for mname, p in preds.items():
        rec = {"dataset": ds.name, "seed": seed, "model": mname}
        rec.update(metrics.evaluate(rt, yt, p, ds.strong_threshold))
        rec["fallback_frac"] = fallbacks[mname]
        rows.append(rec)
    rnd = metrics.random_ranking_reference(rt, yt, seed=seed)
    rows.append({"dataset": ds.name, "seed": seed, "model": "random ranking",
                 "mse": np.nan, "mse_strong": np.nan, "ci_global": 0.5,
                 "fallback_frac": np.nan, **rnd})

    hp = {"mf_rank": rank, "ridge_alpha": alpha, "knn": kp, "gbr_max_iter": best_gb[1],
          "gate_partner": partner_name, "gate_s0": s0, "gate_tau": tau,
          "n_test_targets": int(len(sp["test_targets"])), "n_test_pairs": int(len(te))}
    log(f"  seed {seed}: rank={rank} alpha={alpha:g} knn={kp} partner={partner_name} "
        f"gate(s0={s0:.3f},tau={tau:g}) fallback={fb_t.mean():.1%}")

    # per-pair detail retained for the similarity-bucket analysis
    detail = pd.DataFrame({
        "dataset": ds.name, "seed": seed, "ligand": rt, "target": ct, "y": yt,
        "max_sim": ctx.sim[ct][:, ctx.train_targets].max(axis=1),
        **{f"p_{m.split()[0]}": p for m, p in preds.items()},
    })
    return rows, hp, detail


def run(name, log=print):
    ds = dtadata.load(name)
    sim = features.kmer_similarity(ds.sequences)
    log(f"\n=== {ds.name} cold-target ({len(SEEDS)} seeds) ===")
    rows, hps, details = [], {}, []
    for seed in SEEDS:
        t0 = time.time()
        r, hp, d = run_seed(ds, sim, seed, log)
        rows += r; hps[seed] = hp; details.append(d)
        log(f"          ({time.time()-t0:.1f}s)")
    return pd.DataFrame(rows), hps, pd.concat(details, ignore_index=True)


if __name__ == "__main__":
    allr, allh, alld = [], {}, []
    for name in ("davis", "kiba"):
        r, h, d = run(name)
        allr.append(r); allh[name] = h; alld.append(d)
    per_seed = pd.concat(allr, ignore_index=True)
    per_seed.to_csv(f"{OUT}/task3_per_seed.csv", index=False)
    pd.concat(alld, ignore_index=True).to_csv(f"{OUT}/task3_pair_detail.csv", index=False)
    json.dump(allh, open(f"{OUT}/task3_hparams.json", "w"), indent=2, default=float)

    cols = ["mse", "mse_strong", "ci_global", "ci_per_ligand", "hit@1", "hit@5", "mrr", "fallback_frac"]
    agg = (per_seed.groupby(["dataset", "model"])[cols]
           .agg(["mean", "std"]).reset_index())
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    agg.to_csv(f"{OUT}/task3_summary.csv", index=False)
    show = ["dataset", "model", "mse_mean", "mse_std", "ci_global_mean", "ci_global_std",
            "ci_per_ligand_mean", "hit@1_mean", "mrr_mean", "fallback_frac_mean"]
    print("\n" + agg[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
