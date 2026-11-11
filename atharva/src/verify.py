"""Independent checks on the results before they are reported."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import dtadata, splits, metrics, checkpoint

ok = True
def check(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")

print("1. Split integrity")
for n in ("davis", "kiba"):
    ds = dtadata.load(n); sp = splits.warm_split(ds, 42)
    tr, va, te = set(sp["train"].tolist()), set(sp["valid"].tolist()), set(sp["test"].tolist())
    check(f"{ds.name} warm: folds disjoint", not (tr & va) and not (tr & te) and not (va & te))
    check(f"{ds.name} warm: folds cover every pair", len(tr | va | te) == ds.n_pairs)
    key = lambda idx: set(zip(ds.rows[idx].tolist(), ds.cols[idx].tolist()))
    check(f"{ds.name} warm: no ligand-target pair in two folds",
          not (key(sp['train']) & key(sp['test'])))
    check(f"{ds.name} warm: all test ligands+targets seen in train",
          set(ds.rows[sp['test']]) <= set(ds.rows[sp['train']]) and
          set(ds.cols[sp['test']]) <= set(ds.cols[sp['train']]))
    cs = splits.cold_target_split(ds, 0)
    check(f"{ds.name} cold: test targets absent from training pairs",
          not (set(ds.cols[cs['train']]) & set(cs['test_targets'].tolist())))
    seen = set(np.asarray(ds.sequences)[cs['train_targets']])
    leak = sum(1 for s in np.asarray(ds.sequences)[cs['test_targets']] if s in seen)
    check(f"{ds.name} cold: no identical sequence spans the split", leak == 0, f"{leak} leaks")

print("\n2. Checkpoints reload to identical predictions")
for n in ("davis", "kiba"):
    ds = dtadata.load(n); sp = splits.warm_split(ds, 42); te = sp["test"]
    cf = checkpoint.LoadedCF(f"saved_models/{n}_cf.npz")
    p1 = cf.predict(ds.rows[te], ds.cols[te])
    p2 = checkpoint.LoadedCF(f"saved_models/{n}_cf.npz").predict(ds.rows[te], ds.cols[te])
    check(f"{ds.name} two independent loads agree bit-for-bit", np.array_equal(p1, p2))
    check(f"{ds.name} ligand/target ids stored", len(cf.ligand_ids) == ds.n_ligands)

print("\n3. Reported test MSE recomputed from scratch")
t2 = pd.read_csv("results/task2_test_results.csv")
for n in ("davis", "kiba"):
    ds = dtadata.load(n); sp = splits.warm_split(ds, 42); te = sp["test"]
    cf = checkpoint.LoadedCF(f"saved_models/{n}_cf.npz")
    want = float(t2[(t2.dataset == ds.name) & (t2.model == "M3 kNN + MF blend")]["mse"].iloc[0])
    got = metrics.mse(ds.y[te], cf.predict(ds.rows[te], ds.cols[te]))
    check(f"{ds.name} blend MSE matches table", abs(want - got) < 1e-9, f"{want:.6f} vs {got:.6f}")
    want = float(t2[(t2.dataset == ds.name) & (t2.model == "M2 biased MF")]["mse"].iloc[0])
    got = metrics.mse(ds.y[te], cf.predict_mf(ds.rows[te], ds.cols[te]))
    check(f"{ds.name} MF MSE matches table", abs(want - got) < 1e-9, f"{want:.6f} vs {got:.6f}")

print("\n4. Sanity vs published DeepDTA reference (different split, so only a ballpark)")
for n, ref_mse, ref_ci in (("DAVIS", 0.261, 0.878), ("KIBA", 0.194, 0.863)):
    r = t2[(t2.dataset == n) & (t2.model == "M2 biased MF")].iloc[0]
    print(f"  {n}: ours MSE={r['mse']:.3f} CI={r['ci_global']:.3f} | DeepDTA MSE={ref_mse} CI={ref_ci}")

print("\n5. Ranking baselines behave")
for n in ("DAVIS", "KIBA"):
    g = t2[t2.dataset == n].set_index("model")
    check(f"{n} global mean ranks no better than random (within noise)",
          g.loc["B0 global mean", "hit@1"] - g.loc["random ranking", "hit@1"] < 0.03,
          f"{g.loc['B0 global mean','hit@1']:.4f} vs {g.loc['random ranking','hit@1']:.4f}")
    # KIBA ranks only ~10 candidates per ligand, so a multiplicative bar is the
    # wrong test there; require a large absolute lift plus a strong CI instead.
    lift = g.loc["M3 kNN + MF blend", "hit@1"] - g.loc["random ranking", "hit@1"]
    check(f"{n} best model beats random on Hit@1", lift > 0.15 and g.loc["M3 kNN + MF blend", "ci_global"] > 0.80,
          f"lift={lift:+.3f}, CI={g.loc['M3 kNN + MF blend','ci_global']:.3f}")
    check(f"{n} train MSE <= test MSE for MF", g.loc["M2 biased MF", "train_mse"] <= g.loc["M2 biased MF", "mse"])

print("\n6. Task 3: cold models beat the fallback on every seed")
t3 = pd.read_csv("results/task3_per_seed.csv")
for n in ("DAVIS", "KIBA"):
    g = t3[t3.dataset == n]
    c0 = g[g.model == "C0 ligand-mean fallback"].set_index("seed")["mse"]
    for m in ("C1 guilt-by-association kNN", "C4 similarity-gated blend"):
        c = g[g.model == m].set_index("seed")["mse"]
        check(f"{n} {m.split()[0]} < C0 on all 5 seeds", bool((c < c0).all()),
              f"wins {int((c < c0).sum())}/5")

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
