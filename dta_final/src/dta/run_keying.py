"""Controlled A/B on the one preprocessing decision that moves every number.

Two defensible conventions exist for what counts as a distinct target:

``sequence``  merge identifiers that share an amino-acid sequence, which is what
              this project does, because a kernel or content model reading only
              the sequence cannot tell them apart anyway.
``id``        keep every source identifier separate, because in DAVIS those
              identifiers are kinase mutants and the mutation is real: ABL1 and
              ABL1(T315I) differ by up to 4.75 pKd.

The two conventions produce different test sets, so numbers computed under one
cannot be placed in a table beside numbers computed under the other.  This
script holds the model, the hyperparameters and the seed fixed and varies only
the convention, which is the only way to attribute the difference to it.

Writes results/tables/keying_ab.csv and results/tables/keying_strata.csv.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import data as D
from . import metrics as MT
from . import models as M
from . import splits as S
from .config import SPLIT_SEEDS, TABLES

FIXED = {
    "B3_bias_only": lambda: M.BiasOnly(lam=1.0),
    "M1_item_knn": lambda: M.ItemKNN(k=20, beta=10.0),
    "M2_biased_mf": lambda: M.BiasedMF(k=32, lam=0.05),
}


def _id_keyed(name: str) -> dict:
    """Rebuild the pair table keyed on source identifier instead of chemistry.

    This re-runs the same load, canonicalise and clean steps the project uses
    and stops short of ``deduplicate``, so the only difference from the shipped
    bundle is the merge.  Reconstructing it from the merged table instead would
    hand every identifier the *averaged* score and so would not be ID keying
    at all.
    """
    log = D.CleaningLog(dataset=name)
    raw = D.load_raw(name)
    df = D.to_long(raw, log)
    df["smiles"], _ = D.canonicalise_smiles(list(df["smiles"]))
    df = D.clean(df, name, log)

    pairs = df.copy()
    lig = (pairs[["ligand_id"]].drop_duplicates().reset_index(drop=True)
           .assign(lig_idx=lambda d: np.arange(len(d))))
    tgt = (pairs[["target_id"]].drop_duplicates().reset_index(drop=True)
           .assign(tgt_idx=lambda d: np.arange(len(d))))
    pairs = pairs.merge(lig, on="ligand_id").merge(tgt, on="target_id")
    return {"pairs": pairs, "n_lig": len(lig), "n_tgt": len(tgt),
            "ligands": lig, "targets": tgt}


def _evaluate(bundle: dict, name: str, keying: str, seeds) -> list[dict]:
    pairs, nl, nt = bundle["pairs"], bundle["n_lig"], bundle["n_tgt"]
    rows = []
    for seed in seeds:
        if keying == "sequence":
            assign = S.load_split(name, "S1_pair", seed)
        else:
            # The ID-keyed table has no frozen split, so make the equivalent one.
            rng = np.random.default_rng(seed)
            u = rng.random(len(pairs))
            assign = np.where(u < 0.70, "train", np.where(u < 0.85, "val", "test"))
        tr, va, te = pairs[assign == "train"], pairs[assign == "val"], pairs[assign == "test"]
        variance = float(np.var(te["score"].to_numpy(dtype=float)))
        for label, build in FIXED.items():
            model = build().fit(tr, nl, nt, val=va)
            pred = model.predict(te)
            mse = MT.mse(te["score"].to_numpy(dtype=float), pred)
            rows.append({
                "dataset": name, "keying": keying, "seed": seed, "model": label,
                "n_test": len(te), "test_variance": variance, "mse": mse,
                "r2": 1.0 - mse / variance,
            })
    return rows


def strata(name: str, seed: int = SPLIT_SEEDS[0]) -> pd.DataFrame:
    """Under ID keying, is the advantage concentrated where a twin is in training?

    A test target whose sequence-identical sibling sits in the training set is
    the case the sequence convention is designed to remove.  If the gain over
    the bias baseline is much larger there than elsewhere, the convention is
    doing real work rather than discarding signal for nothing.
    """
    bundle = _id_keyed(name)
    pairs, nl, nt = bundle["pairs"], bundle["n_lig"], bundle["n_tgt"]
    proc = D.load_processed(name)

    # Sequence group of each source identifier, from the merged entity table.
    group_of = {}
    for g, ids in enumerate(proc["targets"]["target_ids"]):
        for one in str(ids).split("|"):
            group_of[one] = g
    pairs = pairs.assign(group=pairs["target_id"].map(group_of))

    rng = np.random.default_rng(seed)
    u = rng.random(len(pairs))
    assign = np.where(u < 0.70, "train", np.where(u < 0.85, "val", "test"))
    tr, va, te = pairs[assign == "train"], pairs[assign == "val"], pairs[assign == "test"]

    train_ids = set(tr["target_id"])
    by_group: dict[int, set] = {}
    for one, g in group_of.items():
        by_group.setdefault(g, set()).add(one)
    has_twin = te["target_id"].map(
        lambda t: any(o != t and o in train_ids for o in by_group[group_of[t]]))
    te = te.assign(twin=has_twin)

    fitted = {k: v().fit(tr, nl, nt, val=va) for k, v in FIXED.items()}
    rows = []
    for twin, part in te.groupby("twin"):
        y = part["score"].to_numpy(dtype=float)
        base = MT.mse(y, fitted["B3_bias_only"].predict(part))
        for label, model in fitted.items():
            mse = MT.mse(y, model.predict(part))
            rows.append({
                "dataset": name,
                "stratum": "sequence twin in training" if twin else "no twin",
                "n_test_pairs": len(part), "model": label, "mse": mse,
                "gain_over_bias": (base - mse) / base,
            })
    return pd.DataFrame(rows)


def main(which=None, seeds=SPLIT_SEEDS) -> None:
    which = which or ["davis", "kiba"]
    ab, st = [], []
    for name in which:
        print(f"\n=== {name.upper()}")
        ab += _evaluate(D.load_processed(name), name, "sequence", seeds)
        ab += _evaluate(_id_keyed(name), name, "id", seeds)
        frame = pd.DataFrame([r for r in ab if r["dataset"] == name])
        pivot = frame.pivot_table(index="model", columns="keying", values="r2")
        pivot["id_minus_sequence"] = pivot["id"] - pivot["sequence"]
        print("  variance explained under each convention:")
        print(pivot.round(4).to_string())

        s = strata(name)
        st.append(s)
        print("  by whether a sequence twin is in training:")
        print(s.round(4).to_string(index=False))

    pd.DataFrame(ab).to_csv(TABLES / "keying_ab.csv", index=False)
    pd.concat(st, ignore_index=True).to_csv(TABLES / "keying_strata.csv", index=False)
    print(f"\nwrote keying tables to {TABLES}")


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")] or None)
