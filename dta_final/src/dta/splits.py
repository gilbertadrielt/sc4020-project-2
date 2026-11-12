"""The four evaluation regimes, frozen to disk and checked for leakage.

A split is the contract the whole project hangs off:

    make_split(pairs, n_lig, n_tgt, setting, seed) -> Series of
        "train" | "val" | "test" | "unused"

Nothing else in the codebase is allowed to partition data.  Every split is
written to disk with a fingerprint hash so that two runs, on two machines, can
prove they used the same partition before their numbers are compared.

The four settings are the standard regimes in the drug-target literature.  The
brief asks for S1 and S3; the other two cost almost nothing once splitting is
a module rather than a few lines inside a training script, and S4 is the one
that matches the real discovery problem.

======  ======================================  =========================
Code    Held out                                Question it answers
======  ======================================  =========================
S1      random pairs                            can we fill gaps for drugs
                                                and proteins we know?
S2      whole drugs                             can we screen a brand new
                                                compound?
S3      whole targets                           can we screen against a
                                                brand new protein?
S4      whole drugs AND whole targets           neither side ever seen
======  ======================================  =========================

S4 keeps only the pairs where both sides fall in the same fold, so pairs that
are half-new are marked ``unused`` rather than quietly leaking into training.

``groups`` generalises S2/S3/S4 from "hold out entities" to "hold out whole
clusters of similar entities".  Passing sequence-similarity clusters gives the
strict-cold regime, where a test protein has no close relative in training
either, which is the honest hardest setting.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .config import SPLIT_FRACTIONS, SPLIT_SEEDS, SPLITS

SETTING_NAMES = ("S1_pair", "S2_new_drug", "S3_new_target", "S4_both_new")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _three_way(items: np.ndarray, rng: np.random.Generator) -> dict[int, str]:
    """Assign each item to train/val/test by the configured fractions."""
    items = np.asarray(items)
    shuffled = rng.permutation(items)
    n = len(shuffled)
    n_train = int(round(SPLIT_FRACTIONS["train"] * n))
    n_val = int(round(SPLIT_FRACTIONS["val"] * n))
    # Guarantee a non-empty val and test whenever there is anything to give.
    n_train = min(n_train, max(n - 2, 1)) if n >= 3 else n
    n_val = min(n_val, max(n - n_train - 1, 0)) if n >= 3 else 0
    out: dict[int, str] = {}
    for i, it in enumerate(shuffled):
        out[int(it)] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return out


def _entity_groups(n: int, groups: np.ndarray | None) -> np.ndarray:
    """Cluster id per entity; the identity mapping when no clustering is given."""
    if groups is None:
        return np.arange(n)
    groups = np.asarray(groups)
    if len(groups) != n:
        raise ValueError(f"groups has length {len(groups)}, expected {n}")
    return groups


# ---------------------------------------------------------------------------
# The four settings
# ---------------------------------------------------------------------------


def _split_s1(pairs: pd.DataFrame, n_lig: int, n_tgt: int, rng) -> tuple[pd.Series, dict]:
    """Random pairs, stratified within each drug.

    Stratifying by drug stops a drug from losing all of its test pairs by
    chance, which would silently shrink the per-drug ranking evaluation.
    """
    assign = pd.Series("train", index=pairs.index, dtype=object)
    for _, idx in pairs.groupby("lig_idx").groups.items():
        idx = np.asarray(idx)
        order = rng.permutation(len(idx))
        n = len(idx)
        n_train = int(round(SPLIT_FRACTIONS["train"] * n))
        n_val = int(round(SPLIT_FRACTIONS["val"] * n))
        for rank, pos in enumerate(order):
            assign.iloc[pairs.index.get_loc(idx[pos])] = (
                "train" if rank < n_train else ("val" if rank < n_train + n_val else "test")
            )

    # Warm-start repair: an evaluation pair whose drug or target has no
    # training row is a cold-start case, which belongs to S2/S3, not here.
    moved = 0
    for _ in range(3):  # a move can orphan another entity, so iterate
        train = assign == "train"
        warm_lig = set(pairs.loc[train, "lig_idx"])
        warm_tgt = set(pairs.loc[train, "tgt_idx"])
        bad = (~train) & (
            ~pairs["lig_idx"].isin(warm_lig) | ~pairs["tgt_idx"].isin(warm_tgt)
        )
        if not bad.any():
            break
        assign[bad] = "train"
        moved += int(bad.sum())

    return assign, {"moved_to_train_for_warm_start": moved}


def _split_entity(
    pairs: pd.DataFrame,
    n_lig: int,
    n_tgt: int,
    rng,
    axis: str,
    groups: np.ndarray | None,
) -> tuple[pd.Series, dict]:
    """S2 and S3: hold out whole drugs (axis='lig') or whole targets ('tgt')."""
    n = n_lig if axis == "lig" else n_tgt
    col = f"{axis}_idx"
    g = _entity_groups(n, groups)
    fold_of_group = _three_way(np.unique(g), rng)
    fold_of_entity = np.array([fold_of_group[int(g[i])] for i in range(n)], dtype=object)
    assign = pd.Series(fold_of_entity[pairs[col].to_numpy()], index=pairs.index, dtype=object)
    return assign, {
        f"n_{axis}_train": int((fold_of_entity == "train").sum()),
        f"n_{axis}_val": int((fold_of_entity == "val").sum()),
        f"n_{axis}_test": int((fold_of_entity == "test").sum()),
        "n_groups": int(len(np.unique(g))),
    }


def _split_s4(
    pairs: pd.DataFrame,
    n_lig: int,
    n_tgt: int,
    rng,
    lig_groups: np.ndarray | None,
    tgt_groups: np.ndarray | None,
) -> tuple[pd.Series, dict]:
    """Both sides new: a pair counts only when drug and target land in the same fold.

    Half-new pairs (new drug with a seen target, or the reverse) are marked
    ``unused``.  Putting them in training would make the test targets warm;
    putting them in test would make it an S2/S3 question instead.
    """
    gl = _entity_groups(n_lig, lig_groups)
    gt = _entity_groups(n_tgt, tgt_groups)
    lig_fold_g = _three_way(np.unique(gl), rng)
    tgt_fold_g = _three_way(np.unique(gt), rng)
    lig_fold = np.array([lig_fold_g[int(gl[i])] for i in range(n_lig)], dtype=object)
    tgt_fold = np.array([tgt_fold_g[int(gt[j])] for j in range(n_tgt)], dtype=object)

    lf = lig_fold[pairs["lig_idx"].to_numpy()]
    tf = tgt_fold[pairs["tgt_idx"].to_numpy()]
    assign = pd.Series(np.where(lf == tf, lf, "unused"), index=pairs.index, dtype=object)
    return assign, {
        "n_lig_train": int((lig_fold == "train").sum()),
        "n_lig_test": int((lig_fold == "test").sum()),
        "n_tgt_train": int((tgt_fold == "train").sum()),
        "n_tgt_test": int((tgt_fold == "test").sum()),
        "n_unused_pairs": int((assign == "unused").sum()),
    }


def make_split(
    pairs: pd.DataFrame,
    n_lig: int,
    n_tgt: int,
    setting: str = "S1_pair",
    seed: int = 2026,
    lig_groups: np.ndarray | None = None,
    tgt_groups: np.ndarray | None = None,
) -> tuple[pd.Series, dict]:
    """Partition ``pairs`` under one of the four regimes.

    Returns the assignment and a dictionary of statistics that goes into the
    split report, including anything the split had to repair.
    """
    rng = np.random.default_rng(seed)
    if setting == "S1_pair":
        assign, stats = _split_s1(pairs, n_lig, n_tgt, rng)
    elif setting == "S2_new_drug":
        assign, stats = _split_entity(pairs, n_lig, n_tgt, rng, "lig", lig_groups)
    elif setting == "S3_new_target":
        assign, stats = _split_entity(pairs, n_lig, n_tgt, rng, "tgt", tgt_groups)
    elif setting == "S4_both_new":
        assign, stats = _split_s4(pairs, n_lig, n_tgt, rng, lig_groups, tgt_groups)
    else:
        raise ValueError(f"unknown setting {setting!r}; expected one of {SETTING_NAMES}")

    stats = dict(stats)
    stats.update(
        {
            "setting": setting,
            "seed": int(seed),
            "n_train": int((assign == "train").sum()),
            "n_val": int((assign == "val").sum()),
            "n_test": int((assign == "test").sum()),
            "n_unused": int((assign == "unused").sum()),
        }
    )
    check_split(pairs, assign, setting)
    stats["fingerprint"] = fingerprint(pairs, assign)
    return assign, stats


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def check_split(pairs: pd.DataFrame, assign: pd.Series, setting: str) -> None:
    """Assert the properties the setting claims. Raises rather than warns.

    These are the assertions that would have caught the leakage that makes
    independent runs of this project incomparable, so they are errors.
    """
    if len(assign) != len(pairs):
        raise AssertionError("assignment and pair table have different lengths")
    if assign.isna().any():
        raise AssertionError("unassigned pairs")

    train = pairs[assign == "train"]
    evalset = pairs[assign.isin(["val", "test"])]
    if len(evalset) == 0:
        raise AssertionError(f"{setting}: empty evaluation set")

    # No pair may appear in two partitions. Pairs are unique by construction
    # after deduplication, so overlap can only come from a broken assignment.
    key = pairs["lig_idx"].astype(str) + "|" + pairs["tgt_idx"].astype(str)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = set(key[assign == a]) & set(key[assign == b])
        if shared:
            raise AssertionError(f"{setting}: {len(shared)} pairs in both {a} and {b}")

    train_lig = set(train["lig_idx"])
    train_tgt = set(train["tgt_idx"])

    if setting == "S1_pair":
        if not set(evalset["lig_idx"]) <= train_lig:
            raise AssertionError("S1: an evaluation drug is absent from training")
        if not set(evalset["tgt_idx"]) <= train_tgt:
            raise AssertionError("S1: an evaluation target is absent from training")

    if setting in ("S2_new_drug", "S4_both_new"):
        leaked = set(evalset["lig_idx"]) & train_lig
        if leaked:
            raise AssertionError(f"{setting}: {len(leaked)} evaluation drugs appear in training")

    if setting in ("S3_new_target", "S4_both_new"):
        leaked = set(evalset["tgt_idx"]) & train_tgt
        if leaked:
            raise AssertionError(f"{setting}: {len(leaked)} evaluation targets appear in training")


def fingerprint(pairs: pd.DataFrame, assign: pd.Series) -> str:
    """Stable hash of the partition, so two runs can prove they match."""
    key = (
        pairs["lig_idx"].astype(str) + ":" + pairs["tgt_idx"].astype(str) + "=" + assign.astype(str)
    )
    return hashlib.sha256("\n".join(sorted(key)).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Freeze and load
# ---------------------------------------------------------------------------


def _path(name: str, setting: str, seed: int):
    return SPLITS / f"{name}__{setting}__seed{seed}.csv"


def freeze(
    name: str,
    pairs: pd.DataFrame,
    n_lig: int,
    n_tgt: int,
    settings=SETTING_NAMES,
    seeds=SPLIT_SEEDS,
    lig_groups=None,
    tgt_groups=None,
) -> pd.DataFrame:
    """Generate and write every (setting, seed) split for one dataset."""
    report = []
    for setting in settings:
        for seed in seeds:
            assign, stats = make_split(
                pairs, n_lig, n_tgt, setting=setting, seed=seed,
                lig_groups=lig_groups, tgt_groups=tgt_groups,
            )
            assign.to_frame("fold").to_csv(_path(name, setting, seed), index=False)
            stats["dataset"] = name
            report.append(stats)
    rep = pd.DataFrame(report)
    rep.to_csv(SPLITS / f"report_{name}.csv", index=False)
    with open(SPLITS / f"report_{name}.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    return rep


def load_split(name: str, setting: str = "S1_pair", seed: int = 2026) -> pd.Series:
    """Read a frozen split. Fails loudly rather than silently regenerating."""
    p = _path(name, setting, seed)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run `python -m dta.splits` first")
    return pd.read_csv(p)["fold"]


if __name__ == "__main__":
    import sys

    from . import data as D
    from .config import DATASETS

    which = [a for a in sys.argv[1:] if not a.startswith("--")] or list(DATASETS)
    for ds in which:
        proc = D.load_processed(ds)
        rep = freeze(ds, proc["pairs"], proc["n_lig"], proc["n_tgt"])
        print(f"\n=== {ds.upper()}")
        cols = [c for c in ["setting", "seed", "n_train", "n_val", "n_test",
                            "n_unused", "moved_to_train_for_warm_start", "fingerprint"]
                if c in rep.columns]
        print(rep[rep.seed == rep.seed.min()][cols].to_string(index=False))
