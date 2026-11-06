"""Train/validation/test partitions for Tasks 2 and 3.

Two splits, one module, because the only thing that makes the two tasks
comparable is that they score the same pairs the same way:

``warm``
    The Task 2 split. 70/15/15 over *pairs*, with every validation and test
    drug and target guaranteed to appear in training - otherwise the model has
    no vector for them and the fold is quietly measuring the fallback.

``cold_target``
    The Task 3 split. Whole targets are held out, so a test target has no
    training measurements at all and plain CF cannot score it.

Both are deterministic given a seed, and both are written to disk, so the whole
group works from byte-identical partitions. Regenerating from the same seed
reproduces the file exactly; the file exists so nobody has to take that on
trust mid-session.

The cold split holds out **sequence groups, not target ids**. DAVIS writes
mutations into the target id but not the sequence, so 81 of its 442 targets
have a byte-identical twin. Holding out ``ABL1(T315I)`` while ``ABL1`` stays in
training would put the identical 1167 aa string on both sides of the split, and
the "cold" target would be one the model has already seen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATA_DIR, Dataset

SPLIT_DIR = DATA_DIR / "processed"

#: Fixed so every group member's run partitions the data identically.
DEFAULT_SEED = 20261115

FOLDS = ("train", "val", "test")

#: Columns that must stay strings across a CSV round-trip.
#: `str`, which this pandas resolves to StringDtype - the same dtype it
#: infers for these columns in memory, so a reloaded split compares equal
#: to the one that wrote it. Forcing `object` here breaks that.
_ID_DTYPES = {"drug_id": str, "target_id": str, "seq_group": str}


@dataclass(frozen=True)
class Split:
    """One partition of one dataset, plus the evidence that it is sound."""

    kind: str
    dataset: str
    seed: int
    #: drug_id, target_id, seq_group, score, fold
    frame: pd.DataFrame
    report: dict = field(default_factory=dict)

    def fold(self, name: str) -> pd.DataFrame:
        if name not in FOLDS:
            raise ValueError(f"fold must be one of {FOLDS}, got {name!r}")
        return self.frame[self.frame["fold"] == name].reset_index(drop=True)

    @property
    def train(self) -> pd.DataFrame:
        return self.fold("train")

    @property
    def val(self) -> pd.DataFrame:
        return self.fold("val")

    @property
    def test(self) -> pd.DataFrame:
        return self.fold("test")

    def save(self, split_dir: Path | str = SPLIT_DIR) -> Path:
        """Write the partition as CSV, with a JSON sidecar for provenance."""
        out_dir = Path(split_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.dataset}_{self.kind}.csv"
        self.frame.to_csv(path, index=False)
        path.with_suffix(".json").write_text(
            json.dumps(
                {"kind": self.kind, "dataset": self.dataset, "seed": self.seed,
                 "report": self.report},
                indent=2, default=float,
            )
        )
        return path

    @classmethod
    def load(cls, dataset: str, kind: str, split_dir: Path | str = SPLIT_DIR) -> "Split":
        path = Path(split_dir) / f"{dataset}_{kind}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Build it with scripts/make_splits.py."
            )
        meta = json.loads(path.with_suffix(".json").read_text())
        # Identifiers are strings, and read_csv would otherwise infer a dtype
        # from how they look. DAVIS drug ids are decimal ("5291", "44259"), so
        # they come back as int64 and stop matching the string keys in
        # ``ligands`` and ``targets`` - silently, as an empty join rather than
        # an error. KIBA's CHEMBL ids hide the problem by not looking numeric.
        return cls(kind=meta["kind"], dataset=meta["dataset"], seed=meta["seed"],
                   frame=pd.read_csv(path, dtype=_ID_DTYPES), report=meta["report"])

    def __repr__(self) -> str:
        sizes = self.frame["fold"].value_counts()
        return (
            f"Split({self.dataset}/{self.kind}, seed={self.seed}, "
            f"train={sizes.get('train', 0):,} val={sizes.get('val', 0):,} "
            f"test={sizes.get('test', 0):,})"
        )


def _check(split: Split) -> None:
    """Assert the properties each split kind promises. Cheap, run every time."""
    frame = split.frame
    assert set(frame["fold"]) <= set(FOLDS), "unknown fold label"
    assert len(frame) == len(frame.drop_duplicates(["drug_id", "target_id"])), (
        "a (drug, target) pair appears twice in the split"
    )

    train = frame[frame["fold"] == "train"]
    held = frame[frame["fold"] != "train"]
    train_drugs = set(train["drug_id"])

    # Drugs are warm in both splits: the query side is always something the
    # model has seen.
    assert set(held["drug_id"]) <= train_drugs, "a held-out drug is missing from train"

    if split.kind == "warm":
        assert set(held["target_id"]) <= set(train["target_id"]), (
            "a held-out target is missing from train; the warm split is not warm"
        )
    elif split.kind == "cold_target":
        overlap = set(held["seq_group"]) & set(train["seq_group"])
        assert not overlap, (
            f"{len(overlap)} sequence groups straddle the cold split, so a "
            f"'new' target is one the model trained on: {sorted(overlap)[:5]}"
        )


def warm_split(
    ds: Dataset,
    seed: int = DEFAULT_SEED,
    fracs: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> Split:
    """70/15/15 over pairs, with every held-out entity present in training.

    The warm-start guarantee is built in rather than repaired afterwards: in
    shuffled order, the first pair carrying each drug and each target is
    reserved for training, and the remaining pairs are divided to hit the
    requested fractions. Repairing instead - splitting first, then sweeping
    violations into train - moves more pairs than necessary, because one
    violating row drags every other row sharing its cold entity along with it.
    """
    if not np.isclose(sum(fracs), 1.0):
        raise ValueError(f"fracs must sum to 1, got {fracs} summing to {sum(fracs)}")

    pairs = ds.pairs.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    shuffled = pairs.iloc[order]

    reserved = (~shuffled.duplicated("drug_id") | ~shuffled.duplicated("target_id")).to_numpy()
    n_reserved = int(reserved.sum())
    n_total = len(pairs)

    n_train = int(round(fracs[0] * n_total))
    n_val = int(round(fracs[1] * n_total))
    if n_reserved > n_train:
        raise ValueError(
            f"{ds.name}: covering every drug and target needs {n_reserved:,} pairs, "
            f"more than the {n_train:,} the train fraction allows"
        )

    fold = np.array(["train"] * n_total, dtype=object)
    free = np.flatnonzero(~reserved)  # positions within `shuffled`
    fold[free[: n_val]] = "val"
    fold[free[n_val: n_val + (n_total - n_train - n_val)]] = "test"

    frame = shuffled.assign(fold=fold).sort_values(["drug_id", "target_id"], kind="stable")
    frame = frame.reset_index(drop=True)

    sizes = frame["fold"].value_counts()
    split = Split(
        kind="warm", dataset=ds.name, seed=seed, frame=frame,
        report={
            "n_pairs": n_total,
            "pairs_reserved_for_coverage": n_reserved,
            "frac_reserved": n_reserved / n_total,
            **{f"n_{f}": int(sizes.get(f, 0)) for f in FOLDS},
            **{f"frac_{f}": float(sizes.get(f, 0) / n_total) for f in FOLDS},
            "train_drugs": int(frame.loc[frame["fold"] == "train", "drug_id"].nunique()),
            "train_targets": int(frame.loc[frame["fold"] == "train", "target_id"].nunique()),
        },
    )
    _check(split)
    return split


def cold_target_split(
    ds: Dataset,
    seed: int = DEFAULT_SEED,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> Split:
    """Hold out whole sequence groups, so a test target is genuinely unseen.

    Fractions are of *sequence groups*, not pairs, because a target is held out
    with all of its measurements. Groups vary in size, so the resulting pair
    fractions differ from the group fractions and both are reported.

    Drugs stay warm. A drug whose every measurement falls inside the held-out
    groups would arrive in validation or test with no training history, which
    is a cold-drug problem and not what Task 3 asks about; those pairs are
    dropped and counted in ``pairs_dropped_cold_drug``.
    """
    pairs = ds.pairs.reset_index(drop=True)
    groups = np.sort(pairs["seq_group"].unique())
    rng = np.random.default_rng(seed)
    shuffled_groups = rng.permutation(groups)

    n_val = int(round(val_frac * len(groups)))
    n_test = int(round(test_frac * len(groups)))
    if n_val < 1 or n_test < 1 or n_val + n_test >= len(groups):
        raise ValueError(
            f"{ds.name}: {len(groups)} sequence groups cannot be split "
            f"{val_frac:.0%}/{test_frac:.0%} into non-empty folds"
        )

    assignment = dict.fromkeys(shuffled_groups[: n_val], "val")
    assignment |= dict.fromkeys(shuffled_groups[n_val: n_val + n_test], "test")

    frame = pairs.assign(fold=pairs["seq_group"].map(assignment).fillna("train"))

    # Keep only held-out pairs whose drug the model has actually trained on.
    train_drugs = set(frame.loc[frame["fold"] == "train", "drug_id"])
    cold_drug = (frame["fold"] != "train") & ~frame["drug_id"].isin(train_drugs)
    n_dropped = int(cold_drug.sum())
    frame = frame[~cold_drug].reset_index(drop=True)

    sizes = frame["fold"].value_counts()
    group_folds = frame.groupby("seq_group")["fold"].first()
    split = Split(
        kind="cold_target", dataset=ds.name, seed=seed, frame=frame,
        report={
            "n_pairs": len(frame),
            "n_seq_groups": len(groups),
            **{f"n_{f}": int(sizes.get(f, 0)) for f in FOLDS},
            **{f"frac_{f}": float(sizes.get(f, 0) / len(frame)) for f in FOLDS},
            **{f"groups_{f}": int((group_folds == f).sum()) for f in FOLDS},
            "targets_held_out": int(
                frame.loc[frame["fold"] != "train", "target_id"].nunique()
            ),
            "pairs_dropped_cold_drug": n_dropped,
        },
    )
    _check(split)
    return split


def build(ds: Dataset, seed: int = DEFAULT_SEED) -> dict[str, Split]:
    """Both splits for one dataset."""
    return {"warm": warm_split(ds, seed=seed), "cold_target": cold_target_split(ds, seed=seed)}


def summary(splits: dict[str, dict[str, Split]]) -> pd.DataFrame:
    """One row per (dataset, split kind) for the report."""
    rows = []
    for dataset, by_kind in splits.items():
        for kind, split in by_kind.items():
            r = split.report
            rows.append({
                "dataset": dataset.upper(), "split": kind,
                "train": r["n_train"], "val": r["n_val"], "test": r["n_test"],
                "frac_test": r["frac_test"],
                "held_out_targets": r.get("targets_held_out", 0),
                "groups_test": r.get("groups_test", float("nan")),
                "reserved_for_coverage": r.get("pairs_reserved_for_coverage", 0),
                "dropped_cold_drug": r.get("pairs_dropped_cold_drug", 0),
            })
    return pd.DataFrame(rows).set_index(["dataset", "split"])
