"""Loading, cleaning and deduplication for the DAVIS and KIBA datasets.

One shared module: Tasks 1-3 all import from here, so every experiment sees
identical pairs, identifiers and score scales.

Source files are the DeepDTA copies (``data/<dataset>/``):

    proteins.txt      JSON dict, target id -> amino-acid sequence
    ligands_can.txt   JSON dict, drug id -> canonical SMILES
    Y                 pickled numpy matrix, ligand-major

``Y`` holds **raw Kd in nM** for DAVIS, so it is converted to pKd here. KIBA's
``Y`` already holds KIBA scores and is used as-is. The two scales are not
comparable and must never be pooled.

Deduplication
-------------
DAVIS writes mutations into the *target id* but not into the sequence, so
442 ids collapse onto 379 unique sequences (e.g. all 15 ``ABL1`` variants share
one 1167 aa string). Averaging those merges genuinely different measurements -
``ABL1`` vs ``ABL1(T315I)`` differ by ~2 log units of peak affinity - so the
default ``dedup_key="id"`` keeps every target distinct and records the sequence
group alongside it. Pass ``dedup_key="sequence"`` to average instead.

Either way every row carries ``seq_group``, which is the grouping unit a
cold-target split must respect so sequence-identical targets cannot straddle
train and test.
"""

from __future__ import annotations

import io
import json
import pickle
import pickletools
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Y is an untrusted download, so its opcodes are checked before unpickling.
_ALLOWED_PICKLE_IMPORTS = {
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
}


@dataclass(frozen=True)
class _Spec:
    score_name: str
    #: True when Y holds raw Kd in nM and needs the pKd transform.
    log_transform: bool


SPECS = {
    "davis": _Spec(score_name="pKd", log_transform=True),
    "kiba": _Spec(score_name="KIBA", log_transform=False),
}


def _safe_load_pickle(path: Path):
    """Unpickle only after confirming every import is a numpy array builder."""
    raw = path.read_bytes()
    found, stack = set(), []
    for op, arg, _ in pickletools.genops(io.BytesIO(raw)):
        if op.name == "GLOBAL":
            found.add(tuple(arg.split(" ", 1)))
        elif op.name in ("BINUNICODE", "SHORT_BINUNICODE"):
            stack.append(arg)
        elif op.name == "STACK_GLOBAL" and len(stack) >= 2:
            found.add((stack[-2], stack[-1]))

    unexpected = found - _ALLOWED_PICKLE_IMPORTS
    if unexpected:
        raise ValueError(
            f"{path} contains unexpected pickle imports {sorted(unexpected)}; "
            "refusing to load. Re-download it from the DeepDTA repository."
        )
    with warnings.catch_warnings():
        # numpy >= 2.4 warns about the py2-era dtype(align=0) in these files.
        # The filter is broad but scoped to this one call.
        warnings.simplefilter("ignore")
        return pickle.loads(raw, encoding="latin1")


def load_raw(dataset: str, data_dir: Path | str = DATA_DIR):
    """Return ``(ligands, proteins, Y)`` exactly as stored on disk.

    ``ligands`` and ``proteins`` are insertion-ordered dicts whose order matches
    ``Y``'s rows and columns respectively.
    """
    if dataset not in SPECS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(SPECS)}")
    root = Path(data_dir) / dataset
    missing = [f for f in ("proteins.txt", "ligands_can.txt", "Y") if not (root / f).exists()]
    if missing:
        raise FileNotFoundError(f"{root} is missing {missing}. See README for the download step.")

    ligands = json.loads((root / "ligands_can.txt").read_text())
    proteins = json.loads((root / "proteins.txt").read_text())
    y = np.asarray(_safe_load_pickle(root / "Y"), dtype=float)

    if y.shape != (len(ligands), len(proteins)):
        raise ValueError(
            f"{dataset}: Y is {y.shape} but found {len(ligands)} ligands and "
            f"{len(proteins)} proteins; the files are out of sync."
        )
    return ligands, proteins, y


def _group_labels(mapping: dict[str, str]) -> dict[str, str]:
    """Map each id to a stable label shared by all ids with the same value.

    The label is the alphabetically first id in the group, which keeps readable
    names in report tables (the ABL1 group is labelled ``ABL1``).
    """
    by_value: dict[str, list[str]] = defaultdict(list)
    for key, value in mapping.items():
        by_value[value].append(key)
    return {key: sorted(ids)[0] for ids in by_value.values() for key in ids}


@dataclass
class Dataset:
    """Cleaned, deduplicated pairs plus the entity tables they reference."""

    name: str
    score_name: str
    dedup_key: str
    #: drug_id, target_id, seq_group, score
    pairs: pd.DataFrame
    #: drug_id, smiles, lig_group
    ligands: pd.DataFrame
    #: target_id, sequence, seq_group, n_aa
    targets: pd.DataFrame
    report: dict = field(default_factory=dict)

    @property
    def n_drugs(self) -> int:
        return self.pairs["drug_id"].nunique()

    @property
    def n_targets(self) -> int:
        return self.pairs["target_id"].nunique()

    def matrix(self) -> pd.DataFrame:
        """Drug x target score matrix; unobserved pairs are NaN, never 0."""
        return self.pairs.pivot(index="drug_id", columns="target_id", values="score")

    def example_rows(self, n: int = 5, smiles_chars: int = 40, seq_chars: int = 40):
        """Pairs joined to SMILES and sequence, truncated for display."""
        out = (
            self.pairs.head(n)
            .merge(self.ligands[["drug_id", "smiles"]], on="drug_id", how="left")
            .merge(self.targets[["target_id", "sequence", "n_aa"]], on="target_id", how="left")
        )
        out["smiles"] = out["smiles"].str.slice(0, smiles_chars) + "..."
        out["sequence"] = out["sequence"].str.slice(0, seq_chars) + "..."
        return out[["drug_id", "smiles", "target_id", "sequence", "n_aa", "score"]]

    def __repr__(self) -> str:
        return (
            f"Dataset({self.name}, {len(self.pairs):,} pairs, "
            f"{self.n_drugs} drugs x {self.n_targets} targets, "
            f"score={self.score_name}, dedup_key={self.dedup_key!r})"
        )


def load(
    dataset: str,
    data_dir: Path | str = DATA_DIR,
    dedup_key: str = "id",
) -> Dataset:
    """Load, clean and deduplicate one dataset.

    Parameters
    ----------
    dataset
        ``"davis"`` or ``"kiba"``.
    dedup_key
        ``"id"`` keeps every target id distinct (default; 442 DAVIS targets).
        ``"sequence"`` averages targets that share an amino-acid sequence
        (379 DAVIS targets), which merges mutant and wild-type measurements.
    """
    if dedup_key not in ("id", "sequence"):
        raise ValueError(f"dedup_key must be 'id' or 'sequence', got {dedup_key!r}")

    spec = SPECS[dataset]
    ligands, proteins, y = load_raw(dataset, data_dir)
    drug_ids = np.array(list(ligands), dtype=object)
    target_ids = np.array(list(proteins), dtype=object)

    # NaN marks a pair the assay never reported; it is missing, not dirty.
    observed = ~np.isnan(y)
    finite = observed & np.isfinite(y)
    n_nonfinite = int((observed & ~finite).sum())

    # A Kd must be positive before it can be log-transformed. KIBA scores may
    # legitimately be zero, so positivity is required only for raw Kd.
    if spec.log_transform:
        valid = finite & (y > 0)
        n_nonpositive = int((finite & (y <= 0)).sum())
    else:
        valid = finite
        n_nonpositive = 0

    rows, cols = np.nonzero(valid)
    raw_scores = y[rows, cols]
    # pKd = 9 - log10(Kd in nM): higher means stronger binding.
    scores = 9.0 - np.log10(raw_scores) if spec.log_transform else raw_scores

    seq_group_of = _group_labels(proteins)
    lig_group_of = _group_labels(ligands)

    pairs = pd.DataFrame(
        {
            "drug_id": drug_ids[rows],
            "target_id": target_ids[cols],
            "score": scores,
        }
    )
    pairs["seq_group"] = pairs["target_id"].map(seq_group_of)

    before = {
        "n_pairs": len(pairs),
        "n_drugs": pairs["drug_id"].nunique(),
        "n_targets": pairs["target_id"].nunique(),
    }

    key_col = "target_id" if dedup_key == "id" else "seq_group"
    # min/max as named aggregations rather than a lambda: pandas runs these in
    # cython, while a lambda is called once per group - 118k groups for KIBA,
    # which turned a 2 s load into a 20 s one.
    merged = pairs.groupby(["drug_id", key_col], as_index=False, sort=False).agg(
        score=("score", "mean"),
        n_merged=("score", "size"),
        score_min=("score", "min"),
        score_max=("score", "max"),
    )
    merged["spread"] = merged["score_max"] - merged["score_min"]
    if dedup_key == "sequence":
        merged = merged.rename(columns={"seq_group": "target_id"})
        merged["seq_group"] = merged["target_id"]
    else:
        merged["seq_group"] = merged["target_id"].map(seq_group_of)

    conflicts = merged.loc[merged["n_merged"] > 1, "spread"]
    report = {
        "dataset": dataset,
        "score_name": spec.score_name,
        "dedup_key": dedup_key,
        "cells_total": int(y.size),
        "cells_observed": int(observed.sum()),
        "cells_missing": int((~observed).sum()),
        "sparsity": float((~observed).mean()),
        "dropped_nonfinite": n_nonfinite,
        "dropped_nonpositive": n_nonpositive,
        "pairs_before_dedup": before["n_pairs"],
        "drugs_before_dedup": before["n_drugs"],
        "targets_before_dedup": before["n_targets"],
        "pairs_after_dedup": len(merged),
        "drugs_after_dedup": merged["drug_id"].nunique(),
        "targets_after_dedup": merged["target_id"].nunique(),
        "unique_sequences": len(set(proteins.values())),
        "unique_smiles": len(set(ligands.values())),
        "merged_cells": int((merged["n_merged"] > 1).sum()),
        "merged_spread_max": float(conflicts.max()) if len(conflicts) else 0.0,
        "merged_spread_mean": float(conflicts.mean()) if len(conflicts) else 0.0,
        "merged_spread_over_1": int((conflicts > 1).sum()),
    }

    pairs = (
        merged[["drug_id", "target_id", "seq_group", "score"]]
        .sort_values(["drug_id", "target_id"], kind="stable")
        .reset_index(drop=True)
    )

    lig_table = pd.DataFrame(
        {"drug_id": list(ligands), "smiles": list(ligands.values())}
    )
    lig_table["lig_group"] = lig_table["drug_id"].map(lig_group_of)

    tgt_table = pd.DataFrame(
        {"target_id": list(proteins), "sequence": list(proteins.values())}
    )
    tgt_table["seq_group"] = tgt_table["target_id"].map(seq_group_of)
    tgt_table["n_aa"] = tgt_table["sequence"].str.len()

    if dedup_key == "sequence":
        tgt_table = (
            tgt_table.drop_duplicates("seq_group")
            .assign(target_id=lambda d: d["seq_group"])
            .reset_index(drop=True)
        )

    # The pairs table must only reference entities we actually kept.
    assert set(pairs["drug_id"]) <= set(lig_table["drug_id"])
    assert set(pairs["target_id"]) <= set(tgt_table["target_id"])

    return Dataset(
        name=dataset,
        score_name=spec.score_name,
        dedup_key=dedup_key,
        pairs=pairs,
        ligands=lig_table,
        targets=tgt_table,
        report=report,
    )


def load_both(data_dir: Path | str = DATA_DIR, dedup_key: str = "id") -> dict[str, Dataset]:
    """Load DAVIS and KIBA under the same settings."""
    return {name: load(name, data_dir, dedup_key) for name in ("davis", "kiba")}


def dedup_table(datasets: dict[str, Dataset]) -> pd.DataFrame:
    """Original vs deduplicated counts, for the Task 1 report table."""
    rows = []
    for ds in datasets.values():
        r = ds.report
        rows.append(
            {
                "dataset": ds.name.upper(),
                "ligands (orig)": r["drugs_before_dedup"],
                "ligands (dedup)": r["drugs_after_dedup"],
                "targets (orig)": r["targets_before_dedup"],
                "targets (dedup)": r["targets_after_dedup"],
                "pairs (orig)": r["pairs_before_dedup"],
                "pairs (dedup)": r["pairs_after_dedup"],
                "unique sequences": r["unique_sequences"],
                "unique SMILES": r["unique_smiles"],
            }
        )
    return pd.DataFrame(rows).set_index("dataset")


def dedup_comparison(
    datasets: dict[str, Dataset] | None = None,
    data_dir: Path | str = DATA_DIR,
) -> pd.DataFrame:
    """What each deduplication key costs, both keys side by side.

    The brief asks for original vs deduplicated counts. Keying by ``id`` on the
    DeepDTA matrices changes nothing, because ``Y`` already holds exactly one
    cell per (drug, target) pair - so a one-key table has identical "before" and
    "after" columns and reads as if no decision was made. Running both keys
    shows the decision instead: DAVIS gives up 63 targets and 4,284 pairs under
    the sequence key, and the 1,224 cells that get averaged include values up to
    4.75 pKd apart.

    ``datasets`` is the already-loaded ``dedup_key="id"`` pair, if the caller has
    it, so only the sequence-keyed load has to be repeated.
    """
    by_key = {
        "id": datasets if datasets is not None else load_both(data_dir, dedup_key="id"),
        "sequence": load_both(data_dir, dedup_key="sequence"),
    }
    rows = []
    for name in by_key["id"]:
        for key, loaded in by_key.items():
            r = loaded[name].report
            rows.append(
                {
                    "dataset": name.upper(),
                    "dedup_key": key,
                    "ligands": r["drugs_after_dedup"],
                    "targets": r["targets_after_dedup"],
                    "pairs": r["pairs_after_dedup"],
                    "cells_merged": r["merged_cells"],
                    "spread_max": r["merged_spread_max"],
                    "spread_mean": r["merged_spread_mean"],
                    "spread_over_1": r["merged_spread_over_1"],
                }
            )
    return pd.DataFrame(rows).set_index(["dataset", "dedup_key"])


def shared_sequences(datasets: dict[str, Dataset]) -> set[str]:
    """Sequences appearing byte-identically in both datasets.

    179 of KIBA's 229 sequences also occur in DAVIS, so protein features
    (k-mer counts, ESM embeddings) can be computed once across 429 unique
    sequences and reused for both datasets in Task 3.
    """
    seqs = [set(ds.targets["sequence"]) for ds in datasets.values()]
    return set.intersection(*seqs) if seqs else set()


if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_columns", 50)
    sets = load_both()

    for ds in sets.values():
        r = ds.report
        print(f"\n=== {ds.name.upper()} ===")
        print(ds)
        print(
            f"matrix {r['cells_observed']:,}/{r['cells_total']:,} observed "
            f"({100 * (1 - r['sparsity']):.1f}% filled, {100 * r['sparsity']:.1f}% sparse)"
        )
        print(
            f"dropped: {r['dropped_nonfinite']} non-finite, "
            f"{r['dropped_nonpositive']} non-positive Kd"
        )
        print(
            f"merged cells: {r['merged_cells']} "
            f"(max spread {r['merged_spread_max']:.2f} {r['score_name']})"
        )
        s = ds.pairs["score"]
        print(f"score: min {s.min():.3f}  max {s.max():.3f}  mean {s.mean():.3f}")
        print(ds.example_rows(3).to_string(index=False))

    print("\n=== dedup counts ===")
    print(dedup_table(sets).to_string())
    print(f"\nsequences shared by both datasets: {len(shared_sequences(sets))}")
    total_unique = len(set().union(*[set(d.targets['sequence']) for d in sets.values()]))
    print(f"unique sequences across both     : {total_unique}")
