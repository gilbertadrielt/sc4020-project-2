"""Raw files to a clean, chemistry-keyed table of observed pairs.

The pipeline is deliberately linear and every stage appends to a cleaning log,
so the counts in the report are generated rather than typed:

    download -> load_raw -> to_long -> canonicalise -> clean -> deduplicate
             -> index_entities -> save

Three decisions in here matter more than the rest of the project put together.

**Missing is not zero.** Three quarters of the KIBA matrix was never measured.
Those cells become no row at all.  Filling them with zero would invent several
hundred thousand confident non-binding claims that nobody ran.

**The assay ceiling is not a measurement.** Around 70% of DAVIS labels sit at
Kd = 10,000 nM, which is where the assay stopped looking.  The true value is
*at most* pKd 5.  These rows are flagged ``censored`` and kept, because
dropping them would discard the genuine finding that those pairs do not bind,
and because a censored-aware loss can use them correctly later.

**Entities are chemistry, not identifiers.** DAVIS lists 442 target IDs that
are only 379 distinct amino-acid sequences: mutants and phosphorylated forms
are separate IDs carrying the same sequence in the benchmark files.  Keying on
IDs lets the identical protein sit on both sides of a train/test split, which
inflates every collaborative method.  Pairs are therefore keyed on
``(canonical SMILES, amino-acid sequence)`` and duplicates are merged, keeping
the repeat count and the spread so the merge can be audited.
"""
from __future__ import annotations

import json
import pickle
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_PROCESSED, DATA_RAW, DATASETS, DAVIS_KD_CEILING_NM, SCORE_NAME

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

DEEPDTA_BASE = "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data"

#: ``Y`` is the affinity matrix: Kd in nM for DAVIS, the composite score with
#: NaN for unmeasured pairs for KIBA.  The similarity files are the
#: benchmark's own published drug-drug and target-target matrices, reused here
#: as kernels rather than recomputed.
RAW_FILES = {
    "davis": [
        "Y",
        "ligands_can.txt",
        "proteins.txt",
        "target-target_similarities_WS.txt",
        "drug-drug_similarities_2D.txt",
    ],
    "kiba": [
        "Y",
        "ligands_can.txt",
        "proteins.txt",
        "kiba_target_sim.txt",
        "kiba_drug_sim.txt",
    ],
}

SIM_FILES = {
    "davis": {"target": "target-target_similarities_WS.txt", "drug": "drug-drug_similarities_2D.txt"},
    "kiba": {"target": "kiba_target_sim.txt", "drug": "kiba_drug_sim.txt"},
}


def download(name: str, force: bool = False) -> Path:
    """Fetch one dataset's raw files into ``data/raw/<name>/``.

    Skips files already present, so re-running is cheap and works offline once
    the data is there.  If the network is unavailable, fetch the files listed
    in ``RAW_FILES`` by hand from the printed URL; nothing downstream cares how
    they arrived.
    """
    import urllib.error
    import urllib.request

    out = DATA_RAW / name
    out.mkdir(parents=True, exist_ok=True)
    for fname in RAW_FILES[name]:
        dest = out / fname
        if dest.exists() and dest.stat().st_size > 0 and not force:
            continue
        url = f"{DEEPDTA_BASE}/{name}/{fname}"
        try:
            with urllib.request.urlopen(url, timeout=180) as r, open(dest, "wb") as fh:
                fh.write(r.read())
        except (urllib.error.URLError, TimeoutError) as exc:
            raise SystemExit(f"could not download {url} ({exc}); save it as {dest}") from exc
    return out


# ---------------------------------------------------------------------------
# Cleaning log
# ---------------------------------------------------------------------------


@dataclass
class CleaningLog:
    """Row-count audit trail, written beside the processed data."""

    dataset: str
    steps: list = field(default_factory=list)

    def add(self, step: str, remaining: int, note: str = "") -> None:
        self.steps.append({"step": step, "pairs_remaining": int(remaining), "note": note})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


@dataclass
class RawDataset:
    name: str
    ligand_ids: list
    smiles: list
    target_ids: list
    sequences: list
    Y: np.ndarray


def load_raw(name: str, root: Path | None = None) -> RawDataset:
    """Read the benchmark files for ``name`` without altering a single value."""
    root = Path(root) if root is not None else DATA_RAW / name
    ligands = json.load(open(root / "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(root / "proteins.txt"), object_pairs_hook=OrderedDict)
    with open(root / "Y", "rb") as fh:
        Y = np.asarray(pickle.load(fh, encoding="latin1"), dtype=float)

    if Y.shape != (len(ligands), len(proteins)):
        raise ValueError(f"{name}: Y is {Y.shape}, expected {(len(ligands), len(proteins))}")

    return RawDataset(
        name=name,
        ligand_ids=list(ligands.keys()),
        smiles=list(ligands.values()),
        target_ids=list(proteins.keys()),
        sequences=list(proteins.values()),
        Y=Y,
    )


def to_long(raw: RawDataset, log: CleaningLog) -> pd.DataFrame:
    """Matrix to one row per *observed* pair.

    A non-finite cell means the experiment was never run, so it produces no
    row.  This is the single place the "missing is not zero" rule is enforced.
    """
    mask = np.isfinite(raw.Y)
    rows, cols = np.where(mask)
    df = pd.DataFrame(
        {
            "ligand_id": [raw.ligand_ids[i] for i in rows],
            "target_id": [raw.target_ids[j] for j in cols],
            "smiles": [raw.smiles[i] for i in rows],
            "sequence": [raw.sequences[j] for j in cols],
            "raw_value": raw.Y[rows, cols],
        }
    )
    log.add("observed cells in the raw matrix", len(df),
            f"matrix {raw.Y.shape}, {100 * (1 - mask.mean()):.1f}% never measured")
    return df


def canonicalise_smiles(smiles: list) -> tuple[list, str]:
    """Put every SMILES in one canonical form so duplicates can be detected.

    The same molecule has many valid SMILES spellings, so string equality on
    the raw text under-counts duplicates.  Without RDKit the raw strings are
    used and the cleaning log records which path was taken, because that
    changes the duplicate count and therefore every downstream number.
    """
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return list(smiles), "raw SMILES (RDKit unavailable)"

    out, n_failed = [], 0
    cache: dict[str, str] = {}
    for s in smiles:
        if s not in cache:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                cache[s] = s
                n_failed += 1
            else:
                cache[s] = Chem.MolToSmiles(mol)
        out.append(cache[s])
    note = "RDKit canonical SMILES"
    if n_failed:
        note += f" ({n_failed} unparseable, kept as written)"
    return out, note


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------


def clean(df: pd.DataFrame, name: str, log: CleaningLog) -> pd.DataFrame:
    """Drop impossible rows, convert the score scale, flag abnormal values.

    Nothing abnormal is removed.  The brief forbids silent removal, and the
    censored values in particular carry the information that those pairs do
    not bind, which is most of what DAVIS knows.
    """
    df = df.copy()

    finite = np.isfinite(df["raw_value"])
    if (~finite).any():
        df = df[finite]
        log.add("drop non-finite values", len(df), "")

    if name == "davis":
        positive = df["raw_value"] > 0
        if (~positive).any():
            df = df[positive]
            log.add("drop Kd <= 0", len(df), "log of a non-positive Kd is undefined")
        # The one and only conversion from Kd in nM to pKd.
        df["score"] = 9.0 - np.log10(df["raw_value"].to_numpy(dtype=float))
        df["censored"] = df["raw_value"] >= DAVIS_KD_CEILING_NM
        log.add(
            "convert Kd (nM) to pKd = 9 - log10(Kd)", len(df),
            f"{df['censored'].mean():.1%} of labels sit on the assay ceiling "
            f"(Kd >= {DAVIS_KD_CEILING_NM:.0f} nM, i.e. pKd <= 5) and are flagged censored",
        )
    else:
        df["score"] = df["raw_value"].astype(float)
        # KIBA scores are a composite with no detection ceiling in the file.
        df["censored"] = False
        log.add("use the supplied KIBA score as given", len(df),
                "composite of Ki, Kd and IC50; not comparable with pKd")

    # Flag, never drop: values far outside the bulk are reported in Task 1.
    q1, q3 = df["score"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df["extreme"] = (df["score"] < q1 - 3 * iqr) | (df["score"] > q3 + 3 * iqr)
    if df["extreme"].any():
        log.add("flag extreme values (kept)", len(df),
                f"{int(df['extreme'].sum())} pairs beyond 3 IQR from the quartiles")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Deduplicate and index
# ---------------------------------------------------------------------------


def deduplicate(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Merge pairs that are the same chemistry under different identifiers.

    Keeps ``n_repeats`` and the min/max/std of the merged labels, so the
    report can show how much the merged duplicates disagreed.  On DAVIS the
    worst disagreement is several pKd units for what the files call the same
    molecule against the same sequence, which is the argument for doing this
    at all.
    """
    before = len(df)
    agg = (
        df.groupby(["smiles", "sequence"], sort=False)
        .agg(
            score=("score", "mean"),
            score_min=("score", "min"),
            score_max=("score", "max"),
            score_std=("score", "std"),
            n_repeats=("score", "size"),
            censored=("censored", "all"),
            any_censored=("censored", "any"),
            extreme=("extreme", "any"),
            ligand_id=("ligand_id", "first"),
            target_id=("target_id", "first"),
            ligand_ids=("ligand_id", lambda s: "|".join(sorted(set(s)))),
            target_ids=("target_id", lambda s: "|".join(sorted(set(s)))),
        )
        .reset_index()
    )
    agg["score_std"] = agg["score_std"].fillna(0.0)
    agg["spread"] = agg["score_max"] - agg["score_min"]

    merged = int((agg["n_repeats"] > 1).sum())
    note = f"{before - len(agg)} rows merged into {merged} groups"
    if merged:
        note += f"; largest label disagreement {agg['spread'].max():.2f}"
    log.add("deduplicate on (canonical SMILES, sequence)", len(agg), note)
    return agg


def index_entities(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign contiguous integer indices to the distinct molecules and proteins.

    Returns the pair table with ``lig_idx``/``tgt_idx`` columns plus the two
    entity tables.  Everything downstream addresses entities by index; the
    original identifiers survive on the entity tables so case studies can
    still quote them.
    """
    ligands = (
        pairs[["smiles", "ligand_id", "ligand_ids"]]
        .drop_duplicates("smiles")
        .reset_index(drop=True)
        .rename(columns={"ligand_id": "ligand_id_first"})
    )
    ligands["lig_idx"] = np.arange(len(ligands))

    targets = (
        pairs[["sequence", "target_id", "target_ids"]]
        .drop_duplicates("sequence")
        .reset_index(drop=True)
        .rename(columns={"target_id": "target_id_first"})
    )
    targets["tgt_idx"] = np.arange(len(targets))
    targets["seq_len"] = targets["sequence"].str.len()

    pairs = pairs.merge(ligands[["smiles", "lig_idx"]], on="smiles", how="left")
    pairs = pairs.merge(targets[["sequence", "tgt_idx"]], on="sequence", how="left")

    assert pairs["lig_idx"].notna().all() and pairs["tgt_idx"].notna().all()
    assert not pairs.duplicated(["lig_idx", "tgt_idx"]).any(), "duplicate pair survived"
    return pairs, ligands, targets


# ---------------------------------------------------------------------------
# Build and load
# ---------------------------------------------------------------------------


def counts_table(name: str, raw: RawDataset, df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Entity and pair counts at ID level and at chemistry level."""
    n_lig_id, n_tgt_id = len(raw.ligand_ids), len(raw.target_ids)
    n_lig_chem = pairs["lig_idx"].nunique()
    n_tgt_chem = pairs["tgt_idx"].nunique()
    rows = [
        {
            "level": "source identifiers",
            "ligands": n_lig_id,
            "targets": n_tgt_id,
            "observed_pairs": int(np.isfinite(raw.Y).sum()),
            "matrix_cells": n_lig_id * n_tgt_id,
        },
        {
            "level": "distinct chemistry",
            "ligands": n_lig_chem,
            "targets": n_tgt_chem,
            # Distinct (SMILES, sequence) combinations, not the ID-level count,
            # so density cannot exceed 1.
            "observed_pairs": int(len(pairs)),
            "matrix_cells": n_lig_chem * n_tgt_chem,
        },
    ]
    out = pd.DataFrame(rows)
    out["density"] = out["observed_pairs"] / out["matrix_cells"]
    out.insert(0, "dataset", name)
    return out


def build(name: str, save: bool = True) -> dict:
    """Run the whole pipeline for one dataset."""
    log = CleaningLog(dataset=name)
    raw = load_raw(name)
    df = to_long(raw, log)

    df["smiles"], note = canonicalise_smiles(list(df["smiles"]))
    log.add("canonicalise SMILES", len(df), note)

    df = clean(df, name, log)
    pairs = deduplicate(df, log)
    pairs, ligands, targets = index_entities(pairs)
    counts = counts_table(name, raw, df, pairs)

    # The pair table is normalised: it stores indices and scores only.  SMILES
    # and sequences live once each on the entity tables and are joined back on
    # demand (``attach_entities``).  Repeating a 2,500-character sequence on
    # every one of 117,000 rows would turn a 6 MB file into 100 MB.
    cols = [
        "lig_idx", "tgt_idx", "ligand_id", "target_id",
        "score", "score_min", "score_max", "score_std", "spread",
        "n_repeats", "censored", "any_censored", "extreme",
    ]
    pairs = pairs[cols].sort_values(["lig_idx", "tgt_idx"]).reset_index(drop=True)

    out = {
        "name": name,
        "pairs": pairs,
        "ligands": ligands,
        "targets": targets,
        "counts": counts,
        "cleaning_log": log.frame(),
        "n_lig": int(pairs["lig_idx"].nunique()),
        "n_tgt": int(pairs["tgt_idx"].nunique()),
    }

    if save:
        pairs.to_csv(DATA_PROCESSED / f"pairs_{name}.csv", index=False)
        ligands.to_csv(DATA_PROCESSED / f"ligands_{name}.csv", index=False)
        targets.to_csv(DATA_PROCESSED / f"targets_{name}.csv", index=False)
        counts.to_csv(DATA_PROCESSED / f"counts_{name}.csv", index=False)
        log.frame().to_csv(DATA_PROCESSED / f"cleaning_{name}.csv", index=False)
    return out


def load_processed(name: str) -> dict:
    """Read what ``build`` wrote, rebuilding it if it is not there yet."""
    p = DATA_PROCESSED / f"pairs_{name}.csv"
    if not p.exists():
        return build(name)
    pairs = pd.read_csv(p)
    for c in ("censored", "any_censored", "extreme"):
        pairs[c] = pairs[c].astype(bool)
    return {
        "name": name,
        "pairs": pairs,
        "ligands": pd.read_csv(DATA_PROCESSED / f"ligands_{name}.csv"),
        "targets": pd.read_csv(DATA_PROCESSED / f"targets_{name}.csv"),
        "counts": pd.read_csv(DATA_PROCESSED / f"counts_{name}.csv"),
        "cleaning_log": pd.read_csv(DATA_PROCESSED / f"cleaning_{name}.csv"),
        "n_lig": int(pairs["lig_idx"].nunique()),
        "n_tgt": int(pairs["tgt_idx"].nunique()),
    }


def attach_entities(pairs: pd.DataFrame, proc: dict) -> pd.DataFrame:
    """Join SMILES and sequence back onto a pair table when they are needed."""
    return (
        pairs.merge(proc["ligands"][["lig_idx", "smiles"]], on="lig_idx", how="left")
        .merge(proc["targets"][["tgt_idx", "sequence"]], on="tgt_idx", how="left")
    )


def to_matrix(pairs: pd.DataFrame, n_lig: int, n_tgt: int, col: str = "score") -> np.ndarray:
    """Dense matrix with NaN where nothing was measured."""
    M = np.full((n_lig, n_tgt), np.nan)
    M[pairs["lig_idx"].to_numpy(), pairs["tgt_idx"].to_numpy()] = pairs[col].to_numpy()
    return M


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    which = [a for a in args if not a.startswith("--")] or list(DATASETS)
    if "--download" in args:
        for ds in which:
            print(f"downloading {ds}")
            download(ds, force="--force" in args)
        if set(args) <= {"--download", "--force"}:
            raise SystemExit(0)
    for ds in which:
        res = build(ds)
        print(f"\n=== {ds.upper()}  ({SCORE_NAME[ds]})")
        print(res["cleaning_log"].to_string(index=False))
        print(res["counts"].to_string(index=False))
