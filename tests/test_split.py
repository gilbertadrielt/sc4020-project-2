"""Correctness tests for the Task 2 and Task 3 partitions.

These run on a small synthetic Dataset rather than on DAVIS, so they are fast
and so the leakage cases can be constructed on purpose. The one that matters is
the last group: a cold-target split must separate sequence-identical targets,
because DAVIS writes mutations into the target id but not into the sequence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import Dataset
from src.split import Split, cold_target_split, warm_split


def make_dataset(n_drugs=12, n_targets=20, twins=4, seed=0) -> Dataset:
    """A complete drug x target matrix where some targets share a sequence.

    ``twins`` targets are given the sequence group of another target, standing
    in for DAVIS's mutant kinases.
    """
    rng = np.random.default_rng(seed)
    drug_ids = [f"d{i}" for i in range(n_drugs)]
    target_ids = [f"t{j}" for j in range(n_targets)]

    seq_group = {t: t for t in target_ids}
    for j in range(twins):
        seq_group[target_ids[-(j + 1)]] = target_ids[j]  # twin of an earlier target

    rows = [
        {"drug_id": d, "target_id": t, "seq_group": seq_group[t],
         "score": float(rng.uniform(5, 10))}
        for d in drug_ids for t in target_ids
    ]
    pairs = pd.DataFrame(rows)
    return Dataset(
        name="synthetic", score_name="pKd", dedup_key="id", pairs=pairs,
        ligands=pd.DataFrame({"drug_id": drug_ids, "smiles": ["C"] * n_drugs,
                              "lig_group": drug_ids}),
        targets=pd.DataFrame({"target_id": target_ids,
                              "sequence": ["M" * 50] * n_targets,
                              "seq_group": [seq_group[t] for t in target_ids],
                              "n_aa": [50] * n_targets}),
        report={"sparsity": 0.0, "cells_missing": 0},
    )


@pytest.fixture
def ds() -> Dataset:
    return make_dataset()


# --------------------------------------------------------------------------
# warm split
# --------------------------------------------------------------------------

def test_warm_split_keeps_every_held_out_entity_in_train(ds):
    split = warm_split(ds, seed=1)
    train, held = split.train, pd.concat([split.val, split.test])
    assert set(held["drug_id"]) <= set(train["drug_id"])
    assert set(held["target_id"]) <= set(train["target_id"])


def test_warm_split_hits_the_requested_proportions(ds):
    split = warm_split(ds, seed=1, fracs=(0.70, 0.15, 0.15))
    r = split.report
    assert r["frac_train"] == pytest.approx(0.70, abs=0.02)
    assert r["frac_val"] == pytest.approx(0.15, abs=0.02)
    assert r["frac_test"] == pytest.approx(0.15, abs=0.02)


def test_warm_split_partitions_every_pair_exactly_once(ds):
    split = warm_split(ds, seed=1)
    assert len(split.frame) == len(ds.pairs)
    assert len(split.frame.drop_duplicates(["drug_id", "target_id"])) == len(ds.pairs)


def test_warm_split_is_reproducible_and_seed_dependent(ds):
    a, b = warm_split(ds, seed=1), warm_split(ds, seed=1)
    pd.testing.assert_frame_equal(a.frame, b.frame)
    assert not a.frame["fold"].equals(warm_split(ds, seed=2).frame["fold"])


def test_warm_split_refuses_a_train_fraction_too_small_to_cover_entities():
    """A tiny train fold cannot hold one pair per drug and per target."""
    small = make_dataset(n_drugs=20, n_targets=20)
    with pytest.raises(ValueError, match="more than"):
        warm_split(small, fracs=(0.05, 0.475, 0.475))


def test_fracs_must_sum_to_one(ds):
    with pytest.raises(ValueError, match="sum to 1"):
        warm_split(ds, fracs=(0.5, 0.2, 0.2))


# --------------------------------------------------------------------------
# cold-target split
# --------------------------------------------------------------------------

def test_cold_split_gives_held_out_targets_no_training_measurements(ds):
    split = cold_target_split(ds, seed=1)
    held_targets = set(pd.concat([split.val, split.test])["target_id"])
    assert held_targets, "nothing was held out"
    assert not (held_targets & set(split.train["target_id"]))


def test_cold_split_never_separates_sequence_identical_targets(ds):
    """The leakage this split exists to prevent.

    Four targets in the fixture share a sequence group with an earlier target.
    If a group straddled the split, a 'new' target would be one whose identical
    sequence the model had already trained on.
    """
    split = cold_target_split(ds, seed=3)
    by_group = split.frame.groupby("seq_group")["fold"].nunique()
    assert (by_group == 1).all(), "a sequence group landed in more than one fold"


def test_cold_split_keeps_drugs_warm(ds):
    split = cold_target_split(ds, seed=1)
    held = pd.concat([split.val, split.test])
    assert set(held["drug_id"]) <= set(split.train["drug_id"])


def test_cold_split_holds_out_whole_groups_not_whole_ids(ds):
    """Twins travel together, so the held-out target count exceeds the groups."""
    split = cold_target_split(ds, seed=3)
    r = split.report
    assert r["groups_val"] + r["groups_test"] >= 2
    assert r["targets_held_out"] >= r["groups_val"] + r["groups_test"]


def test_cold_split_is_reproducible(ds):
    a, b = cold_target_split(ds, seed=5), cold_target_split(ds, seed=5)
    pd.testing.assert_frame_equal(a.frame, b.frame)


def test_cold_split_refuses_folds_that_would_be_empty():
    tiny = make_dataset(n_drugs=4, n_targets=4, twins=0)
    with pytest.raises(ValueError, match="non-empty folds"):
        cold_target_split(tiny, val_frac=0.01, test_frac=0.01)


# --------------------------------------------------------------------------
# persistence: the group works from one file, so it has to round-trip exactly
# --------------------------------------------------------------------------

def test_split_round_trips_through_disk(ds, tmp_path):
    original = warm_split(ds, seed=4)
    original.save(tmp_path)
    loaded = Split.load("synthetic", "warm", tmp_path)

    pd.testing.assert_frame_equal(original.frame, loaded.frame)
    assert loaded.seed == original.seed
    assert loaded.report["n_train"] == original.report["n_train"]


def test_loading_a_missing_split_says_how_to_build_it(ds, tmp_path):
    with pytest.raises(FileNotFoundError, match="make_splits"):
        Split.load("synthetic", "warm", tmp_path)


def test_fold_accessor_rejects_an_unknown_name(ds):
    with pytest.raises(ValueError, match="fold must be one of"):
        warm_split(ds).fold("holdout")


def test_numeric_looking_ids_survive_the_csv_round_trip(tmp_path):
    """Regression: DAVIS drug ids are decimal strings like ``5291``.

    ``read_csv`` infers int64 for those, and the loaded split then stops
    matching the string keys in ``ligands`` and ``targets``. It fails as an
    empty join rather than an error, so nothing announces it - the content
    models simply find no structure for any DAVIS drug. KIBA's CHEMBL ids are
    not numeric, which is why this hides on one dataset and not the other.
    """
    ds = make_dataset(n_drugs=4, n_targets=6, twins=0)
    ds.pairs["drug_id"] = ds.pairs["drug_id"].str.replace("d", "", regex=False)
    original = warm_split(ds, seed=1)
    original.save(tmp_path)

    loaded = Split.load("synthetic", "warm", tmp_path)
    assert pd.api.types.is_string_dtype(loaded.frame["drug_id"]), (
        "ids came back as a number dtype and will no longer join"
    )
    assert loaded.frame["drug_id"].iloc[0] == str(loaded.frame["drug_id"].iloc[0])
    assert set(loaded.frame["drug_id"]) == set(ds.pairs["drug_id"])
    pd.testing.assert_frame_equal(original.frame, loaded.frame)
