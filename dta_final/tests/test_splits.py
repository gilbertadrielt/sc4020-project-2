"""Splits are where leakage enters, so each regime's claim is asserted.

These tests exist because four independent implementations of this project
produced numbers that could not be compared, and the cause was a splitting
decision nobody had written down. Every property a setting claims is checked
here rather than trusted.
"""
import numpy as np
import pandas as pd
import pytest

from dta import splits as S


def _grid(n_lig=12, n_tgt=9, density=1.0, seed=0):
    rng = np.random.default_rng(seed)
    rows, cols = np.meshgrid(np.arange(n_lig), np.arange(n_tgt), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    keep = rng.random(len(rows)) < density
    rows, cols = rows[keep], cols[keep]
    return pd.DataFrame(
        {"lig_idx": rows, "tgt_idx": cols, "score": rng.normal(size=len(rows))}
    ), n_lig, n_tgt


@pytest.mark.parametrize("setting", S.SETTING_NAMES)
def test_every_pair_is_assigned_exactly_once(setting):
    pairs, nl, nt = _grid()
    assign, _ = S.make_split(pairs, nl, nt, setting=setting, seed=1)
    assert len(assign) == len(pairs)
    assert set(assign.unique()) <= {"train", "val", "test", "unused"}


@pytest.mark.parametrize("setting", S.SETTING_NAMES)
def test_no_pair_appears_in_two_partitions(setting):
    pairs, nl, nt = _grid()
    assign, _ = S.make_split(pairs, nl, nt, setting=setting, seed=2)
    key = pairs["lig_idx"].astype(str) + "|" + pairs["tgt_idx"].astype(str)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (set(key[assign == a]) & set(key[assign == b]))


def test_s1_keeps_every_evaluation_entity_warm():
    """S1 is the warm regime: a cold entity there belongs to S2 or S3."""
    pairs, nl, nt = _grid(density=0.45, seed=3)
    assign, stats = S.make_split(pairs, nl, nt, setting="S1_pair", seed=3)
    train = pairs[assign == "train"]
    ev = pairs[assign.isin(["val", "test"])]
    assert set(ev["lig_idx"]) <= set(train["lig_idx"])
    assert set(ev["tgt_idx"]) <= set(train["tgt_idx"])
    assert "moved_to_train_for_warm_start" in stats


def test_s2_evaluation_drugs_are_never_in_training():
    pairs, nl, nt = _grid()
    assign, _ = S.make_split(pairs, nl, nt, setting="S2_new_drug", seed=4)
    train = set(pairs.loc[assign == "train", "lig_idx"])
    ev = set(pairs.loc[assign.isin(["val", "test"]), "lig_idx"])
    assert not (train & ev)


def test_s3_evaluation_targets_are_never_in_training():
    pairs, nl, nt = _grid()
    assign, _ = S.make_split(pairs, nl, nt, setting="S3_new_target", seed=5)
    train = set(pairs.loc[assign == "train", "tgt_idx"])
    ev = set(pairs.loc[assign.isin(["val", "test"]), "tgt_idx"])
    assert not (train & ev)


def test_s4_holds_out_both_sides_and_discards_half_new_pairs():
    pairs, nl, nt = _grid()
    assign, stats = S.make_split(pairs, nl, nt, setting="S4_both_new", seed=6)
    train = pairs[assign == "train"]
    test = pairs[assign == "test"]
    assert not (set(train["lig_idx"]) & set(test["lig_idx"]))
    assert not (set(train["tgt_idx"]) & set(test["tgt_idx"]))
    # A pair that is new on only one side must be dropped, not silently kept.
    assert stats["n_unused"] > 0


def test_grouping_prevents_near_duplicates_splitting_apart():
    """The strict-cold regime: whole clusters move together.

    Grouping is what stops an identical sequence under two identifiers from
    landing on both sides, which is the specific leak this project is about.
    """
    pairs, nl, nt = _grid(n_tgt=10)
    groups = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4])  # targets 0,1,2 identical
    assign, _ = S.make_split(pairs, nl, nt, setting="S3_new_target", seed=7,
                             tgt_groups=groups)
    fold_of = {}
    for t in range(nt):
        folds = set(assign[pairs["tgt_idx"] == t])
        if folds:
            fold_of[t] = folds.pop()
    for g in np.unique(groups):
        members = [t for t in np.where(groups == g)[0] if t in fold_of]
        assert len({fold_of[t] for t in members}) == 1


def test_check_split_catches_injected_leakage():
    """The guard must fail loudly on a deliberately corrupted split."""
    pairs, nl, nt = _grid()
    assign, _ = S.make_split(pairs, nl, nt, setting="S3_new_target", seed=8)
    bad = assign.copy()
    leaked_target = pairs.loc[bad == "test", "tgt_idx"].iloc[0]
    victim = pairs.index[(pairs["tgt_idx"] == leaked_target) & (bad == "test")][0]
    bad.loc[victim] = "train"
    with pytest.raises(AssertionError):
        S.check_split(pairs, bad, "S3_new_target")


def test_fingerprint_detects_any_change_and_is_stable():
    pairs, nl, nt = _grid()
    a1, _ = S.make_split(pairs, nl, nt, setting="S1_pair", seed=9)
    a2, _ = S.make_split(pairs, nl, nt, setting="S1_pair", seed=9)
    assert S.fingerprint(pairs, a1) == S.fingerprint(pairs, a2)
    a3, _ = S.make_split(pairs, nl, nt, setting="S1_pair", seed=10)
    assert S.fingerprint(pairs, a1) != S.fingerprint(pairs, a3)


def test_same_seed_reproduces_the_partition_exactly():
    pairs, nl, nt = _grid()
    for setting in S.SETTING_NAMES:
        a, _ = S.make_split(pairs, nl, nt, setting=setting, seed=11)
        b, _ = S.make_split(pairs, nl, nt, setting=setting, seed=11)
        assert (a == b).all()


def test_unknown_setting_is_rejected():
    pairs, nl, nt = _grid()
    with pytest.raises(ValueError, match="unknown setting"):
        S.make_split(pairs, nl, nt, setting="S9_nonsense", seed=0)
