"""Warm (random-pair) and cold-target partitioning."""
import numpy as np


def warm_split(ds, seed=42, fracs=(0.70, 0.15, 0.15)):
    """Random 70/15/15 split over observed pairs.

    Every ligand and target appearing in validation or test is forced to also
    appear in training, as the brief requires, by promoting offending pairs
    back into the training fold. No pair ever lands in two folds.
    """
    rng = np.random.default_rng(seed)
    n = ds.n_pairs
    perm = rng.permutation(n)
    n_tr = int(round(fracs[0] * n))
    n_va = int(round(fracs[1] * n))
    fold = np.empty(n, dtype=np.int8)
    fold[perm[:n_tr]] = 0
    fold[perm[n_tr:n_tr + n_va]] = 1
    fold[perm[n_tr + n_va:]] = 2

    promoted = 0
    for _ in range(10):
        tr = fold == 0
        missing_lig = np.setdiff1d(np.unique(ds.rows[~tr]), np.unique(ds.rows[tr]))
        missing_tgt = np.setdiff1d(np.unique(ds.cols[~tr]), np.unique(ds.cols[tr]))
        if not len(missing_lig) and not len(missing_tgt):
            break
        move = []
        for lig in missing_lig:
            cand = np.flatnonzero((ds.rows == lig) & ~tr)
            if len(cand):
                move.append(rng.choice(cand))
        for tgt in missing_tgt:
            cand = np.flatnonzero((ds.cols == tgt) & ~tr)
            if len(cand):
                move.append(rng.choice(cand))
        if not move:
            break
        move = np.unique(move)
        fold[move] = 0
        promoted += len(move)

    tr = fold == 0
    assert not np.setdiff1d(np.unique(ds.rows[~tr]), np.unique(ds.rows[tr])).size
    assert not np.setdiff1d(np.unique(ds.cols[~tr]), np.unique(ds.cols[tr])).size
    return {
        "fold": fold,
        "train": np.flatnonzero(fold == 0),
        "valid": np.flatnonzero(fold == 1),
        "test": np.flatnonzero(fold == 2),
        "promoted_to_train": int(promoted),
        "seed": seed,
    }


def sequence_groups(ds):
    """Map each target index to a group id shared by targets with identical sequence.

    DAVIS lists 442 target IDs but only 379 distinct sequences (kinase domain
    variants and isoforms). Splitting on raw IDs would place a byte-identical
    sequence in both folds, so cold-target splits must hold out whole groups.
    """
    lookup, groups = {}, np.empty(ds.n_targets, dtype=np.int32)
    for j, s in enumerate(ds.sequences):
        groups[j] = lookup.setdefault(s, len(lookup))
    return groups


def cold_target_split(ds, seed=0, test_frac=0.15, valid_frac=0.15):
    """Hold out whole target sequence-groups: those targets have no training rows.

    Returns pair indices plus the target indices in each fold. Validation
    targets are also cold, so cold-specific hyperparameters are tuned without
    ever touching the test targets.
    """
    rng = np.random.default_rng(seed)
    groups = sequence_groups(ds)
    uniq = np.unique(groups)
    perm = rng.permutation(uniq)
    n_te = max(1, int(round(test_frac * len(uniq))))
    n_va = max(1, int(round(valid_frac * len(uniq))))
    g_test, g_valid = set(perm[:n_te].tolist()), set(perm[n_te:n_te + n_va].tolist())

    tgt_fold = np.zeros(ds.n_targets, dtype=np.int8)
    for j in range(ds.n_targets):
        if groups[j] in g_test:
            tgt_fold[j] = 2
        elif groups[j] in g_valid:
            tgt_fold[j] = 1
    fold = tgt_fold[ds.cols]
    return {
        "fold": fold,
        "train": np.flatnonzero(fold == 0),
        "valid": np.flatnonzero(fold == 1),
        "test": np.flatnonzero(fold == 2),
        "target_fold": tgt_fold,
        "train_targets": np.flatnonzero(tgt_fold == 0),
        "valid_targets": np.flatnonzero(tgt_fold == 1),
        "test_targets": np.flatnonzero(tgt_fold == 2),
        "n_groups": len(uniq),
        "seed": seed,
    }
