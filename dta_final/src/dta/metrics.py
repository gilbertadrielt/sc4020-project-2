"""Every number the project reports, with the awkward cases handled explicitly.

Four families:

**Error.** MSE overall, plus MSE restricted to the measured (uncensored)
subset and to strong binders.  On DAVIS roughly 70% of labels sit on the assay
ceiling, so overall MSE is partly a measure of how well a model reproduces a
detection limit; the restricted versions are where the difficulty lives.

**Concordance index.** The fraction of comparable pairs a model orders
correctly, with half credit for predicted ties.  Computed in O(n log n) from
Kendall's tau-b, because the brute-force definition is O(n^2) and the KIBA
test set alone has around 155 million comparable pairs.  Verified against the
brute force in the test suite.

**Ranking.** Hit@k, MRR and NDCG@k computed per drug and averaged, with the
random-ranking baseline printed beside every one of them.  Without that
baseline the numbers are unreadable: a KIBA drug often has only four or five
held-out candidates, so Hit@5 is nearly free, while a DAVIS drug has dozens.

**Uncertainty.** Bootstrap intervals, resampling drugs rather than pairs,
because pairs within a drug are not independent.

Two conventions matter enough to state in the report.

*Ties in prediction* are broken at random.  A model that scores every
candidate identically cannot rank, and random tie-breaking makes it score
exactly like the random baseline, which is the truthful statement; counting
ties as misses would make it look worse than random, which it is not.

*Ties in truth* mean several candidates share the best score, which is common
on DAVIS where most labels are the same censored value.  A hit counts if any
tied-best candidate is in the top k, and a drug whose candidates are all tied
is skipped and counted, because for that drug the question is meaningless.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.stats import kendalltau

# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


def mse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def masked_mse(y_true, y_pred, mask) -> float:
    """MSE over a subset; NaN when the subset is empty."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("nan")
    return mse(np.asarray(y_true)[mask], np.asarray(y_pred)[mask])


# ---------------------------------------------------------------------------
# Concordance index
# ---------------------------------------------------------------------------


def _tie_pairs(values: np.ndarray) -> int:
    """Number of index pairs sharing a value: sum of t(t-1)/2 over tied groups."""
    _, counts = np.unique(values, return_counts=True)
    return int(np.sum(counts * (counts - 1) // 2))


def concordance_index(y_true, y_pred) -> float:
    """CI = (concordant + 0.5 * predicted ties) / comparable pairs.

    Derived from Kendall's tau-b so it runs in O(n log n).  With
    ``A`` comparable pairs (different true values) and ``B`` pairs with
    different predictions::

        tau_b = (C - D) / sqrt(A * B)
        CI    = (tau_b * sqrt(B / A) + 1) / 2
              = ((C - D) / A + 1) / 2
              = (C + 0.5 * T_pred) / A

    Two edge cases are handled explicitly rather than inherited from scipy:

    * ``A == 0`` means every true value is tied, so there is nothing to order
      and CI is undefined -> NaN.
    * ``B == 0`` means the model predicted one constant, so every comparable
      pair is a tie and CI is exactly 0.5.  scipy returns NaN here, but 0.5 is
      the correct and important answer: it is what a cold-start fallback
      scores, and reporting NaN would hide the failure.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]

    n = len(y_true)
    if n < 2:
        return float("nan")

    n0 = n * (n - 1) // 2
    A = n0 - _tie_pairs(y_true)
    B = n0 - _tie_pairs(y_pred)
    if A == 0:
        return float("nan")
    if B == 0:
        return 0.5

    tau = kendalltau(y_true, y_pred, variant="b").statistic
    if not np.isfinite(tau):
        return float("nan")
    return float((tau * np.sqrt(B / A) + 1.0) / 2.0)


def concordance_index_bruteforce(y_true, y_pred) -> float:
    """The definition, counted pair by pair. O(n^2); for tests only."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    concordant = ties = comparable = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            comparable += 1
            hi, lo = (i, j) if y_true[i] > y_true[j] else (j, i)
            if y_pred[hi] > y_pred[lo]:
                concordant += 1
            elif y_pred[hi] == y_pred[lo]:
                ties += 1
    if comparable == 0:
        return float("nan")
    return (concordant + 0.5 * ties) / comparable


def grouped_concordance_index(group_ids, y_true, y_pred, min_size: int = 2) -> dict:
    """CI computed inside each group and averaged over groups.

    Per-*ligand* CI (group by drug) ranks a drug's candidate targets, which is
    the quantity that matches the use case and the one that exposes cold-start
    failure: a model with no information about unseen targets scores every
    candidate alike and lands on exactly 0.5.

    Per-*target* CI (group by protein) does not expose that, because the drug
    biases still order drugs sensibly within any one target, so a useless
    model can score around 0.8 there.  Both are reported.
    """
    group_ids = np.asarray(group_ids)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    scores, skipped = [], 0
    for g in np.unique(group_ids):
        m = group_ids == g
        if m.sum() < min_size:
            skipped += 1
            continue
        c = concordance_index(y_true[m], y_pred[m])
        if np.isnan(c):  # every true value in this group is tied
            skipped += 1
            continue
        scores.append(c)
    return {
        "mean_ci": float(np.mean(scores)) if scores else float("nan"),
        "n_groups_scored": len(scores),
        "n_groups_skipped": int(skipped),
    }


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _rank_desc(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Positions 0..n-1 of each item when sorted best first, ties at random."""
    jitter = rng.permutation(len(scores))
    order = np.lexsort((jitter, -scores))
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(len(scores))
    return ranks


def _tie_block(y_pred: np.ndarray, is_best: np.ndarray) -> tuple[int, int, int]:
    """Where the best true item sits in the predicted ordering, counting ties.

    Returns ``(b, t, m)``: ``b`` items predicted strictly above the tie block the
    best-ranked correct item falls in, ``t`` items inside that block, and ``m`` of
    those that are themselves correct.
    """
    top = float(np.max(y_pred[is_best]))
    b = int(np.sum(y_pred > top))
    t = int(np.sum(y_pred == top))
    m = int(np.sum(is_best & (y_pred == top)))
    return b, t, m


def _p_rank_above(b: int, t: int, m: int, j: int) -> float:
    """P(best correct item lands strictly below position b + j) under random ties.

    Within the tie block the m correct items take m of the t slots uniformly, so
    the probability none of them is in the first j slots is the product of
    (t - j - i) / (t - i) over i < m.
    """
    if j <= 0:
        return 1.0
    if j > t - m:
        return 0.0
    p = 1.0
    for i in range(m):
        p *= (t - j - i) / (t - i)
    return float(p)


def _expected_hit_at_k(b: int, t: int, m: int, k: int) -> float:
    """E[Hit@k] over random tie-breaking, exactly rather than by one draw.

    A model that ties every candidate gets its true expectation instead of a
    sample from it, so the degenerate baselines land exactly on the random
    row rather than scattering around it with a standard deviation as large as
    the value itself.
    """
    return 1.0 - _p_rank_above(b, t, m, min(max(k - b, 0), t))


def _expected_reciprocal_rank(b: int, t: int, m: int) -> float:
    """E[1 / rank] over random tie-breaking, exactly."""
    total = 0.0
    previous = 1.0
    for j in range(1, t - m + 2):
        current = _p_rank_above(b, t, m, j)
        total += (previous - current) / (b + j)
        previous = current
    return float(total)


def _random_hit_at_k(n: int, n_best: int, k: int) -> float:
    """Exact probability a random top-k contains at least one of n_best items.

    1 - C(n - m, k) / C(n, k), computed as a product to avoid overflow.
    """
    if k >= n:
        return 1.0
    p_miss = 1.0
    for i in range(k):
        p_miss *= (n - n_best - i) / (n - i)
        if p_miss <= 0:
            return 1.0
    return float(1.0 - p_miss)


@dataclass
class RankingResult:
    hit_at_1: float = float("nan")
    hit_at_5: float = float("nan")
    hit_at_10: float = float("nan")
    mrr: float = float("nan")
    ndcg_at_5: float = float("nan")
    random_hit_at_1: float = float("nan")
    random_hit_at_5: float = float("nan")
    random_hit_at_10: float = float("nan")
    random_mrr: float = float("nan")
    n_groups_scored: int = 0
    n_groups_skipped: int = 0
    mean_candidates: float = float("nan")
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("extra")
        return d


def ranking_metrics(
    group_ids,
    y_true,
    y_pred,
    ks=(1, 5, 10),
    seed: int = 2026,
    n_random_repeats: int = 32,
) -> RankingResult:
    """Hit@k, MRR and NDCG@k per group, averaged, with random baselines.

    A group is one drug and its candidate targets.  Groups with fewer than two
    candidates, or whose candidates all share the same true score, are skipped
    and counted: for those the question "is the best one near the top?" has no
    answer.
    """
    group_ids = np.asarray(group_ids)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rng = np.random.default_rng(seed)

    hits = {k: [] for k in ks}
    rnd_hits = {k: [] for k in ks}
    rr, rnd_rr, ndcg, sizes = [], [], [], []
    skipped = 0

    for g in np.unique(group_ids):
        m = group_ids == g
        t, p = y_true[m], y_pred[m]
        n = len(t)
        if n < 2 or np.ptp(t) == 0:
            skipped += 1
            continue

        best = t.max()
        is_best = t == best
        n_best = int(is_best.sum())
        sizes.append(n)

        # Hit@k and MRR as exact expectations over the predicted tie block,
        # not as one random draw from it.
        n_above, n_tied, n_tied_best = _tie_block(p, is_best)
        for k in ks:
            hits[k].append(_expected_hit_at_k(n_above, n_tied, n_tied_best, k))
            rnd_hits[k].append(_random_hit_at_k(n, n_best, k))
        rr.append(_expected_reciprocal_rank(n_above, n_tied, n_tied_best))

        # A random ranking is the all-tied case: one block of n holding n_best.
        rnd_rr.append(_expected_reciprocal_rank(0, n, n_best))

        ranks = _rank_desc(p, rng)

        # NDCG with linear gain on min-shifted relevance, so gains are never
        # negative on a scale (like pKd) that does not start at zero.
        rel = t - t.min()
        k5 = min(5, n)
        order = np.argsort(ranks)
        disc = 1.0 / np.log2(np.arange(2, k5 + 2))
        dcg = float(np.sum(rel[order][:k5] * disc))
        idcg = float(np.sum(np.sort(rel)[::-1][:k5] * disc))
        ndcg.append(dcg / idcg if idcg > 0 else np.nan)

    if not sizes:
        return RankingResult(n_groups_skipped=skipped)

    out = RankingResult(
        hit_at_1=float(np.mean(hits[1])) if 1 in hits else float("nan"),
        hit_at_5=float(np.mean(hits[5])) if 5 in hits else float("nan"),
        hit_at_10=float(np.mean(hits[10])) if 10 in hits else float("nan"),
        mrr=float(np.mean(rr)),
        ndcg_at_5=float(np.nanmean(ndcg)),
        random_hit_at_1=float(np.mean(rnd_hits[1])) if 1 in rnd_hits else float("nan"),
        random_hit_at_5=float(np.mean(rnd_hits[5])) if 5 in rnd_hits else float("nan"),
        random_hit_at_10=float(np.mean(rnd_hits[10])) if 10 in rnd_hits else float("nan"),
        random_mrr=float(np.mean(rnd_rr)),
        n_groups_scored=len(sizes),
        n_groups_skipped=int(skipped),
        mean_candidates=float(np.mean(sizes)),
    )
    return out


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


def bootstrap_interval(
    group_ids,
    y_true,
    y_pred,
    statistic,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 2026,
) -> tuple[float, float]:
    """Percentile interval, resampling whole groups.

    Pairs sharing a drug are not independent, so resampling pairs would give
    an interval that is too narrow.  Groups (drugs) are the resampling unit.
    """
    group_ids = np.asarray(group_ids)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    groups = np.unique(group_ids)
    index_of = {g: np.where(group_ids == g)[0] for g in groups}
    rng = np.random.default_rng(seed)

    vals = []
    for _ in range(n_boot):
        drawn = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([index_of[g] for g in drawn])
        try:
            v = statistic(y_true[idx], y_pred[idx])
        except Exception:
            v = np.nan
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


# ---------------------------------------------------------------------------
# One call for everything
# ---------------------------------------------------------------------------


def evaluate(
    pairs,
    y_pred,
    dataset: str,
    strong_threshold: float | None = None,
    seed: int = 2026,
    bootstrap: bool = False,
    n_boot: int = 1000,
) -> dict:
    """Full metric set for one model on one evaluation fold.

    ``pairs`` is the evaluation slice of the pair table, so the censored and
    strong-binder masks come from the data rather than being recomputed.
    """
    from .config import STRONG_THRESHOLD

    y_true = pairs["score"].to_numpy(dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lig = pairs["lig_idx"].to_numpy()
    tgt = pairs["tgt_idx"].to_numpy()
    thr = strong_threshold if strong_threshold is not None else STRONG_THRESHOLD[dataset]
    censored = pairs["censored"].to_numpy(dtype=bool) if "censored" in pairs else np.zeros(len(pairs), bool)

    out = {
        "n_pairs": int(len(pairs)),
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mse_strong": masked_mse(y_true, y_pred, y_true >= thr),
        "mse_measured": masked_mse(y_true, y_pred, ~censored),
        "ci": concordance_index(y_true, y_pred),
        "ci_measured": (
            concordance_index(y_true[~censored], y_pred[~censored]) if (~censored).any() else float("nan")
        ),
    }

    per_lig = grouped_concordance_index(lig, y_true, y_pred)
    out["ci_per_ligand"] = per_lig["mean_ci"]
    out["n_ligands_scored"] = per_lig["n_groups_scored"]
    per_tgt = grouped_concordance_index(tgt, y_true, y_pred)
    out["ci_per_target"] = per_tgt["mean_ci"]

    rank = ranking_metrics(lig, y_true, y_pred, seed=seed)
    out.update(rank.as_dict())

    if bootstrap:
        for key, fn in (("mse", mse), ("ci", concordance_index)):
            lo, hi = bootstrap_interval(lig, y_true, y_pred, fn, n_boot=n_boot, seed=seed)
            out[f"{key}_lo"], out[f"{key}_hi"] = lo, hi

    return out
