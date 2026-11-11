"""Shared measured-pair regression, global CI and held-out per-drug ranking."""
import math

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _fenwick_ci(y, ranks, size):
    tree = np.zeros(size + 1, dtype=np.int64)
    credit, comparable, seen = 0., 0, 0
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and y[end] == y[start]:
            end += 1
        # Query an entire true-score tie group before inserting any of it.
        for i in range(start, end):
            rank = ranks[i]
            below, upto = 0, 0
            j = rank - 1
            while j > 0:
                below += tree[j]
                j -= j & -j
            j = rank
            while j > 0:
                upto += tree[j]
                j -= j & -j
            credit += below + .5 * (upto - below)
            comparable += seen
        for i in range(start, end):
            j = ranks[i]
            while j <= size:
                tree[j] += 1
                j += j & -j
        seen += end - start
        start = end
    return credit / comparable if comparable else np.nan


def concordance_index(y, prediction):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != prediction.shape or y.ndim != 1 or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError('CI expects equally shaped finite vectors')
    order = np.argsort(y, kind='stable')
    _, ranks = np.unique(prediction, return_inverse=True)
    return float(_fenwick_ci(y[order], (ranks[order] + 1).astype(np.int64), int(ranks.max()) + 1)) if len(y) else float('nan')


def _hit_probability(scores, relevant, k):
    """Expected Hit@k under uniform random ordering inside predicted-score ties."""
    k = min(k, len(scores))
    boundary = np.sort(scores)[-k]
    above, tied = scores > boundary, scores == boundary
    if relevant[above].any():
        return 1.
    slots = k - int(above.sum())
    total, good = int(tied.sum()), int(relevant[tied].sum())
    return 1. - (math.comb(total - good, slots) / math.comb(total, slots) if total - good >= slots else 0.)


def expected_reciprocal_rank(scores, relevant):
    """Expected reciprocal rank of first relevant item, averaging score ties."""
    best_score = np.max(scores[relevant])
    above = int(np.sum(scores > best_score))
    tied = scores == best_score
    m, r = int(tied.sum()), int(relevant[tied].sum())
    probability, result = r / m, 0.
    for j in range(1, m - r + 2):
        result += probability / (above + j)
        if j < m - r + 1:
            probability *= (m - j - r + 1) / (m - j)
    return result


def ranking_metrics(frame, prediction):
    work = frame[['drug_id', 'target_id', 'affinity']].copy()
    p = np.asarray(prediction, dtype=float)
    if p.shape != (len(work),) or not np.isfinite(p).all():
        raise ValueError('Ranking requires finite predictions matching frame length')
    work['prediction'] = p
    rows = []
    for drug, group in work.groupby('drug_id', sort=True):
        y, p = group.affinity.to_numpy(), group.prediction.to_numpy()
        best = y == y.max()
        ci = concordance_index(y, p)
        rows.append({'drug_id': int(drug), 'candidates': len(y), 'best_ties': int(best.sum()),
                     'hit1': _hit_probability(p, best, 1), 'hit5': _hit_probability(p, best, 5),
                     'hit10': _hit_probability(p, best, 10), 'mrr': expected_reciprocal_rank(p, best),
                     'ci': ci, 'eligible': len(y) >= 2 and not best.all()})
    per_drug = pd.DataFrame(rows)
    if not len(per_drug):
        return {'ranking_eligible_drugs': 0, 'hit1': None, 'hit5': None, 'hit10': None, 'mrr': None, 'ci_per_ligand': None}, per_drug
    eligible = per_drug[per_drug.eligible]
    def avg(values):
        return float(values.mean()) if len(values) and values.notna().any() else None
    summary = {'hit1': avg(eligible.hit1), 'hit5': avg(eligible.loc[eligible.candidates >= 5, 'hit5']),
               'hit10': avg(eligible.loc[eligible.candidates >= 10, 'hit10']), 'mrr': avg(eligible.mrr),
               'ci_per_ligand': avg(eligible.ci),
               'ranking_eligible_drugs': len(eligible),
               'ranking_singleton_drugs': int((per_drug.candidates == 1).sum()),
               'ranking_all_tied_drugs': int(((per_drug.best_ties == per_drug.candidates) & (per_drug.candidates >= 2)).sum()),
               'hit5_gt5': avg(eligible.loc[eligible.candidates > 5, 'hit5']),
               'hit1_gt5': avg(eligible.loc[eligible.candidates > 5, 'hit1']),
               'ranking_gt5_drugs': int((eligible.candidates > 5).sum()),
               'ranking_hit5_drugs': int((eligible.candidates >= 5).sum()),
               'ranking_hit10_drugs': int((eligible.candidates >= 10).sum()),
               'trivial_hit5_drugs': int((eligible.candidates <= 5).sum()),
               'candidate_min': int(per_drug.candidates.min()), 'candidate_median': float(per_drug.candidates.median()),
               'candidate_max': int(per_drug.candidates.max())}
    return summary, per_drug


def evaluate(frame, prediction, secondary=True):
    y, p = frame.affinity.to_numpy(), np.asarray(prediction, dtype=float)
    if y.shape != p.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Evaluation requires equally shaped nonempty finite vectors')
    ci = concordance_index(y, p)
    ranking, _ = ranking_metrics(frame, p)
    result = {'pairs': len(y), 'mse': float(np.mean((y - p) ** 2)),
              'rmse': float(np.sqrt(np.mean((y - p) ** 2))), 'mae': float(np.mean(np.abs(y - p))),
              'ci_global': ci if np.isfinite(ci) else None, **ranking}
    if secondary:
        swapped = frame.rename(columns={'drug_id': 'target_id', 'target_id': 'drug_id'})
        target_ranking, _ = ranking_metrics(swapped, p)
        for key in ['hit1', 'hit5', 'hit10', 'mrr', 'ranking_eligible_drugs']:
            result['per_target_' + key] = target_ranking[key]
        result['ci_per_target'] = target_ranking['ci_per_ligand']
    return result


def random_ranking_reference(frame):
    # All predictions tied is the exact expectation of a uniform random ranking.
    return ranking_metrics(frame, np.zeros(len(frame)))[0]
