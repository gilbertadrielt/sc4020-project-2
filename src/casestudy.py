"""Case studies: what the recommender actually hands back for one drug.

A results table says a model reaches CI 0.89. It does not say whether the
targets it puts at the top are the right ones, and that is the question the
brief's case study asks. For three test drugs per dataset this module produces
the ranked list the model would recommend, the measured affinity beside each
prediction, and a verdict on whether the drug's truly strongest held-out target
made the cut.

The predictions come from the reloaded ``.npz`` checkpoint, not from the model
still in memory, so the case study doubles as evidence that the saved artefact
is the one that produced the numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import TIE_BREAK_SEED, concordance_index, mse

#: The ids the brief suggests, used when the split puts them in test.
SUGGESTED = {
    "davis": ["10127622", "11338033", "44259"],
    "kiba": ["CHEMBL1967538", "CHEMBL2004544", "CHEMBL565612"],
}


def pick_drugs(
    test: pd.DataFrame,
    train: pd.DataFrame,
    preferred=(),
    n: int = 3,
    min_candidates: int = 5,
) -> tuple[list[str], str]:
    """Choose the case-study drugs, preferring the brief's ids when usable.

    A suggested id is only usable if the split actually put it in test with
    enough held-out targets to rank. When they are not all available, the
    replacements are chosen to span the conditions the report has to discuss
    rather than at random: the most promiscuous drug in test, the most
    selective, and the one with the least training history.

    Returns the ids and a sentence naming how they were chosen, so the report
    can state it rather than leave the choice looking arbitrary.
    """
    candidates = test.groupby("drug_id").size()
    eligible = set(candidates[candidates >= min_candidates].index)

    usable = [d for d in preferred if d in eligible]
    if len(usable) >= n:
        return usable[:n], f"the {n} ids suggested in the brief, all present in test"

    # Rank the test drugs on the two axes Task 1 found to matter, plus history.
    strength = test.groupby("drug_id")["score"].mean()
    history = train.groupby("drug_id").size()
    pool = [d for d in eligible if d not in usable]
    if not pool:
        return usable, "every suggested id that the split placed in test"

    picks = list(usable)
    for label, series, biggest in (
        ("most promiscuous", strength, True),
        ("most selective", strength, False),
        ("least training history", history.reindex(pool).fillna(0), False),
    ):
        if len(picks) >= n:
            break
        ordered = series.reindex(pool).dropna().sort_values(ascending=not biggest)
        for drug in ordered.index:
            if drug not in picks:
                picks.append(drug)
                break

    how = (f"{len(usable)} of the brief's suggested ids were in test; the rest "
           "span the most promiscuous, most selective and least-measured drugs")
    return picks[:n], how


def ranked_candidates(
    model,
    test: pd.DataFrame,
    drug_id: str,
    k: int = 5,
    seed: int = TIE_BREAK_SEED,
) -> pd.DataFrame:
    """The drug's held-out targets, ordered as the model would recommend them.

    ``verdict`` marks the target whose *measured* affinity is the highest:
    "correct" is the one the recommender should have put first, and every other
    row is "wrong" in the brief's sense, however plausible its prediction.
    """
    rows = test[test["drug_id"] == drug_id].copy()
    if rows.empty:
        raise ValueError(f"{drug_id!r} has no pairs in this fold")

    rows["pred"] = model.predict(rows)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(rows)), -rows["pred"].to_numpy()))
    rows = rows.iloc[order].reset_index(drop=True)
    rows["rank"] = np.arange(1, len(rows) + 1)

    best = rows["score"] == rows["score"].max()
    rows["verdict"] = np.where(best, "correct", "wrong")
    rows["error"] = rows["pred"] - rows["score"]

    columns = ["rank", "target_id", "pred", "score", "error", "verdict"]
    return rows[columns].head(k)


def summarise(model, test: pd.DataFrame, drug_ids, k: int = 5) -> pd.DataFrame:
    """One row per case-study drug: how well the model served that query."""
    rows = []
    for drug_id in drug_ids:
        frame = test[test["drug_id"] == drug_id]
        pred = model.predict(frame)
        truth = frame["score"].to_numpy(float)

        rng = np.random.default_rng(TIE_BREAK_SEED)
        order = np.lexsort((rng.random(len(pred)), -pred))
        rank_of = np.empty(len(pred), int)
        rank_of[order] = np.arange(1, len(pred) + 1)
        best_rank = int(rank_of[truth == truth.max()].min())

        rows.append({
            "drug_id": drug_id,
            "candidates": len(frame),
            "mse": mse(truth, pred),
            "ci": concordance_index(truth, pred),
            "best_target_rank": best_rank,
            f"hit@{k}": int(best_rank <= k),
            "true_best_score": float(truth.max()),
            "predicted_for_it": float(pred[np.argmax(truth)]),
        })
    return pd.DataFrame(rows).set_index("drug_id")


def report(model, test: pd.DataFrame, train: pd.DataFrame, dataset: str,
           k: int = 5, min_candidates: int = 5):
    """Everything the report needs: the chosen drugs, their tables, the summary."""
    drugs, how = pick_drugs(test, train, preferred=SUGGESTED.get(dataset, ()),
                            min_candidates=min_candidates)
    details = {d: ranked_candidates(model, test, d, k=k) for d in drugs}
    return drugs, how, details, summarise(model, test, drugs, k=k)
