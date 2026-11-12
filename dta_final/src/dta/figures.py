"""Draw the result figures from the tables the sweep already wrote.

Separating drawing from fitting means a caption, a colour or a layout can
change in seconds without retraining anything, and it guarantees the figures
and the tables come from the same numbers.

Run ``dta.protocol`` and ``dta.run_analysis`` first; this module only reads.
"""
from __future__ import annotations

import sys

import pandas as pd

from . import plots as P
from .config import DATASETS, RESULTS, TABLES


def _read(path):
    return pd.read_csv(path) if path.exists() else None


def main(which=None) -> None:
    which = which or list(DATASETS)
    P.use_style()
    made = []

    summary = _read(TABLES / "summary_all.csv")
    if summary is None:
        print("no summary_all.csv; run `python -m dta.protocol` first")
        return

    for ds in which:
        if (summary.dataset == ds).any():
            made.append(P.four_settings(summary, ds))

    if (summary.setting == "S1_pair").any():
        made.append(P.warm_comparison(summary[summary.dataset.isin(which)]))
        fig = P.censored_effect(summary[summary.dataset.isin(which)])
        if fig is not None:
            made.append(fig)

    tuning = _read(TABLES / "tuning_all.csv")
    if tuning is not None:
        made.append(P.tuning_curve(tuning[tuning.dataset.isin(which)]))

    buckets = {ds: b for ds in which
               if (b := _read(TABLES / f"buckets_{ds}.csv")) is not None and len(b)}
    if buckets:
        made.append(P.similarity_buckets(buckets))

    for p in made:
        print(f"  wrote {p.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")] or None)
