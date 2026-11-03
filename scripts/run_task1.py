"""Run the whole of Task 1 end to end.

    python scripts/run_task1.py

Writes every figure into ``results/figures/`` and every table into
``results/tables/``, and prints the numbers to quote in the report. Re-running
is idempotent, so the report and the figures can never disagree.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src import data, eda, viz

STEPS = [
    ("heatmaps", "4. ligand x target heatmaps", eda.figure_heatmap),
    ("score_distributions", "5. observed score distributions", eda.figure_score_distribution),
    ("observations_per_entity", "6a. observations per ligand and target", eda.figure_observations_per_entity),
    ("svd_spectrum", "6b. SVD explained variance", eda.figure_svd_spectrum),
    ("selectivity", "6c. per-drug selectivity", eda.figure_selectivity),
    ("target_correlation", "6d. target profile correlation", eda.figure_target_correlation),
    ("structure_vs_affinity", "6e. structure similarity vs affinity correlation", eda.figure_structure_vs_affinity),
]


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main() -> int:
    pd.set_option("display.width", 200, "display.max_columns", 60,
                  "display.float_format", lambda v: f"{v:,.4g}")
    viz.apply_style()

    banner("1-3. load, clean, deduplicate")
    datasets = data.load_both(dedup_key="id")
    for ds in datasets.values():
        print(f"\n{ds}")
        r = ds.report
        print(f"  observed {r['cells_observed']:,}/{r['cells_total']:,} cells "
              f"({100 * (1 - r['sparsity']):.1f}% filled, {100 * r['sparsity']:.1f}% sparse)")
        print(f"  dropped {r['dropped_nonfinite']} non-finite, "
              f"{r['dropped_nonpositive']} non-positive Kd")
        print(f"  {r['targets_before_dedup']} target ids -> "
              f"{r['unique_sequences']} unique sequences")
        print("  example rows:")
        print(ds.example_rows(3).to_string(index=False))

    counts = data.dedup_table(datasets)
    print("\noriginal vs deduplicated counts:")
    print(counts.to_string())
    viz.save_table(counts, "task1_dedup_counts")

    shared = data.shared_sequences(datasets)
    all_seqs = set().union(*[set(d.targets["sequence"]) for d in datasets.values()])
    print(f"\nsequences in both datasets: {len(shared)}")
    print(f"unique sequences overall  : {len(all_seqs)}  "
          "(embed once, reuse for both in Task 3)")

    banner("deduplication evidence: what averaging by sequence would cost")
    collapse = eda.sequence_collapse_table(datasets["davis"])
    if collapse.empty:
        print("no sequence-identical target groups found")
    else:
        print(collapse.to_string())
        viz.save_table(collapse, "task1_sequence_collapse")
        print(f"\nworst single (drug, group) spread: "
              f"{collapse['max_spread'].max():.2f} pKd")
        print(f"groups affected: {len(collapse)}, "
              f"drug-cells over 1 pKd apart: {int(collapse['drugs_over_1_unit'].sum())}")

    for key, title, fn in STEPS:
        banner(title)
        start = time.perf_counter()
        table = fn(datasets)
        print(table.to_string())
        if not table.empty:
            viz.save_table(table, f"task1_{key}")
        print(f"\n[{time.perf_counter() - start:.1f}s]")

    banner("done")
    figs = sorted((viz.RESULTS_DIR / "figures").glob("*.png"))
    tabs = sorted((viz.RESULTS_DIR / "tables").glob("*.csv"))
    print(f"{len(figs)} figures in results/figures/:")
    for p in figs:
        print(f"  {p.name}")
    print(f"{len(tabs)} tables in results/tables/:")
    for p in tabs:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
