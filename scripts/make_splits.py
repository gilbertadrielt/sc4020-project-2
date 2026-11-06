"""Build and save the Task 2 and Task 3 partitions.

    python scripts/make_splits.py

Writes ``data/processed/<dataset>_<kind>.csv`` plus a JSON sidecar, for both
datasets and both split kinds. Everything downstream loads these files rather
than re-splitting, so every experiment in the group scores the same pairs.

``data/`` is gitignored, so the files are not shared through the repo - the
seed is. Re-running this script anywhere reproduces the same partitions
byte for byte; the saved file just means nobody has to take that on trust
halfway through an experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src import data, split, viz


def main() -> int:
    pd.set_option("display.width", 200, "display.max_columns", 60,
                  "display.float_format", lambda v: f"{v:,.4g}")

    datasets = data.load_both(dedup_key="id")
    built = {}
    for name, ds in datasets.items():
        built[name] = split.build(ds, seed=split.DEFAULT_SEED)
        for kind, s in built[name].items():
            path = s.save()
            print(f"{s!r}\n  -> {path}")

    table = split.summary(built)
    print("\nsplit summary:")
    print(table.to_string())
    viz.save_table(table, "task2_splits")

    print("\ncold-target folds hold out whole sequence groups, so a test "
          "target has no training measurements and no identical twin in train.")
    for name, by_kind in built.items():
        r = by_kind["cold_target"].report
        print(f"  {name.upper()}: {r['groups_test']} of {r['n_seq_groups']} groups "
              f"({r['targets_held_out']} target ids) held out, "
              f"{r['pairs_dropped_cold_drug']} pairs dropped for cold drugs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
