"""Shared figure style for every report figure.

One visual system across Tasks 1-3: same ink, same hairline grid, same blue
sequential ramp, same treatment of missing data. Import ``apply_style`` once at
the top of a script or notebook and use ``save`` to write into ``results/``.

Colour rules this module enforces, so individual figures cannot drift:

* **Sequential magnitude is one hue, light to dark.** No rainbow and no viridis -
  a multi-hue ramp invents category boundaries inside a continuous scale.
* **Missing is not a low value.** Unobserved cells get a desaturated grey that
  cannot be mistaken for the light end of the blue ramp, plus an explicit legend
  swatch. A missing affinity means "never measured", not "does not bind".
* **DAVIS and KIBA never share an axis.** pKd and KIBA score are different
  scales, so they always go in separate panels with their own axis label.

Categorical slots are the first three of a palette validated for colour-vision
deficiency (worst all-pairs CVD dE 9.2, normal-vision dE 24.0). Slot 3 (aqua)
sits below 3:1 contrast on this surface, so anything drawn in it also carries a
direct label.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

#: Categorical slots, in fixed order. Never cycle past these; fold a tail into
#: "other" or facet into small multiples instead.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BLUE, ORANGE, AQUA = SERIES

#: Blue sequential ramp, steps 200-700. The ramp deliberately starts at 200
#: rather than 100 so its light end stays visibly blue against MISSING.
_BLUE_STEPS = [
    "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]

#: Unobserved cells. Desaturated warm grey - a different hue family from the
#: ramp, so "not measured" never reads as "weak binding".
MISSING = "#cfcdc5"

#: Diverging pair for signed quantities (residuals, deltas), neutral midpoint.
DIVERGING = ["#1c5cab", "#f0efec", "#d03b3b"]


def apply_style() -> None:
    """Install the report's matplotlib defaults. Call once per script."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.titlepad": 8,
            "axes.labelsize": 9,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            # Recessive chrome: only the left and bottom rules, hairline weight.
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",  # never dashed: dashes read as a threshold
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "legend.labelcolor": INK_SECONDARY,
            "lines.linewidth": 1.6,
            "lines.markersize": 4,
            "figure.dpi": 110,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
        }
    )


def sequential_cmap(missing: str = MISSING) -> LinearSegmentedColormap:
    """One-hue blue ramp for continuous magnitude, with missing set apart."""
    cmap = LinearSegmentedColormap.from_list("report_blue", _BLUE_STEPS, N=256)
    cmap = cmap.copy()
    cmap.set_bad(missing)
    return cmap


def missing_handle(label: str = "not measured") -> Patch:
    """Legend swatch so the grey is named rather than guessed at."""
    return Patch(facecolor=MISSING, edgecolor="none", label=label)


def save(fig, name: str, subdir: str = "figures", formats=("png", "pdf")) -> list[Path]:
    """Write a figure into ``results/<subdir>/`` as PNG (review) and PDF (report)."""
    out_dir = RESULTS_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths


def save_table(df, name: str, subdir: str = "tables") -> Path:
    """Write a table as CSV for the report, and echo it to stdout."""
    out_dir = RESULTS_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    df.to_csv(path)
    return path
