"""The two figures for the analyses added in the merge.

    fig09_representation   which protein descriptor the cold-target models read
    fig10_keying           what the deduplication convention is worth

Both follow the house style in ``plots``, so they sit beside the other figures
in the report without looking like they came from somewhere else.
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plots as P
from .config import FIGURES, TABLES

LABEL = {
    "target": "composition\n(420)",
    "target_kmer_svd": "3-mer SVD\n(256)",
    "target_esm2_35m": "ESM-2 35M\n(480)",
    "target_esm2_650m": "ESM-2 650M\n(1280)",
    "target_sw_profile": "Smith-Waterman\n(n targets)",
}
ORDER = ["target", "target_kmer_svd", "target_esm2_35m",
         "target_esm2_650m", "target_sw_profile"]


def representation_figure(which=("davis", "kiba")):
    frames = {}
    for ds in which:
        path = TABLES / f"representation_{ds}.csv"
        if path.exists():
            frames[ds] = pd.read_csv(path)
    if not frames:
        return None

    fig, axes = plt.subplots(1, len(frames), figsize=(3.6 * len(frames), 3.0),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (ds, frame) in zip(axes, frames.items()):
        agg = frame.groupby(["representation", "mixup"]).agg(
            mse=("mse", "mean"), sd=("mse", "std")).reset_index()
        present = [r for r in ORDER if r in set(agg["representation"])]
        x = np.arange(len(present))
        width = 0.38
        for offset, (mixup, colour, label) in enumerate(
                [("none", P.PALETTE["accent"], "no mixup"),
                 ("drug", P.PALETTE["muted"], "GBA-Mixup")]):
            sub = agg[agg["mixup"] == mixup].set_index("representation").reindex(present)
            ax.bar(x + (offset - 0.5) * width, sub["mse"], width,
                   yerr=sub["sd"], color=colour, label=label,
                   error_kw=dict(lw=0.8, capsize=2, ecolor=P.PALETTE["ink"]))
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[r] for r in present], fontsize=6.5,
                           rotation=18, ha="right")
        ax.set_ylabel("Cold-target MSE" if ax is axes[0] else "")
        ax.set_title(f"{ds.upper()}: S3, 5 seeds")
        P._despine(ax)
        if ax is axes[0]:
            ax.legend(loc="upper right")
    fig.suptitle("Alignment beats a protein language model on both panels; "
                 "mixup changes nothing", fontsize=9.5, fontweight="bold")
    return P._save(fig, "fig09_representation")


def keying_figure(which=("davis", "kiba")):
    path = TABLES / "keying_ab.csv"
    if not path.exists():
        return None
    ab = pd.read_csv(path)
    strata_path = TABLES / "keying_strata.csv"

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)

    # Left: variance explained under each convention, both datasets.
    agg = ab.groupby(["dataset", "keying", "model"])["r2"].mean().reset_index()
    models = ["B3_bias_only", "M1_item_knn", "M2_biased_mf"]
    names = ["bias only", "item kNN", "biased MF"]
    x = np.arange(len(models))
    width = 0.2
    for i, (ds, keying, colour, hatch) in enumerate([
            ("davis", "sequence", P.PALETTE["accent"], ""),
            ("davis", "id", P.PALETTE["accent"], "//"),
            ("kiba", "sequence", P.PALETTE["muted"], ""),
            ("kiba", "id", P.PALETTE["muted"], "//")]):
        sub = (agg[(agg.dataset == ds) & (agg.keying == keying)]
               .set_index("model").reindex(models))
        axes[0].bar(x + (i - 1.5) * width, sub["r2"], width, color=colour,
                    hatch=hatch, edgecolor="white", linewidth=0.5,
                    label=f"{ds.upper()} {keying}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("Variance explained on held-out pairs")
    axes[0].set_title("Hatched = keyed on source identifier")
    axes[0].legend(ncol=2, fontsize=6.5)
    P._despine(axes[0])

    # Right: where the extra advantage comes from.
    if strata_path.exists():
        st = pd.read_csv(strata_path)
        st = st[(st.dataset == "davis") & (st.model != "B3_bias_only")]
        pivot = st.pivot_table(index="model", columns="stratum",
                               values="gain_over_bias")
        cols = [c for c in ["no twin", "sequence twin in training"] if c in pivot]
        xb = np.arange(len(pivot))
        for j, col in enumerate(cols):
            axes[1].bar(xb + (j - 0.5) * 0.38, pivot[col] * 100, 0.38,
                        color=[P.PALETTE["muted"], P.PALETTE["accent"]][j],
                        label=col)
        axes[1].set_xticks(xb)
        axes[1].set_xticklabels([n for m, n in zip(models, names) if m in pivot.index])
        axes[1].set_ylabel("MSE reduction over bias only (%)")
        axes[1].set_title("DAVIS: the gain sits with the twins")
        axes[1].legend(fontsize=7)
        P._despine(axes[1])
    fig.suptitle("Keeping sequence-identical identifiers apart inflates "
                 "collaborative filtering, and only where twins exist",
                 fontsize=9.5, fontweight="bold")
    return P._save(fig, "fig10_keying")


def main(which=None) -> None:
    which = tuple(which or ("davis", "kiba"))
    P.use_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for fn in (representation_figure, keying_figure):
        path = fn(which)
        if path:
            print(f"  wrote {path.name}")


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")] or None)
