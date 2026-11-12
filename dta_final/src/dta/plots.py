"""Every figure, drawn from the saved tables.

House rules, applied throughout so the set reads as one piece of work:

* Magnitude uses a single-hue sequential ramp, one hue per dataset, never a
  rainbow.  Unmeasured cells are a flat grey that is not on the ramp and
  always carries its own legend entry, because an unmeasured pair is not a
  zero.
* Identity uses a fixed categorical order, assigned in order and never
  cycled.  Where more than five things are compared it becomes small
  multiples or a single-hue bar chart with the model of interest emphasised.
* Text wears ink, never a series colour; grid lines stay recessive.
* One y axis per panel.  Two measures on different scales become two panels.
* Captions state the finding, not the variable, so a reader who only looks
  at the figures still leaves with the result.

Each function writes a vector PDF for the report and a PNG for slides.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from scipy.stats import gaussian_kde

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .config import CATEGORICAL, FIG_DPI, FIGURES, PALETTE, SCORE_LABEL, SEQUENTIAL  # noqa: E402

SHORT = {"davis": "pKd", "kiba": "KIBA score"}
SETTING_TITLE = {
    "S1_pair": "S1  random pairs",
    "S2_new_drug": "S2  unseen drugs",
    "S3_new_target": "S3  unseen targets",
    "S4_both_new": "S4  both unseen",
}


def use_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": FIG_DPI, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.labelsize": 8.5,
        "axes.edgecolor": PALETTE["grid"], "axes.labelcolor": PALETTE["ink"],
        "axes.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": PALETTE["grid"], "grid.linewidth": 0.6, "grid.alpha": 0.9,
        "xtick.color": PALETTE["muted"], "ytick.color": PALETTE["muted"],
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.frameon": False, "legend.fontsize": 7.5,
        "text.color": PALETTE["ink"], "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _despine(ax, keep=("left", "bottom")) -> None:
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def _save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / f"{name}.pdf"
    fig.savefig(pdf)
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)
    return pdf


def _pretty(model: str) -> str:
    return model.split("_", 1)[-1].replace("_", " ")


def _order(M: np.ndarray, max_n: int = 1500) -> np.ndarray:
    """Cluster order, so structure shows instead of file order."""
    n = M.shape[0]
    if n < 3:
        return np.arange(n)
    if n > max_n:
        return np.argsort(-np.nan_to_num(M, nan=-np.inf).mean(axis=1))
    X = np.nan_to_num(M, nan=float(np.nanmean(M)))
    d = np.nan_to_num(pdist(X, metric="correlation"), nan=1.0, posinf=1.0, neginf=1.0)
    return leaves_list(linkage(d, method="average"))


# ---------------------------------------------------------------------------
# Task 1
# ---------------------------------------------------------------------------


def heatmaps(matrices: dict, name="fig01_heatmaps") -> Path:
    use_style()
    fig, axes = plt.subplots(1, len(matrices), figsize=(7.2, 3.3))
    axes = np.atleast_1d(axes)
    for ax, (ds, M) in zip(axes, matrices.items()):
        M = np.asarray(M, dtype=float)
        Ms = M[np.ix_(_order(M), _order(M.T))]
        cmap = plt.get_cmap(SEQUENTIAL[ds]).copy()
        cmap.set_bad(PALETTE["missing"])
        im = ax.imshow(np.ma.masked_invalid(Ms), aspect="auto", cmap=cmap,
                       interpolation="nearest", rasterized=True)
        missing = float(np.mean(~np.isfinite(M)))
        ax.set_title(f"{ds.upper()}  ({M.shape[0]} drugs x {M.shape[1]} targets)")
        ax.set_xlabel("Targets, clustered")
        ax.set_ylabel("Drugs, clustered")
        ax.grid(False); ax.set_xticks([]); ax.set_yticks([]); _despine(ax, keep=())
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.set_label(SHORT[ds], fontsize=7.5); cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7)
        if missing > 0:
            ax.legend(handles=[Patch(facecolor=PALETTE["missing"], edgecolor=PALETTE["grid"],
                                     label=f"never measured ({missing:.1%})")],
                      loc="upper center", bbox_to_anchor=(0.5, -0.09), handlelength=1.2)
    fig.suptitle("DAVIS is fully measured; three quarters of KIBA was never tested",
                 fontsize=10, fontweight="bold", y=1.02)
    fig.subplots_adjust(wspace=0.42)
    return _save(fig, name)


def distributions(scores: dict, censored: dict | None = None, name="fig02_distributions") -> Path:
    """Score distributions, with the censored spike drawn as its own bar."""
    use_style()
    censored = censored or {}
    fig, axes = plt.subplots(1, len(scores), figsize=(7.2, 2.9))
    axes = np.atleast_1d(axes)
    for ax, (ds, y) in zip(axes, scores.items()):
        y = np.asarray(y, dtype=float); y = y[np.isfinite(y)]
        colour = PALETTE[ds]
        cmask = np.asarray(censored.get(ds, np.zeros(len(y), bool)), dtype=bool)
        free = y[~cmask] if cmask.any() else y
        n_all = len(y)

        # Everything on one density scale: that of the full sample. Densities
        # over the measured subset alone would make the censored spike look
        # like a footnote instead of most of the dataset.
        bins = np.histogram_bin_edges(free, bins=55)
        width = float(bins[1] - bins[0])
        ax.hist(free, bins=bins, weights=np.full(len(free), 1.0 / (n_all * width)),
                color=to_rgba(colour, 0.30), edgecolor="none", label="measured values")
        if len(free) > 10 and np.ptp(free) > 0:
            grid = np.linspace(free.min(), free.max(), 400)
            ax.plot(grid, gaussian_kde(free)(grid) * len(free) / n_all,
                    color=colour, lw=1.7, label="KDE, measured only")
        if cmask.any():
            frac = float(cmask.mean()); x0 = float(y[cmask][0])
            # Drawn to scale this bar is ~50x taller than the rest, so the
            # axis is clipped to the measured density and the bar is labelled
            # as running off the top. Clipping the view, not the data.
            ax.bar([x0], [frac / width], width=width, align="edge",
                   color=to_rgba(PALETTE["muted"], 0.55), edgecolor=PALETTE["ink"],
                   linewidth=0.7, hatch="///", zorder=3,
                   label=f"censored at the ceiling ({frac:.0%}, off scale)")
            top = float(np.nanmax(gaussian_kde(free)(np.linspace(free.min(), free.max(), 200))
                                  * len(free) / n_all)) if len(free) > 10 else 1.0
            hist_top = float((np.histogram(free, bins=bins)[0] / (n_all * width)).max())
            ax.set_ylim(0, max(top, hist_top) * 1.45)
            ax.annotate(f"{frac:.0%} of all labels\nare this one value",
                        xy=(x0 + width, ax.get_ylim()[1] * 0.55), xytext=(20, 0),
                        textcoords="offset points", fontsize=7.5, color=PALETTE["ink"],
                        va="center",
                        arrowprops=dict(arrowstyle="->", color=PALETTE["muted"], lw=0.9))
        ax.set_title(ds.upper()); ax.set_xlabel(SCORE_LABEL[ds])
        ax.set_ylabel("Density (full sample)")
        _despine(ax); ax.legend(loc="upper right", borderpad=0.2)
    fig.suptitle("Most DAVIS labels are a detection limit, not a measurement",
                 fontsize=10, fontweight="bold", y=1.06)
    return _save(fig, name)


def structure(variance: dict, spectra: dict, name="fig03_structure") -> Path:
    """Why factorisation is the right family: headroom, and low rank."""
    use_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    order = ["ligand identity alone", "target identity alone", "additive (mu + b + c)"]
    xs = np.arange(len(order)); width = 0.36
    for i, (ds, df) in enumerate(variance.items()):
        d = df.set_index("component")["variance_explained"]
        vals = [float(d.get(k, np.nan)) for k in order]
        bars = ax1.bar(xs + (i - 0.5) * width, vals, width, color=PALETTE[ds],
                       label=ds.upper(), edgecolor="none")
        # Label only the additive bar: the number that frames Task 2.
        ax1.bar_label(bars, labels=["", "", f"{vals[-1]:.0%}"], fontsize=7.5,
                      padding=2, color=PALETTE["ink"])
    ax1.set_xticks(xs); ax1.set_xticklabels(["drug\nonly", "target\nonly", "additive\nmodel"])
    ax1.set_ylabel("Share of score variance explained"); ax1.set_ylim(0, 1)
    ax1.set_title("What bias models can reach"); ax1.legend(loc="upper left"); _despine(ax1)

    markers = {"davis": "o", "kiba": "s"}
    offsets = {"davis": (14, 26), "kiba": (26, -22)}
    for ds, sp in spectra.items():
        ax2.plot(sp["component"], sp["cumulative_energy"], color=PALETTE[ds], lw=1.6,
                 marker=markers.get(ds, "o"), ms=3.5, label=ds.upper())
        five = sp.loc[sp["component"] == 5, "cumulative_energy"]
        if len(five):
            ax2.annotate(f"{ds.upper()}: {float(five.iloc[0]):.0%} in 5 components",
                         xy=(5, float(five.iloc[0])), xytext=offsets.get(ds, (8, 8)),
                         textcoords="offset points", fontsize=7.5, color=PALETTE[ds],
                         arrowprops=dict(arrowstyle="-", color=PALETTE[ds], lw=0.7))
    ax2.set_xlabel("Number of components"); ax2.set_ylabel("Cumulative share of energy")
    ax2.set_ylim(0, 1.02); ax2.set_title("The matrix is close to low rank")
    ax2.legend(loc="lower right"); _despine(ax2)
    fig.suptitle("Half to two thirds of the signal is drug x target interaction, and it is low rank",
                 fontsize=10, fontweight="bold", y=1.04)
    return _save(fig, name)


# ---------------------------------------------------------------------------
# The centrepiece: four regimes
# ---------------------------------------------------------------------------


def four_settings(summary: pd.DataFrame, dataset: str, metric="ci_per_ligand",
                  name=None) -> Path:
    """Per-ligand CI by model, one panel per regime.

    Per-ligand CI is the metric that exposes cold-start failure: a model with
    no information about an unseen entity scores every candidate alike, so
    every ranking is a tie and the value is exactly 0.5.  Global CI would
    hide that, because the drug biases still order drugs sensibly.
    """
    use_style()
    name = name or f"fig04_four_settings_{dataset}"
    settings = [s for s in SETTING_TITLE if (summary.setting == s).any()]
    fig, axes = plt.subplots(1, len(settings), figsize=(7.4, 3.0))
    axes = np.atleast_1d(axes)
    mcol, scol = f"{metric}_mean", f"{metric}_std"

    for ax, setting in zip(axes, settings):
        d = summary[(summary.dataset == dataset) & (summary.setting == setting)]
        d = d.sort_values(mcol)
        y = np.arange(len(d))
        best = int(np.argmax(d[mcol].to_numpy()))
        colours = [PALETTE[dataset] if i == best else to_rgba(PALETTE[dataset], 0.33)
                   for i in range(len(d))]
        ax.barh(y, d[mcol], xerr=d.get(scol), color=colours, edgecolor="none",
                error_kw={"ecolor": PALETTE["muted"], "elinewidth": 0.9, "capsize": 2})
        ax.set_yticks(y); ax.set_yticklabels([_pretty(m) for m in d["model"]], fontsize=7)
        ax.axvline(0.5, color=PALETTE["muted"], lw=0.9, ls="--")
        ax.set_xlim(0.45, 1.0)
        ax.set_title(SETTING_TITLE[setting], fontsize=8.5)
        ax.set_xlabel("per-ligand CI")
        ax.grid(axis="y", visible=False); _despine(ax)
    axes[0].annotate("chance", xy=(0.5, -0.9), fontsize=7, color=PALETTE["muted"], ha="center")
    fig.suptitle(
        f"{dataset.upper()}: as more of the matrix is hidden, only the kernel model stays above chance",
        fontsize=10, fontweight="bold", y=1.03)
    fig.tight_layout()
    return _save(fig, name)


def warm_comparison(summary: pd.DataFrame, name="fig05_warm_models") -> Path:
    """Test MSE and per-ligand CI for every warm-start model, both datasets."""
    use_style()
    datasets = [d for d in ("davis", "kiba") if (summary.dataset == d).any()]
    fig, axes = plt.subplots(2, len(datasets), figsize=(7.2, 5.0))
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1:
        axes = axes.T
    for j, ds in enumerate(datasets):
        d = summary[(summary.dataset == ds) & (summary.setting == "S1_pair")]
        for i, (metric, label, best_is_min) in enumerate(
            [("mse", "Test MSE (lower is better)", True),
             ("ci_per_ligand", "Per-ligand CI (higher is better)", False)]
        ):
            ax = axes[i, j]
            col = f"{metric}_mean"
            dd = d.sort_values(col, ascending=not best_is_min)
            vals = dd[col].to_numpy()
            pick = int(np.argmin(vals) if best_is_min else np.argmax(vals))
            colours = [PALETTE[ds] if t == pick else to_rgba(PALETTE[ds], 0.33)
                       for t in range(len(dd))]
            bars = ax.barh(np.arange(len(dd)), vals, xerr=dd.get(f"{metric}_std"),
                           color=colours, edgecolor="none",
                           error_kw={"ecolor": PALETTE["muted"], "elinewidth": 0.8, "capsize": 2})
            ax.set_yticks(np.arange(len(dd)))
            ax.set_yticklabels([_pretty(m) for m in dd["model"]], fontsize=7)
            ax.bar_label(bars, labels=[f"{v:.3f}" if t == pick else "" for t, v in enumerate(vals)],
                         fontsize=7, padding=2, color=PALETTE["ink"])
            ax.set_xlabel(label)
            if i == 0:
                ax.set_title(ds.upper())
            if metric == "ci_per_ligand":
                ax.axvline(0.5, color=PALETTE["muted"], lw=0.9, ls="--")
                ax.set_xlim(0.45, 1.0)
            ax.grid(axis="y", visible=False); _despine(ax)
    fig.suptitle("Interaction terms beat every bias baseline on both datasets",
                 fontsize=10, fontweight="bold", y=1.0)
    fig.tight_layout()
    return _save(fig, name)


def censored_effect(summary: pd.DataFrame, name="fig06_censored_loss") -> Path:
    """The predicted trade-off from training with a censored loss.

    Stated before the experiment: fitting the detection limit as if it were a
    measurement should buy MSE and cost ranking.  Each panel shows the
    censored model minus the identical model trained on squared error, so a
    bar to the right favours the censored loss.
    """
    use_style()
    # Error and ranking metrics are different measures on different scales, so
    # they get their own panel rather than sharing an axis where the largest
    # one would flatten the rest.
    GROUPS = {
        "Error  (lower is better)": [("mse", "MSE"), ("mse_strong", "MSE on strong binders")],
        "Ranking  (higher is better)": [("ci", "CI"), ("ci_per_ligand", "CI per-drug"),
                                        ("hit_at_1", "Hit@1"), ("mrr", "MRR")],
    }

    def deltas(ds):
        d = summary[(summary.dataset == ds) & (summary.setting == "S1_pair")].set_index("model")
        if not {"M4_tobit_mf", "M4b_tobit_squared"} <= set(d.index):
            return None
        return {m: float(d.loc["M4_tobit_mf", f"{m}_mean"] - d.loc["M4b_tobit_squared", f"{m}_mean"])
                for group in GROUPS.values() for m, _ in group if f"{m}_mean" in d.columns}

    davis = deltas("davis")
    if davis is None:
        return None
    kiba = deltas("kiba")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, (title, metrics) in zip(axes, GROUPS.items()):
        metrics = [(m, lab) for m, lab in metrics if m in davis]
        vals = [davis[m] for m, _ in metrics]
        # Right always means "the censored loss helped", whichever direction
        # the underlying metric improves in.
        signed = [-v if "Error" in title else v for v in vals]
        y = np.arange(len(metrics))
        ax.barh(y, signed, color=[PALETTE["accent"] if s > 0 else PALETTE["muted"] for s in signed],
                edgecolor="none", height=0.6)
        ax.set_yticks(y); ax.set_yticklabels([lab for _, lab in metrics], fontsize=7.5)
        ax.invert_yaxis()
        pad = max(abs(min(signed)), abs(max(signed))) * 0.55 or 1.0
        ax.set_xlim(min(min(signed), 0) - pad, max(max(signed), 0) + pad)
        for yy, (s, v) in enumerate(zip(signed, vals)):
            ax.annotate(f"{v:+.4f}", xy=(s, yy), xytext=(6 if s >= 0 else -6, 0),
                        textcoords="offset points", fontsize=7.5,
                        ha="left" if s >= 0 else "right", va="center", color=PALETTE["ink"])
        ax.axvline(0, color=PALETTE["ink"], lw=1.0)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("change from using the censored loss")
        ax.grid(axis="y", visible=False); _despine(ax)

    note = "DAVIS, 71% of labels censored."
    if kiba is not None and all(abs(v) < 1e-12 for v in kiba.values()):
        note += "  KIBA has no censored labels, and the two models there are identical "
        note += "to machine precision, as the loss requires."
    fig.text(0.5, -0.06, note, ha="center", fontsize=7.5, color=PALETTE["muted"])
    fig.suptitle("The censored loss costs squared error and buys ranking, exactly as predicted",
                 fontsize=10, fontweight="bold", y=1.04)
    fig.tight_layout()
    return _save(fig, name)


def similarity_buckets(buckets: dict, name="fig07_similarity_buckets") -> Path:
    """Where borrowing from relatives stops working."""
    use_style()
    fig, axes = plt.subplots(1, len(buckets), figsize=(7.2, 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    from .analysis import BUCKET_LABELS

    for ax, (ds, df) in zip(axes, buckets.items()):
        piv = df.pivot_table(index="bucket", columns="model", values="ci_per_ligand",
                             aggfunc="mean", observed=False)
        # The buckets are an ordered scale, but read back from CSV they are
        # plain strings and would sort alphabetically, putting "<0.10
        # (orphan)" in the middle. That inverts the story the figure tells,
        # so the canonical order is imposed explicitly.
        piv = piv.reindex([b for b in BUCKET_LABELS if b in piv.index]).dropna(how="all")
        for i, m in enumerate(piv.columns[: len(CATEGORICAL)]):
            ax.plot(np.arange(len(piv.index)), piv[m], lw=1.7,
                    color=CATEGORICAL[i % len(CATEGORICAL)],
                    marker=["o", "s", "^", "D", "v"][i % 5], ms=4, label=_pretty(str(m)))
        ax.set_xticks(np.arange(len(piv.index)))
        ax.set_xticklabels([str(b) for b in piv.index], rotation=18, ha="right", fontsize=7)
        ax.axhline(0.5, color=PALETTE["muted"], lw=0.9, ls="--")
        ax.set_xlabel("Similarity to nearest training target")
        ax.set_ylabel("Per-ligand CI")
        ax.set_title(ds.upper()); ax.legend(loc="best"); _despine(ax)
    fig.suptitle(
        "Borrowing from neighbours fails on orphan proteins; the kernel model still generalises",
        fontsize=10, fontweight="bold", y=1.04)
    fig.tight_layout()
    return _save(fig, name)


def tuning_curve(tuning: pd.DataFrame, name="fig08_tuning") -> Path:
    """Validation error against the number of factors, at the best lambda."""
    use_style()
    fig, ax = plt.subplots(figsize=(3.8, 2.7))
    d = tuning[(tuning.model == "M2_biased_mf") & (tuning.setting == "S1_pair")]
    for ds in [x for x in ("davis", "kiba") if (d.dataset == x).any()]:
        s = d[d.dataset == ds].dropna(subset=["val_mse"])
        if not len(s):
            continue
        best = s.loc[s.groupby("k")["val_mse"].idxmin()].sort_values("k")
        ax.plot(best["k"], best["val_mse"], color=PALETTE[ds], lw=1.7,
                marker="o" if ds == "davis" else "s", ms=4, label=ds.upper())
        lo = best.loc[best["val_mse"].idxmin()]
        ax.scatter([lo["k"]], [lo["val_mse"]], s=55, facecolors="none",
                   edgecolors=PALETTE[ds], lw=1.4, zorder=5)
    ax.set_xlabel("Latent factors k   (k = 0 is the bias-only model)")
    ax.set_ylabel("Validation MSE")
    ax.set_title("Chosen on validation, never on test")
    ax.legend(loc="best"); _despine(ax)
    return _save(fig, name)
