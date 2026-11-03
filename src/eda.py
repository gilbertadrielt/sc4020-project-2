"""Task 1 exploratory analysis for DAVIS and KIBA.

Every function takes the loaded datasets, writes one figure into
``results/figures/`` and returns the numbers that go in the report, so no figure
and no quoted statistic can drift apart.

Run everything with ``python scripts/run_task1.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import gaussian_kde, spearmanr

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from . import viz
from .data import Dataset

#: Score above which a pair counts as strong binding, per score scale.
#: DAVIS pKd >= 7 is Kd <= 100 nM; KIBA >= 12.1 is the threshold the KIBA
#: literature uses to binarise the score.
STRONG_BINDING = {"pKd": 7.0, "KIBA": 12.1}

#: DAVIS censors weak binders at Kd = 10,000 nM, which is exactly pKd 5.
DAVIS_CEILING = 5.0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def masked_corr(x: np.ndarray, min_overlap: int = 5) -> np.ndarray:
    """Pairwise Pearson correlation of rows, over each pair's shared columns.

    NaN means unobserved. Pairs sharing fewer than ``min_overlap`` columns are
    returned as NaN rather than as a correlation computed from 2 or 3 points,
    which would be noise presented as signal.
    """
    mask = (~np.isnan(x)).astype(np.float64)
    filled = np.nan_to_num(x, nan=0.0)

    n = mask @ mask.T
    sx = filled @ mask.T
    sy = sx.T
    sxx = (filled**2) @ mask.T
    syy = sxx.T
    sxy = filled @ filled.T

    with np.errstate(invalid="ignore", divide="ignore"):
        num = n * sxy - sx * sy
        den = np.sqrt((n * sxx - sx**2) * (n * syy - sy**2))
        corr = num / den

    corr[n < min_overlap] = np.nan
    np.fill_diagonal(corr, 1.0)
    return corr


def _reorder(corr: np.ndarray) -> np.ndarray:
    """Hierarchical-clustering leaf order, so block structure becomes visible.

    Unusable pairs (too little overlap) are treated as uncorrelated for the
    clustering only; they stay NaN in the figure.
    """
    filled = np.nan_to_num(corr, nan=0.0)
    dist = np.clip(1.0 - filled, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    condensed = squareform(dist, checks=False)
    link = hierarchy.linkage(condensed, method="average")
    # Optimal leaf ordering is O(n^3); past a few hundred leaves the plain
    # dendrogram order is close enough and far cheaper.
    if len(corr) <= 300:
        link = hierarchy.optimal_leaf_ordering(link, condensed)
    return hierarchy.leaves_list(link)


def _panels(datasets: dict[str, Dataset], width=7.0, height=3.1):
    fig, axes = plt.subplots(1, len(datasets), figsize=(width, height))
    return fig, dict(zip(datasets, np.atleast_1d(axes)))


# --------------------------------------------------------------------------
# step 4 - the affinity matrices
# --------------------------------------------------------------------------

def figure_heatmap(datasets: dict[str, Dataset]) -> pd.DataFrame:
    """Ligand x target heatmap per dataset, unobserved cells explicitly grey."""
    fig, axes = _panels(datasets, width=7.2, height=3.4)
    cmap = viz.sequential_cmap()
    rows = []

    for name, ds in datasets.items():
        ax = axes[name]
        mat = ds.matrix().to_numpy(dtype=float)
        # Sort both axes by mean score so structure is visible instead of the
        # arbitrary order the source files happen to use.
        with np.errstate(invalid="ignore"):
            r_order = np.argsort(-np.nanmean(mat, axis=1))
            c_order = np.argsort(-np.nanmean(mat, axis=0))
        mat = mat[np.ix_(r_order, c_order)]

        im = ax.imshow(
            np.ma.masked_invalid(mat),
            cmap=cmap, aspect="auto", interpolation="nearest",
        )
        ax.set_title(f"{name.upper()}  ({mat.shape[0]} ligands x {mat.shape[1]} targets)")
        ax.set_xlabel("targets, sorted by mean score")
        ax.set_ylabel("ligands, sorted by mean score")
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cbar.set_label(ds.score_name, color=viz.INK_SECONDARY, fontsize=8)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=7, color=viz.MUTED, labelcolor=viz.INK_SECONDARY)

        miss = float(np.isnan(mat).mean())
        rows.append({"dataset": name.upper(), "missing_fraction": miss,
                     "observed": int((~np.isnan(mat)).sum()), "cells": int(mat.size)})

    # Name the grey once, for both panels.
    fig.legend(handles=[viz.missing_handle("not measured")],
               loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    viz.save(fig, "task1_heatmaps")
    return pd.DataFrame(rows).set_index("dataset")


# --------------------------------------------------------------------------
# step 5 - score distributions
# --------------------------------------------------------------------------

def figure_score_distribution(datasets: dict[str, Dataset]) -> pd.DataFrame:
    """Observed-score density per dataset.

    DAVIS is handled specially: 69.6% of its labels sit exactly at the assay
    ceiling pKd = 5. A KDE over that point mass would smear a censored spike
    into a smooth bump and misrepresent the data, so the spike is drawn as its
    own labelled bar and the density is estimated on the uncensored values only.
    """
    fig, axes = _panels(datasets, width=7.2, height=3.0)
    rows = []

    for name, ds in datasets.items():
        ax = axes[name]
        s = ds.pairs["score"].to_numpy(dtype=float)
        censored = np.isclose(s, DAVIS_CEILING) if ds.score_name == "pKd" else np.zeros_like(s, bool)
        free = s[~censored]

        ax.hist(free, bins=60, density=True, color=viz.BLUE, alpha=0.30,
                edgecolor="none", label="observed")
        if len(free) > 1:
            grid = np.linspace(free.min(), free.max(), 400)
            ax.plot(grid, gaussian_kde(free)(grid), color=viz.BLUE, lw=1.6, label="KDE")

        if censored.any():
            frac = censored.mean()
            # The spike is a point mass, so it gets a bar, not a density curve.
            ax.axvline(DAVIS_CEILING, color=viz.ORANGE, lw=2.0)
            ax.annotate(
                f"{frac:.1%} of labels at the\nassay ceiling (pKd = 5)",
                xy=(DAVIS_CEILING, ax.get_ylim()[1] * 0.92),
                xytext=(0.30, 0.86), textcoords="axes fraction",
                color=viz.ORANGE, fontsize=8, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=viz.ORANGE, lw=1.0),
            )
            ax.set_title(f"{name.upper()}  (density excludes the ceiling spike)")
        else:
            low = int((s < 5).sum())
            ax.set_title(f"{name.upper()}  ({len(s):,} observed pairs)")
            if low:
                ax.annotate(
                    f"{low} scores below 5\n(extreme low outliers)",
                    xy=(0.04, 0.86), xycoords="axes fraction",
                    color=viz.ORANGE, fontsize=8, va="top",
                )

        ax.set_xlabel(f"{ds.score_name}  (higher = stronger binding)")
        ax.set_ylabel("density")
        ax.legend(loc="upper right")

        rows.append({
            "dataset": name.upper(), "score": ds.score_name, "n": len(s),
            "min": s.min(), "median": float(np.median(s)), "max": s.max(),
            "mean": s.mean(), "std": s.std(),
            "at_ceiling_frac": float(censored.mean()),
            "strong_frac": float((s >= STRONG_BINDING[ds.score_name]).mean()),
        })

    fig.tight_layout()
    viz.save(fig, "task1_score_distributions")
    return pd.DataFrame(rows).set_index("dataset")


# --------------------------------------------------------------------------
# step 6a - how much evidence per entity (drives Task 2 fallbacks)
# --------------------------------------------------------------------------

def figure_observations_per_entity(datasets: dict[str, Dataset]) -> pd.DataFrame:
    """Observations per ligand and per target.

    DAVIS is a complete matrix, so its counts are constants (442 per ligand,
    68 per target) and carry no information; only datasets with missing data get
    a panel. KIBA's long tail is what forces neighbour fallbacks in Task 2.
    """
    sparse = {n: d for n, d in datasets.items() if d.report["cells_missing"] > 0}
    rows = []
    for name, ds in datasets.items():
        per_drug = ds.pairs.groupby("drug_id").size()
        per_target = ds.pairs.groupby("target_id").size()
        rows.append({
            "dataset": name.upper(),
            "per_ligand_min": int(per_drug.min()), "per_ligand_median": float(per_drug.median()),
            "per_ligand_max": int(per_drug.max()),
            "per_target_min": int(per_target.min()), "per_target_median": float(per_target.median()),
            "per_target_max": int(per_target.max()),
            "ligands_under_10": int((per_drug < 10).sum()),
            "ligands_under_5": int((per_drug < 5).sum()),
        })
    table = pd.DataFrame(rows).set_index("dataset")

    if not sparse:
        return table

    fig, axes = plt.subplots(1, 2 * len(sparse), figsize=(7.2, 2.8))
    axes = np.atleast_1d(axes)
    for i, (name, ds) in enumerate(sparse.items()):
        for j, (label, counts) in enumerate(
            [("ligand", ds.pairs.groupby("drug_id").size()),
             ("target", ds.pairs.groupby("target_id").size())]
        ):
            ax = axes[2 * i + j]
            ax.hist(counts, bins=40, color=viz.BLUE, alpha=0.75, edgecolor="none")
            ax.set_yscale("log")
            ax.set_title(f"{name.upper()}: per {label}")
            ax.set_xlabel("observed pairs")
            ax.set_ylabel(f"{label}s (log)" if j == 0 else "")
            ax.axvline(counts.median(), color=viz.ORANGE, lw=1.4)
            ax.annotate(f"median {counts.median():.0f}", xy=(counts.median(), 0.88),
                        xycoords=("data", "axes fraction"), color=viz.ORANGE,
                        fontsize=7.5, ha="left", va="top", xytext=(4, 0),
                        textcoords="offset points")
    fig.tight_layout()
    viz.save(fig, "task1_observations_per_entity")
    return table


# --------------------------------------------------------------------------
# step 6b - does a low-rank model make sense? (justifies MF in Task 2)
# --------------------------------------------------------------------------

def figure_svd_spectrum(datasets: dict[str, Dataset], max_k: int = 60) -> pd.DataFrame:
    """Cumulative explained variance of the affinity matrix.

    Both curves are dimensionless fractions of variance, so they legitimately
    share one axis. KIBA has to be imputed before an SVD exists at all: missing
    cells are filled with the target mean, which biases the spectrum *towards*
    low rank, so its curve is an optimistic bound and is labelled as such.
    """
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    rows = []

    for i, (name, ds) in enumerate(datasets.items()):
        mat = ds.matrix().to_numpy(dtype=float)
        imputed = ds.report["cells_missing"] > 0
        if imputed:
            col_mean = np.nanmean(mat, axis=0)
            col_mean = np.where(np.isnan(col_mean), np.nanmean(mat), col_mean)
            mat = np.where(np.isnan(mat), col_mean[None, :], mat)
        centred = mat - mat.mean()

        sv = np.linalg.svd(centred, compute_uv=False)
        cum = np.cumsum(sv**2) / np.sum(sv**2)
        k = min(max_k, len(cum))
        label = f"{name.upper()}" + (" (target-mean imputed)" if imputed else " (complete)")
        ax.plot(np.arange(1, k + 1), cum[:k], color=viz.SERIES[i], lw=1.6, label=label)

        rows.append({
            "dataset": name.upper(), "imputed": imputed,
            "rank": int(min(mat.shape)),
            "k_for_80pct": int(np.searchsorted(cum, 0.80) + 1),
            "k_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
            "var_at_k10": float(cum[min(9, len(cum) - 1)]),
            "var_at_k20": float(cum[min(19, len(cum) - 1)]),
        })

    for level in (0.80, 0.90):
        ax.axhline(level, color=viz.GRID, lw=0.8, zorder=0)
        ax.annotate(f"{level:.0%}", xy=(0.995, level), xycoords=("axes fraction", "data"),
                    fontsize=7, color=viz.MUTED, ha="right", va="bottom")
    ax.set_xlabel("number of latent factors")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title("A few factors carry most of the signal")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    fig.tight_layout()
    viz.save(fig, "task1_svd_spectrum")
    return pd.DataFrame(rows).set_index("dataset")


# --------------------------------------------------------------------------
# step 6c - per-drug selectivity (picks the Task 2 case studies)
# --------------------------------------------------------------------------

def figure_selectivity(datasets: dict[str, Dataset]) -> pd.DataFrame:
    """How many targets each drug binds strongly.

    A promiscuous drug and a selective one are different recommendation
    problems, so Task 2's case studies should cover both ends of this.
    """
    fig, axes = _panels(datasets, width=7.2, height=2.9)
    rows = []

    for name, ds in datasets.items():
        ax = axes[name]
        thresh = STRONG_BINDING[ds.score_name]
        strong = (
            ds.pairs.assign(hit=ds.pairs["score"] >= thresh)
            .groupby("drug_id")["hit"].sum()
        )
        ax.hist(strong, bins=40, color=viz.BLUE, alpha=0.75, edgecolor="none")
        ax.set_yscale("log")
        ax.set_title(f"{name.upper()}: targets with {ds.score_name} >= {thresh:g}")
        ax.set_xlabel("strongly bound targets per drug")
        ax.set_ylabel("drugs (log)")

        rows.append({
            "dataset": name.upper(), "threshold": thresh,
            "drugs": int(len(strong)),
            "drugs_with_no_strong_target": int((strong == 0).sum()),
            "median_strong": float(strong.median()),
            "max_strong": int(strong.max()),
            "most_promiscuous": str(strong.idxmax()),
        })
    fig.tight_layout()
    viz.save(fig, "task1_selectivity")
    return pd.DataFrame(rows).set_index("dataset")


# --------------------------------------------------------------------------
# step 6d - target families (motivates borrowing neighbours in Task 3)
# --------------------------------------------------------------------------

def figure_target_correlation(datasets: dict[str, Dataset], min_overlap: int = 10) -> pd.DataFrame:
    """Target-target affinity-profile correlation, reordered by clustering.

    Correlation is signed, so this uses the diverging palette with a neutral
    midpoint at zero. Visible blocks mean targets fall into families that share
    a binding profile - the premise Task 3's "borrow from similar targets"
    approach depends on.
    """
    cmap = LinearSegmentedColormap.from_list("report_div", viz.DIVERGING, N=256)
    cmap = cmap.copy()
    cmap.set_bad(viz.MISSING)
    fig, axes = _panels(datasets, width=7.2, height=3.3)
    rows = []

    for name, ds in datasets.items():
        ax = axes[name]
        mat = ds.matrix().to_numpy(dtype=float).T  # targets x drugs
        corr = masked_corr(mat, min_overlap=min_overlap)
        order = _reorder(corr)
        shown = corr[np.ix_(order, order)]

        im = ax.imshow(np.ma.masked_invalid(shown), cmap=cmap,
                       norm=TwoSlopeNorm(vmin=-1, vcenter=0.0, vmax=1),
                       aspect="auto", interpolation="nearest")
        ax.set_title(f"{name.upper()}: {corr.shape[0]} targets, clustered")
        ax.set_xlabel("targets (clustered order)")
        ax.set_ylabel("targets (clustered order)")
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cbar.set_label("profile correlation", color=viz.INK_SECONDARY, fontsize=8)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=7, color=viz.MUTED, labelcolor=viz.INK_SECONDARY)

        off = corr[~np.eye(len(corr), dtype=bool)]
        rows.append({
            "dataset": name.upper(),
            "targets": int(corr.shape[0]),
            "pairs_with_overlap": int(np.isfinite(off).sum()),
            "pairs_unusable": int(np.isnan(off).sum()),
            "median_corr": float(np.nanmedian(off)),
            "frac_corr_over_0.5": float(np.nanmean(off > 0.5)),
        })

    fig.legend(handles=[viz.missing_handle(f"fewer than {min_overlap} shared drugs")],
               loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    viz.save(fig, "task1_target_correlation")
    return pd.DataFrame(rows).set_index("dataset")


# --------------------------------------------------------------------------
# step 6e - do similar drugs bind similarly? (the Task 3 premise, tested)
# --------------------------------------------------------------------------

def figure_structure_vs_affinity(
    datasets: dict[str, Dataset],
    min_overlap: int = 10,
    max_drugs: int = 700,
    seed: int = 0,
) -> pd.DataFrame:
    """Morgan-fingerprint Tanimoto similarity vs affinity-profile correlation.

    If structurally similar drugs really do have correlated binding profiles,
    a content-based model can generalise from structure - which is exactly what
    Task 3 needs. A weak relationship is just as useful a result, and is
    reported rather than hidden.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")  # unparseable SMILES are counted, not printed
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rng = np.random.default_rng(seed)

    fig, axes = _panels(datasets, width=7.2, height=3.0)
    rows = []

    for name, ds in datasets.items():
        ax = axes[name]
        mat_df = ds.matrix()
        drug_ids = list(mat_df.index)

        # Keep drugs with enough measurements for a meaningful profile, then
        # cap the count so the pair matrix stays tractable.
        counts = ds.pairs.groupby("drug_id").size()
        eligible = [d for d in drug_ids if counts.get(d, 0) >= min_overlap]
        if len(eligible) > max_drugs:
            eligible = list(rng.choice(eligible, size=max_drugs, replace=False))
        keep = mat_df.index.isin(eligible)

        smiles = ds.ligands.set_index("drug_id")["smiles"]
        bits, kept = [], []
        n_unparsed = 0
        for d in mat_df.index[keep]:
            mol = Chem.MolFromSmiles(smiles.get(d, ""))
            if mol is None:
                n_unparsed += 1
                continue
            bits.append(gen.GetFingerprintAsNumPy(mol))
            kept.append(d)

        fp = np.asarray(bits, dtype=np.float32)
        inter = fp @ fp.T
        size = fp.sum(axis=1)
        union = size[:, None] + size[None, :] - inter
        with np.errstate(invalid="ignore", divide="ignore"):
            tanimoto = np.where(union > 0, inter / union, np.nan)

        profiles = mat_df.loc[kept].to_numpy(dtype=float)
        corr = masked_corr(profiles, min_overlap=min_overlap)

        iu = np.triu_indices(len(kept), k=1)
        x, y = tanimoto[iu], corr[iu]
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]

        hb = ax.hexbin(x, y, gridsize=34, mincnt=1, cmap=viz.sequential_cmap(),
                       linewidths=0.0)
        cbar = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.03)
        cbar.set_label("drug pairs", color=viz.INK_SECONDARY, fontsize=8)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=7, color=viz.MUTED, labelcolor=viz.INK_SECONDARY)

        rho, pval = spearmanr(x, y) if len(x) > 2 else (np.nan, np.nan)
        ax.set_title(f"{name.upper()}: {len(kept)} drugs, {len(x):,} pairs")
        ax.set_xlabel("Morgan fingerprint Tanimoto similarity")
        ax.set_ylabel("affinity profile correlation")
        ax.annotate(f"Spearman rho = {rho:.3f}", xy=(0.04, 0.95), xycoords="axes fraction",
                    fontsize=8.5, color=viz.INK, va="top", weight="semibold")
        ax.grid(False)

        rows.append({
            "dataset": name.upper(), "drugs_used": len(kept),
            "unparsed_smiles": n_unparsed, "pairs": int(len(x)),
            "spearman_rho": float(rho), "p_value": float(pval),
            "median_tanimoto": float(np.median(x)),
        })

    fig.tight_layout()
    viz.save(fig, "task1_structure_vs_affinity")
    return pd.DataFrame(rows).set_index("dataset")


# --------------------------------------------------------------------------
# the deduplication evidence
# --------------------------------------------------------------------------

def sequence_collapse_table(ds: Dataset) -> pd.DataFrame:
    """Per-group spread when sequence-identical targets would be averaged.

    This is the evidence for the deduplication decision: each row is a set of
    target ids sharing one amino-acid sequence, and ``max_spread`` is how far
    apart their measured affinities get for a single drug. Large spreads mean
    averaging would destroy real signal.
    """
    mat = ds.matrix()
    groups = ds.targets.groupby("seq_group")["target_id"].apply(list)
    rows = []
    for label, ids in groups.items():
        ids = [i for i in ids if i in mat.columns]
        if len(ids) < 2:
            continue
        sub = mat[ids].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            spread = np.nanmax(sub, axis=1) - np.nanmin(sub, axis=1)
        spread = spread[np.isfinite(spread)]
        if not len(spread):
            continue
        rows.append({
            "seq_group": label, "n_ids": len(ids),
            "max_spread": float(spread.max()), "mean_spread": float(spread.mean()),
            "drugs_over_1_unit": int((spread > 1).sum()),
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("max_spread", ascending=False).set_index("seq_group")
