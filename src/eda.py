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

# Protein features live in features.py, where Task 3 consumes them; they are
# re-exported here because Task 1's figures were their first caller.
from .features import AMINO_ACIDS, KMER_USEFUL_COSINE, kmer_profiles  # noqa: F401


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
        var_x = n * sxx - sx**2
        var_y = n * syy - sy**2
        corr = (n * sxy - sx * sy) / np.sqrt(var_x * var_y)

    # This closed form loses precision when a profile is near-constant over the
    # shared columns - KIBA scores cluster tightly around 11, so the variance
    # terms can cancel to noise and push |r| past 1. Those pairs have no
    # meaningful correlation, so they are dropped rather than clipped into
    # looking like strong ones; the rest are clipped against pure rounding.
    #
    # The variance test has to be *relative*: a genuinely constant profile does
    # not cancel to exactly zero in floating point, it cancels to a tiny
    # positive residual, which `var > 0` would wave through. Real profiles sit
    # many orders of magnitude above this cut-off.
    tiny = np.finfo(np.float64).tiny
    degenerate = (
        (var_x <= 1e-12 * np.maximum(n * sxx, tiny))
        | (var_y <= 1e-12 * np.maximum(n * syy, tiny))
        | ~np.isfinite(corr)
        | (np.abs(corr) > 1 + 1e-6)
    )
    corr[degenerate] = np.nan
    corr = np.clip(corr, -1.0, 1.0)

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

    ``obs_floor`` is the evidence for a preprocessing filter rather than a
    property of the chemistry. KIBA's floor is 11 observations on *both* axes,
    with zero ligands and zero targets below 10, which is the fingerprint of the
    "keep entities with at least 10 measurements" filter DeepDTA inherited from
    Tang et al. Two consequences for the report: the sparsity we measure is
    post-filter, and Task 3's cold-target scores will be optimistic, because
    every held-out target still arrives with at least 11 measurements.
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
            # Both counts are zero for KIBA and the floor is 11 on both axes:
            # that is the >= 10 observation filter, not the chemistry.
            "ligands_under_10": int((per_drug < 10).sum()),
            "targets_under_10": int((per_target < 10).sum()),
            "obs_floor": int(min(per_drug.min(), per_target.min())),
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
            # Both panels count different entities, so both are labelled.
            ax.set_ylabel(f"{label}s (log)")
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

        # Upper triangle only. Counting every off-diagonal cell would report
        # each target pair twice and invite a 229**2 denominator, which also
        # wrongly includes the 229 self-correlations.
        off = corr[np.triu_indices(len(corr), k=1)]
        rows.append({
            "dataset": name.upper(),
            "targets": int(corr.shape[0]),
            "target_pairs": int(len(off)),
            "pairs_with_overlap": int(np.isfinite(off).sum()),
            "pairs_unusable": int(np.isnan(off).sum()),
            "frac_pairs_unusable": float(np.isnan(off).mean()),
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
        n_eligible = len(eligible)
        sampled = n_eligible > max_drugs
        if sampled:
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
        # The drug count is a sample for KIBA, so the title says so rather than
        # letting "700 drugs" read as the whole dataset.
        sample_note = f" of {n_eligible} (seed {seed})" if sampled else ""
        ax.set_title(f"{name.upper()}: {len(kept)} drugs{sample_note}, {len(x):,} pairs")
        ax.set_xlabel("Morgan fingerprint Tanimoto similarity")
        ax.set_ylabel("affinity profile correlation")
        ax.annotate(f"Spearman rho = {rho:.3f}", xy=(0.04, 0.95), xycoords="axes fraction",
                    fontsize=8.5, color=viz.INK, va="top", weight="semibold")
        ax.grid(False)

        rows.append({
            "dataset": name.upper(), "drugs_used": len(kept),
            "drugs_eligible": n_eligible, "sampled": sampled, "seed": seed,
            "unparsed_smiles": n_unparsed, "pairs": int(len(x)),
            "spearman_rho": float(rho), "p_value": float(pval),
            "median_tanimoto": float(np.median(x)),
        })

    fig.tight_layout()
    viz.save(fig, "task1_structure_vs_affinity")
    return pd.DataFrame(rows).set_index("dataset")


# --------------------------------------------------------------------------
# step 6f - do similar *proteins* bind similarly? (what Task 3 rests on)
# --------------------------------------------------------------------------

def figure_sequence_vs_affinity(
    datasets: dict[str, Dataset],
    min_overlap: int = 10,
    k: int = 3,
    bin_width: float = 0.05,
    min_bin: int = 20,
) -> pd.DataFrame:
    """Protein k-mer cosine similarity vs target affinity-profile correlation.

    The drug-side version of this test comes back weak, and every Task 3 option
    we are considering leans on the *protein* side instead: borrowing a
    neighbour's scores, averaging neighbours' latent vectors, or learning a map
    from a protein representation to one. All three assume sequence-similar
    targets have correlated affinity profiles. This measures that assumption
    before any of them is built.

    Two numbers decide it. ``spearman_rho`` is the overall relationship, read
    the same way as the drug-side figure. ``corr_median_top1`` is the one that
    matters for the cheapest option: for each target, take its single most
    similar neighbour and record how correlated their profiles are. If that sits
    near ``corr_median_all``, then picking the nearest sequence neighbour is no
    better than picking any target at all, and neighbour-borrowing cannot beat
    the per-drug mean by much.

    Pairs from the same ``seq_group`` are excluded throughout, so the two
    numbers describe neighbours that would still be available once a cold-target
    split has separated the sequence-identical variants.
    """
    fig, axes = _panels(datasets, width=7.2, height=3.0)
    rows = []

    for name, ds in datasets.items():
        ax = axes[name]
        mat_df = ds.matrix()
        seq_of = ds.targets.set_index("target_id")["sequence"]
        seqs = [seq_of[t] for t in mat_df.columns]

        profiles = mat_df.to_numpy(dtype=float).T  # targets x drugs
        corr = masked_corr(profiles, min_overlap=min_overlap)

        vectors = kmer_profiles(seqs, k=k)
        cosine = np.clip(vectors @ vectors.T, 0.0, 1.0).astype(np.float64)

        # DAVIS writes mutations into the target id but not the sequence, so 81
        # of its 442 targets have a byte-identical twin sitting at cosine 1.0.
        # A cold-target split groups on seq_group, so that twin is never
        # available to borrow from, and counting it here would measure the
        # leakage the split exists to prevent. Both the scatter and the nearest
        # neighbour are restricted to pairs from different sequence groups.
        group = ds.targets.set_index("target_id")["seq_group"].reindex(mat_df.columns).to_numpy()
        cross_group = group[:, None] != group[None, :]

        iu = np.triu_indices(len(seqs), k=1)
        x, y = cosine[iu], corr[iu]
        ok = np.isfinite(x) & np.isfinite(y) & cross_group[iu]
        n_same_group = int((~cross_group[iu]).sum())
        x, y = x[ok], y[ok]

        # The nearest *usable* sequence neighbour per target: the one option 1
        # would actually borrow from under a cold-target split.
        searchable = np.where(cross_group, cosine, -np.inf)
        rows_idx = np.arange(len(seqs))
        nbr = searchable.argmax(axis=1)
        has_nbr = np.isfinite(searchable[rows_idx, nbr])
        nbr_corr = np.where(has_nbr, corr[rows_idx, nbr], np.nan)
        nbr_cos = np.where(has_nbr, searchable[rows_idx, nbr], np.nan)

        hb = ax.hexbin(x, y, gridsize=34, mincnt=1, cmap=viz.sequential_cmap(),
                       linewidths=0.0)
        cbar = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.03)
        cbar.set_label("target pairs", color=viz.INK_SECONDARY, fontsize=8)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=7, color=viz.MUTED, labelcolor=viz.INK_SECONDARY)

        # Median correlation per similarity band. Fixed-width bins, not
        # quantiles: 90% of kinase pairs sit below cosine 0.2, so decile edges
        # would stop the line before the sparse high-similarity tail - which is
        # the only part of this plot that decides anything. Bins thinner than
        # `min_bin` pairs are dropped so the tail is a trend, not one pair.
        edges = np.arange(0.0, x.max() + bin_width, bin_width)
        slot = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
        filled = [b for b in range(len(edges) - 1) if (slot == b).sum() >= min_bin]
        if filled:
            ax.plot([x[slot == b].mean() for b in filled],
                    [np.median(y[slot == b]) for b in filled],
                    color=viz.ORANGE, lw=1.6, marker="o", markersize=3.2,
                    label=f"median per {bin_width} band", zorder=5)
            ax.legend(loc="lower right")

        rho, pval = spearmanr(x, y) if len(x) > 2 else (np.nan, np.nan)
        ax.set_title(f"{name.upper()}: {len(seqs)} targets, {len(x):,} pairs")
        ax.set_xlabel(f"protein {k}-mer cosine similarity")
        ax.set_ylabel("affinity profile correlation")
        ax.annotate(f"Spearman rho = {rho:.3f}", xy=(0.04, 0.95), xycoords="axes fraction",
                    fontsize=8.5, color=viz.INK, va="top", weight="semibold")
        ax.grid(False)

        rows.append({
            "dataset": name.upper(), "targets": len(seqs), "pairs": int(len(x)),
            "pairs_same_seq_group": n_same_group,
            "spearman_rho": float(rho), "p_value": float(pval),
            "median_cosine": float(np.median(x)),
            "p90_cosine": float(np.quantile(x, 0.90)),
            "median_top1_cosine": float(np.nanmedian(nbr_cos)),
            # How often a neighbour method would actually have something worth
            # borrowing from, rather than firing on an uninformative match.
            "frac_top1_above_threshold": float(np.nanmean(nbr_cos > KMER_USEFUL_COSINE)),
            "corr_median_all": float(np.median(y)),
            "corr_median_top1": float(np.nanmedian(nbr_corr)),
            "top1_lift": float(np.nanmedian(nbr_corr) - np.median(y)),
        })

    fig.tight_layout()
    viz.save(fig, "task1_sequence_vs_affinity")
    return pd.DataFrame(rows).set_index("dataset")


# --------------------------------------------------------------------------
# step 6g - is a cell missing because nobody expected it to bind?
# --------------------------------------------------------------------------

def figure_missingness_bias(datasets: dict[str, Dataset], n_bins: int = 10) -> pd.DataFrame:
    """Does the number of measurements an entity has predict its scores?

    A random 70/15/15 pair split assumes a cell is missing for reasons unrelated
    to the score it hides. KIBA's assays were chosen by chemists, who measure
    the pairs they expect to bind, so the entities with the most measurements
    may also be the ones scoring highest. If they are, the observed cells are a
    biased sample of the matrix, Task 2's test error is optimistic, and the
    report has to say so - this is the standard missing-not-at-random result in
    recommender evaluation (Steck, KDD 2010).

    DAVIS is a complete matrix, so nothing is missing and no such bias can
    exist. It still gets a row, with ``missing_frac`` 0 and an undefined
    correlation, so the table records that the check was run on both datasets
    rather than quietly covering one.
    """
    rows = []
    for name, ds in datasets.items():
        for entity, key in (("ligand", "drug_id"), ("target", "target_id")):
            grouped = ds.pairs.groupby(key)["score"]
            counts = grouped.size().to_numpy(dtype=float)
            means = grouped.mean().to_numpy(dtype=float)

            # A complete matrix gives every entity the same count, so there is
            # no "most measured" decile to compare: sorting constant counts
            # just returns them in index order, and the difference between the
            # two ends of that order is noise. Every derived number is NaN
            # rather than a figure someone could quote.
            varies = len(np.unique(counts)) > 1
            if varies:
                rho, pval = spearmanr(counts, means)
                order = np.argsort(counts, kind="stable")
                edge = max(1, len(order) // n_bins)
                fewest = float(means[order[:edge]].mean())
                most = float(means[order[-edge:]].mean())
            else:
                rho = pval = fewest = most = np.nan

            rows.append({
                "dataset": name.upper(), "entity": entity,
                "n_entities": len(counts),
                "missing_frac": ds.report["sparsity"],
                "counts_vary": varies,
                "spearman_rho": float(rho), "p_value": float(pval),
                "mean_score_fewest_decile": fewest,
                "mean_score_most_decile": most,
                "delta": most - fewest,
            })
    table = pd.DataFrame(rows).set_index(["dataset", "entity"])

    sparse = {n: d for n, d in datasets.items() if d.report["cells_missing"] > 0}
    if not sparse:
        return table

    fig, axes = plt.subplots(1, 2 * len(sparse), figsize=(7.2, 2.9))
    axes = np.atleast_1d(axes)
    for i, (name, ds) in enumerate(sparse.items()):
        for j, (entity, key) in enumerate((("ligand", "drug_id"), ("target", "target_id"))):
            ax = axes[2 * i + j]
            grouped = ds.pairs.groupby(key)["score"]
            counts = grouped.size().to_numpy(dtype=float)
            means = grouped.mean().to_numpy(dtype=float)

            edges = np.unique(np.quantile(counts, np.linspace(0, 1, n_bins + 1)))
            slot = np.clip(np.searchsorted(edges, counts, side="right") - 1, 0, len(edges) - 2)
            present = [b for b in range(len(edges) - 1) if (slot == b).any()]
            bands = [means[slot == b] for b in present]

            # Standard error per bin, and the dataset mean as a reference. The
            # y-range here is a third of a KIBA unit, so without both of these
            # the bin-to-bin wobble reads as a trend - which is the mistake
            # this whole figure exists to test for.
            overall = float(ds.pairs["score"].mean())
            ax.axhline(overall, color=viz.GRID, lw=0.9, zorder=0)
            # Left edge: the first decile is the lowest point in both panels,
            # so there is always clear space just above the line there.
            ax.annotate("dataset mean", xy=(0.02, overall), xycoords=("axes fraction", "data"),
                        xytext=(0, 2), textcoords="offset points",
                        fontsize=7, color=viz.MUTED, ha="left", va="bottom")
            ax.errorbar([counts[slot == b].mean() for b in present],
                        [b.mean() for b in bands],
                        yerr=[b.std(ddof=1) / np.sqrt(len(b)) if len(b) > 1 else 0.0 for b in bands],
                        color=viz.BLUE, lw=1.6, marker="o", markersize=3.4,
                        capsize=2.5, elinewidth=0.9)
            ax.set_xscale("log")
            ax.set_title(f"{name.upper()}: per {entity}")
            ax.set_xlabel(f"measurements per {entity} (log)")
            ax.set_ylabel(f"mean {ds.score_name}")
            rho = table.loc[(name.upper(), entity), "spearman_rho"]
            ax.annotate(f"rho = {rho:.3f}", xy=(0.96, 0.94), xycoords="axes fraction",
                        fontsize=8.5, color=viz.INK, ha="right", va="top",
                        weight="semibold")
    fig.tight_layout()
    viz.save(fig, "task1_missingness_bias")
    return table


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
