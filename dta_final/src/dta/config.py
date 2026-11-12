"""Paths, constants and the few global conventions the whole project obeys.

Nothing here is a model choice.  Anything that could reasonably be tuned lives
in a model's constructor or in the experiment grid, not in this file, so that
reading this module tells you how the project is wired and never what it
concluded.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
FEATURES = ROOT / "features"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
MODELS = ROOT / "saved_models"

for _d in (DATA_INTERIM, DATA_PROCESSED, SPLITS, FEATURES, TABLES, FIGURES, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

DATASETS = ("davis", "kiba")

#: Master seed.  Every stochastic step derives from it, so a single change
#: here re-randomises the whole project reproducibly.
SEED = 2026

#: Seeds for repeated splits.  Five is enough to report a spread without the
#: sweep becoming the bottleneck; Task 2 gets the same treatment as Task 3,
#: which single-split protocols usually skip.
SPLIT_SEEDS = (2026, 2027, 2028, 2029, 2030)

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# ---------------------------------------------------------------------------
# Score scales.  The two datasets are NOT on a comparable scale and must never
# be pooled; every table in the report is per dataset.
# ---------------------------------------------------------------------------

SCORE_NAME = {"davis": "pKd", "kiba": "KIBA score"}
SCORE_LABEL = {
    "davis": "pKd  (higher = stronger binding)",
    "kiba": "KIBA score  (higher = stronger binding)",
}

#: DAVIS is published as Kd in nM.  pKd = 9 - log10(Kd), so bigger is stronger
#: and a unit step is a tenfold change in affinity anywhere on the scale.
#: Converting twice is silent and ruinous, so it happens in exactly one place
#: (``data.clean``) and the cleaning log records that it happened.
DAVIS_KD_CEILING_NM = 10_000.0

#: The assay could not detect binding weaker than the ceiling, so every label
#: sitting on it means "at most this", not "exactly this".  Roughly 70% of
#: DAVIS labels are censored in this sense, which is why the project carries a
#: censored-loss model and reports metrics on the measured subset separately.
DAVIS_CENSOR_PKD = 9.0 - 5.0  # 5.0

#: Above this score a pair counts as a strong binder.  DAVIS pKd 7 is the
#: usual medicinal-chemistry cut for a genuinely potent interaction; the KIBA
#: cut is the upper quartile of its own scale, since the two are unrelated.
STRONG_THRESHOLD = {"davis": 7.0, "kiba": 12.1}

# ---------------------------------------------------------------------------
# Evaluation settings.  The brief asks for S1 and S3; S2 and S4 are the other
# two standard regimes in the drug-target literature and cost almost nothing
# once splitting is a first-class module.
# ---------------------------------------------------------------------------

SETTINGS = {
    "S1_pair": "random held-out pairs; every drug and target also appears in training",
    "S2_new_drug": "whole drugs held out; their targets are all seen",
    "S3_new_target": "whole targets held out; their drugs are all seen",
    "S4_both_new": "drugs and targets both held out; neither side was ever seen",
}

#: Ties are broken at RANDOM when ranking.  A model that scores every
#: candidate identically cannot rank, and random tie-breaking makes it score
#: exactly like the random baseline, which is the truthful statement.  Counting
#: ties as misses would instead make it look *worse* than random.
TIE_POLICY = "random"

#: Bootstrap resamples for interval estimates on test metrics.
N_BOOTSTRAP = 1000

# ---------------------------------------------------------------------------
# Figures.  Categorical colours verified for colour-vision separation and
# contrast; assigned in order and never cycled.
# ---------------------------------------------------------------------------

CATEGORICAL = ["#3A6FB0", "#D97706", "#0F8A6A", "#9333EA", "#BE123C"]
PALETTE = {
    "davis": CATEGORICAL[0],
    "kiba": CATEGORICAL[1],
    "accent": CATEGORICAL[2],
    "ink": "#1F2328",
    "muted": "#5A6169",
    "grid": "#DADEE3",
    "missing": "#E4E7EA",
}
SEQUENTIAL = {"davis": "Blues", "kiba": "YlOrBr"}
FIG_DPI = 200
