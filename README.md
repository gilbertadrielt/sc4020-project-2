# SC4020 Project 2 — DAVIS & KIBA drug–target recommender

Drugs are users, protein targets are items, binding affinity is the rating.

## Setup

```bash
pip install -r requirements.txt
python scripts/download_data.py    # fetches DeepDTA files into data/, checksum-verified
python scripts/run_task1.py        # Task 1 figures + tables into results/
python scripts/make_splits.py      # warm + cold-target splits into data/processed/
python scripts/run_task2.py        # tune on validation, score test once, case studies
python scripts/run_task3.py        # the cold-target ladder
python -m pytest tests/ -q
```

Run them in that order: `run_task2.py` and `run_task3.py` both load the saved
splits rather than re-splitting, so the whole group scores identical pairs.

`data/` is gitignored — everyone regenerates it from the download script, which
pins a SHA-256 per file so the whole group provably works from identical bytes.

## Layout

```
data/            davis/, kiba/ (raw), processed/
src/
  data.py        load, clean, dedup — the one module everything imports
  viz.py         shared figure style for every report figure
  eda.py         Task 1 analyses
  split.py       warm 70/15/15 + cold-target split, saved to data/processed/
  metrics.py     mse, rmse, ci, hit@k, mrr
  cf.py          baselines, biased MF (ALS), item-kNN, save/load .npz
  tuning.py      validation grid search — never sees the test fold
  casestudy.py   (Task 2) top-k recommendations for three test drugs per dataset
  features.py    (Task 3) protein k-mer profiles, Morgan fingerprints, similarities
  cold.py        (Task 3) the four cold-target models
scripts/         download_data.py, run_task1.py, make_splits.py, run_task2.py,
                 run_task3.py
tests/           correctness tests for every shared helper
results/         figures/, tables/
saved_models/    davis_cf.npz, kiba_cf.npz
```

## Two things that will bite you

**1. DAVIS from DeepDTA is raw Kd in nanomolar, so it needs converting once.**
`data.py` applies `pKd = 9 − log10(Kd)` and the result spans 5.000–10.796. The
Kaggle mirror of DAVIS is *already* pKd — if you mix the two sources you will
double-convert. Never pool pKd with KIBA scores; they are different scales.

**2. DAVIS writes mutations into the target ID but not into the sequence.**
All 15 `ABL1` variants — wild-type, `E255K`, `T315I`, the phosphorylated forms —
share one byte-identical 1167 aa string. 442 target IDs collapse onto 379 unique
sequences.

We key by **ID** (`dedup_key="id"`, the default), keeping all 442 targets and
30,056 pairs, because averaging by sequence destroys real pharmacology.
`results/tables/task1_dedup_counts.csv` reports **both** keys side by side —
keying by ID merges nothing (the DeepDTA matrix already holds one cell per
pair), so a one-key table would have identical before and after columns and
hide the decision rather than show it:

| | |
|---|---|
| (drug, group) cells that would be merged | 1,224 |
| merged values > 1 pKd apart (>10× in Kd) | 260 |
| merged values > 2 pKd apart (>100× in Kd) | 77 |
| worst single case | **4.75 pKd** (~56,000× in Kd) |

The clearest case: wild-type `ABL1` reaches pKd 10.54 while `ABL1(T315I)` tops
out at 8.55. T315I is the gatekeeper mutation that confers imatinib resistance,
so that ~2-log gap *is* the measurement — averaging it away would be wrong.

Every row still carries `seq_group`, and **a cold-target split must group on
that column**, or a "new" ABL1 mutant in test is really a target already seen in
training. Pass `dedup_key="sequence"` to compare the alternative (25,772 pairs).

## Dataset facts

| | DAVIS | KIBA |
|---|---|---|
| ligands × targets | 68 × 442 | 2,111 × 229 |
| observed pairs | 30,056 (complete) | 118,254 of 483,419 |
| sparsity | 0% | **75.5%** |
| score | pKd, 5.000–10.796 | KIBA score, 0–17.2 |
| unique sequences | 379 of 442 IDs | 229 of 229 IDs |
| unique SMILES | 68 of 68 | **2,068 of 2,111** |
| observations per ligand | 442 (constant) | min 11, median 30, max 166 |

- **69.6% of DAVIS labels sit exactly at pKd 5**, the 10,000 nM assay ceiling.
  That is censoring, not weak binding measured precisely, so a near-constant
  predictor will score well on MSE — lean on the ranking metrics in Task 2.
- **KIBA has 7 scores below 5**, far outside its 10–16 bulk. Inspect, don't drop.
- **KIBA also has duplicate ligands**: 43 ChEMBL IDs share a SMILES with another
  (one group of 8, one of 6, one of 3, 29 pairs), and their scores for the same
  target differ by up to 2.78. Same problem as the targets, mirrored; `lig_group`
  carries the grouping.
- **179 sequences are byte-identical across the two datasets**, so there are 429
  unique sequences in total — embed them once and reuse for both in Task 3.

## What Task 1 found

| Finding | Number | Why it matters |
|---|---|---|
| Low-rank structure | DAVIS: 10 factors → 80% variance, 20 → 90% | Justifies matrix factorization in Task 2 |
| Targets form families | Median profile correlation: DAVIS 0.27, **KIBA 0.29** | Borrowing from similar targets is sound on *both* |
| KIBA neighbours are thin | **35% of target pairs (9,166 of 26,106) share < 10 drugs** | KIBA's problem is neighbour *coverage*, not similarity quality — build a fallback ladder in Task 2, not a cleverer metric |
| Drug structure predicts binding weakly | Tanimoto vs profile correlation: DAVIS ρ = 0.275, KIBA ρ = 0.141 (700 of 2,111 drugs, seed 0) | Fingerprints alone will not carry Task 3 |
| **Protein similarity is a threshold, not a trend** | 3-mer cosine vs profile correlation: ρ = 0.059 (DAVIS), 0.222 (KIBA) — but the median correlation jumps from ~0.25 to ~0.9 above cosine 0.3 | The low ρ is a red herring. Never read this relationship as linear |
| **The nearest sequence neighbour is worth borrowing from** | Top-1 cross-group neighbour: median profile correlation **0.86** (DAVIS), **0.84** (KIBA), against 0.27 / 0.29 for a random pair. Above threshold for 71% / 72% of targets | Task 3 should borrow from the *few* nearest neighbours above the threshold and fall back for the other ~29% — a similarity-weighted average over all targets would be drowned by the uninformative majority |
| **KIBA is missing not at random** | More-measured ligands score higher: ρ = 0.169, p = 5e-15, +0.53 KIBA units from the least- to most-measured decile. Targets show nothing (ρ = 0.034, p = 0.61) | A random pair split evaluates on a biased sample, so Task 2's test error is optimistic on the ligand axis — say so in the report |
| **KIBA was pre-filtered to ≥ 10 observations** | Floor of 11 on *both* axes; zero ligands and zero targets below 10 | The 75.5% sparsity is post-filter, and Task 3's cold-target scores are optimistic: every held-out target still arrives with ≥ 11 measurements |
| Selectivity varies hugely | DAVIS median 20.5 strong targets/drug, max 254 (drug `44259`); KIBA 338 drugs with none | Pick Task 2 case studies from both ends |

KIBA's SVD curve is computed on a target-mean-imputed matrix, which biases it
towards low rank — treat its 14-factors-for-80% as an optimistic bound, not a
measurement.

Two counting conventions worth stating once, because both are easy to get
wrong: target pairs are counted **unordered** (229 targets give 26,106 pairs,
not 229² = 52,441), and the sequence-similarity analysis **excludes pairs from
the same `seq_group`**. Without that exclusion, 81 of DAVIS's 442 targets have a
byte-identical twin sitting at cosine 1.0 and the neighbour result measures the
very leakage a cold-target split exists to prevent.

## Group decisions

- [x] **Deduplication key** — ID, with `seq_group` carried for splitting
- [x] **One saved split file** — `data/processed/<dataset>_{warm,cold_target}.csv`,
  written by `make_splits.py` and loaded by both `run_task2.py` and `run_task3.py`
- [ ] Main CF model: matrix factorization or kNN — biased MF wins validation CI
  on both datasets (`task2_leaderboard.csv`); the group still needs to confirm
- [ ] Owner of `split.py` and `metrics.py`

Deadline: 00:00 SGT 16 Nov 2026 (end of Sunday 15 Nov). Keep the repo private;
all code is screened with MOSS.

## Task 2 — collaborative filtering

**The protocol, which the report should state in these words.** Fit every
hyperparameter configuration on **train**; pick the winner by **validation
CI**; score the winners on **test**, once. `src/tuning.py` takes `train` and
`val` only — there is no argument through which it could reach test.

CI rather than MSE decides, and that choice matters here. 69.6% of DAVIS
labels sit at the pKd 5 censoring floor, so MSE rewards a model for predicting
"weak" everywhere; selecting on it would tune for the assay's detection limit
rather than for the ranking a recommender is asked to produce. Every
configuration's MSE and Hit@k are kept in `results/tables/task2_grid_*.csv`
regardless, so the report can show both and say which one chose.

**The number that should drive the discussion** is `mse_strong` — error over
pairs above the strong-binding threshold. On DAVIS the tuned models sit near
0.25 overall but near 1.4 on real binders: roughly six times worse on the pairs
anyone cares about. Overall MSE is flattered by the censored majority, exactly
as Task 1 predicted. Quote both columns or the result is misleading.

Two things worth saying because they are measured, not assumed:

- **Hyperparameters barely matter for item-kNN.** All 30 KIBA configurations
  land within 0.0044 CI of each other. Do not present `k=10, min_overlap=20` as
  a discovery.
- **Capacity and regularization interact.** CI rises steadily with the factor
  count only at `reg=5` (DAVIS 0.866 → 0.891 from 5 to 40 factors, KIBA
  0.838 → 0.854 from 10 to 90). At `reg=1` extra factors buy nothing reliable —
  KIBA *drops* from 0.830 to 0.809 between 10 and 40 — and at `reg=20` the
  factor count stops mattering at all (0.856 at every size on DAVIS).
- **The chosen factor count is not an optimum.** Both winners sit at the top of
  the factor axis: 55 on DAVIS, where it ties 40 to four decimals, and 90 on
  KIBA, still creeping upward (0.852 → 0.854 from 60). Report the factor count
  as "large enough", not as tuned.

Reference points: DeepDTA reports 0.261 / 0.878 on DAVIS and 0.194 / 0.863 on
KIBA. Our splits are not theirs, so this is **not** a head-to-head — "a tuned
CF model lands in the same range as the published deep model" is what the
evidence supports.

## Task 3 — cold targets

The cold split holds out whole **sequence groups**, so a held-out target has no
training measurements *and* no byte-identical twin in training. On it, every
Task 2 model — bias-only, matrix factorisation, item-kNN — returns byte-identical
numbers, because none has a `c_j` or `v_j` for an unseen target and all collapse
to `mu + b_i`, one value per drug. That identical row is the baseline, and it is
a far better motivation for the task than any paragraph.

Four rungs, in `src/cold.py`:

| Model | Idea | Falls back? |
|---|---|---|
| `SequenceNeighbour` | borrow the neighbours' residuals for this drug | yes, below the cosine threshold |
| `MFSequenceTransfer` | average the neighbours' `v_j` and `c_j`, reuse the MF formula | yes, same rule |
| `LatentRegressor` | ridge from k-mer profile to `(v_j, c_j)`, learned on warm targets | no rule, but an orthogonal target gets zero latents and reverts on its own |
| `ContentRegressor` | Morgan ⊕ k-mer → LightGBM, no collaborative term at all | never |

**Task 1's threshold finding predicted a modelling result.** `min_cosine=0` —
take the k nearest neighbours whatever their similarity — was included as an
ablation and lost at every k for both neighbour models on DAVIS and for
sequence kNN on KIBA. The one exception is MF + transfer on KIBA at k=3
(0.770 against 0.766), which is not the configuration that wins. On DAVIS at
k=10 it scores 0.797 against 0.815 at the 0.3 threshold, which is the elbow
Task 1 measured. Declining to guess on
40% of cold targets beats guessing on all of them.

**Score transfer and parameter transfer fail differently.** On KIBA,
`MFSequenceTransfer` falls back 7.8% of the time against `SequenceNeighbour`'s
24%, because it needs only the neighbour's fitted vector, not an observed
(drug, neighbour) measurement — a real advantage on a 75%-sparse matrix. On
DAVIS, a complete matrix, both are *identically* 38.2%, since the only failure
left is "no neighbour above threshold". The two numbers matching exactly where
theory says they must is the confirmation that the mechanism is understood.

## Revised Tasks 2 and 3 workspace

See [drug_target_recommender_fixed](drug_target_recommender_fixed/README.md) for the revised implementation, reproducibility instructions, executed notebooks, and combined Task 2/3 report.
