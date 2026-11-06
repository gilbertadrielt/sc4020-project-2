# SC4020 Project 2 — DAVIS & KIBA drug–target recommender

Drugs are users, protein targets are items, binding affinity is the rating.

## Setup

```bash
pip install -r requirements.txt
python scripts/download_data.py    # fetches DeepDTA files into data/, checksum-verified
python scripts/run_task1.py        # all Task 1 figures + tables into results/
python -m pytest tests/ -q
```

`data/` is gitignored — everyone regenerates it from the download script, which
pins a SHA-256 per file so the whole group provably works from identical bytes.

## Layout

```
data/            davis/, kiba/ (raw), processed/
src/
  data.py        load, clean, dedup — the one module everything imports
  viz.py         shared figure style for every report figure
  eda.py         Task 1 analyses, plus kmer_profiles() which Task 3 reuses
  split.py       warm 70/15/15 + cold-target split, saved to data/processed/
  metrics.py     mse, rmse, ci, hit@k, mrr
  cf.py          baselines, biased MF (ALS), item-kNN, save/load .npz
  tuning.py      validation grid search — never sees the test fold
  features.py    (Task 3) Morgan FP, k-mer, ESM embeddings
scripts/         download_data.py, run_task1.py, make_splits.py, run_task2.py
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
- [ ] One saved split file so everyone uses identical partitions (`src/split.py`)
- [ ] Main CF model: matrix factorization or kNN
- [ ] Owner of `split.py` and `metrics.py`

Deadline: 00:00 SGT 16 Nov 2026 (end of Sunday 15 Nov). Keep the repo private;
all code is screened with MOSS.
