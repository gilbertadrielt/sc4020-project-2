# What this merge added, and what it changed

Everything below sits on top of the existing pipeline. No split, no kernel and no
existing model was re-specified, so every KronRLS, kNN and MF number is unchanged to
four decimals. `python -m pytest` goes from 53 tests to 60.

## 1. Four more protein representations (`kernels.py`)

`build_features` previously emitted one protein block, `target`, holding 420-dimensional
amino-acid and dipeptide composition. It now emits five:

| Block | Dim (DAVIS / KIBA) | Source |
| --- | --- | --- |
| `target` | 420 / 420 | unchanged, composition |
| `target_kmer_svd` | 256 / 228 | 3-mer TF-IDF reduced by SVD |
| `target_sw_profile` | 379 / 229 | normalised Smith-Waterman kernel rows |
| `target_esm2_35m` | 480 / 480 | mean-pooled ESM-2, 35M parameters |
| `target_esm2_650m` | 1280 / 1280 | mean-pooled ESM-2, 650M parameters |

The two ESM-2 blocks load from `features/esm2_{size}_{dataset}.npy`, produced on a GPU
by `scripts/colab_esm2.py`. They are indexed by `proteins.txt` identifier and mapped onto
this project's distinct-sequence entities by sequence lookup; members of a merged group
have a byte-identical sequence so their embeddings are identical and the mapping is exact.
**If the files are absent the pipeline runs unchanged** and the representation drops out
of the ablation, so no GPU is required to reproduce anything else.

## 2. GBA-Mixup (`cold.py`)

`gba_mixup()` implements the augmentation from MixingDTA (Kim et al., *Bioinformatics*
41:i105-i114, 2025), which the project brief cites and which nothing in the repo
previously used. Pairs sharing a drug or a target are treated as graph neighbours, a
partner is sampled with a preference for a similar label, and feature vectors and labels
are interpolated with lambda drawn from Beta(0.4, 0.4). `ContentRegressor` gained
`mixup=` and `mixup_ratio=`; virtual rows are appended before fitting. Registered as
`C5_content_mixup`.

## 3. Representation ablation (`run_analysis.py`)

`representation_ablation()` runs `ContentRegressor` over all five blocks with mixup on
and off, five seeds, S3. Writes `results/tables/representation_{dataset}.csv`.

## Changes to existing behaviour, listed so nobody is surprised

1. **`ContentRegressor` and `FactorMap` default to `target_sw_profile`, not composition.**
   This is the one default that moved. It affects S2 and S4, where the block is not tuned.
2. **`protocol.py` tunes `target_block` on validation** for C2, C3 and C5 in S3. Tuning
   selects `target_sw_profile` for all three on both datasets.
3. **Hit@k and MRR are exact expectations over predicted ties**, not one random draw.
   `_expected_hit_at_k` and `_expected_reciprocal_rank` replace the single-permutation
   path. Non-degenerate models are unaffected because they have no ties. A constant
   predictor now scores *exactly* the random row with zero seed-to-seed variance, where
   before it scattered around it with a standard deviation as large as the value.
4. **The case study uses the three ligands the brief names per dataset**, which all
   survive deduplication, with the previous heuristic picks kept as extra cases.
5. **The headline table's "Best MSE" column is split** into "Its MSE" (the best-by-CI
   model's own MSE) and "Lowest MSE here". The two differ in four of the eight regimes,
   so the old single column overstated what was being claimed.

## What the changes bought

Cold-target S3, mean of five seeds. Kernel and neighbour models are unchanged.

| Model | DAVIS MSE before | after | DAVIS CI/drug before | after |
| --- | --- | --- | --- | --- |
| C3 content regression | 0.4428 | **0.2962** | 0.6149 | **0.7702** |
| C2 factor map | 0.4744 | **0.4171** | 0.5634 | **0.6764** |

| Model | KIBA MSE before | after | KIBA CI/drug before | after |
| --- | --- | --- | --- | --- |
| C3 content regression | 0.4199 | **0.3250** | 0.5692 | **0.6566** |
| C2 factor map | 0.4180 | **0.3957** | 0.5678 | 0.5860 |

The feature-based models were not weak ideas; they were reading the weakest of the five
descriptors available. On DAVIS S3, `C5_content_mixup` now takes the top per-drug CI
(0.778) from KronRLS (0.769).

## Two findings the report should carry

**Smith-Waterman beats ESM-2, consistently.** Mean over five seeds, S3, same model and
split, only the representation varying:

| Representation | DAVIS MSE | DAVIS CI/drug | KIBA MSE | KIBA CI/drug |
| --- | --- | --- | --- | --- |
| composition (the old default) | 0.4428 | 0.6149 | 0.4199 | 0.5692 |
| 3-mer SVD | 0.4196 | 0.6965 | 0.4350 | 0.6005 |
| ESM-2 35M | 0.3827 | 0.6960 | 0.3919 | 0.5904 |
| ESM-2 650M | 0.3591 | 0.7032 | 0.3565 | 0.6539 |
| **Smith-Waterman** | **0.2962** | **0.7702** | **0.3250** | **0.6566** |

A plausible mechanism, offered as such: both panels are kinases, so what separates them
is a few dozen residues lining the ATP pocket against a background of high global
homology. Local alignment scores that directly; mean-pooling a language model over a
700-residue multi-domain protein averages it away. This is a claim about mean-pooled
whole-sequence embeddings on a single-family panel, not about protein language models in
general, and a binding-site-pooled ESM-2 is the obvious untested alternative.

**GBA-Mixup does not help here.** Differences land in the third decimal with inconsistent
sign (DAVIS 0.2962 against 0.2968, KIBA 0.3250 against 0.3290). Report it as implemented,
ablated and null rather than quietly dropping it: the brief cites the paper, and a
negative result that was properly tested is worth more than a method left out.
