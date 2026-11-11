# Task 2 — warm-pair report evidence (v1)

This experiment holds out canonical ligand–sequence pairs while keeping every evaluation entity in training. Seed 2026; per-ligand 70/15/15 allocation with pre-freeze target-coverage repair. Final counts and repair moves are in data/processed/splits/task2/*/manifest.json. The official brief and accompanying design plan were unavailable; the supplied agenda governs this implementation.

## Data and scales

| dataset | raw_ligands | raw_targets | raw_finite | unique_ligands | unique_targets | unique_pairs | missing_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| davis | 68 | 442 | 30056 | 68 | 379 | 25772 | 0.0000 |
| kiba | 2111 | 229 | 118254 | 2068 | 229 | 117657 | 0.7516 |

DAVIS converts positive raw Kd in nM once using pKd=9−log10(Kd). Keep screening-limit measurements; any-censored flags and censored fractions survive mean aggregation. A merged score can include censored and uncensored repeats, so a score different from 5 can still have a censored source. KIBA uses supplied transformed scores unchanged. Canonical isomeric SMILES and exact normalized sequences define identity; pair repeats are mean-merged after conversion, with source cells/IDs, sample std and screening fractions saved.

## Methods and selection

Global, ligand and target means use warm-train only. Additive bias regularization is selected from {1,10,50} by validation. Target kNN uses overlap-restricted cosine on training-only additive-bias residuals, shrinkage n_common/(n_common+beta), K={5,10,20,50}, beta={0,10,50,100}, and positive available neighbours; no usable neighbour falls back to additive bias. MF is mu+b_ligand+b_target+U·V, trained on observed pairs by our NumPy/Numba SGD (no recommender package). The per-observation SGD objective includes L2 factor/bias penalties, hence a degree-weighted total regularizer. Ranks {0,4,8,16,32}, reg {.01,.1}, learning rate .02, at most 160 epochs and patience 16 use validation MSE, with validation concordance as trial tie-breaker. Rank zero is an SGD bias-only ablation. Predictions are not clipped; training bounds are saved. No test score selects parameters or epochs.

## Test comparison

| dataset | model | mse | ci_global | ci_per_ligand | hit1 | hit5 | hit10 | mrr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| davis | global_mean | 0.6432 | 0.5000 | 0.5000 | 0.0178 | 0.0889 | 0.1776 | 0.0819 |
| davis | ligand_mean | 0.4843 | 0.7679 | 0.5000 | 0.0178 | 0.0889 | 0.1776 | 0.0819 |
| davis | target_mean | 0.5993 | 0.6385 | 0.6985 | 0.0746 | 0.3881 | 0.5522 | 0.2368 |
| davis | bias_only | 0.4321 | 0.8052 | 0.6981 | 0.1045 | 0.3881 | 0.5522 | 0.2554 |
| davis | target_knn | 0.2465 | 0.8654 | 0.8134 | 0.3284 | 0.6716 | 0.8209 | 0.4827 |
| davis | MF | 0.2559 | 0.8654 | 0.8021 | 0.4030 | 0.6418 | 0.8060 | 0.5276 |
| davis | random_ranking | — | — | 0.5000 | 0.0178 | 0.0889 | 0.1776 | 0.0819 |
| kiba | global_mean | 0.7123 | 0.5000 | 0.5000 | 0.2222 | 0.4603 | 0.6828 | 0.4234 |
| kiba | ligand_mean | 0.4250 | 0.7075 | 0.5000 | 0.2222 | 0.4603 | 0.6828 | 0.4234 |
| kiba | target_mean | 0.6177 | 0.6807 | 0.6767 | 0.3781 | 0.6805 | 0.8637 | 0.5645 |
| kiba | bias_only | 0.3531 | 0.7854 | 0.6848 | 0.3910 | 0.6879 | 0.8718 | 0.5737 |
| kiba | target_knn | 0.2190 | 0.8419 | 0.7633 | 0.5300 | 0.8476 | 0.9457 | 0.6923 |
| kiba | MF | 0.2012 | 0.8410 | 0.7687 | 0.5683 | 0.8726 | 0.9573 | 0.7209 |
| kiba | random_ranking | — | — | 0.5000 | 0.2222 | 0.4603 | 0.6828 | 0.4234 |

MSE uses observed labels only. Higher pKd and supplied transformed KIBA mean stronger affinity. CI denotes concordance: exclude true ties, award predicted ties half credit, compare with brute force in tests. Primary rankings recommend targets for each ligand within held-out measured candidates, with all true-best ties relevant. Hit and reciprocal rank are expected values over uniform ordering of predicted ties. Singleton and all-truth-tied groups are excluded and counted. Hit@5 requires >=5 candidates and Hit@10 requires >=10; hit5_gt5 further excludes the trivial five-candidate case. Per-target rankings/CI are a secondary interpretation of “recommend a drug for a target”. Random rankings use exact expectations, with no regression MSE. Candidate sets differ across datasets/splits, so raw Hit values are not directly comparable. No unobserved KIBA pairs are labelled negatives. Group bootstrap intervals describe uncertainty within a fixed split and are distinct from concordance and five-split standard deviation.

## DAVIS: answers to the Task 2 questions

**Does it learn beyond lookup?** All evaluated test pairs were absent from training. MF test MSE is 0.255887, versus ligand mean 0.484315 and additive bias 0.432096; its relative reduction over bias is 40.78%. Pearson association of held-out true bias residuals with the latent dot product is 0.6370. This is descriptive evidence of interaction structure; MF also learns its own biases, so the dot-product correlation alone is not a causal ablation.

**Is CF necessary?** MF improves regression over the selected additive bias on this split. The training matrix density is 0.6992; the selected rank is 32, with 14752 parameters versus 448 for bias-only. Matrix completion is useful for deliberately withheld interactions, whereas observed complete DAVIS pairs can already be looked up. The parameter count/density does not establish clinical utility or superiority for unseen proteins.

**Sparsity and abnormal values.** Training co-observation quantiles (target pairs) are {'0.0': 16.0, '0.25': 30.0, '0.5': 33.0, '0.75': 36.0, '1.0': 48.0}; selected kNN is {'k': 5, 'beta': 100.0}, test fallback fraction 0.0000%. See the overlap distribution, sparsity support table, score-bin errors and screening audit below. Dense/sparse performance uses training observation counts only.

| min_train_count | max_train_count | pairs | mf_mse | bias_mse | knn_fallback_fraction |
| --- | --- | --- | --- | --- | --- |
| 100 | 100000 | 3876 | 0.2559 | 0.4321 | 0.0000 |

**Case studies from reload.** Three eligible ligands were chosen using suggested IDs when available, otherwise observed sparse/median/dense training histories. Only held-out measured targets are candidates. Top-10 needs at least ten candidates; no training labels are substituted.

| original_drug_ids | train_observations | test_candidates | true_best_predicted_rank | true_best_bias_rank | mse | ci_global | hit1 | hit5 | mrr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 44259 | 265 | 57 | 6 | 21 | 0.8889 | 0.7987 | 0.0000 | 0.0000 | 0.1667 |
| 11338033 | 265 | 57 | 4 | 46 | 0.1605 | 0.9697 | 0.0000 | 1.0000 | 0.2500 |
| 10127622 | 265 | 57 | 19 | 14 | 0.0497 | 0.6786 | 0.0000 | 0.0000 | 0.0526 |

Figures: `figures/task2_davis_comparison.png`, `figures/task2_davis_diagnostics.png`, `figures/task2_davis_learning_curve.png`. Tables: `tables/task2_davis_case_rankings.csv`, `tables/task2_davis_label_bins.csv`, `tables/task2_davis_diagnostics.csv`, `tables/task2_davis_overlap.csv`. Original-ID alias lists in case tables are the observed sources for that pair; entity tables contain the complete aliases.

## KIBA: answers to the Task 2 questions

**Does it learn beyond lookup?** All evaluated test pairs were absent from training. MF test MSE is 0.201213, versus ligand mean 0.424951 and additive bias 0.353102; its relative reduction over bias is 43.02%. Pearson association of held-out true bias residuals with the latent dot product is 0.6371. This is descriptive evidence of interaction structure; MF also learns its own biases, so the dot-product correlation alone is not a causal ablation.

**Is CF necessary?** MF improves regression over the selected additive bias on this split. The training matrix density is 0.1719; the selected rank is 32, with 75802 parameters versus 2298 for bias-only. Matrix completion is useful for deliberately withheld interactions, whereas observed complete DAVIS pairs can already be looked up. The parameter count/density does not establish clinical utility or superiority for unseen proteins.

**Sparsity and abnormal values.** Training co-observation quantiles (target pairs) are {'0.0': 0.0, '0.25': 2.0, '0.5': 20.0, '0.75': 226.0, '1.0': 504.0}; selected kNN is {'k': 10, 'beta': 50.0}, test fallback fraction 0.1082%. See the overlap distribution, sparsity support table, score-bin errors and screening audit below. Dense/sparse performance uses training observation counts only.

| min_train_count | max_train_count | pairs | mf_mse | bias_mse | knn_fallback_fraction |
| --- | --- | --- | --- | --- | --- |
| 0 | 20 | 2657 | 0.4088 | 0.6281 | 0.0075 |
| 20 | 50 | 3161 | 0.2145 | 0.3456 | 0.0000 |
| 50 | 100 | 11280 | 0.1461 | 0.2854 | 0.0000 |
| 100 | 100000 | 1390 | 0.2213 | 0.3935 | 0.0000 |

**Case studies from reload.** Three eligible ligands were chosen using suggested IDs when available, otherwise observed sparse/median/dense training histories. Only held-out measured targets are candidates. Top-10 needs at least ten candidates; no training labels are substituted.

| original_drug_ids | train_observations | test_candidates | true_best_predicted_rank | true_best_bias_rank | mse | ci_global | hit1 | hit5 | mrr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CHEMBL2004544 | 69 | 15 | 2 | 3 | 0.0991 | 0.7723 | 0.0000 | 1.0000 | 0.5000 |
| CHEMBL565612 | 116 | 25 | 1 | 22 | 0.4856 | 0.6724 | 1.0000 | 1.0000 | 1.0000 |
| CHEMBL2007574 | 42 | 10 | 1 | 9 | 0.0410 | 0.8140 | 1.0000 | 1.0000 | 1.0000 |

Figures: `figures/task2_kiba_comparison.png`, `figures/task2_kiba_diagnostics.png`, `figures/task2_kiba_learning_curve.png`. Tables: `tables/task2_kiba_case_rankings.csv`, `tables/task2_kiba_label_bins.csv`, `tables/task2_kiba_diagnostics.csv`, `tables/task2_kiba_overlap.csv`. Original-ID alias lists in case tables are the observed sources for that pair; entity tables contain the complete aliases.

## Anomaly audit and scientific limits

The following finite nonpositive KIBA observations occur in the pinned upstream file. Their presence and provenance are verified; assay-level correctness cannot be established from these files. They remain flagged and retained, without relabelling missing values or deleting an inconvenient score.

| drug_id | target_id | affinity | original_drug_id | original_target_id | source_cell | censored | suspicious |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1323 | 118 | 0 | CHEMBL373751 | P53350 | 1336:118 | False | True |

DAVIS screening-bound values are benchmark labels, not exact measured affinities. Mean-merging aliases changes the usual benchmark counts and loses assay/isoform distinctions. This is a new canonical-identity split, not the official DeepDTA folds. Only one warm seed was used. Rankings within measured candidates do not establish whole-proteome retrieval.

## Optional train-only thinning

| fraction | model | training_pairs | unseen_test_pairs | mse | ci_global |
| --- | --- | --- | --- | --- | --- |
| 1.0000 | bias_only | 18020 | 0 | 0.4321 | 0.8052 |
| 1.0000 | MF | 18020 | 0 | 0.2559 | 0.8654 |
| 0.5000 | bias_only | 9010 | 0 | 0.4384 | 0.8027 |
| 0.5000 | MF | 9010 | 0 | 0.3422 | 0.8340 |
| 0.2500 | bias_only | 4505 | 0 | 0.4504 | 0.7935 |
| 0.2500 | MF | 4505 | 0 | 0.4379 | 0.7938 |
| 0.1000 | bias_only | 1802 | 31 | 0.4914 | 0.7624 |
| 0.1000 | MF | 1802 | 31 | 0.4836 | 0.7612 |
| 0.0500 | bias_only | 901 | 363 | 0.5252 | 0.7360 |
| 0.0500 | MF | 901 | 363 | 0.5103 | 0.7357 |

Nested 100/50/25/10/5% subsets change training only; the full warm test is fixed. Full-training validation-selected hyperparameters are retained, with checkpoint epoch selected on the same validation partition for each subset. Unseen entity factors/biases are zero, giving partial/global additive fallback. Figure: `figures/task2_davis_thinning.png`.

## Paired uncertainty

| dataset | model | baseline | ligands | replicates | mse_difference | bootstrap_95_low | bootstrap_95_high | seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| davis | MF | bias_only | 68 | 1000 | -0.1762 | -0.2316 | -0.1248 | 2026 |
| kiba | MF | bias_only | 2068 | 1000 | -0.1519 | -0.1661 | -0.1391 | 2026 |

Intervals use 1,000 paired ligand-group bootstrap replicates, and describe MF-minus-bias MSE on the fixed split. They do not account for split choice.

## References and assistance


- Öztürk et al. (2018), DeepDTA, https://doi.org/10.1093/bioinformatics/bty593. Raw files from hkmztrk/DeepDTA commit a546a8433a6822e958f36171c4356ad6f414d623, with ordered mappings and SHA-256 in data/raw/*/source.json.
- DeepDTA data documentation: https://github.com/hkmztrk/DeepDTA/blob/master/data/README.md (raw Kd units, transformed KIBA, NaN missingness).
- He et al. (2017), SimBoost, https://doi.org/10.1186/s13321-017-0209-z (transformed KIBA direction and DAVIS screening-bound provenance).
- Biased MF is the standard additive-bias plus dot-product model. SGD, residual target-kNN, cold factor ridge, tie-expectation metrics and experiment orchestration are implemented locally. NumPy/SciPy/scikit-learn/RDKit/Numba provide numerical, fixed-feature and acceleration support; no pretrained recommender is used.
- Local reuse: CF SGD, Fenwick CI, 3-mer features and tests were adapted from ../drug_target_recommender/src and tests. That sibling implementation and Task 1 artifacts were preserved. Codex provided substantial implementation, analysis, testing and writing assistance. Human contributor names, contributions and the course's assistance policy must be checked by the authors; no identities or approval claims are invented.
