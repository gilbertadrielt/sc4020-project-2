# SC4020 Tasks 2 and 3 — executed evidence


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


# Task 3 — cold-target report evidence (v1)

Five separately trained target-sequence holdouts use seeds 2026–2030. Approximately 70/15/15% of unique exact normalized sequences go to train/validation/test. Every affinity measurement for a sequence remains in one partition. All model selection uses cold validation only; test sequences have zero training affinity observations. Evaluation ligands without training observations are explicitly flagged and retain a global/partial-bias fallback; exact counts are in split manifests and saved prediction files.

## Algorithms

**C0:** ligand means from each cold-training partition, with global training mean for unknown ligands. **C1:** compare a cold full sequence with training sequences using L2-normalized fixed A–Z 3-mer count cosine. Take top K training proteins; transfer available observations for that ligand, weighted by cosine^power, otherwise use C0. This is algebraically C0 plus weighted ligand-mean residuals. K={5,15,50}, power={1,4} are selected on cold validation. Missing ligand–neighbour measurements contribute no weight.

**C2 learned factor ridge:** fit a separate rank-16 biased MF for up to 80 fixed-budget epochs on cold-train only (train-error checkpoint, reg .01 and LR .02); never import warm-task factors. Represent sequences with fixed 3-mer, 20-AA composition and 400-dipeptide frequencies. Train-target-centered multioutput ridge maps these sequence features to learned training-target bias and factors. Tune ridge alpha {.1,1,10,100} on cold validation. Predict mu+b_ligand+predicted_bias+U_ligand·predicted_factor. Cold queries do not use target-ID factors. Unobserved ligand factors are zero. The MF optimizer seed is fixed at 2026 for all split seeds. Cached radius-2 2048-bit Morgan features are available; C2 uses cold-train ligand factors instead. No SVD, imputation, pretrained embedding or test-label fitting is used.

Neither dataset supplied an S-W similarity file locally, so this implementation computes label-free sequence cosine instead of downloading or assuming a normalization. Cosine is bounded 0..1, but is not alignment identity. Pre-specified <.3/.3–.5/.5–.7/>=.7 buckets describe this feature similarity; do not interpret them as sequence-identity family thresholds.

## Five-split aggregate

| dataset | model | mse_mean | mse_std | ci_global_mean | ci_global_std | hit1_mean | hit5_mean | mrr_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| davis | factor_ridge | 0.3374 | 0.0429 | 0.8365 | 0.0084 | 0.2657 | 0.5184 | 0.3953 |
| davis | ligand_mean | 0.4800 | 0.0711 | 0.7675 | 0.0076 | 0.0177 | 0.0885 | 0.0816 |
| davis | sequence_neighbors | 0.3025 | 0.0395 | 0.8448 | 0.0122 | 0.3407 | 0.6143 | 0.4645 |
| kiba | factor_ridge | 0.3240 | 0.0161 | 0.7564 | 0.0222 | 0.3614 | 0.6672 | 0.5380 |
| kiba | ligand_mean | 0.3868 | 0.0358 | 0.7077 | 0.0221 | 0.1969 | 0.4784 | 0.3918 |
| kiba | sequence_neighbors | 0.3061 | 0.0281 | 0.7694 | 0.0244 | 0.3755 | 0.7265 | 0.5573 |

The table gives arithmetic mean and sample standard deviation (ddof=1) across five repeated target splits; they are not independent five-fold CV, confidence intervals, or bootstrap variability. Every model is retrained per split. Full metric means/std/counts, including per-ligand and per-target concordance, are in tables/task3_aggregate.csv. Per-split primary model selection is saved before test evaluation and its aggregate is in tables/task3_validation_selected_aggregate.csv.

MSE uses observed labels only. Higher pKd and supplied transformed KIBA mean stronger affinity. CI denotes concordance: exclude true ties, award predicted ties half credit, compare with brute force in tests. Primary rankings recommend targets for each ligand within held-out measured candidates, with all true-best ties relevant. Hit and reciprocal rank are expected values over uniform ordering of predicted ties. Singleton and all-truth-tied groups are excluded and counted. Hit@5 requires >=5 candidates and Hit@10 requires >=10; hit5_gt5 further excludes the trivial five-candidate case. Per-target rankings/CI are a secondary interpretation of “recommend a drug for a target”. Random rankings use exact expectations, with no regression MSE. Candidate sets differ across datasets/splits, so raw Hit values are not directly comparable. No unobserved KIBA pairs are labelled negatives. Group bootstrap intervals describe uncertainty within a fixed split and are distinct from concordance and five-split standard deviation.

## Per-split results

| dataset | seed | model | selected_by_validation | mse | ci_global | ci_per_ligand | ci_per_target | hit1 | hit5 | mrr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| davis | 2026 | ligand_mean | False | 0.3969 | 0.7682 | 0.5000 | 0.8009 | 0.0175 | 0.0877 | 0.0812 |
| davis | 2026 | sequence_neighbors | True | 0.2794 | 0.8327 | 0.7405 | 0.8638 | 0.3030 | 0.5152 | 0.4018 |
| davis | 2026 | factor_ridge | False | 0.2860 | 0.8288 | 0.7121 | 0.8596 | 0.2273 | 0.4697 | 0.3535 |
| davis | 2027 | ligand_mean | False | 0.4441 | 0.7658 | 0.5000 | 0.8053 | 0.0178 | 0.0889 | 0.0819 |
| davis | 2027 | sequence_neighbors | True | 0.2793 | 0.8498 | 0.7612 | 0.8718 | 0.2794 | 0.6471 | 0.4396 |
| davis | 2027 | factor_ridge | False | 0.3285 | 0.8264 | 0.7229 | 0.8585 | 0.1618 | 0.4706 | 0.3012 |
| davis | 2028 | ligand_mean | False | 0.4995 | 0.7639 | 0.5000 | 0.8011 | 0.0175 | 0.0877 | 0.0812 |
| davis | 2028 | sequence_neighbors | True | 0.3161 | 0.8325 | 0.7527 | 0.8505 | 0.3333 | 0.5606 | 0.4441 |
| davis | 2028 | factor_ridge | False | 0.3374 | 0.8406 | 0.7688 | 0.8511 | 0.3030 | 0.4848 | 0.4160 |
| davis | 2029 | ligand_mean | False | 0.5876 | 0.7595 | 0.5000 | 0.8069 | 0.0175 | 0.0877 | 0.0812 |
| davis | 2029 | sequence_neighbors | True | 0.3661 | 0.8479 | 0.7604 | 0.8729 | 0.2879 | 0.5909 | 0.4166 |
| davis | 2029 | factor_ridge | False | 0.4050 | 0.8407 | 0.7642 | 0.8618 | 0.2424 | 0.5303 | 0.3797 |
| davis | 2030 | ligand_mean | False | 0.4721 | 0.7799 | 0.5000 | 0.8097 | 0.0181 | 0.0902 | 0.0827 |
| davis | 2030 | sequence_neighbors | True | 0.2718 | 0.8611 | 0.7871 | 0.8750 | 0.5000 | 0.7576 | 0.6206 |
| davis | 2030 | factor_ridge | False | 0.3301 | 0.8457 | 0.7464 | 0.8630 | 0.3939 | 0.6364 | 0.5264 |
| kiba | 2026 | ligand_mean | False | 0.4395 | 0.7010 | 0.5000 | 0.7656 | 0.2163 | 0.5013 | 0.4216 |
| kiba | 2026 | sequence_neighbors | False | 0.3220 | 0.7845 | 0.6656 | 0.8106 | 0.4192 | 0.7973 | 0.6135 |
| kiba | 2026 | factor_ridge | True | 0.3309 | 0.7802 | 0.6726 | 0.8036 | 0.4567 | 0.8017 | 0.6318 |
| kiba | 2027 | ligand_mean | False | 0.3826 | 0.7226 | 0.5000 | 0.8069 | 0.2249 | 0.5020 | 0.4240 |
| kiba | 2027 | sequence_neighbors | True | 0.2783 | 0.7993 | 0.6898 | 0.8168 | 0.4385 | 0.7827 | 0.6185 |
| kiba | 2027 | factor_ridge | False | 0.3169 | 0.7678 | 0.6824 | 0.8244 | 0.4117 | 0.7042 | 0.5934 |
| kiba | 2028 | ligand_mean | False | 0.3952 | 0.6780 | 0.5000 | 0.7543 | 0.2534 | 0.5124 | 0.4583 |
| kiba | 2028 | sequence_neighbors | True | 0.3213 | 0.7373 | 0.6314 | 0.7820 | 0.4178 | 0.7556 | 0.6056 |
| kiba | 2028 | factor_ridge | False | 0.3017 | 0.7408 | 0.6663 | 0.7933 | 0.4330 | 0.7067 | 0.6107 |
| kiba | 2029 | ligand_mean | False | 0.3760 | 0.7354 | 0.5000 | 0.7494 | 0.1512 | 0.4613 | 0.3383 |
| kiba | 2029 | sequence_neighbors | True | 0.3351 | 0.7717 | 0.6180 | 0.7896 | 0.2638 | 0.5959 | 0.4409 |
| kiba | 2029 | factor_ridge | False | 0.3451 | 0.7672 | 0.6023 | 0.7764 | 0.2520 | 0.5547 | 0.4285 |
| kiba | 2030 | ligand_mean | False | 0.3405 | 0.7014 | 0.5000 | 0.7710 | 0.1384 | 0.4148 | 0.3169 |
| kiba | 2030 | sequence_neighbors | True | 0.2735 | 0.7543 | 0.6386 | 0.8014 | 0.3380 | 0.7009 | 0.5079 |
| kiba | 2030 | factor_ridge | False | 0.3252 | 0.7261 | 0.5820 | 0.7912 | 0.2537 | 0.5686 | 0.4255 |

## Coverage, costs and failure analysis

| dataset | seed | primary_model | knn_test_fallback_fraction | unseen_ligand_test_pairs | cold_mf_seconds | whole_run_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| davis | 2026 | sequence_neighbors | 0.0000 | 0 | 0.2360 | 5.4425 |
| davis | 2027 | sequence_neighbors | 0.0000 | 0 | 0.2262 | 4.1375 |
| davis | 2028 | sequence_neighbors | 0.0000 | 0 | 0.2277 | 4.1803 |
| davis | 2029 | sequence_neighbors | 0.0000 | 0 | 0.2338 | 4.1206 |
| davis | 2030 | sequence_neighbors | 0.0000 | 0 | 0.2320 | 4.0985 |
| kiba | 2026 | factor_ridge | 0.0015 | 0 | 0.4964 | 6.5873 |
| kiba | 2027 | sequence_neighbors | 0.0036 | 0 | 0.5042 | 5.8722 |
| kiba | 2028 | sequence_neighbors | 0.0009 | 0 | 0.4970 | 5.7696 |
| kiba | 2029 | sequence_neighbors | 0.0019 | 0 | 0.5477 | 5.7960 |
| kiba | 2030 | sequence_neighbors | 0.0009 | 0 | 0.4790 | 5.8702 |

Sequence neighbours may have no observed measurement for a query ligand, especially in sparse KIBA; C0 then provides no target-specific adjustment. The factor-ridge method supplies an adjustment through sequence features, but a linear sequence-to-latent mapping can miss target selectivity and overfit small training target sets. Composition and 3-mers lose order beyond three residues; variable full sequence lengths and domains alter cosine. Distribution shift and related sequences remaining across splits limit claims of remote-homology generalization. Per-target errors, sequence lengths, neighbour support and nearest-training similarities are exported for each split; bucket counts must accompany any performance claim. Empty buckets remain empty.

Files: tables/task3_{dataset}_seed{seed}_similarity_buckets.csv, *_per_target.csv and *_sequence_audit.csv; figures/task3_{dataset}_seed{seed}_similarity.png and *_comparison.png. Saved neighbours and ridge archives contain training-only target sequences, observations or latent outputs, provenance, parameters and split hashes. Every archive was checked in a fresh interpreter with label-free queries, and identical score order after reload.

## Scope and limitations

All core and priority-one models/splits/diagnostics in the agenda are implemented. The optional C3 pair regressor, ESM-2, strict family cold split, mixup/ensemble and censored-loss MF were not run; no results are asserted for them. Original SW matrices were not supplied and were not assumed available. The official rubric and project plan were not found, so exact rubric compliance beyond the agenda cannot be certified. Warm-pair and cold-target errors answer different questions and are not compared as the same protocol.

## References and assistance


- Öztürk et al. (2018), DeepDTA, https://doi.org/10.1093/bioinformatics/bty593. Raw files from hkmztrk/DeepDTA commit a546a8433a6822e958f36171c4356ad6f414d623, with ordered mappings and SHA-256 in data/raw/*/source.json.
- DeepDTA data documentation: https://github.com/hkmztrk/DeepDTA/blob/master/data/README.md (raw Kd units, transformed KIBA, NaN missingness).
- He et al. (2017), SimBoost, https://doi.org/10.1186/s13321-017-0209-z (transformed KIBA direction and DAVIS screening-bound provenance).
- Biased MF is the standard additive-bias plus dot-product model. SGD, residual target-kNN, cold factor ridge, tie-expectation metrics and experiment orchestration are implemented locally. NumPy/SciPy/scikit-learn/RDKit/Numba provide numerical, fixed-feature and acceleration support; no pretrained recommender is used.
- Local reuse: CF SGD, Fenwick CI, 3-mer features and tests were adapted from ../drug_target_recommender/src and tests. That sibling implementation and Task 1 artifacts were preserved. Codex provided substantial implementation, analysis, testing and writing assistance. Human contributor names, contributions and the course's assistance policy must be checked by the authors; no identities or approval claims are invented.
