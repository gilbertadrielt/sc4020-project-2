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
