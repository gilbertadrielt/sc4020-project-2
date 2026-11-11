# Agenda completion record — v1

The original `agenda (1).md` is preserved. This record maps its required work to executed artifacts; optional extensions are distinguished from requirements.

| Phase / gate | Status | Evidence |
| --- | --- | --- |
| 0: repository, data scale and environment audit | PASS | `implementation_log.md`, `requirements.lock.txt`, raw source manifests, README; sibling project and Task 1 preserved |
| 1: canonical observed-pair cleaning on both datasets | PASS | `data/processed/pairs_*.csv`, ligand/target entity and original-ID mapping tables, `cleaning_log.csv`, per-dataset summaries, duplicate/anomaly/exclusion audits |
| 1: DAVIS screening and KIBA missingness | PASS | Screening fractions/flags retained through merging; nonpositive Kd rejection tested; KIBA source missing key set preserved; single finite zero flagged/retained without invented assay validity |
| 2: warm pair split seed 2026 | PASS | Frozen pair keys, per-ligand stratification, target repair log, entity mappings, SHA-256 and automated coverage/intersection checks |
| 2: exact-sequence cold holdout | PASS | Five frozen splits per dataset, seeds 2026–2030; all partition sequence/target intersections empty, unseen-ligand counts recorded |
| 3: metrics and baselines | PASS | Exact MSE; Fenwick concordance versus brute force; prediction ties half-credit; true ties excluded; expected tie-aware Hit@1/5/10 and MRR checked by permutation enumeration; all-tied/singleton/small candidate counts; mean/bias/random baselines |
| 4A: NumPy core CF on both datasets | PASS | Ten MF configurations per dataset (including rank 0); validation-selected bias; sixteen target-kNN configurations; `results/hparams.csv`; no missing cells used as labels |
| 4A: saved and independently reloaded checkpoints | PASS | `{davis,kiba}_cf.npz`, auxiliary baseline/kNN archives; original-ID map length assertions, float64 arrays and non-pickle strings; fresh-process score and exact ordering checks |
| 4B: selected-model evaluation and brief questions | PASS against agenda | All train/validation/test metrics, latent residual correlation, density/parameter counts, label/censor bins, overlap/support/fallback diagnostics and actual comparisons in `task2_report_notes.md`; official brief absent |
| 4C: three ligand cases per dataset | PASS | Six eligible cases with >=10 observed test candidates, reload predictions, actual source target IDs, true-best and bias ranks, MSE/concordance/Hit/MRR, CSV and Markdown plus executed case notebook |
| 5A: cold baselines and features | PASS | C0 ligand means, C1 sequence-neighbour residual transfer, fixed sequence 3-mer/AAC/dipeptide matrices and canonical Morgan fingerprints, keyed caches and hashes; training-only label use |
| 5B: learned sequence-aware model | PASS | C2 cold-trained MF plus train-centered multioutput sequence-to-bias/factor ridge on both datasets; no query target-ID factors; cold-validation alpha selection |
| 5C: five-split evaluation | PASS | Per-split JSON/checkpoints/CSV, test MSE/global and grouped concordance/Hit/MRR, five-split sample std, nearest-training cosine audits and pre-specified similarity buckets/counts/plots |
| 6: execution and evidence | PASS | Unit/integrity tests; actual-source subset smoke; full pipeline; three notebooks executed in fresh kernels; independent prediction/metric reconciliation; reproducibility manifest and generated reports |
| Optional: paired uncertainty | DONE | 1,000 paired ligand-group bootstrap replicates; confidence bounds separate from concordance and split std |
| Optional: DAVIS train density | DONE | Nested 100/50/25/10/5% train-only subsets, fixed test, explicit unseen-entity fallback/counts and plot |
| Optional: C3 pair regressor / ESM-2 / strict-family split / ensemble / mixup / censored-loss MF | NOT RUN | C2 satisfies learned-model requirement; no claims made for optional alternatives |
| Supplied Smith–Waterman matrix | UNAVAILABLE LOCALLY | No such files found; fixed-vocabulary sequence cosine used instead, documented as not alignment identity |
| Official rubric, accompanying design plan, human contribution details | UNAVAILABLE | Agenda used as specification; authors must supply/check actual course requirements and human attribution |

Model selection uses validation MSE, with validation concordance as trial tie-breaker. Fixed test scores are independently re-read to verify saved artifacts, never to tune. Warm-pair and cold-target protocols are reported separately. Artifact reconciliation uses exact round-trip score parsing to preserve ranking ties. `verification.json` and `reproducibility_manifest.json` record the final checks and hashes.
