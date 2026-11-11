# SC4020 Project 2 — Codex Implementation Agenda

## Mission

Implement **Task 2 (collaborative filtering on known ligands/targets)** and **Task 3 (cold-target prediction)** on **DAVIS and KIBA**, starting with a **shared, auditable data-cleaning pipeline**. Use the accompanying project plan as a design reference, but **verify all numerical claims against the actual files** rather than hard-coding them. Preserve Task 1 outputs if already present; do not unnecessarily rewrite teammates' work.

**Primary priorities:** (1) no leakage; (2) working end-to-end training/evaluation on both datasets; (3) answers to the project questions with saved results and figures; (4) reproducible notebook/report evidence. The planned submission deadline is **16 Nov 2026, 00:00 SGT**, so prioritize the required pipeline and models before stretch experiments.

## Instructions to Codex

1. **Inspect the repository first**: list files, read README/requirements/existing notebooks/scripts, identify data formats, and reuse working code where possible. Do not delete or overwrite source data or teammates' results.
2. **Work incrementally, in the phase order below**. After each phase, run a small test and record what passed, failed, or is unavailable in `results/implementation_log.md`.
3. Use **seed 2026**, centralized configuration, deterministic splits, and versioned output files. Add assertions and informative errors. Do not access test labels for tuning or training.
4. Never fabricate dataset counts, performance, references, timings, or success claims. If dependencies or data are missing, document them and supply executable code rather than synthetic 'final' results.
5. Implement core CF methods ourselves in NumPy (no pretrained recommender package); standard libraries such as NumPy, scipy, scikit-learn, RDKit, and LightGBM are fine for supporting components. Cite borrowed ideas/code where required by course policy.
6. Keep notebooks thin; reusable logic belongs in `src/`. Avoid data leakage through preprocessing, duplicate entities, target similarity, feature scaling, hyperparameter selection, and split reuse.
7. Keep **Task 2 warm-pair splits** and **Task 3 cold-target splits** separate; do not compare their metrics as if they were the same test protocol.
8. Where a choice is ambiguous in the project brief, implement the primary interpretation in the plan and clearly flag the ambiguity (especially prediction direction and metric tie conventions).

## Expected repository structure

```text
project2/
  data/
    raw/davis/                  # immutable downloaded originals
    raw/kiba/
    processed/
      pairs_davis.csv
      pairs_kiba.csv
      ligands_davis.csv
      targets_davis.csv
      ligands_kiba.csv
      targets_kiba.csv
      cleaning_log.csv
      splits/
  src/
    config.py
    data.py
    splits.py
    metrics.py
    baselines.py
    cf.py
    features.py
    cold.py
    plots.py
  scripts/
    prepare_data.py
    make_splits.py
    train_cf.py
    evaluate_cf.py
    run_case_studies.py
    train_cold.py
    evaluate_cold.py
    export_results.py
  notebooks/
    02_task2_cf.ipynb
    03_task2_case_study.ipynb
    04_task3_cold.ipynb
  saved_models/
  results/
    tables/
    figures/
    implementation_log.md
    hparams.csv
  tests/
  requirements.txt
  README.md
  agenda.md
```

Adapt paths to the actual repository; do **not** create duplicate functionality just to match names.

---

## Phase 0 — Repository audit and setup

- [ ] Find DAVIS/KIBA raw files, prior preprocessing, existing Task 1 work, and any grading rubric/brief.
- [ ] Confirm DAVIS raw labels are **Kd in nM** versus already-transformed **pKd**; do not transform twice.
- [ ] Confirm KIBA score orientation (whether larger/smaller means better binding) and consistently apply it to ranking, case studies, and labels.
- [ ] Install/pin dependencies, record Python/package versions, and add concise run commands to README.
- [ ] Establish a smoke-test command that runs a few epochs or a small subset without changing final frozen splits.

**Gate:** The source format and score scales are known; raw files are immutable.

## Phase 1 — Clean and standardize the data FIRST

Implement `src/data.py` and `scripts/prepare_data.py`.

- [ ] Convert each dataset to long-form observed pairs with `ligand_id`, `target_id`, `smiles`, `sequence`, `score` and provenance fields.
- [ ] Keep missing KIBA matrix entries **missing**; never convert NaN to a zero/negative/non-binder label.
- [ ] Remove non-finite *observed* scores and invalid structures/sequences where needed; report all exclusions.
- [ ] For DAVIS raw Kd, require `Kd > 0` and calculate `pKd = 9 - log10(Kd_nM)`; if already pKd, preserve it.
- [ ] Flag DAVIS `Kd = 10000 nM` / `pKd = 5` as **censored/screening-limit**; **keep** them. Do not assume measured affinity equals the screening bound exactly in scientific interpretation, but use the supplied label for standard benchmark MSE.
- [ ] Flag (do not silently drop) suspicious KIBA scores including a possible 0; verify whether each is valid before treating it as anomalous.
- [ ] Canonicalize valid ligand SMILES with RDKit; retain mapping to original drug IDs.
- [ ] Normalize protein sequence formatting and deduplicate on **(canonical SMILES, exact sequence)**, not on source IDs. If repeated measurements conflict, aggregate observed scores by mean and save repeat count, std, source IDs, and censored fraction.
- [ ] Build unique ligand and target entity tables, preserving original IDs and stable processed keys.
- [ ] Log counts after each cleaning step, raw versus unique ligand/target/pair counts, missingness, score distribution, and all flags.
- [ ] Validate: unique pair keys, finite observed labels, consistent score scale, no empty SMILES/sequence, no accidental filled KIBA gaps.

**Outputs:** cleaned pair files, entity files, cleaning log, small summary JSON. **Gate:** Each dataset reloads cleanly, with no duplicate canonical pair and an explainable count change. Expected counts in the supplied plan are **sanity checks only**, not assertions.

## Phase 2 — Frozen, leakage-safe splits

Implement `src/splits.py` and `scripts/make_splits.py`. Persist indices/keys to disk; reuse them in every experiment.

### Task 2 warm split

- [ ] 70% train / 15% validation / 15% test of **unique ligand-target pairs** with seed 2026.
- [ ] Aim for per-ligand stratification; ensure **every ligand and target in validation/test is present in train**, moving/adjusting assignments only before freeze. Log final sizes and moves.
- [ ] Assert train/val/test pair intersections are empty, and all held-out entities are warm.
- [ ] Freeze the resulting split files and record SHA-256 hashes; never recreate silently in evaluation.

### Task 3 cold-target split

- [ ] Split by **unique protein sequence**, not target ID; all measurements for a sequence must be in a single partition. Approximate 70/15/15 by target entities, 5 reproducible seeds if time permits.
- [ ] Ensure train/val/test target sets and sequences are disjoint, with cold validation and cold test. Ensure evaluation ligands have training observations or explicitly flag/exclude them with counts.
- [ ] Keep target-similarity computation from using held-out *affinity labels*; externally supplied protein sequence similarity can be used as an input feature.
- [ ] Persist all five splits, hashes, entity mappings and metadata. If short on time, first complete **one** valid split for both datasets before running additional seeds.

**Gate:** Automated leakage assertions pass for every frozen split.

## Phase 3 — Metrics and simple baselines (shared)

Implement `src/metrics.py`, `src/baselines.py`, and unit tests.

- [ ] MSE for train, validation, test; optional score-bin MSE and DAVIS censored-versus-uncensored MSE.
- [ ] **Concordance index (CI)** with true-label ties excluded and predicted ties awarded half credit; verify the formula and tie handling against a brute-force implementation on a small example. Do not confuse concordance index with confidence intervals.
- [ ] Per-ligand ranking metrics: Hit@1, Hit@5, Hit@10 where defined, MRR and optionally NDCG@5/per-ligand CI. Define the 'best' true target by the confirmed score direction, make ties explicit, skip/report undersized or all-tied candidate sets.
- [ ] Evaluate per-target ranking as a secondary view if needed to address the wording 'recommend drug for target'.
- [ ] Baselines: **global mean**, **ligand mean**, **target mean**, **regularized additive bias** `mu + b_ligand + b_target`, plus **random ranking** for Hit@k context.
- [ ] Never use validation/test labels in mean/bias estimates. Fallback consistently for unknown entities.
- [ ] Optional 95% bootstrap intervals by resampling ligand groups (respect dependence); report bootstrap intervals separately from CI.

**Gate:** Unit tests cover exact MSE, concordant/discordant/prediction-tied/truth-tied pairs, rankings with ties and cold/warm edge cases.

---

## Phase 4 — Task 2: Collaborative Filtering (40%)

**Goal:** Predict unseen drug-target **pairs among known entities**, showing the model learns beyond lookup/bias tables. Implement on **both DAVIS and KIBA**.

### 4A. Implement models

- [ ] **M1: item/target-based kNN** over *training-only bias residuals*, with co-observation overlap shrinkage: `sim_shrunk = sim * n_common / (n_common + beta)`; tune `K={5,10,20,50}`, `beta={0,10,50,100}` as feasible. If no usable neighbours, fall back to the additive-bias baseline and count fallbacks.
- [ ] **M2: biased matrix factorization (primary)** using training-only observed pairs. Implement ALS or seeded SGD in NumPy:

  `prediction(i,j) = mu + b_i + c_j + U[i] dot V[j]`

  Minimize observed-pair squared error plus regularization for factors and biases. Never treat unobserved KIBA cells as training labels.
- [ ] Search rank `k` (start with `{0, 4, 8, 16, 32}`; extend to `{0,2,4,8,16,32,64}` if feasible), regularization, and epochs/early stopping using **validation only**.
- [ ] Clip predictions to the **training-label** range if appropriate and log whether clipping was used.
- [ ] Save all hyperparameter trials in `results/hparams.csv`, select final model using validation MSE (CI as tie-breaker), and log settings.
- [ ] Save `saved_models/davis_cf.npz` and `saved_models/kiba_cf.npz`: `mu`, biases, factors, mapping keys/IDs, score name and bounds, hyperparameters, seed, split hash; use non-pickle string arrays.
- [ ] Reload each `.npz` in a **fresh process**, reproduce predictions (`allclose` with tight tolerance) and identical rankings, and print/save a reload-check result.

### 4B. Evaluate, compare, and answer the brief

- [ ] Produce **both-dataset** tables for baselines, kNN and MF: train/validation/test MSE and CI; test Hit@1, Hit@5, MRR, with ranking candidate counts and random-ranking context.
- [ ] Evaluate **test once per selected final model**. Log date, git commit, split hash, parameters and metrics.
- [ ] Demonstrate **not a lookup**: predictions are for held-out pairs; compare MF vs ligand mean and bias-only, and compute held-out residual association between true residuals and `U·V` (interpret with caution).
- [ ] Address **is CF necessary?** with observed improvement (or lack thereof) over bias-only, plus the density and parameter-count argument. Do not claim CF is better unless the results support it.
- [ ] Address **KIBA sparsity and abnormal values** with observed-count/overlap distributions, neighbour availability, fallback rates, and changes to performance across dense vs sparse ligands.
- [ ] Generate test true-versus-predicted plot and error by label bin, ensuring bins reflect the correct dataset scale.
- [ ] Optional high-value experiment: hold Task 2 test fixed on DAVIS; subsample only the **training** interactions to 100%, 50%, 25%, 10%, 5%; retrain both bias-only and MF; plot test performance vs training density. Clearly flag if thinning makes some evaluation entities unseen and define a consistent fallback.

### 4C. Case study from the **reloaded** models

- [ ] Pick **three ligands per dataset with enough test candidates**. Suggested candidates from the plan (verify IDs and eligibility before use): DAVIS `44259`, `11338033`, `10127622`; KIBA `CHEMBL1967538`, `CHEMBL2004544`, `CHEMBL565612`.
- [ ] Explain why each was picked (selectivity/promiscuity or sparse/dense history), based on actual data.
- [ ] For each: top-ranked held-out target table (`target_id`, prediction, true score, true-best indicator/correct-wrong), predicted rank of actual best, per-ligand MSE, CI where defined, Hit@1, Hit@5, MRR, and bias-baseline rank.
- [ ] If too few held-out candidates for meaningful top-10 or CI, **state this**; never substitute training labels as 'test' cases.
- [ ] Save concise CSV/Markdown tables and generate `notebooks/03_task2_case_study.ipynb` that loads models rather than refitting them.

**Task 2 completion gate:** Both datasets have an honest comparison table, saved/reloaded CF models, cases, and explicit written answers to all Task 2 questions.

---

## Phase 5 — Task 3: Unseen target cold start (30%)

**Goal:** Predict interactions for proteins whose **exact sequences have zero affinity observations in training**. This is a different evaluation from Task 2: **do not fit target-ID factors for test proteins**.

### 5A. Baselines and features

- [ ] C0: per-ligand-mean / `mu + b_ligand` cold-start fallback learned on cold-train.
- [ ] C1: **sequence-similarity kNN**. For each cold protein, rank only **training proteins** by sequence similarity; for each ligand, aggregate available ligand-training-target residuals, with a robust fallback if none exist. Tune K and similarity exponent on **cold validation** only.
- [ ] If using provided Smith-Waterman matrix for DAVIS, normalize by self-scores `s(a,b)/sqrt(s(a,a)*s(b,b))`; verify available target-key alignment. Check whether KIBA similarities are already normalized rather than assuming.
- [ ] Protein features: amino-acid composition; dipeptide composition and/or 3-mer TF-IDF with optional train-fit SVD. Compute from sequence only. Save keyed feature matrices.
- [ ] Drug features: canonical Morgan fingerprints (radius 2, 2048 bits), or cold-train CF drug factors. Save keyed feature matrices.
- [ ] Fit transformations, imputers, PCA/SVD and model selection **only on cold-train/validation as appropriate**, never on test labels.

### 5B. Implement a learned cold-start method

- [ ] C2 (good if time-limited): fit biased MF on **cold-train** and learn ridge regression to map protein features to its trained target bias and factors; predict via `mu + b_ligand + predicted_target_bias + U_ligand dot predicted_target_factor`.
- [ ] C3 (alternative or enhancement): train a pairwise regressor (e.g., LightGBM, ridge or modest MLP) on concatenated drug and protein features. Tune on cold validation; report computational cost and feature choices.
- [ ] Ensure at least **one working sequence-aware method** beats or fairly compares against C0 for **both datasets** before attempting expensive embeddings.
- [ ] Optional: ESM-2 protein embeddings with explicit model version and chunking for sequences exceeding context length; cache deterministically. Note that pretrained protein sequence knowledge is external representation learning, **not affinity-label training**.
- [ ] Optional only after essentials: kNN/learned-model ensemble, similarity-family strict-cold split, mixup, extended ablations.

### 5C. Cold-target evaluation

- [ ] On each split, retrain every model solely on that split's cold-train; tune on cold validation, and compute test metrics only for final configurations.
- [ ] Report test **MSE, global CI, per-ligand CI, Hit@1/5, MRR**, and **per-target CI** if enough drug candidates exist; document skipped groups and ranking candidate construction.
- [ ] Run **5 cold splits** as resources permit; report mean ± standard deviation **across splits** (not bootstrap intervals mislabeled as split variability).
- [ ] For each test protein, compute max normalized similarity to **training proteins** and group into pre-specified similarity buckets (`<0.3`, `0.3–0.5`, `0.5–0.7`, `>=0.7`). Plot performance per bucket and sample sizes. If similarity scale differs, document normalization and whether buckets are meaningful.
- [ ] Explain failure modes: proteins remote from training sequences, few per-ligand interactions, KIBA measurement sparsity, sequence length/feature limitations, distribution shift.
- [ ] Optional ablations: composition vs k-mer vs ESM, fingerprints vs CF factors, with vs without similarity kNN.
- [ ] Save per-split JSON, aggregate CSV, figures, trained-model configuration, and a notebook that loads frozen splits and produces tables.

**Task 3 completion gate:** Results on **both datasets** with provably unseen target sequences, C0 and at least one sequence-based method, properly explained method/evidence/limitations.

---

## Phase 6 — Tests, documentation and report-ready outputs

- [ ] Run `pytest` (or equivalent) on data conversion, dedup, split leakage, metric tie cases, baseline leakage, save/reload parity and cold-target prediction.
- [ ] Run smoke pipeline, then complete dataset pipelines; freeze artifacts and write a reproducibility manifest (seed, package versions, git commit, data/split hashes, commands).
- [ ] Ensure thin notebooks open and run top to bottom from a clean kernel using saved artifacts; no hidden notebook-only state.
- [ ] Export Task 2 and Task 3 tables/figures from result files, not manually transcribed values.
- [ ] Generate `results/task2_report_notes.md` and `results/task3_report_notes.md`: methods, results with units/score directions, evidence for each required question, limitations, and figure/table paths.
- [ ] Check that all externally sourced data, pretrained representations, reused code and substantial assistance are cited consistent with the course's rules.
- [ ] Leave untouched deliverables outside our scope unless needed for successful integration; communicate any inconsistencies with Task 1.

## Priority if time is running out

**P0 — Must finish:** cleaned data; no-duplicate warm and sequence-grouped cold splits; tested metrics; simple baselines; biased MF trained/evaluated on DAVIS and KIBA; NPZ save/reload; three cases each where feasible; C0 + one sequence-aware cold method evaluated on both datasets; report-ready results.

**P1 — Strong improvements:** bias-residual kNN CF, 5 cold seeds, similarity buckets, CI/bootstrap intervals, sparsity and score-bin analysis, automated figures.

**P2 — Stretch only:** censored-label-aware MF, 5-level DAVIS sparsity experiment, ESM-2, strict family-cold split, ensemble/mixup.

## Final acceptance checklist

- [ ] No raw files modified; data-cleaning counts logged, screening limits and KIBA missingness handled correctly.
- [ ] No overlapping canonical pair in Task 2 splits; all warm eval entities appear in warm train.
- [ ] No identical target sequences across Task 3 train/val/test; no test-target affinity used for training/tuning.
- [ ] Models fit and evaluated separately on DAVIS and KIBA using their proper label scales.
- [ ] Task 2 CF saves to both required `.npz` files; predictions from reload match original.
- [ ] Task 2: MSE, CI and ranking metrics plus baselines, cases and all four brief questions answered.
- [ ] Task 3: functional cold-start method, test metrics and clearly documented algorithm on both datasets.
- [ ] Tests and execution commands documented; report-ready artifacts exist; no invented results.

## How to execute this agenda with Codex

Start by telling Codex:

> Read `agenda.md` thoroughly. Inspect this repository and the actual DAVIS/KIBA files before editing. Implement Phase 0 and Phase 1 first, run tests, then progress through the remaining phases in priority order. Keep me informed of what changed, the commands to reproduce it, and any unresolved blockers. Prioritize real verified results and leakage safety over optional models. Do not invent metrics or rewrite working code unnecessarily.
