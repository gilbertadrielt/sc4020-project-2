# SC4020 Tasks 2 and 3: DAVIS and KIBA

Executed, leakage-checked collaborative filtering and cold-target experiments, implementing [agenda (1).md](<agenda%20(1).md>). This directory is the **v1** experiment workspace. The sibling project and parent Task 1 files are untouched.

Start with [Task 2 report notes](results/task2_report_notes.md), [Task 3 report notes](results/task3_report_notes.md), and the [combined editable report](reports/task2_task3_report.md). Figures and all numerical tables are generated from saved result files. The official assignment brief and accompanying plan were not present; remaining rubric checks and real student contribution details require the authors.

## Reproduce

Python 3.13.7 was used. From this directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/run_pipeline.py --smoke
.venv/bin/python scripts/run_pipeline.py
```

The download command retrieves only six small files at a pinned DeepDTA commit, checks existing source hashes, and never replaces originals. Raw files are already available in this workspace. Source matrices are trusted upstream pickle artifacts; do not replace them with arbitrary pickle files. Requirements lock records exact executed package versions; requirements.txt records compatible direct dependencies.

The smoke run uses a deterministic, support-selected subset of actual data under `results/smoke/v1/`. It fits only a few epochs and never changes final frozen splits or benchmark outputs. Smoke metrics are diagnostics, never final results. Full runs use one warm split and five cold splits per dataset. Completed evaluations are cached; rerunning the pipeline verifies frozen data and reuses their metrics. Use another `DTR_ROOT` or an archived copy for a new experiment version; do not edit frozen split CSVs. Data/model/result fingerprints are in [the reproducibility manifest](results/reproducibility_manifest.json).

Individual commands:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/make_splits.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/train_cf.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/train_cold.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/sparsity_experiment.py
.venv/bin/python scripts/export_results.py
.venv/bin/python scripts/build_notebooks.py --execute
.venv/bin/python scripts/verify_outputs.py
```

Training scripts accept `--dataset davis|kiba|all`. Cold training accepts `--seeds 2026 2027 2028 2029 2030`. All numerical policy lives in `src/config.py`; split seeds are 2026–2030 and the optimizer seed is fixed at 2026. Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` for consistent resource use; the pipeline sets these itself. Models are NumPy implementations with Numba accelerating explicit SGD/CI loops, not pretrained recommender packages.

## Data and protocols

Immutable originals are in `data/raw/{davis,kiba}`. Each source JSON records URLs, commit, score scale and SHA-256. RDKit canonical isomeric SMILES and exact whitespace-normalized uppercase full sequences define entities. Pair measurements are mean-merged **after** score conversion. Exclusions, original entity maps, actual source row/column cells, repeats, sample std, conflicts, score quantiles and screening fractions are retained. Cleaning verifies equality of the observed canonical key set before/after aggregation, so missing KIBA measurements never become labels.

DAVIS raw Kd is in nM: positive finite values become `pKd = 9 - log10(Kd_nM)` exactly once. Supplied 10,000 nM screening-bound labels remain in MSE and are flagged as censored; they are not scientifically interpreted as exact affinity measurements. KIBA uses the supplied **transformed** DeepDTA score, higher meaning stronger binding; finite nonpositive values are flagged and retained without inventing assay-level reasons to remove them. See the [upstream data documentation](https://github.com/hkmztrk/DeepDTA/blob/master/data/README.md) and [SimBoost's transformation description](https://doi.org/10.1186/s13321-017-0209-z).

Task 2 holds out 70/15/15% of observed canonical pairs, stratified by ligand with target-coverage repair before freezing. All validation/test entities occur in warm train. Task 3 holds out whole **exact sequences**, approximately 70/15/15% by target entities, and retrains on each of five splits. Unseen evaluation ligands are explicitly flagged and receive defined fallbacks. Split manifests include sizes, entity mappings, repair moves and hashes. Near-homologous sequences can remain across cold partitions; these are not strict family splits or official DeepDTA folds. Never compare warm and cold scores as the same protocol.

## Models and evidence

Task 2 compares global/ligand/target means, validation-tuned additive bias, overlap-shrunk bias-residual target kNN, and biased MF with ranks 0/4/8/16/32. All fit on observed warm-training pairs only. Early stopping and model selection use validation MSE with concordance as tie-breaker. Rank zero is an explicit bias-only SGD ablation. All selected methods report train/validation/test metrics. Checkpoints store float64 factors/biases, non-pickle original-ID maps, scale/bounds, hyperparameters, seed and split hash. Three eligible held-out ligand case studies per dataset use reloaded models and include baseline ranks. The optional DAVIS 100/50/25/10/5% train-only thinning experiment holds its original test fixed, counts unseen entities and uses zero-factor/bias fallback.

Task 3 compares cold-training ligand means (C0), training-protein sequence-neighbour observation transfer (C1), and a learned sequence-to-bias/factor ridge model on independently cold-trained MF (C2). Sequence features are fixed-vocabulary 3-mer frequencies, amino-acid and dipeptide composition; ridge centers only on cold-training targets. C2 never looks up test-target factors. Canonical 2048-bit radius-2 Morgan fingerprints are cached, while C2 uses cold-trained ligand factors. No supplied Smith–Waterman matrices were found: similarity is full-sequence 3-mer cosine, bounded 0..1 and **not alignment identity**. Per-split nearest-training audits and pre-specified similarity-bucket counts/errors are saved.

MSE, global/per-ligand/per-target concordance, Hit@1/5/10, MRR, candidate counts, support/fallback diagnostics, train-density effects and score-bin errors are exported. True ties are excluded from concordance; prediction ties earn half credit. Ranking uses all tied true-best items and exact expectation over predicted ties. Singleton/all-truth-tied groups are skipped and counted. Hit@5/10 require at least 5/10 measured held-out candidates, with `hit5_gt5` excluding trivial five-candidate sets. Primary ranking direction is targets per ligand; per-target rankings address the alternate wording. Uniform random ranking is calculated analytically. Bootstrap intervals resample ligand groups and are separate from five-cold-split standard deviation and concordance.

Prediction CSVs retain 17 significant digits; use `src.io.read_predictions` to preserve adjacent floating-point scores and exact tie credit. `scripts/export_predictions.py` restores exports from selected checkpoints without refitting. Saved models are checked in fresh interpreter processes using label-free queries; score tolerance is 1e-12 and ordering must match. Final verification independently reconstructs predictions, reconciles saved metrics and candidate totals, checks data/split hashes and confirms notebook execution. It never fits or selects models.

## Files

- `src/`: reusable data, split, metric, baseline, CF, feature, cold-model, plotting and reporting logic.
- `scripts/`: preparation, training/evaluation, subset smoke, diagnostics, exports, notebook execution and artifact verification. Training and selected-model evaluation are intentionally one orchestrated step, avoiding duplicated commands that silently recreate splits.
- `data/processed/`: `pairs_*`, `ligands_*`, `targets_*`, cleaning logs/audits, keyed feature caches and frozen splits.
- `saved_models/`: `{dataset}_cf.npz`, baseline/kNN archives and cold `{neighbors,ridge}_seed{seed}.npz`.
- `results/tables/`, `results/predictions/`, `results/figures/`: machine-readable evidence and PNG/PDF figures. `results/hparams.csv` unifies every tuning trial.
- [Implementation log](results/implementation_log.md), [verification](results/verification.json), and [agenda completion record](results/agenda_completion.md).
- [Task 2 notebook](notebooks/02_task2_cf.ipynb), [case-study notebook](notebooks/03_task2_case_study.ipynb), [Task 3 notebook](notebooks/04_task3_cold.ipynb): executed fresh kernels; saved artifacts only, no refitting.

Generated data/models are ignored by Git to avoid committing large artifacts; they are present locally and reproducible. Keep the data, checkpoints, results and notebooks together when sharing an executed workspace.

## Scope and attribution

Core and priority-one requirements are implemented, including five cold seeds, target-kNN, bootstraps, similarity buckets and figures. Optional C3, ESM-2, strict family-cold splits, ensembles/mixup and censored-loss MF were not run. No synthetic final performance, clinical conclusion, unknown contributor identity or missing rubric requirement is asserted.

DeepDTA/SimBoost and assistance references are included in both report notes. Some local SGD, CI, sequence features and tests were adapted from the existing sibling implementation. Codex supplied substantial implementation, analysis, tests and writing; the authors must verify their course disclosure policy and add real human contributions.
