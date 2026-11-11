# DAVIS / KIBA — Tasks 2 and 3

Warm-split collaborative filtering (Task 2) and cold-target generalisation (Task 3)
on the DAVIS and KIBA drug–target affinity benchmarks.

## Layout

    src/            source
      dtadata.py      loading, pKd conversion, cleaning and dedup reporting
      splits.py       warm 70/15/15 split; cold-target split grouped by sequence
      metrics.py      MSE, Concordance Index, Hit@k, MRR, random-ranking reference
      models.py       B0-B3 baselines, M1 target kNN, M2 biased MF (ALS), M3 blend
      features.py     protein descriptors and k-mer similarity
      cold_models.py  C0-C4 cold-target models
      checkpoint.py   save/restore CF state
      task2.py        Task 2 runner (tune on validation, score test once)
      case_study.py   Task 2 case study, predictions reloaded from disk
      task3.py        Task 3 runner, 5 cold seeds
      buckets.py      per-target metrics grouped by homology
      verify.py       independent checks on everything reported
      make_report.py  renders results/results.html from the CSVs

    saved_models/   davis_cf.npz, kiba_cf.npz
    results/        all result CSVs plus results.html

## Reproducing

    cd atharva
    python3 src/task2.py        # warm split, writes checkpoints + results
    python3 src/case_study.py   # case study, loads checkpoints from disk
    python3 src/task3.py        # cold targets, 5 seeds
    python3 src/buckets.py      # similarity-bucket analysis
    python3 src/verify.py       # all checks must pass
    python3 src/make_report.py  # render the page

Data is the DeepDTA benchmark release. Download `ligands_can.txt`, `proteins.txt` and `Y` for each of
`davis` and `kiba` from https://github.com/hkmztrk/DeepDTA/tree/master/data into
`atharva/data/deepdta/{davis,kiba}/` (git-ignored), or point `DTA_DATA_ROOT` at wherever they live.

`results/task3_pair_detail.csv` (14.8 MB, per-pair predictions) is not committed. `buckets.py`
needs it, so run `task3.py` first, which regenerates it.

## Protocol notes

* Ranking is per ligand over that ligand's held-out targets. Prediction ties are broken
  at random, otherwise a constant predictor inherits the array order and beats chance.
* Cold-target splits hold out whole *sequence groups*: DAVIS has 442 target IDs but only
  379 distinct sequences, so an ID-level split leaks identical proteins across folds.
* "MSE strong" is restricted to true strong binders (DAVIS pKd >= 7, KIBA >= 12.1),
  because 69.6% of DAVIS labels sit at the pKd = 5 screening floor.
