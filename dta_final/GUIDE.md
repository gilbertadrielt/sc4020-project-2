# What to do next

Three days to the deadline. Here is the order I would do things in.

---

## 1. Run it once yourself (30 to 50 minutes)

```bash
cd dta
pip install -r requirements.txt
bash run_all.sh
```

Nothing needs a GPU and nothing needs more than about 3 GB of RAM, so your laptop is
fine. Colab works too, and the raw data download is 35 MB.

If you are short on time, `bash run_all.sh --quick` does DAVIS with 2 seeds in about
five minutes, which is enough to confirm your machine runs it. Note that `--quick`
overwrites `results/` with the 2-seed numbers, so run the full thing before you take
numbers for the report.

The run I did is already in `results/`, so if a stage fails you can keep writing while
you debug.

**What to check when it finishes.** Stage 0 must say `53 passed`. Stage 3 must print no
assertion errors, because the leakage checks raise rather than warn. Then compare your
`results/tables/summary_all.csv` against the numbers in the README table. They should
match exactly, because every seed is fixed and nothing samples at run time.

---

## 2. Decide what goes in the report

The report is what gets marked, not the repository. The code's job is to make every
claim in the report checkable.

I would build it around one argument: **the metric everyone reports hides the failure
that matters, and only one model class survives once you stop hiding it.** Everything
in `results/` is evidence for some step of that argument.

Here is a section plan with the specific exhibit for each claim. The numbers are
already in `results/tables/tex/` as LaTeX you can `\input` directly, so you should not
be retyping any figure into the report by hand.

| Section | The claim | Exhibit |
|---|---|---|
| Problem | Drugs are users, targets are items, affinity is the rating, and 75% of the KIBA matrix was never measured | `fig01_heatmaps` |
| Data, Task 1 | Two datasets, different scales, one of them 71% censored | `task1_comparison.tex`, `fig02_distributions` |
| Data cleaning | 442 DAVIS target IDs are 379 sequences, and keying on IDs leaks | `task1_cleaning_davis.tex` |
| Structure | Additive effects explain 37% of DAVIS variance, so 63% is interaction worth modelling | `task1_variance_davis.tex`, `fig03_structure` |
| Protocol | Four regimes, five seeds, frozen splits, tuning never touches test | `data/splits/report_*.csv` |
| Metric | Per-drug CI exposes what pooled CI hides | `fig04_four_settings_*` |
| Warm results | MF beats biases, kernels beat MF | `results_davis_S1_pair.tex`, `fig05_warm_models` |
| Censoring | Modelling the detection limit trades error for ranking | `fig06_censored_loss` |
| Cold start | Guilt-by-association goes below chance on orphans, the kernel model does not | `fig07_similarity_buckets` |
| Case study | What a recommendation actually looks like for three drugs | `case_detail_*.csv` |
| Limitations | Five of them, stated in the README | |

The four findings I would lead with, in this order, are in the README under
**"The findings worth reporting."** Each one is a sentence you can defend with a number
and, for two of them, with a unit test.

---

## 3. The three things most likely to earn marks

**The 0.500 prediction.** We said before measuring that a target-blind model must score
exactly one half per-drug CI, because every unseen target gets the same score so every
ranking is a tie. It measures 0.500 with standard deviation 0.000, on both datasets, in
every similarity bucket. A test asserts the property directly. Deriving a number and
then hitting it exactly reads very differently from reporting a number you found.

**The leakage experiment.** We ran the controlled version: same pipeline, same seed, same
hyperparameters, deduplication the only thing toggled. ID-level keying inflates the
collaborative-filtering advantage by 0.071 to 0.105 R² on DAVIS, where 14.3% of columns
are duplicates, and by roughly zero on KIBA, where 0.5% are. It only moves models that
can exploit a duplicate, which is the signature of leakage rather than noise. That is a
real experiment with a mechanism and a control, not a methodology complaint.

**The orphan result.** Sequence-kNN scores 0.440 on KIBA proteins with no training
relative above 0.10 similarity. Below chance means it orders candidates backwards, which
is worse than refusing to answer. KronRLS, given the identical similarities, holds 0.690.
Finding where your own best idea fails, and explaining why the other method survives,
is the part most reports skip.

---

## 4. Honesty items to carry into the report

Put these in rather than hoping nobody asks.

- **Strong-binder error.** The censored model is worse than squared error on MSE for
  strong binders (1.797 against 1.587), not just on overall MSE. We predicted the overall
  trade and not this one. The README explains why it happens and why ranking still
  justifies the choice, but write it as a trade and not a win.
- **DAVIS S2 barely moves.** 0.732 against a 0.706 fallback. Unseen drugs are the regime
  where we add least, because a drug's own mean affinity is already a strong predictor
  and chemical similarity adds little on top.
- **S4 is weak everywhere.** 0.620 and 0.618. Above chance, which no other method here
  manages, but nobody should call it useful.
- **The case study is mixed.** On the most promiscuous DAVIS drug the true best target
  ranks 13th of 57. Report that alongside the two where it ranks 1st.
- **Hyperparameters come from the first seed only.** So the spread across seeds is spread
  in data, not in tuning. That is the honest reading and it is also why the spread is
  narrower than it would otherwise be.

---

## 5. Splitting the work across four people

The split contract is what makes this parallelisable. A split is a pure function written
to disk with a content hash, and a model is two methods. So:

- One person owns Task 1 and the data sections. Everything they need is in
  `run_task1` output and `analysis.py`.
- One person owns the warm regime: `models.py`, `fig05`, the S1 tables.
- One person owns cold start: `cold.py`, `kronrls.py`, `fig07`, the S2 to S4 tables.
- One person owns the protocol and the write-up spine: `splits.py`, `metrics.py`, `fig04`,
  and the argument that holds the sections together.

Everyone can re-run their own piece alone, because each stage reads what the previous one
wrote. Compare split fingerprints before comparing numbers; that is the whole point of
them.

`results/tables/tex/contribution_table.tex` is the required contribution table with
placeholders to fill in.

---

## 6. Academic integrity

- The repository must stay private. Code gets MOSS-screened against other submissions.
- Everything here is written from scratch against published method descriptions, not
  copied from an implementation. The nine references in the README are the papers the
  methods come from, and KronRLS-MKL in particular should be cited where you describe it.
- `requirements.txt` is numpy, pandas, scipy, scikit-learn, matplotlib, and optionally
  rdkit and lightgbm. Nothing in it implements a drug-target model, so there is no
  borrowed method hiding in a dependency.
- Say in the report that you used an AI assistant if your course requires it. Check the
  SC4020 brief; several NTU courses now ask for a declaration.

---

## 7. If something breaks

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: dta` | run from the repo root, or `export PYTHONPATH=$PWD/src` |
| rdkit import fails | optional; the code falls back to a SMILES-string fingerprint and says so |
| lightgbm import fails | optional; `ContentRegressor` falls back to ridge |
| `AssertionError` in stage 3 | a split violated its own regime. Do not work around it, that assert is doing its job |
| KIBA kernels slow | 2068 × 2068 kernels, about 90 seconds and 40 MB. Expected |
| numbers differ from the README | check you did not leave `--quick` on |

---

## What I would still add, in priority order

1. **The report itself.** I can draft it with every number and figure wired in, if you want.
2. **ESM-2 protein embeddings.** `kernels.py` has the hook. Would likely help S3 and S4,
   which are the weakest results. Half a day, and needs a GPU to be quick.
3. **A fifth regime**, holding out whole protein families rather than individual targets,
   which is the honest version of S3 and would almost certainly look worse.

The first one is worth more marks than the other two combined.
