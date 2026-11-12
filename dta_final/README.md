# Drug–target binding affinity: a recommender built from scratch

Drugs are users, protein targets are items, binding affinity is the rating.
Most of the matrix was never measured, because measuring one cell means running a
laboratory experiment.

This repository asks one question in four increasingly hard regimes, and answers all
four with the same code path.

| Regime | Held out | Real-world question |
|---|---|---|
| **S1** | random pairs | fill a gap for a drug and protein we know |
| **S2** | whole drugs | screen a brand-new compound |
| **S3** | whole targets | screen against a brand-new protein |
| **S4** | both | neither side ever seen |

---

## Quick start

```bash
pip install -r requirements.txt
bash run_all.sh              # everything, ~45 min on 2 cores
bash run_all.sh --quick      # DAVIS only, 2 seeds, ~5 min
```

Outputs:

```
results/figures/      fig01..fig08, PDF for the report and PNG for slides
results/tables/       every number as CSV
results/tables/tex/   the same tables as LaTeX, ready to \input
saved_models/         checkpoints that reload without pickle
data/splits/          frozen partitions, one per (regime, seed), with fingerprints
```

Stage by stage:

```bash
python -m dta.data --download     # fetch raw DeepDTA files
python -m dta.data                # clean, deduplicate, index
python -m dta.splits              # freeze 4 regimes x 5 seeds, assert no leakage
python -m dta.kernels             # similarity matrices and feature vectors
python -m dta.run_task1           # Task 1 analysis, tables, figures
python -m dta.protocol            # tune on validation, evaluate across seeds
python -m dta.run_analysis        # buckets, lookup check, case study
python -m dta.tables              # CSV -> LaTeX
python -m pytest -q               # 53 tests
```

---

## Design: two contracts, everything else swappable

The failure mode this architecture is built to prevent is four people producing four
sets of numbers that cannot be compared. So exactly two things are fixed.

**A split** is a pure function `(pairs, entities, setting, seed) -> fold labels`,
written to disk with a content hash. Nothing else partitions data.

**A model** implements `fit(train, n_lig, n_tgt, val, kernels, features) -> self` and
`predict(pairs) -> array`. Adding a model is one file; comparing models is a loop.

```
src/dta/
  config.py      paths, seeds, score scales, the validated palette
  data.py        download, clean, censor flags, chemistry-keyed dedup
  splits.py      S1-S4 + leakage assertions + fingerprints
  kernels.py     sequence / chemical / profile similarity, PSD projection, MKL
  metrics.py     MSE, CI, Hit@k, MRR, bootstrap
  analysis.py    Task 1 analytics, similarity buckets
  models.py      baselines, item-kNN, biased MF, censored MF, blend
  kronrls.py     Kronecker kernel ridge (+ multiple kernels)
  cold.py        cold-start comparison models
  protocol.py    tune on validation, evaluate across seeds
  plots.py       every figure
  tables.py      CSV -> LaTeX
```

---

## Three decisions that drive every number

### 1. Entities are chemistry, not identifiers

DAVIS lists **442 target IDs that are only 379 distinct sequences**: mutants and
phosphorylated forms (`ABL1`, `ABL1p`, `ABL1(T315I)`, …) carry an identical sequence in
the benchmark files. Keying on IDs puts the same protein on both sides of a split.

Pairs are keyed on `(canonical SMILES, amino-acid sequence)`. DAVIS goes 30,056 → 25,772
pairs, KIBA 118,254 → 117,657. The worst disagreement between merged DAVIS duplicates is
**4.75 pKd units** for what the files call the same molecule against the same sequence,
so these are real conflicts rather than harmless repeats.

Collapsing protein modifications is a [known flaw in the standard
benchmark](https://www.themoonlight.io/review/towards-precision-protein-ligand-affinity-prediction-benchmark-a-complete-and-modification-aware-davis-dataset),
not a quirk of this analysis.

### 2. The assay ceiling is not a measurement

69.6% of raw DAVIS labels sit at Kd = 10,000 nM, where the assay stopped looking, so the
truth is *at most* pKd 5. These are flagged `censored` and kept, and the project includes
a model whose loss knows about them (below).

### 3. A missing KIBA cell is not a zero

75.2% of the matrix was never tested. Missing cells are `NaN` and never become rows.
Treating them as zeros would invent ~356,000 strong negative claims nobody measured.

---

## Results

Mean over five seeds. **Per-ligand CI** ranks a drug's candidate targets, which is the
quantity that matches the use case and the one that exposes cold-start failure.

### DAVIS

| Regime | Fallback CI per-drug | Best model | Best CI per-drug | Best MSE |
|---|---|---|---|---|
| S1 random pairs | 0.500 | KronRLS, multiple kernels | **0.840** | 0.226 |
| S2 unseen drugs | 0.706 | KronRLS | **0.732** | 0.537 |
| S3 unseen targets | 0.500 | KronRLS | **0.769** | 0.292 |
| S4 both unseen | 0.500 | KronRLS | **0.620** | 0.568 |

### KIBA

| Regime | Fallback CI per-drug | Best model | Best CI per-drug | Best MSE |
|---|---|---|---|---|
| S1 random pairs | 0.500 | KronRLS, multiple kernels | **0.792** | 0.167 |
| S2 unseen drugs | 0.675 | KronRLS | **0.758** | 0.370 |
| S3 unseen targets | 0.500 | KronRLS, multiple kernels | **0.681** | 0.298 |
| S4 both unseen | 0.500 | KronRLS | **0.618** | 0.502 |

Full tables per regime are in `results/tables/`, and `summary_all.csv` carries every model
rather than just the winner.

### The findings worth reporting

**The fallback scores exactly 0.500, predicted before it was measured.** With no training
data for a target, the model never updates that target's parameters, so the prediction
reduces to a quantity that does not depend on the target at all. Every unseen target gets
the same score, every ranking is a tie, and the expected concordance is exactly one half.
Measured: 0.500 with standard deviation 0.000 across five seeds, on both datasets, in
every similarity bucket. A unit test asserts the property directly
(`test_per_ligand_ci_is_exactly_half_for_a_target_blind_model`).

**Global CI hides this completely.** The same fallback scores 0.70–0.78 on pooled CI,
because the drug biases still order drugs sensibly within any one target. Reporting only
pooled CI would make a model that cannot rank at all look respectable.

**One model spans all four regimes.** KronRLS is best or tied-best everywhere, and it is
the *only* method that beats chance in S4, where neither the drug nor the protein was
ever seen. Collaborative filtering and content regression each solve half the problem;
the Kronecker kernel solves both because the only thing that changes between regimes is
which kernel it is handed.

**The censored loss trades error for ranking, as predicted.** Stated in the docstring
before running: fitting the detection limit as if it were a measurement should buy MSE
and cost ranking, so declining to fit it should do the reverse. On DAVIS, against the
identical model under squared error, the censored model gives up error and gains order:

| | censored loss | squared error | |
|---|---|---|---|
| MSE | 0.2479 | **0.2458** | squared error wins, as predicted |
| MSE on strong binders | 1.7971 | **1.5873** | squared error wins |
| CI per-drug | **0.8298** | 0.8118 | censored wins, as predicted |
| Hit@1 | **0.3490** | 0.3303 | censored wins |
| MRR | **0.4981** | 0.4891 | censored wins |

The strong-binder error going the same way as overall MSE was not predicted and is worth
a sentence in the report: releasing the censored cells downward lets the model spread its
predictions over a wider range, which costs calibration everywhere, including at the top,
while still ordering candidates better. Ranking is what the use case needs, so the trade
is the right one, but it is a trade and not a free win.

On KIBA, which has no censoring, the two collapse onto each other to machine precision —
a correctness check as much as a result.

**Borrowing from relatives is a homology lookup, and the kernel model is not.** `fig07`
splits cold-target performance by distance to the nearest training protein. Guilt-by-
association does well on near-homologues (CI per-drug 0.732 on KIBA above 0.35 similarity)
and then falls *below chance* on orphans (**0.440** under 0.10 similarity), which is worse
than useless: it confidently orders candidates backwards. KronRLS, handed the same
similarities, holds **0.690** on exactly those orphans. The reason is that the kernel model
is regularised toward the global structure rather than committing to one nearest neighbour,
so when no relative exists it degrades to a sensible prior instead of trusting noise.

---

## What makes the comparison trustworthy

* **53 tests**, and the important ones check mathematics rather than regressions:
  the fast O(n log n) concordance index is verified against the O(n²) definition under
  heavy ties; the KronRLS conjugate-gradient solver is verified against both a closed-form
  eigendecomposition and an explicit dense Kronecker solve; the bias model is verified to
  be *exactly* a factorisation with zero factors; the censored loss is verified to be
  *exactly* squared error when nothing is censored.
* **Leakage is asserted, not assumed.** Every split checks the property its regime claims
  and raises rather than warns. A test injects leakage deliberately to confirm the guard fires.
* **Tuning never touches test.** Hyperparameters are chosen by grid search on the
  validation fold of the first seed and then held fixed; every test evaluation is logged.
* **Five seeds everywhere**, including the warm regime, which single-split protocols skip.
* **Frozen splits with fingerprints**, so two runs can prove they used the same partition
  before their numbers are compared.
* **Checkpoints reload without pickle** and are asserted to reproduce predictions exactly.
* **Verified reproducible from a clean checkout.** The pipeline was re-run end to end in a
  fresh directory containing nothing but the source and the raw downloads. Every one of
  the 50 overlapping (regime, seed, model) rows came back bit-identical, to 0.00e+00 in
  MSE, CI, per-drug CI, Hit@1 and MRR, and all 20 split fingerprints matched. Nothing
  samples at run time.

---

## Models

| | Model | Idea |
|---|---|---|
| B0–B3 | means and biases | `mu + b_i + c_j`; B3 is `BiasedMF(k=0)` so the comparison is exact |
| M1 | item-based kNN | similarity on bias residuals, shrunk by overlap |
| M2 | biased MF (ALS) | `+ u_i·v_j`; count-scaled regularisation |
| M3 | kNN + MF blend | weight chosen on validation |
| M4 | **censored MF** | Tobit loss by iteratively re-targeted ALS |
| M5 | **KronRLS** | Kronecker kernel ridge; handles all four regimes |
| M6 | **KronRLS-MKL** | several kernels per side, weights from kernel alignment |
| C0–C4 | cold-start set | fallback, guilt-by-association, factor transfer, content regression, gated blend |

Two implementation notes worth knowing:

*Count-scaled regularisation* is the default for MF. The two sides of this matrix are
unbalanced (a DAVIS drug has 265 training observations, a target only 48), so one constant
penalty leaves targets underdetermined at k=32 and the factors diverge. Scaling by
observation count makes a single lambda mean the same thing on both sides.

*The censored loss is fitted by re-targeting, not gradient descent.* Because
`max(0, pred − c)² = (pred − min(pred, c))²`, substituting the pseudo-target
`min(pred, c)` turns the Tobit objective into ordinary least squares, which the exact ALS
solver already handles. Alternating re-target and re-fit is a majorise-minimise scheme, so
it converges rather than drifting, and with `use_censored_loss=False` it collapses exactly
onto the ordinary model.

---

## Limitations

* **Cold splits are small.** About 57 DAVIS and 34 KIBA targets per seed, hence five seeds
  with the spread reported on everything.
* **Sequence similarity is whole-protein**, not binding-site. All targets are kinases
  sharing a conserved domain, so two can look similar and differ where it matters.
* **KIBA's measured cells are not a random sample.** Drugs with more measurements score
  slightly higher (Spearman ρ ≈ 0.17), so held-out estimates inherit that bias.
* **Hyperparameters are tuned on the first seed only** and reused, so the spread across
  seeds is a spread in data, not in tuning.
* **ESM-2 protein embeddings are not included.** The feature module is structured to
  accept them, but the defaults need nothing beyond scikit-learn so the project runs
  anywhere.

---

## References

1. Öztürk, Özgür & Ozkirimli. *DeepDTA.* Bioinformatics 34(17), 2018, i821–i829.
2. Davis et al. *Comprehensive analysis of kinase inhibitor selectivity.* Nat. Biotechnol. 29, 2011.
3. Tang et al. *Making sense of large-scale kinase inhibitor bioactivity data sets.* JCIM 54(3), 2014.
4. Nascimento, Prudêncio & Costa. *A multiple kernel learning algorithm for drug-target
   interaction prediction.* BMC Bioinformatics 17, 2016. (KronRLS-MKL)
5. Koren, Bell & Volinsky. *Matrix factorization techniques for recommender systems.* IEEE Computer 42(8), 2009.
6. Zhou et al. *Large-scale parallel collaborative filtering for the Netflix Prize.* AAIM 2008. (ALS-WR)
7. Candès & Recht. *Exact matrix completion via convex optimization.* FoCM 9, 2009.
8. Tobin. *Estimation of relationships for limited dependent variables.* Econometrica 26(1), 1958.
9. Gantner et al. *Learning attribute-to-feature mappings for cold-start recommendations.* ICDM 2010.
