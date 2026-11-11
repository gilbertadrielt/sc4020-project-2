"""Generate all report numbers from saved CSV/JSON results, never hand-enter them."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.config import ROOT, DATASETS, COLD_SEEDS, SEED
from src.common import setup, log
from src.reporting import markdown_table, ligand_bootstrap_mse
from src.io import read_predictions

T = ROOT / 'results/tables'
REFERENCES = '''
- Öztürk et al. (2018), DeepDTA, https://doi.org/10.1093/bioinformatics/bty593. Raw files from hkmztrk/DeepDTA commit a546a8433a6822e958f36171c4356ad6f414d623, with ordered mappings and SHA-256 in data/raw/*/source.json.
- DeepDTA data documentation: https://github.com/hkmztrk/DeepDTA/blob/master/data/README.md (raw Kd units, transformed KIBA, NaN missingness).
- He et al. (2017), SimBoost, https://doi.org/10.1186/s13321-017-0209-z (transformed KIBA direction and DAVIS screening-bound provenance).
- Biased MF is the standard additive-bias plus dot-product model. SGD, residual target-kNN, cold factor ridge, tie-expectation metrics and experiment orchestration are implemented locally. NumPy/SciPy/scikit-learn/RDKit/Numba provide numerical, fixed-feature and acceleration support; no pretrained recommender is used.
- Local reuse: CF SGD, Fenwick CI, 3-mer features and tests were adapted from ../drug_target_recommender/src and tests. That sibling implementation and Task 1 artifacts were preserved. Codex provided substantial implementation, analysis, testing and writing assistance. Human contributor names, contributions and the course's assistance policy must be checked by the authors; no identities or approval claims are invented.
'''
METRICS = '''MSE uses observed labels only. Higher pKd and supplied transformed KIBA mean stronger affinity. CI denotes concordance: exclude true ties, award predicted ties half credit, compare with brute force in tests. Primary rankings recommend targets for each ligand within held-out measured candidates, with all true-best ties relevant. Hit and reciprocal rank are expected values over uniform ordering of predicted ties. Singleton and all-truth-tied groups are excluded and counted. Hit@5 requires >=5 candidates and Hit@10 requires >=10; hit5_gt5 further excludes the trivial five-candidate case. Per-target rankings/CI are a secondary interpretation of “recommend a drug for a target”. Random rankings use exact expectations, with no regression MSE. Candidate sets differ across datasets/splits, so raw Hit values are not directly comparable. No unobserved KIBA pairs are labelled negatives. Group bootstrap intervals describe uncertainty within a fixed split and are distinct from concordance and five-split standard deviation.'''


def main():
    setup()
    warm = pd.concat([pd.read_csv(T / f'task2_{ds}_metrics.csv') for ds in DATASETS], ignore_index=True)
    cold = pd.concat([pd.read_csv(T / f'task3_{ds}_seed{seed}_metrics.csv') for ds in DATASETS for seed in COLD_SEEDS], ignore_index=True)
    warm.to_csv(T / 'task2_summary.csv', index=False)
    cold.to_csv(T / 'task3_summary.csv', index=False)
    test = cold[(cold.split == 'test') & (cold.model != 'random_ranking')]
    metrics = ['mse', 'ci_global', 'ci_per_ligand', 'ci_per_target', 'hit1', 'hit5', 'hit10', 'mrr']
    aggregate = test.groupby(['dataset', 'model'])[metrics].agg(['mean', 'std', 'count'])
    aggregate.columns = ['_'.join(c) for c in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate.to_csv(T / 'task3_aggregate.csv', index=False)
    selected = test[test.selected_by_validation == True]
    selected.groupby('dataset')[metrics].agg(['mean', 'std']).to_csv(T / 'task3_validation_selected_aggregate.csv')
    hparams = []
    for path in sorted(T.glob('*_tuning.csv')):
        frame = pd.read_csv(path)
        frame.insert(0, 'experiment', path.stem.removesuffix('_tuning'))
        hparams.append(frame)
    pd.concat(hparams, ignore_index=True).to_csv(ROOT / 'results/hparams.csv', index=False)
    bootstraps = []
    for ds in DATASETS:
        frame = read_predictions(ROOT / f'results/predictions/task2_{ds}_test.csv')
        bootstraps.append({'dataset': ds, 'task': 'task2', **ligand_bootstrap_mse(frame, 'MF', 'bias_only')})
        for seed in COLD_SEEDS:
            record = json.loads((T / f'task3_{ds}_seed{seed}_run.json').read_text())
            name = record['selected_model']
            frame = read_predictions(ROOT / f'results/predictions/task3_{ds}_seed{seed}_test.csv')
            bootstraps.append({'dataset': ds, 'task': 'task3', 'split_seed': seed,
                               **ligand_bootstrap_mse(frame, name, 'ligand_mean')})
    pd.DataFrame(bootstraps).to_csv(T / 'paired_ligand_bootstrap.csv', index=False)
    clean = pd.read_csv(ROOT / 'data/processed/cleaning_log.csv')
    cleaning_table = markdown_table(clean[['dataset', 'raw_ligands', 'raw_targets', 'raw_finite', 'unique_ligands', 'unique_targets', 'unique_pairs', 'missing_fraction']])
    warm_cols = ['dataset', 'model', 'mse', 'ci_global', 'ci_per_ligand', 'hit1', 'hit5', 'hit10', 'mrr']
    note2 = ['# Task 2 — warm-pair report evidence (v1)', '',
             'This experiment holds out canonical ligand–sequence pairs while keeping every evaluation entity in training. Seed 2026; per-ligand 70/15/15 allocation with pre-freeze target-coverage repair. Final counts and repair moves are in data/processed/splits/task2/*/manifest.json. The official brief and accompanying design plan were unavailable; the supplied agenda governs this implementation.', '',
             '## Data and scales', '', cleaning_table, '',
             'DAVIS converts positive raw Kd in nM once using pKd=9−log10(Kd). Keep screening-limit measurements; any-censored flags and censored fractions survive mean aggregation. A merged score can include censored and uncensored repeats, so a score different from 5 can still have a censored source. KIBA uses supplied transformed scores unchanged. Canonical isomeric SMILES and exact normalized sequences define identity; pair repeats are mean-merged after conversion, with source cells/IDs, sample std and screening fractions saved.', '',
             '## Methods and selection', '',
             'Global, ligand and target means use warm-train only. Additive bias regularization is selected from {1,10,50} by validation. Target kNN uses overlap-restricted cosine on training-only additive-bias residuals, shrinkage n_common/(n_common+beta), K={5,10,20,50}, beta={0,10,50,100}, and positive available neighbours; no usable neighbour falls back to additive bias. MF is mu+b_ligand+b_target+U·V, trained on observed pairs by our NumPy/Numba SGD (no recommender package). The per-observation SGD objective includes L2 factor/bias penalties, hence a degree-weighted total regularizer. Ranks {0,4,8,16,32}, reg {.01,.1}, learning rate .02, at most 160 epochs and patience 16 use validation MSE, with validation concordance as trial tie-breaker. Rank zero is an SGD bias-only ablation. Predictions are not clipped; training bounds are saved. No test score selects parameters or epochs.', '',
             '## Test comparison', '', markdown_table(warm[(warm.split == 'test')][warm_cols]), '', METRICS, '']
    for ds in DATASETS:
        record = json.loads((T / f'task2_{ds}_run.json').read_text())
        dataset_rows = warm[(warm.dataset == ds) & (warm.split == 'test')].set_index('model')
        bias, mf, ligand = [float(dataset_rows.loc[name, 'mse']) for name in ['bias_only', 'MF', 'ligand_mean']]
        gain = 100 * (bias - mf) / bias
        note2 += [f'## {ds.upper()}: answers to the Task 2 questions', '',
                  f'**Does it learn beyond lookup?** All evaluated test pairs were absent from training. MF test MSE is {mf:.6f}, versus ligand mean {ligand:.6f} and additive bias {bias:.6f}; its relative reduction over bias is {gain:.2f}%. Pearson association of held-out true bias residuals with the latent dot product is {record["latent_residual_correlation"]:.4f}. This is descriptive evidence of interaction structure; MF also learns its own biases, so the dot-product correlation alone is not a causal ablation.', '',
                  f'**Is CF necessary?** MF {"improves" if mf < bias else "does not improve"} regression over the selected additive bias on this split. The training matrix density is {record["train_pair_density"]:.4f}; the selected rank is {record["selected"]["factors"]}, with {record["mf_parameter_count"]} parameters versus {record["bias_parameter_count"]} for bias-only. Matrix completion is useful for deliberately withheld interactions, whereas observed complete DAVIS pairs can already be looked up. The parameter count/density does not establish clinical utility or superiority for unseen proteins.', '',
                  f'**Sparsity and abnormal values.** Training co-observation quantiles (target pairs) are {record["overlap_quantiles"]}; selected kNN is {record["selected_knn"]}, test fallback fraction {record["knn_fallback_fraction"]["test"]:.4%}. See the overlap distribution, sparsity support table, score-bin errors and screening audit below. Dense/sparse performance uses training observation counts only.', '',
                  markdown_table(pd.read_csv(T / f'task2_{ds}_sparsity.csv')), '',
                  f'**Case studies from reload.** Three eligible ligands were chosen using suggested IDs when available, otherwise observed sparse/median/dense training histories. Only held-out measured targets are candidates. Top-10 needs at least ten candidates; no training labels are substituted.', '',
                  markdown_table(pd.read_csv(T / f'task2_{ds}_case_summary.csv')[['original_drug_ids', 'train_observations', 'test_candidates', 'true_best_predicted_rank', 'true_best_bias_rank', 'mse', 'ci_global', 'hit1', 'hit5', 'mrr']]), '',
                  f'Figures: `figures/task2_{ds}_comparison.png`, `figures/task2_{ds}_diagnostics.png`, `figures/task2_{ds}_learning_curve.png`. Tables: `tables/task2_{ds}_case_rankings.csv`, `tables/task2_{ds}_label_bins.csv`, `tables/task2_{ds}_diagnostics.csv`, `tables/task2_{ds}_overlap.csv`. Original-ID alias lists in case tables are the observed sources for that pair; entity tables contain the complete aliases.', '']
        cases = pd.read_csv(T / f'task2_{ds}_case_rankings.csv')
        case_notes = [f'# {ds.upper()} held-out case rankings', '']
        for drug, group in cases.groupby('drug_id'):
            case_notes += [f'## Processed ligand {drug}', '', markdown_table(group.head(10)), '']
        (ROOT / f'results/task2_{ds}_cases.md').write_text('\n'.join(case_notes))
    abnormal = pd.read_csv(ROOT / 'data/processed/kiba_suspicious_audit.csv')
    note2 += ['## Anomaly audit and scientific limits', '',
              'The following finite nonpositive KIBA observations occur in the pinned upstream file. Their presence and provenance are verified; assay-level correctness cannot be established from these files. They remain flagged and retained, without relabelling missing values or deleting an inconvenient score.', '', markdown_table(abnormal), '',
              'DAVIS screening-bound values are benchmark labels, not exact measured affinities. Mean-merging aliases changes the usual benchmark counts and loses assay/isoform distinctions. This is a new canonical-identity split, not the official DeepDTA folds. Only one warm seed was used. Rankings within measured candidates do not establish whole-proteome retrieval.', '']
    if (T / 'task2_davis_thinning.csv').exists():
        note2 += ['## Optional train-only thinning', '', markdown_table(pd.read_csv(T / 'task2_davis_thinning.csv')[['fraction', 'model', 'training_pairs', 'unseen_test_pairs', 'mse', 'ci_global']]), '',
                  'Nested 100/50/25/10/5% subsets change training only; the full warm test is fixed. Full-training validation-selected hyperparameters are retained, with checkpoint epoch selected on the same validation partition for each subset. Unseen entity factors/biases are zero, giving partial/global additive fallback. Figure: `figures/task2_davis_thinning.png`.', '']
    note2 += ['## Paired uncertainty', '', markdown_table(pd.DataFrame(bootstraps).query("task == 'task2'").drop(columns=['task', 'split_seed'], errors='ignore')), '',
              'Intervals use 1,000 paired ligand-group bootstrap replicates, and describe MF-minus-bias MSE on the fixed split. They do not account for split choice.', '', '## References and assistance', '', REFERENCES]
    (ROOT / 'results/task2_report_notes.md').write_text('\n'.join(note2))
    agg_short = aggregate[['dataset', 'model', 'mse_mean', 'mse_std', 'ci_global_mean', 'ci_global_std', 'hit1_mean', 'hit5_mean', 'mrr_mean']]
    note3 = ['# Task 3 — cold-target report evidence (v1)', '',
             'Five separately trained target-sequence holdouts use seeds 2026–2030. Approximately 70/15/15% of unique exact normalized sequences go to train/validation/test. Every affinity measurement for a sequence remains in one partition. All model selection uses cold validation only; test sequences have zero training affinity observations. Evaluation ligands without training observations are explicitly flagged and retain a global/partial-bias fallback; exact counts are in split manifests and saved prediction files.', '',
             '## Algorithms', '',
             '**C0:** ligand means from each cold-training partition, with global training mean for unknown ligands. **C1:** compare a cold full sequence with training sequences using L2-normalized fixed A–Z 3-mer count cosine. Take top K training proteins; transfer available observations for that ligand, weighted by cosine^power, otherwise use C0. This is algebraically C0 plus weighted ligand-mean residuals. K={5,15,50}, power={1,4} are selected on cold validation. Missing ligand–neighbour measurements contribute no weight.', '',
             '**C2 learned factor ridge:** fit a separate rank-16 biased MF for up to 80 fixed-budget epochs on cold-train only (train-error checkpoint, reg .01 and LR .02); never import warm-task factors. Represent sequences with fixed 3-mer, 20-AA composition and 400-dipeptide frequencies. Train-target-centered multioutput ridge maps these sequence features to learned training-target bias and factors. Tune ridge alpha {.1,1,10,100} on cold validation. Predict mu+b_ligand+predicted_bias+U_ligand·predicted_factor. Cold queries do not use target-ID factors. Unobserved ligand factors are zero. The MF optimizer seed is fixed at 2026 for all split seeds. Cached radius-2 2048-bit Morgan features are available; C2 uses cold-train ligand factors instead. No SVD, imputation, pretrained embedding or test-label fitting is used.', '',
             'Neither dataset supplied an S-W similarity file locally, so this implementation computes label-free sequence cosine instead of downloading or assuming a normalization. Cosine is bounded 0..1, but is not alignment identity. Pre-specified <.3/.3–.5/.5–.7/>=.7 buckets describe this feature similarity; do not interpret them as sequence-identity family thresholds.', '',
             '## Five-split aggregate', '', markdown_table(agg_short), '',
             'The table gives arithmetic mean and sample standard deviation (ddof=1) across five repeated target splits; they are not independent five-fold CV, confidence intervals, or bootstrap variability. Every model is retrained per split. Full metric means/std/counts, including per-ligand and per-target concordance, are in tables/task3_aggregate.csv. Per-split primary model selection is saved before test evaluation and its aggregate is in tables/task3_validation_selected_aggregate.csv.', '', METRICS, '',
             '## Per-split results', '', markdown_table(test[['dataset', 'seed', 'model', 'selected_by_validation', 'mse', 'ci_global', 'ci_per_ligand', 'ci_per_target', 'hit1', 'hit5', 'mrr']]), '']
    coverage_rows = []
    for ds in DATASETS:
        for seed in COLD_SEEDS:
            r = json.loads((T / f'task3_{ds}_seed{seed}_run.json').read_text())
            coverage_rows.append({'dataset': ds, 'seed': seed, 'primary_model': r['selected_model'],
                                  'knn_test_fallback_fraction': r['coverage']['test']['fallback_fraction'],
                                  'unseen_ligand_test_pairs': r['coverage']['test']['unseen_ligand_pairs'],
                                  'cold_mf_seconds': r['cold_mf_seconds'], 'whole_run_seconds': r['elapsed_seconds']})
    pd.DataFrame(coverage_rows).to_csv(T / 'task3_coverage_cost.csv', index=False)
    note3 += ['## Coverage, costs and failure analysis', '', markdown_table(pd.DataFrame(coverage_rows)), '',
              'Sequence neighbours may have no observed measurement for a query ligand, especially in sparse KIBA; C0 then provides no target-specific adjustment. The factor-ridge method supplies an adjustment through sequence features, but a linear sequence-to-latent mapping can miss target selectivity and overfit small training target sets. Composition and 3-mers lose order beyond three residues; variable full sequence lengths and domains alter cosine. Distribution shift and related sequences remaining across splits limit claims of remote-homology generalization. Per-target errors, sequence lengths, neighbour support and nearest-training similarities are exported for each split; bucket counts must accompany any performance claim. Empty buckets remain empty.', '',
              'Files: tables/task3_{dataset}_seed{seed}_similarity_buckets.csv, *_per_target.csv and *_sequence_audit.csv; figures/task3_{dataset}_seed{seed}_similarity.png and *_comparison.png. Saved neighbours and ridge archives contain training-only target sequences, observations or latent outputs, provenance, parameters and split hashes. Every archive was checked in a fresh interpreter with label-free queries, and identical score order after reload.', '',
              '## Scope and limitations', '',
              'All core and priority-one models/splits/diagnostics in the agenda are implemented. The optional C3 pair regressor, ESM-2, strict family cold split, mixup/ensemble and censored-loss MF were not run; no results are asserted for them. Original SW matrices were not supplied and were not assumed available. The official rubric and project plan were not found, so exact rubric compliance beyond the agenda cannot be certified. Warm-pair and cold-target errors answer different questions and are not compared as the same protocol.', '', '## References and assistance', '', REFERENCES]
    (ROOT / 'results/task3_report_notes.md').write_text('\n'.join(note3))
    report = ROOT / 'reports'
    report.mkdir(exist_ok=True)
    (report / 'task2_task3_report.md').write_text('# SC4020 Tasks 2 and 3 — executed evidence\n\n' + '\n'.join(note2[1:]) + '\n\n' + '\n'.join(note3))
    log('Phase 6 export', 'Generated aggregate tables, 1000-replicate paired ligand bootstraps, case Markdown and Task 2/3 report notes directly from saved result files.')
    print(agg_short.to_string(index=False))
if __name__ == '__main__':
    main()
