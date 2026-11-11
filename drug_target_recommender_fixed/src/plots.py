"""Report artifacts derived from saved experiment predictions."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .common import ROOT
from .metrics import concordance_index, ranking_metrics, evaluate

plt.rcParams.update({'font.size': 9, 'figure.dpi': 130, 'savefig.dpi': 200,
                     'axes.spines.top': False, 'axes.spines.right': False})


def save_figure(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(ROOT / f'results/figures/{name}.{ext}', bbox_inches='tight')
    plt.close(fig)


def export_predictions(tag, split, frame, predictions, details=None):
    table = frame[['drug_id', 'target_id', 'affinity', 'original_drug_ids', 'original_target_ids']].copy()
    table.insert(0, 'pair_index', frame.index)
    for name, p in predictions.items():
        table[name] = p
        _, ranking = ranking_metrics(frame, p)
        ranking.to_csv(ROOT / f'results/tables/{tag}_{split}_{name}_ranking.csv', index=False)
    for key, values in (details or {}).items():
        table[key] = values
    table.to_csv(ROOT / f'results/predictions/{tag}_{split}.csv', index=False, float_format='%.17g')


def comparison_plot(table, tag, scale):
    test = table[(table.split == 'test') & table.mse.notna()]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    colors = plt.cm.viridis(np.linspace(.15, .85, len(test)))
    for ax, metric in zip(axes, ['mse', 'ci_global', 'mrr']):
        ax.bar(test.model, test[metric], color=colors)
        ax.set_title({'mse': f'MSE ({scale} squared)', 'ci_global': 'Global concordance', 'mrr': 'Per-ligand MRR'}[metric])
        ax.tick_params(axis='x', labelrotation=55)
    axes[1].set_ylim(0, 1)
    axes[2].set_ylim(0, 1)
    fig.suptitle(tag + ' · held-out measured candidates')
    fig.tight_layout()
    save_figure(fig, tag + '_comparison')


def diagnostics(dataset, test, train, predictions, bias_prediction, latent, knn_details):
    tag = f'task2_{dataset}'
    work = test[['drug_id', 'target_id', 'affinity', 'is_censored']].copy()
    work['prediction'] = predictions
    work['squared_error'] = (work.affinity - predictions) ** 2
    work['train_count'] = work.drug_id.map(train.groupby('drug_id').size()).fillna(0)
    # Label bins are dataset-specific, use only for descriptive test analysis.
    edges = [4.99, 5.01, 6, 7, 8, 9, 11] if dataset == 'davis' else [-.01, 5, 10, 12, 14, 16, 18, 25]
    work['label_bin'] = pd.cut(work.affinity, edges, include_lowest=True)
    rows = []
    for name, group in work.groupby('label_bin', observed=False):
        rows.append({'label_bin': str(name), 'pairs': len(group), 'mse': group.squared_error.mean()})
    pd.DataFrame(rows).to_csv(ROOT / f'results/tables/{tag}_label_bins.csv', index=False)
    sparse = []
    for low, high in [(0, 20), (20, 50), (50, 100), (100, 100000)]:
        mask = (work.train_count >= low) & (work.train_count < high)
        if mask.any():
            sparse.append({'min_train_count': low, 'max_train_count': high, 'pairs': int(mask.sum()),
                           'mf_mse': float(work.loc[mask, 'squared_error'].mean()),
                           'bias_mse': float(np.mean((work.loc[mask, 'affinity'] - bias_prediction[mask]) ** 2)),
                           'knn_fallback_fraction': float(knn_details['used_fallback'][mask].mean())})
    pd.DataFrame(sparse).to_csv(ROOT / f'results/tables/{tag}_sparsity.csv', index=False)
    residual = test.affinity.to_numpy() - bias_prediction
    association = float(np.corrcoef(residual, latent)[0, 1]) if np.std(latent) > 0 and np.std(residual) > 0 else None
    masks = {'all': np.ones(len(work), dtype=bool)}
    if dataset == 'davis':
        masks.update({'any_censored_source': test.is_censored.to_numpy(),
                      'no_censored_source': ~test.is_censored.to_numpy(), 'pKd_at_least_7': test.affinity.to_numpy() >= 7})
    summary = [{'group': key, 'pairs': int(mask.sum()), 'mse': float(work.loc[mask, 'squared_error'].mean()) if mask.any() else None}
               for key, mask in masks.items()]
    pd.DataFrame(summary).to_csv(ROOT / f'results/tables/{tag}_diagnostics.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].hexbin(work.affinity, predictions, gridsize=35, mincnt=1)
    lo, hi = min(work.affinity.min(), predictions.min()), max(work.affinity.max(), predictions.max())
    axes[0].plot([lo, hi], [lo, hi], '--', color='grey')
    axes[0].set(xlabel='True score', ylabel='Predicted score', title='MF on held-out pairs')
    bins = pd.DataFrame(rows)
    axes[1].bar(bins.label_bin, bins.mse)
    axes[1].tick_params(axis='x', labelrotation=65)
    axes[1].set(ylabel='MSE', title='Error by true-score bin')
    fig.suptitle(dataset.upper())
    fig.tight_layout()
    save_figure(fig, tag + '_diagnostics')
    return association


def case_studies(dataset, train, test, model, bias):
    counts = train.groupby('drug_id').size()
    eligibility = test.groupby('drug_id').affinity.agg(['size', 'nunique'])
    eligible = eligibility[(eligibility['size'] >= 10) & (eligibility['nunique'] > 1)].index
    if len(eligible) < 3:
        eligible = eligibility[(eligibility['size'] >= 2) & (eligibility['nunique'] > 1)].index
    candidates = counts.reindex(eligible).sort_values(kind='stable')
    suggested = {'davis': ['44259', '11338033', '10127622'], 'kiba': ['CHEMBL1967538', 'CHEMBL2004544', 'CHEMBL565612']}[dataset]
    chosen, reasons = [], {}
    for original in suggested:
        ids = test.loc[test.original_drug_ids.astype(str).str.split('|').apply(lambda v: original in v), 'drug_id'].unique()
        for drug in ids:
            if drug in eligible and drug not in chosen:
                chosen.append(int(drug)); reasons[int(drug)] = 'Suggested source ID; >=10 measured test candidates and nonconstant true scores' if eligibility.loc[drug, 'size'] >= 10 else 'Suggested source ID; >=2 measured test candidates'
    for index in ([0, len(candidates) // 2, len(candidates) - 1] if len(candidates) else []):
        drug = int(candidates.index[index])
        if drug not in chosen and len(chosen) < 3:
            chosen.append(drug); reasons[drug] = 'Sparse/median/dense training history among eligible ligands'
    for drug in candidates.index:
        if len(chosen) >= 3:
            break
        if drug not in chosen:
            chosen.append(int(drug)); reasons[int(drug)] = 'Additional eligible held-out candidate set'
    rows, summaries = [], []
    for drug in chosen[:3]:
        part = test[test.drug_id == drug].copy()
        p, b = model.predict(part), bias.predict(part)
        out = part[['drug_id', 'target_id', 'original_drug_ids', 'original_target_ids', 'affinity']].copy()
        out['prediction'], out['bias_prediction'] = p, b
        out['predicted_rank'] = pd.Series(p).rank(ascending=False, method='min').to_numpy().astype(int)
        out['bias_rank'] = pd.Series(b).rank(ascending=False, method='min').to_numpy().astype(int)
        out['is_true_best'] = part.affinity.to_numpy() == part.affinity.max()
        out['top1_correct'] = (out.predicted_rank == 1) & out.is_true_best
        summaries.append({'drug_id': drug, 'original_drug_ids': str(part.original_drug_ids.iloc[0]),
                          'selection_reason': reasons[drug], 'train_observations': int(counts[drug]),
                          'test_candidates': len(part), 'true_best_predicted_rank': int(out.loc[out.is_true_best, 'predicted_rank'].min()),
                          'true_best_bias_rank': int(out.loc[out.is_true_best, 'bias_rank'].min()),
                          'top10_defined': len(part) >= 10, **evaluate(part, p, secondary=False)})
        rows.append(out.sort_values(['predicted_rank', 'target_id']))
    pd.concat(rows).to_csv(ROOT / f'results/tables/task2_{dataset}_case_rankings.csv', index=False)
    pd.DataFrame(summaries).to_csv(ROOT / f'results/tables/task2_{dataset}_case_summary.csv', index=False)
    return len(summaries)
