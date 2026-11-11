"""Experiment orchestration with validation-only selection and cached test results."""
import itertools
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, SEED, setup, write_json, log, provenance, sha256
from .config import CODE_ROOT, MF_GRID, MF_EPOCHS, MF_PATIENCE, KNN_K, KNN_BETA
from .cf import Means, BiasOnly, MatrixFactorization, TargetKNN, dimensions
from .baselines import TargetMean
from .splits import load_splits
from .metrics import evaluate, concordance_index, random_ranking_reference
from .plots import export_predictions, comparison_plot, diagnostics, case_studies, save_figure


def fresh_reload(kind, checkpoint, query, expected, tag):
    # Isolated, label-free query artifact sent to a fresh Python process.
    query_path = ROOT / f'results/logs/{tag}_reload_query.csv'
    output_path = ROOT / f'results/logs/{tag}_reload_predictions.npy'
    query[[c for c in ['drug_id', 'target_id', 'sequence'] if c in query]].to_csv(query_path, index=False)
    subprocess.run([sys.executable, str(CODE_ROOT / 'scripts/reload_check.py'), kind, str(checkpoint),
                    str(query_path), str(output_path)], check=True)
    reloaded = np.load(output_path, allow_pickle=False)
    np.testing.assert_allclose(expected, reloaded, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(np.argsort(-expected, kind='stable'), np.argsort(-reloaded, kind='stable'))
    # Entire prediction order equality implies identical per-ligand rankings too.
    result = {'fresh_process': True, 'allclose_rtol': 1e-12, 'allclose_atol': 1e-12,
              'ranking_order_identical': True, 'max_absolute_difference': float(np.max(np.abs(expected - reloaded))),
              'model_sha256': sha256(checkpoint), 'pairs': len(query)}
    write_json(ROOT / f'results/tables/{tag}_reload.json', result)
    return result


def selection_key(y, p):
    return float(np.mean((y - p) ** 2)), -concordance_index(y, p)


def cached(tag, manifest):
    path = ROOT / f'results/tables/{tag}_run.json'
    if path.exists():
        record = json.loads(path.read_text())
        if record['split_sha256'] != manifest['split_sha256']:
            raise RuntimeError(f'{tag}: existing evaluation uses different split')
        print(tag, 'already evaluated; reusing frozen metrics', flush=True)
        return True
    return False


def run_cf(dataset, smoke=False):
    setup()
    frame, parts, manifest = load_splits(dataset, 'task2')
    tag = f'task2_{dataset}'
    if cached(tag, manifest):
        return
    train, val = parts['train'], parts['validation']
    nd, nt = dimensions(frame)
    means, target_mean = Means(train, nd), TargetMean(train, nt)
    bias_trials, best_bias_key, bias = [], (np.inf, np.inf), None
    for reg in [1., 10., 50.]:
        candidate = BiasOnly(train, nd, nt, regularization=reg)
        key = selection_key(val.affinity.to_numpy(), candidate.predict(val))
        bias_trials.append({'model': 'bias_only', 'regularization': reg, 'validation_mse': key[0], 'validation_ci': -key[1]})
        if key < best_bias_key:
            best_bias_key, bias, bias_reg = key, candidate, reg
    trials, histories, best_key = [], [], (np.inf, np.inf)
    ranks = [0, 4] if smoke else MF_GRID['factors']
    for trial, (rank, reg, lr) in enumerate(itertools.product(ranks, MF_GRID['regularization'], MF_GRID['learning_rate'])):
        model = MatrixFactorization(nd, nt, factors=rank, regularization=reg, learning_rate=lr,
                                    max_epochs=5 if smoke else MF_EPOCHS, patience=MF_PATIENCE)
        history = model.fit(train, val)
        history['trial'] = trial
        histories.append(history)
        key = selection_key(val.affinity.to_numpy(), model.predict(val))
        trials.append({'model': 'MF', 'trial': trial, **model.params, 'best_epoch': model.best_epoch,
                       'validation_mse': key[0], 'validation_ci': -key[1]})
        print(dataset, 'MF', trials[-1], flush=True)
        if key < best_key:
            best_key, best, selected_trial = key, model, trial
    knn = TargetKNN(train, nd, nt, bias=bias)
    knn_trials, knn_key = [], (np.inf, np.inf)
    for k, beta in itertools.product(KNN_K, KNN_BETA):
        knn.k, knn.beta = k, beta
        p, detail = knn.predict(val)
        key = selection_key(val.affinity.to_numpy(), p)
        knn_trials.append({'model': 'target_knn', 'k': k, 'beta': beta, 'validation_mse': key[0],
                           'validation_ci': -key[1], 'fallback_fraction': float(detail['used_fallback'].mean())})
        if key < knn_key:
            knn_key, selected_knn = key, (k, beta)
    knn.k, knn.beta = selected_knn
    pd.DataFrame(bias_trials + trials + knn_trials).to_csv(ROOT / f'results/tables/{tag}_tuning.csv', index=False)
    pd.concat(histories).to_csv(ROOT / f'results/tables/{tag}_learning_curves.csv', index=False)
    scale = 'pKd' if dataset == 'davis' else 'DeepDTA_KIBA'
    metadata = {**provenance(), 'dataset': dataset, 'task': 'task2', 'score_name': scale,
                'score_bounds': [float(train.affinity.min()), float(train.affinity.max())], 'clipping': False,
                'split_sha256': manifest['split_sha256'], 'clean_sha256': manifest['clean_sha256']}
    checkpoint = ROOT / f'saved_models/{dataset}_cf.npz'
    best.save(checkpoint, metadata,
              pd.read_csv(ROOT / f'data/processed/ligands_{dataset}.csv').sort_values('drug_id').original_drug_id,
              pd.read_csv(ROOT / f'data/processed/targets_{dataset}.csv').sort_values('target_id').original_target_id)
    reloaded = MatrixFactorization.load(checkpoint)
    reload_result = fresh_reload('mf', checkpoint, frame, best.predict(frame), tag)
    np.savez_compressed(ROOT / f'saved_models/{dataset}_baselines_knn.npz',
                        mu=bias.mu, drug_bias=bias.drug_bias, target_bias=bias.target_bias,
                        residual=knn.residual, observed=knn.observed, similarity=knn.similarity, overlap=knn.overlap,
                        drug_mean=means.drug_mean, target_mean=target_mean.means,
                        metadata=json.dumps({**metadata, 'bias_regularization': bias_reg, 'k': knn.k, 'beta': knn.beta}))
    rows, fallback = [], {}
    for split, part in parts.items():
        pknn, detail = knn.predict(part)
        predictions = {'global_mean': np.full(len(part), means.mu), 'ligand_mean': means.predict(part),
                       'target_mean': target_mean.predict(part), 'bias_only': bias.predict(part),
                       'target_knn': pknn, 'MF': reloaded.predict(part)}
        for name, p in predictions.items():
            rows.append({'dataset': dataset, 'task': 'task2', 'seed': SEED, 'split': split, 'model': name, **evaluate(part, p)})
        rows.append({'dataset': dataset, 'task': 'task2', 'seed': SEED, 'split': split, 'model': 'random_ranking', **random_ranking_reference(part)})
        export_predictions(tag, split, part, predictions, {'knn_support': detail['neighbor_support'], 'knn_fallback': detail['used_fallback']})
        fallback[split] = float(detail['used_fallback'].mean())
        if split == 'test':
            d, t = part.drug_id.to_numpy(), part.target_id.to_numpy()
            latent = np.einsum('ij,ij->i', reloaded.U[d], reloaded.V[t])
            association = diagnostics(dataset, part, train, predictions['MF'], predictions['bias_only'], latent, detail)
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / f'results/tables/{tag}_metrics.csv', index=False)
    comparison_plot(table, tag, scale)
    cases = case_studies(dataset, train, parts['test'], reloaded, bias)
    overlap = knn.overlap[np.triu_indices(nt, 1)]
    pd.DataFrame({'co_observed_ligands': overlap}).to_csv(ROOT / f'results/tables/{tag}_overlap.csv', index=False)
    history = histories[selected_trial]
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(history.epoch, history.train_mse, label='Train')
    ax.plot(history.epoch, history.validation_mse, label='Validation')
    ax.axvline(best.best_epoch, color='grey', linestyle='--')
    ax.set(xlabel='Epoch', ylabel=f'MSE ({scale} squared)', title=dataset.upper() + ' selected MF')
    ax.legend()
    save_figure(fig, tag + '_learning_curve')
    record = {**metadata, 'selected': trials[selected_trial], 'selected_knn': {'k': knn.k, 'beta': knn.beta},
              'bias_regularization': bias_reg, 'reload': reload_result, 'knn_fallback_fraction': fallback,
              'latent_residual_correlation': association, 'cases': cases,
              'train_pair_density': len(train) / (nd * nt), 'mf_parameter_count': 1 + (nd + nt) * (1 + best.params['factors']),
              'bias_parameter_count': 1 + nd + nt, 'test_evaluation_policy': 'One evaluation per selected model; reruns reuse cached files',
              'overlap_quantiles': {str(q): float(v) for q, v in pd.Series(overlap).quantile([0, .25, .5, .75, 1]).items()}}
    write_json(ROOT / f'results/tables/{tag}_run.json', record)
    log('Phase 4', f'{dataset}: baseline/MF/kNN evaluation saved; {cases} cases; fresh-process prediction and rank parity PASS. Split {manifest["split_sha256"]}.')
    print(table[table.split == 'test'][['model', 'mse', 'ci_global', 'hit1', 'hit5', 'mrr']].to_string(index=False), flush=True)


def run_cold(dataset, seed=SEED, smoke=False):
    import time
    import matplotlib.pyplot as plt
    from .cold import ColdTargetNeighbors, ColdFactorRidge
    from .config import COLD_K, COLD_POWER, RIDGE_ALPHA, SIMILARITY_BINS
    from .features import cache_features, sequence_similarity
    started = time.perf_counter()
    setup()
    frame, parts, manifest = load_splits(dataset, 'task3', seed)
    tag = f'task3_{dataset}_seed{seed}'
    if cached(tag, manifest):
        return
    train, val = parts['train'], parts['validation']
    nd, nt = dimensions(frame)
    if seed == SEED:
        cache_features(dataset, frame)
    means = Means(train, nd)
    trials, best_key = [], (np.inf, np.inf)
    neighbors = ColdTargetNeighbors().fit(train, nd)
    for k, power in itertools.product(COLD_K, COLD_POWER):
        neighbors.params.update(k=k, power=power)
        tick = time.perf_counter()
        prediction, details = neighbors.predict(val)
        key = selection_key(val.affinity.to_numpy(), prediction)
        trials.append({'model': 'sequence_neighbors', 'k': k, 'power': power, 'validation_mse': key[0],
                       'validation_ci': -key[1], 'seconds': time.perf_counter() - tick,
                       'fallback_fraction': float(details['used_fallback'].mean())})
        if key < best_key:
            best_key, neighbor_params = key, neighbors.params.copy()
    neighbors.params = neighbor_params
    tick = time.perf_counter()
    ridge = ColdFactorRidge().fit(train, nd, nt, factors=4 if smoke else 16, epochs=5 if smoke else 80)
    mf_seconds = time.perf_counter() - tick
    ridge.history.to_csv(ROOT / f'results/tables/{tag}_cold_mf_training.csv', index=False)
    ridge_key = (np.inf, np.inf)
    for alpha in RIDGE_ALPHA:
        tick = time.perf_counter()
        ridge.set_alpha(alpha)
        prediction = ridge.predict(val)
        key = selection_key(val.affinity.to_numpy(), prediction)
        trials.append({'model': 'factor_ridge', 'alpha': alpha, 'validation_mse': key[0],
                       'validation_ci': -key[1], 'seconds': time.perf_counter() - tick})
        if key < ridge_key:
            ridge_key, ridge_alpha = key, alpha
    ridge.set_alpha(ridge_alpha)
    pd.DataFrame(trials).to_csv(ROOT / f'results/tables/{tag}_tuning.csv', index=False)
    scale = 'pKd' if dataset == 'davis' else 'DeepDTA_KIBA'
    metadata = {**provenance(), 'seed': seed, 'dataset': dataset, 'task': 'task3', 'score_name': scale,
                'score_bounds': [float(train.affinity.min()), float(train.affinity.max())], 'clipping': False,
                'split_sha256': manifest['split_sha256'], 'clean_sha256': manifest['clean_sha256'],
                'trained_target_count': int(train.target_id.nunique()), 'evaluation_target_training_observations': 0,
                'representation_fit_policy': 'Fixed sequence vocabulary; ridge centering from cold-train targets only; MF from cold-train affinities only'}
    paths = {name: ROOT / f'saved_models/{dataset}_{name}_seed{seed}.npz' for name in ['neighbors', 'ridge']}
    neighbors.save(paths['neighbors'], metadata)
    ridge.save(paths['ridge'], metadata)
    query = pd.concat([val, parts['test']])
    reloads = {'neighbors': fresh_reload('neighbors', paths['neighbors'], query, neighbors.predict(query)[0], tag + '_neighbors'),
               'ridge': fresh_reload('ridge', paths['ridge'], query, ridge.predict(query), tag + '_ridge')}
    neighbors = ColdTargetNeighbors.load(paths['neighbors'])
    ridge = ColdFactorRidge.load(paths['ridge'])
    baseline_val = selection_key(val.affinity.to_numpy(), means.predict(val))
    # Freeze one primary cold model by VALIDATION before calculating any test metric.
    selection = min([('ligand_mean', baseline_val), ('sequence_neighbors', best_key), ('factor_ridge', ridge_key)], key=lambda item: item[1])[0]
    rows, buckets, coverage, per_targets = [], [], {}, []
    for split in ['validation', 'test']:
        part = parts[split]
        neighbor_pred, details = neighbors.predict(part)
        predictions = {'ligand_mean': means.predict(part), 'sequence_neighbors': neighbor_pred, 'factor_ridge': ridge.predict(part)}
        for name, prediction in predictions.items():
            rows.append({'dataset': dataset, 'task': 'task3', 'seed': seed, 'split': split,
                         'model': name, 'selected_by_validation': name == selection, **evaluate(part, prediction)})
        rows.append({'dataset': dataset, 'task': 'task3', 'seed': seed, 'split': split,
                     'model': 'random_ranking', 'selected_by_validation': False, **random_ranking_reference(part)})
        export_predictions(tag, split, part, predictions, details)
        coverage[split] = {'fallback_pairs': int(details['used_fallback'].sum()),
                           'fallback_fraction': float(details['used_fallback'].mean()),
                           'unseen_ligand_pairs': int(details['unseen_drug'].sum())}
        # Per-protein metrics and pre-specified similarity bins; never tune on these.
        for target, indices in part.groupby('target_id', sort=True).indices.items():
            group = part.iloc[indices]
            for name, prediction in predictions.items():
                p = prediction[indices]
                ci = concordance_index(group.affinity, p)
                per_targets.append({'dataset': dataset, 'seed': seed, 'split': split, 'model': name,
                                    'target_id': int(target), 'original_target_ids': str(group.original_target_ids.iloc[0]),
                                    'pairs': len(group), 'mse': float(np.mean((group.affinity.to_numpy() - p) ** 2)),
                                    'ci': ci, 'max_train_similarity': float(details['max_train_similarity'][indices[0]]),
                                    'sequence_length': len(group.sequence.iloc[0]),
                                    'neighbor_fallback_fraction': float(details['used_fallback'][indices].mean())})
        if split == 'test':
            for low, high, label in zip(SIMILARITY_BINS[:-1], SIMILARITY_BINS[1:], ['<0.3', '0.3–0.5', '0.5–0.7', '>=0.7']):
                mask = (details['max_train_similarity'] >= low) & (details['max_train_similarity'] < high)
                for name, prediction in predictions.items():
                    p, y = prediction[mask], part.affinity.to_numpy()[mask]
                    buckets.append({'dataset': dataset, 'seed': seed, 'model': name, 'bucket': label,
                                    'pairs': int(mask.sum()), 'targets': int(part.loc[mask, 'target_id'].nunique()),
                                    'mse': float(np.mean((p - y) ** 2)) if len(p) else None,
                                    'ci': concordance_index(y, p) if len(p) else None})
    table, targets_table = pd.DataFrame(rows), pd.DataFrame(per_targets)
    table.to_csv(ROOT / f'results/tables/{tag}_metrics.csv', index=False)
    targets_table.to_csv(ROOT / f'results/tables/{tag}_per_target.csv', index=False)
    bucket_table = pd.DataFrame(buckets)
    bucket_table.to_csv(ROOT / f'results/tables/{tag}_similarity_buckets.csv', index=False)
    comparison_plot(table, tag, scale)
    chosen = targets_table[(targets_table.split == 'test') & (targets_table.model == selection)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    sc = axes[0].scatter(chosen.max_train_similarity, chosen.mse, c=chosen.neighbor_fallback_fraction, vmin=0, vmax=1)
    axes[0].set(xlabel='Max 3-mer cosine to cold-train', ylabel=f'MSE ({scale} squared)', title=selection)
    fig.colorbar(sc, ax=axes[0], label='Neighbour fallback fraction')
    b = bucket_table[bucket_table.model == selection]
    axes[1].bar(b.bucket, b.mse)
    axes[1].set(ylabel='MSE', title='Pre-specified similarity buckets')
    for i, row in enumerate(b.itertuples()):
        if pd.notna(row.mse):
            axes[1].text(i, row.mse, f'n={row.pairs}', ha='center', va='bottom', fontsize=8)
    fig.suptitle(tag)
    fig.tight_layout()
    save_figure(fig, tag + '_similarity')
    # Save a label-free train/test nearest-sequence audit (cosine is not alignment identity).
    tr = train[['target_id', 'sequence']].drop_duplicates().sort_values('target_id')
    audit = []
    for split in ['validation', 'test']:
        queries = parts[split][['target_id', 'sequence']].drop_duplicates().sort_values('target_id')
        sim = sequence_similarity(queries.sequence, tr.sequence)
        nearest = sim.argmax(axis=1)
        for i, target in enumerate(queries.target_id):
            audit.append({'target_id': int(target), 'split': split, 'nearest_training_target': int(tr.target_id.iloc[nearest[i]]),
                          'max_3mer_cosine': float(sim[i, nearest[i]])})
    pd.DataFrame(audit).to_csv(ROOT / f'results/tables/{tag}_sequence_audit.csv', index=False)
    write_json(ROOT / f'results/tables/{tag}_run.json',
               {**metadata, 'selected_model': selection, 'neighbor_params': neighbor_params, 'ridge_alpha': ridge_alpha,
                'ridge_mf_params': ridge.params, 'cold_mf_seconds': mf_seconds, 'coverage': coverage,
                'fresh_reload': reloads, 'elapsed_seconds': time.perf_counter() - started,
                'similarity_policy': 'L2-normalized fixed-vocabulary 3-mer cosine (0..1), not Smith-Waterman identity; buckets are descriptive',
                'test_evaluation_policy': 'One evaluation per selected configuration; cached on rerun'})
    log('Phase 5', f'{dataset}/seed={seed}: C0, sequence kNN, learned factor ridge evaluated; primary={selection} by validation; fresh-process reload assertions PASS; no evaluation sequence/target in train.')
    print(tag, table[table.split == 'test'][['model', 'mse', 'ci_global', 'hit1', 'hit5', 'mrr']].to_string(index=False), flush=True)
