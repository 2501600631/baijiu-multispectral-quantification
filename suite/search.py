import copy
import numpy as np
from sklearn.model_selection import KFold
from .common import subset, iqr, digest
from .models import fit_model

def candidates(modes, profile, full=True, representation='direct'):
    fams = ['pls1', 'pls2', 'ridge', 'pcr', 'svr', 'kernel_ridge', 'extra_trees', 'random_forest', 'hist_gradient_boosting', 'multitask_elasticnet', 'reduced_rank', 'regressor_chain', 'graph_ridge']
    if not full:
        fams = ['pls2', 'ridge', 'kernel_ridge']
    if profile.get('is_smoke'):
        fams = ['ridge', 'pls2']
    if representation != 'direct':
        fams = ['ridge', 'kernel_ridge'] if not profile.get('is_smoke') else ['ridge']
    answer = []
    count = profile['candidates_per_family']
    for family in fams:
        for i in range(count):
            spec = {'family': family, 'modes': list(modes), 'pre': ['smooth', 'raw', 'snv', 'derivative'][i % 4], 'representation': representation, 'params': {'n_components': [2, 4, 8, 12][i % 4], 'pca_components': [16, 24, 32, 20][i % 4], 'alpha': [0.1, 1, 10, 100][i % 4], 'C': [0.3, 3, 30, 100][i % 4], 'gamma': ['scale', 0.01, 0.1, 1][i % 4], 'n_estimators': profile['trees'], 'min_samples_leaf': [2, 3, 5, 1][i % 4], 'rank': [2, 4, 6, 8][i % 4], 'l1_ratio': [0.1, 0.5, 0.9, 0.5][i % 4], 'graph_strength': [0.1, 1, 10, 100][i % 4], 'max_leaf_nodes': [3, 7, 15, 7][i % 4], 'max_iter': 20000}}
            if family in ('svr', 'kernel_ridge'):
                spec['params']['pca_components'] = [16, 24, 32, 20][i % 4]
            answer.append(spec)
    return answer

def family_groups(catalog):
    result = {}
    for spec in catalog:
        result.setdefault(spec['family'], []).append(spec)
    return result

def tune(X, Y, catalog, profile, seed, ethanol=None):
    Y = np.asarray(Y, float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    k = min(profile['inner_folds'], len(Y) // 4)
    if k < 2:
        raise ValueError('Too few training bottles for inner tuning')
    splits = list(KFold(k, shuffle=True, random_state=seed).split(Y))
    valid = []
    audit = []
    failures = []
    for spec in catalog:
        loss = []
        error = []
        for fold, (tr, va) in enumerate(splits):
            try:
                model = fit_model(spec, subset(X, tr), Y[tr], None if ethanol is None else ethanol[tr], seed + fold)
                from .common import COMBINATIONS
                views = [subset(X, va)] if spec['family'] != 'masked_ridge' else [{m: np.asarray(X[m])[va] for m in modes} for modes in COMBINATIONS.values()]
                view_scores = []
                for view in views:
                    p = model.predict(view, None if ethanol is None else ethanol[va])
                    if p.shape != Y[va].shape or not np.isfinite(p).all():
                        raise ValueError('Invalid candidate output')
                    view_scores.append(float(np.sqrt(np.mean(((p - Y[va]) / iqr(Y[tr])) ** 2, axis=0)).mean()))
                loss.append(float(np.mean(view_scores)))
            except Exception as e:
                error.append({'fold': fold, 'error': repr(e)})
        if error:
            failures.append({'spec': spec, 'errors': error})
            continue
        score = float(np.mean(loss))
        audit.append({'spec': spec, 'score': score, 'fold_scores': loss})
        valid.append((score, digest(spec), spec))
    if not valid:
        raise RuntimeError('All regression candidates failed: ' + repr(failures))
    for score, _, spec in sorted(valid, key=lambda row: (round(row[0], 12), row[1])):
        try:
            model = fit_model(spec, X, Y, ethanol, seed)
            return (model, {'selected': spec, 'score': score, 'candidates': audit, 'failures': failures, 'inner_folds': k, 'metadata': model.metadata, 'metric': 'equal-target mean inner-fold training-IQR NRMSE'})
        except Exception as e:
            failures.append({'spec': spec, 'stage': 'final_fit', 'error': repr(e)})
    raise RuntimeError('All regression refits failed: ' + repr(failures))

def strict_crossfit(X, Y, catalog, profile, seed, ethanol=None):
    Y = np.asarray(Y, float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    k = min(profile['inner_folds'], len(Y) // 8)
    if k < 2:
        raise ValueError('Insufficient training bottles for strict crossfit')
    p = np.zeros(Y.shape)
    audits = []
    for fold, (tr, va) in enumerate(KFold(k, shuffle=True, random_state=seed).split(Y)):
        model, audit = tune(subset(X, tr), Y[tr], catalog, profile, seed + 101 + fold, None if ethanol is None else ethanol[tr])
        p[va] = model.predict(subset(X, va), None if ethanol is None else ethanol[va])
        audits.append(audit)
    return (p, audits)

def chemistry_descriptors(Y, names):
    Y = np.asarray(Y, float)
    col = {name: Y[:, i] for i, name in enumerate(names)}
    est = sum((col[t] for t in ['ethyl_acetate', 'ethyl_butyrate', 'ethyl_lactate', 'ethyl_hexanoate']))
    alc = sum((col[t] for t in ['n_propanol', 'isobutanol', 'n_butanol', 'isoamyl_alcohol']))
    pairs = [('ethyl_acetate', 'ethyl_hexanoate'), ('ethyl_lactate', 'ethyl_hexanoate'), ('ethyl_butyrate', 'ethyl_hexanoate'), ('isobutanol', 'isoamyl_alcohol')]
    desc = [est, alc]
    labels = ['four_ester_sum', 'four_higher_alcohol_sum']
    for a, b in pairs:
        good = (col[a] > 0) & (col[b] > 0)
        v = np.full(len(Y), np.nan)
        v[good] = np.log(col[a][good] / col[b][good])
        desc.append(v)
        labels.append('log_' + a + '_over_' + b)
    good = (est > 0) & (alc > 0)
    v = np.full(len(Y), np.nan)
    v[good] = np.log(est[good] / alc[good])
    desc.append(v)
    labels.append('log_four_esters_over_four_higher_alcohols')
    return (np.column_stack(desc), labels)
