from pathlib import Path
import copy
import traceback
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from .common import COMBINATIONS, subset, iqr, save_json, read_json, save_csv, digest
from .design import labels, frozen_plan
from .models import fit_model
from .search import candidates, family_groups, tune, strict_crossfit, chemistry_descriptors
from . import classification as clf

def regression_records(data, idx, p, method, scale, task, target_names=None, true=None):
    target_names = data['target_names'] if target_names is None else target_names
    true = data['Y'][idx] if true is None else true
    if not np.isfinite(p).all():
        raise ValueError('Non-finite predictions: ' + method)
    return pd.DataFrame([{'sample_id': str(data['ids'][i]), 'repeat': task['repeat'], 'fold': task['fold'], 'method': method, 'target': name, 'y_true': float(true[k, j]), 'y_pred': float(p[k, j]), 'train_iqr': float(scale[j]), 'fit_n': len(task['train'])} for k, i in enumerate(idx) for j, name in enumerate(target_names)])

def classification_records(data, idx, p, method, classes, task, y):
    pred = np.asarray(classes)[np.argmax(p, axis=1)]
    rows = []
    for k, i in enumerate(idx):
        row = {'sample_id': str(data['ids'][i]), 'repeat': task['repeat'], 'fold': task['fold'], 'method': method, 'y_true': str(y[i]), 'y_pred': str(pred[k]), 'fit_n': len(task['train'])}
        row.update({'p_' + c: float(p[k, j]) for j, c in enumerate(classes)})
        rows.append(row)
    return pd.DataFrame(rows)

def _core_regression(data, task, p):
    tr = np.array(task['train'])
    te = np.array(task['test'])
    seed = task['seed']
    X = subset(data['X'], tr)
    Xt = subset(data['X'], te)
    Y = data['Y'][tr]
    scale = iqr(Y)
    records = []
    audit = {}
    best = {}
    for key, modes in COMBINATIONS.items():
        print('  Regression subset: ' + key, flush=True)
        cat = candidates(modes, p, full=key == 'all')
        winners = []
        for family, options in family_groups(cat).items():
            try:
                m, a = tune(X, Y, options, p, seed)
            except Exception as e:
                raise RuntimeError('Required regression family failed for ' + key + ' / ' + family + ': ' + repr(e))
            name = key + '__' + family
            records.append(regression_records(data, te, m.predict(Xt), name, scale, task))
            audit[name] = a
            winners.append((a['score'], name, m, a))
        if not winners:
            raise RuntimeError('Every family failed for ' + key)
        _, name, m, a = min(winners, key=lambda row: (round(row[0], 12), row[1]))
        records.append(regression_records(data, te, m.predict(Xt), key + '__best', scale, task))
        audit[key + '__best'] = {'selected_from': name, 'score': a['score'], 'metadata': m.metadata}
        best[key] = (a['score'], m, a)
        if key == 'all':
            common = [row for row in winners if row[3]['selected']['family'] in ('pls2', 'ridge', 'kernel_ridge')]
            _, name, cm, ca = min(common, key=lambda row: (round(row[0], 12), row[1]))
            records.append(regression_records(data, te, cm.predict(Xt), 'all__common_best', scale, task))
            audit['all__common_best'] = {'selected_from': name, 'score': ca['score'], 'same_families_as_other_subsets': True}
    for group, keys in [('best_single', ['nir', 'raman', 'uv']), ('best_double', ['nir_raman', 'nir_uv', 'raman_uv'])]:
        selected = min(keys, key=lambda k: (round(best[k][0], 12), k))
        m = best[selected][1]
        records.append(regression_records(data, te, m.predict(Xt), group, scale, task))
        audit[group] = {'selected_subset': selected, 'selection_score': best[selected][0]}
    records.append(regression_records(data, te, np.tile(Y.mean(axis=0), (len(te), 1)), 'training_mean', scale, task))
    ax = dict(X)
    at = dict(Xt)
    abvcat = []
    for degree in [1, 2, 3]:
        name = 'abv_degree%d' % degree
        ax[name] = np.column_stack([data['ethanol'][tr] ** j for j in range(1, degree + 1)])
        at[name] = np.column_stack([data['ethanol'][te] ** j for j in range(1, degree + 1)])
        for spec in candidates([name], p, full=False):
            if spec['family'] in ('ridge', 'kernel_ridge'):
                spec['pre'] = 'raw'
                abvcat.append(spec)
    m, a = tune(ax, Y, abvcat, p, seed)
    a['requires_measured_ethanol'] = True
    audit['abv_only'] = a
    records.append(regression_records(data, te, m.predict(at), 'abv_only', scale, task))
    cat = [dict(s, use_ethanol=True) for s in candidates(COMBINATIONS['all'], p, full=False) if s['family'] == 'ridge']
    m, a = tune(X, Y, cat, p, seed, ethanol=data['ethanol'][tr])
    audit['all_plus_abv'] = a
    records.append(regression_records(data, te, m.predict(Xt, data['ethanol'][te]), 'all_plus_abv', scale, task))
    cat = [dict(s, target_transform='log1p') for s in candidates(COMBINATIONS['all'], p, full=False)]
    if np.all(Y >= 0):
        m, a = tune(X, Y, cat, p, seed)
        audit['all_log1p'] = a
        records.append(regression_records(data, te, m.predict(Xt), 'all_log1p', scale, task))
    return ({'regression': pd.concat(records, ignore_index=True)}, audit)

def _core_classification(data, task, p, classes):
    tr = np.array(task['train'])
    te = np.array(task['test'])
    seed = task['seed']
    y = labels(data)
    X = subset(data['X'], tr)
    Xt = subset(data['X'], te)
    records = []
    audit = {}
    winners = {}
    for key, modes in COMBINATIONS.items():
        print('  Grade subset: ' + key, flush=True)
        cat = clf.catalog(modes, p)
        if key != 'all':
            cat = [s for s in cat if s['family'] in ('logistic', 'svm', 'plsda')]
        m, a = clf.tune(X, y[tr], classes, cat, p, seed)
        audit[key] = a
        records.append(classification_records(data, te, m.predict_proba(Xt), key + '__best', classes, task, y))
        selected_metrics = a.get('selected_metrics')
        score = selected_metrics['macro_f1'] if selected_metrics is not None else 0
        winners[key] = (score, m)
        if key == 'all':
            shared = [s for s in cat if s['family'] in ('logistic', 'svm', 'plsda')]
            cm, ca = clf.tune(X, y[tr], classes, shared, p, seed)
            records.append(classification_records(data, te, cm.predict_proba(Xt), 'all__common_best', classes, task, y))
            audit['all__common_best'] = ca
    for name, keys in [('best_single', ['nir', 'raman', 'uv']), ('best_double', ['nir_raman', 'nir_uv', 'raman_uv'])]:
        key = sorted(keys, key=lambda x: (-winners[x][0], x))[0]
        audit[name] = {'selected_subset': key, 'training_macro_F1': winners[key][0]}
        records.append(classification_records(data, te, winners[key][1].predict_proba(Xt), name, classes, task, y))
    measured = {'chemistry': data['Y'], 'ethanol': data['ethanol'][:, None]}
    specs = [('measured_chemistry', ['chemistry']), ('measured_abv', ['ethanol']), ('chemistry_plus_abv', ['chemistry', 'ethanol'])]
    for name, modes in specs:
        cat = clf.catalog(modes, p)
        for spec in cat:
            spec['pre'] = 'raw'
        m, a = clf.tune(subset(measured, tr), y[tr], classes, cat, p, seed)
        audit[name] = a
        records.append(classification_records(data, te, m.predict_proba(subset(measured, te)), name, classes, task, y))
    spec = {'family': 'majority', 'modes': ['uv'], 'pre': 'raw', 'pca_components': 2}
    m = clf.FittedClassifier(spec, X, y[tr], classes, None, seed)
    records.append(classification_records(data, te, m.predict_proba(Xt), 'majority', classes, task, y))
    return ({'classification': pd.concat(records, ignore_index=True)}, audit)

def _advanced_regression(data, task, p, cfg):
    tr = np.array(task['train'])
    te = np.array(task['test'])
    seed = task['seed']
    X = subset(data['X'], tr)
    Xt = subset(data['X'], te)
    Y = data['Y'][tr]
    scale = iqr(Y)
    records = []
    audit = {}
    other = []
    availability = []
    for rep in ['direct', 'pca_matched', 'paired_pls', 'paired_shuffled', 'paired_cca']:
        print('  Representation: ' + rep, flush=True)
        direct = rep in ('direct', 'pca_matched')
        modes = ['nir'] if direct else COMBINATIONS['all']
        cat = candidates(modes, p, full=False, representation='direct' if direct else rep)
        if direct:
            cat = [s for s in cat if s['family'] in ('ridge', 'kernel_ridge')]
        if rep == 'pca_matched':
            for spec in cat:
                spec['params']['pca_components'] = spec['params']['n_components']
        m, a = tune(X, Y, cat, p, seed)
        audit['representation_' + rep] = a
        pred = m.predict({'nir': Xt['nir']})
        records.append(regression_records(data, te, pred, 'representation_' + rep, scale, task))
    if len(data['target_names']) == 11:
        desc, names = chemistry_descriptors(data['Y'], data['target_names'])
        if not np.isfinite(desc).all():
            raise ValueError('Nonpositive reference prevents prespecified log balances; do not impute')
        dscale = iqr(desc[tr])
        m, a = tune(X, desc[tr], candidates(COMBINATIONS['all'], p, full=False), p, seed)
        other.append(regression_records(data, te, m.predict(Xt), 'direct_descriptors', dscale, task, names, desc[te]))
        audit['direct_descriptors'] = a
        m, a = tune(X, Y, candidates(COMBINATIONS['all'], p, full=False), p, seed)
        pred = m.predict(Xt)
        d, names = chemistry_descriptors(pred, data['target_names'])
        for j, name in enumerate(names):
            valid = np.isfinite(d[:, j])
            other.append(pd.DataFrame([{'sample_id': str(data['ids'][i]), 'repeat': task['repeat'], 'fold': task['fold'], 'method': 'derive_from_concentrations', 'target': name, 'y_true': float(desc[i, j]), 'y_pred': float(d[k, j]) if valid[k] else np.nan, 'train_iqr': float(dscale[j]), 'valid_prediction': bool(valid[k]), 'fit_n': len(tr)} for k, i in enumerate(te)]))
        audit['derive_from_concentrations'] = a
    mask_specs = [{'family': 'masked_ridge', 'modes': COMBINATIONS['all'], 'pre': 'smooth', 'params': {'alpha': v, 'pca_components': 8}} for v in [0.1, 1, 10, 100][:p['candidates_per_family']]]
    mm, ma = tune(X, Y, mask_specs, p, seed)
    audit['masked_ridge'] = ma
    oof = {}
    test = {}
    scores = {}
    for key, modes in COMBINATIONS.items():
        print('  Missing-view / acquisition models: ' + key, flush=True)
        cat = candidates(modes, p, full=False)
        model, choice = tune(X, Y, cat, p, seed)
        test[key] = model.predict(Xt)
        scores[key] = choice['score']
        audit['subset_' + key] = choice
        records.append(regression_records(data, te, test[key], 'specialist_' + key, scale, task))
        ridge_specs = [{'family': 'ridge', 'modes': modes, 'pre': 'smooth', 'params': {'alpha': v, 'pca_components': 8}} for v in [0.1, 1, 10, 100][:p['candidates_per_family']]]
        rm, ra = tune(X, Y, ridge_specs, p, seed)
        audit['matched_ridge_' + key] = ra
        records.append(regression_records(data, te, rm.predict(Xt), 'matched_ridge_' + key, scale, task))
        maskpred = mm.predict({k: Xt[k] for k in modes})
        records.append(regression_records(data, te, maskpred, 'masked_' + key, scale, task))
        pp, aa = strict_crossfit(X, Y, cat, p, seed)
        oof[key] = pp
        audit['crossfit_' + key] = aa
    for missing_key, available in COMBINATIONS.items():
        allowed = [k for k, m in COMBINATIONS.items() if set(m).issubset(available)]
        key = min(allowed, key=lambda k: (round(scores[k], 12), len(COMBINATIONS[k]), k))
        records.append(regression_records(data, te, test[key], 'available_route_' + missing_key, scale, task))
        availability.append({'availability': missing_key, 'selected': key, 'selection': 'inner CV only'})
    audit['availability_routing'] = availability
    singles = ['nir', 'raman', 'uv']
    mat = np.stack([oof[k] / scale for k in singles], axis=2)
    yscaled = Y / scale
    loss = lambda w: float(np.mean((np.tensordot(mat, w, axes=(2, 0)) - yscaled) ** 2))
    fit = minimize(loss, np.ones(3) / 3, bounds=[(0, 1)] * 3, constraints={'type': 'eq', 'fun': lambda w: float(w.sum() - 1)}, method='SLSQP')
    w = fit.x if fit.success else np.ones(3) / 3
    fused = sum((w[j] * test[key] for j, key in enumerate(singles)))
    records.append(regression_records(data, te, fused, 'oof_late_fusion', scale, task))
    audit['oof_late_fusion'] = {'weights': dict(zip(singles, w.tolist())), 'optimizer_success': bool(fit.success)}
    route_rows = []
    randomization_rows = []
    for scenario, costs in cfg['cost_scenarios'].items():
        total = sum(costs.values())
        for penalty in cfg['cost_penalties']:
            paths = []
            pred = np.zeros((len(te), Y.shape[1]))
            spent = np.zeros(len(te))
            states = ['uv'] * len(te)
            method = 'acquisition_%s_lambda_%g' % (scenario, penalty)
            policy = {}
            for state, modes in COMBINATIONS.items():
                extensions = [k for k, m in COMBINATIONS.items() if set(modes).issubset(m) and len(m) == len(modes) + 1]
                if not extensions:
                    continue
                feature = np.hstack([X[k] for k in modes])
                ft = np.hstack([Xt[k] for k in modes])
                trans = make_pipeline(StandardScaler(), PCA(n_components=min(8, len(Y) - 1, feature.shape[1]), svd_solver='full'))
                z = trans.fit_transform(feature)
                zt = trans.transform(ft)
                current = np.mean(((oof[state] - Y) / scale) ** 2, axis=1)
                for nxt in extensions:
                    target = current - np.mean(((oof[nxt] - Y) / scale) ** 2, axis=1)
                    risk = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(z, target)
                    add = sum((costs[k] for k in set(COMBINATIONS[nxt]) - set(modes))) / total
                    policy[state, nxt] = risk.predict(zt) - penalty * add
            for row in range(len(te)):
                state = cfg['acquisition_start']
                path = [state]
                while True:
                    options = [(values[row], nxt) for (prev, nxt), values in policy.items() if prev == state]
                    if not options:
                        break
                    score, nxt = max(options, key=lambda v: (v[0], v[1]))
                    if score <= 0:
                        break
                    state = nxt
                    path.append(state)
                pred[row] = test[state][row]
                spent[row] = sum((costs[k] for k in COMBINATIONS[state]))
                states[row] = state
                route_rows.append({'sample_id': str(data['ids'][te[row]]), 'repeat': task['repeat'], 'fold': task['fold'], 'method': method, 'policy': 'learned', 'scenario': scenario, 'penalty': penalty, 'final_subset': state, 'path': ' > '.join(path), 'relative_cost': spent[row] / total, 'cost_is_illustrative': True})
            records.append(regression_records(data, te, pred, method, scale, task))
            random_states = np.random.RandomState(seed + int(10000 * penalty) + len(scenario)).permutation(states)
            random_pred = np.vstack([test[state][row] for row, state in enumerate(random_states)])
            random_method = 'acquisition_random_%s_lambda_%g' % (scenario, penalty)
            records.append(regression_records(data, te, random_pred, random_method, scale, task))
            for row, state in enumerate(random_states):
                route_rows.append({'sample_id': str(data['ids'][te[row]]), 'repeat': task['repeat'], 'fold': task['fold'], 'method': random_method, 'policy': 'matched_random', 'scenario': scenario, 'penalty': penalty, 'final_subset': state, 'path': 'random assignment, same final-subset counts', 'relative_cost': sum((costs[k] for k in COMBINATIONS[state])) / total, 'cost_is_illustrative': True})
            repetitions = int(cfg.get('matched_random_repetitions', 30))
            for b in range(repetitions):
                allocation = np.random.RandomState(seed + 830003 + 1009 * b + int(penalty * 10000) + len(scenario)).permutation(states)
                rp = np.vstack([test[state][row] for row, state in enumerate(allocation)])
                for j, target in enumerate(data['target_names']):
                    randomization_rows.append({'repeat': task['repeat'], 'fold': task['fold'], 'method': method, 'scenario': scenario, 'penalty': penalty, 'randomization': b, 'target': target, 'n': len(te), 'normalized_sse': float(np.sum(((rp[:, j] - data['Y'][te, j]) / scale[j]) ** 2)), 'learned_normalized_sse': float(np.sum(((pred[:, j] - data['Y'][te, j]) / scale[j]) ** 2)), 'relative_cost': float(np.mean(spent / total)), 'strict_same_final_subset_counts': bool(sorted(allocation) == sorted(states)), 'conditional_on_fitted_models': True})
    frames = {'regression': pd.concat(records, ignore_index=True), 'acquisition': pd.DataFrame(route_rows), 'acquisition_randomization': pd.DataFrame(randomization_rows)}
    if other:
        frames['descriptors'] = pd.concat(other, ignore_index=True)
    return (frames, audit)

def _advanced_classification(data, task, p, classes):
    tr = np.array(task['train'])
    te = np.array(task['test'])
    seed = task['seed']
    y = labels(data)
    X = subset(data['X'], tr)
    Xt = subset(data['X'], te)
    Y = data['Y'][tr]
    records = []
    audit = {}
    for key in ['nir', 'uv', 'raman_uv', 'all']:
        print('  Nested chemistry-to-grade bridge: ' + key, flush=True)
        cat = candidates(COMBINATIONS[key], p, full=False)
        cc = clf.catalog(['predicted_chemistry'], p)
        for spec in cc:
            spec['pre'] = 'raw'
        k = min(p['inner_folds'], int(np.unique(y[tr], return_counts=True)[1].min()))
        bridge_folds = []
        inneraudit = []
        for f, (aidx, bidx) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed).split(Y, y[tr])):
            innerx = subset(X, aidx)
            ip, ia = strict_crossfit(innerx, Y[aidx], cat, p, seed + 100 + f)
            im, ib = tune(innerx, Y[aidx], cat, p, seed + 100 + f)
            iv = im.predict(subset(X, bidx))
            bridge_folds.append((aidx, bidx, ip, iv))
            inneraudit.append({'crossfit': ia, 'final': ib})
        ranking = []
        cfail = []
        for spec in cc:
            prob = np.zeros((len(tr), len(classes)))
            failure = []
            for f, (aidx, bidx, ip, iv) in enumerate(bridge_folds):
                try:
                    cm = clf.FittedClassifier(spec, {'predicted_chemistry': ip}, y[tr][aidx], classes, None, seed + f)
                    prob[bidx] = cm.predict_proba({'predicted_chemistry': iv})
                except Exception as e:
                    failure.append(repr(e))
            if failure:
                cfail.append({'spec': spec, 'errors': failure})
                continue
            score = clf.metrics(y[tr], prob, classes)
            ranking.append((-score['macro_f1'], -score['balanced_accuracy'], digest(spec), spec, score))
        if not ranking:
            raise RuntimeError('Every nested chemistry bridge classifier failed')
        selected = sorted(ranking, key=lambda r: r[:3])[0]
        oof, a = strict_crossfit(X, Y, cat, p, seed)
        full, b = tune(X, Y, cat, p, seed)
        pred = full.predict(Xt)
        cx = {'predicted_chemistry': oof}
        ct = {'predicted_chemistry': pred}
        cm = clf.FittedClassifier(selected[3], cx, y[tr], classes, None, seed)
        c = {'selected': selected[3], 'inner_macro_f1': selected[4]['macro_f1'], 'entire_bridge_tuned_in_inner_folds': True, 'failures': cfail, 'inner_bridge_audits': inneraudit}
        records.append(classification_records(data, te, cm.predict_proba(ct), 'chemistry_bridge_' + key, classes, task, y))
        audit['bridge_' + key] = {'chemistry_crossfit': a, 'chemistry_final': b, 'classifier': c, 'test_input': 'spectra only', 'crossfit_training_chemistry': True}
        dm, d = clf.tune(X, y[tr], classes, clf.catalog(COMBINATIONS[key], p), p, seed)
        records.append(classification_records(data, te, dm.predict_proba(Xt), 'direct_grade_' + key, classes, task, y))
        audit['direct_' + key] = d
    return ({'classification': pd.concat(records, ignore_index=True)}, audit)

def finite_quantile(scores, alpha):
    a = np.sort(np.asarray(scores, float), axis=0)
    n = len(a)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    return np.full(a.shape[1:] or (), np.inf) if rank > n else a[rank - 1]

def _interval(data, task, p, cfg, classes):
    tr = np.array(task['train'])
    cal = np.array(task['calibration'])
    te = np.array(task['test'])
    seed = task['seed']
    if not len(cal):
        raise ValueError('No disjoint calibration bottles')
    X = subset(data['X'], tr)
    Xc = subset(data['X'], cal)
    Xt = subset(data['X'], te)
    alpha = 1 - cfg['nominal_coverage']
    audit = {}
    records = []
    calibration_records = []
    if task['kind'] == 'regression':
        Y = data['Y']
        scale = iqr(Y[tr])
        for key in ['nir', 'raman_uv', 'all']:
            m, a = tune(X, Y[tr], candidates(COMBINATIONS[key], p, full=False), p, seed)
            cp = m.predict(Xc)
            tp = m.predict(Xt)
            r = np.abs(cp - Y[cal]) / scale
            qm = finite_quantile(r, alpha)
            qj = float(finite_quantile(np.max(r, axis=1), alpha))
            cf = regression_records(data, cal, cp, key, scale, task)
            cf['normalized_abs_residual'] = r.reshape(-1)
            cf['marginal_quantile'] = np.tile(qm, len(cal))
            cf['joint_quantile'] = qj
            calibration_records.append(cf)
            for kind, q in [('marginal', qm), ('joint', np.repeat(qj, Y.shape[1]))]:
                lo = tp - q * scale
                hi = tp + q * scale
                frame = regression_records(data, te, tp, key + '__' + kind, scale, task)
                frame['lower'] = lo.reshape(-1)
                frame['upper'] = hi.reshape(-1)
                frame['calibration_n'] = len(cal)
                frame['nominal_coverage'] = cfg['nominal_coverage']
                frame['above_train_q75'] = (Y[te] > np.percentile(Y[tr], 75, axis=0)).reshape(-1)
                abv_out = (data['ethanol'][te] < np.percentile(data['ethanol'][tr], 10)) | (data['ethanol'][te] > np.percentile(data['ethanol'][tr], 90))
                frame['ethanol_tail'] = np.repeat(abv_out, Y.shape[1])
                records.append(frame)
            audit[key] = {'choice': a, 'marginal_quantile': qm, 'joint_quantile': qj, 'calibration_labels_used_only_for_quantile': True}
        return ({'interval_regression': pd.concat(records, ignore_index=True), 'calibration_regression': pd.concat(calibration_records, ignore_index=True)}, audit)
    y = labels(data)
    for key in ['nir', 'raman_uv', 'all']:
        m, a = clf.tune(X, y[tr], classes, clf.catalog(COMBINATIONS[key], p), p, seed)
        cp = m.predict_proba(Xc)
        tp = m.predict_proba(Xt)
        trueindex = np.array([classes.index(c) for c in y[cal]])
        q = float(finite_quantile(1 - cp[np.arange(len(cal)), trueindex], alpha))
        cf = classification_records(data, cal, cp, key, classes, task, y)
        cf['nonconformity'] = 1 - cp[np.arange(len(cal)), trueindex]
        cf['quantile'] = q
        calibration_records.append(cf)
        include = 1 - tp <= q
        frame = classification_records(data, te, tp, key, classes, task, y)
        frame['prediction_set'] = [' | '.join(np.asarray(classes)[v]) for v in include]
        frame['set_size'] = include.sum(axis=1)
        frame['covered'] = [bool(include[i, classes.index(c)]) for i, c in enumerate(y[te])]
        frame['calibration_n'] = len(cal)
        frame['nominal_coverage'] = cfg['nominal_coverage']
        records.append(frame)
        audit[key] = {'choice': a, 'quantile': q, 'score': '1 - uncalibrated class score', 'marginal_only_no_class_conditional_guarantee': True}
    return ({'interval_classification': pd.concat(records, ignore_index=True), 'calibration_classification': pd.concat(calibration_records, ignore_index=True)}, audit)

def _learning(data, task, p):
    tr = np.array(task['train'])
    te = np.array(task['test'])
    seed = task['seed']
    rng = np.random.RandomState(seed)
    pool = data['X']['nir'][tr]
    sc = StandardScaler().fit(pool)
    pc = PCA(n_components=min(8, len(tr) - 1, pool.shape[1]), svd_solver='full').fit(sc.transform(pool))
    z = pc.transform(sc.transform(pool))
    center = z.mean(axis=0)
    first = int(np.argmax(np.sum((z - center) ** 2, axis=1)))
    chosen = [first]
    dist = np.sum((z - z[first]) ** 2, axis=1)
    while len(chosen) < len(tr):
        dist[chosen] = -1
        idx = int(np.argmax(dist))
        chosen.append(idx)
        dist = np.minimum(dist, np.sum((z - z[idx]) ** 2, axis=1))
    random = rng.permutation(len(tr))
    records = []
    audit = {}
    for fraction in [0.4, 0.6, 0.8, 1.0]:
        count = max(16, int(round(len(tr) * fraction)))
        count = min(count, len(tr))
        for name, order in [('spectral_kcenter', np.array(chosen)), ('random', random)]:
            selected = np.sort(tr[order[:count]])
            model, a = tune(subset(data['X'], selected), data['Y'][selected], candidates(['nir'], p, full=False), p, seed)
            t = dict(task, train=selected.tolist())
            key = '%s_fraction_%03d' % (name, round(100 * fraction))
            frame = regression_records(data, te, model.predict(subset(data['X'], te)), key, iqr(data['Y'][tr]), t)
            frame['training_fraction'] = fraction
            records.append(frame)
            audit[key] = {'selected_ids': data['ids'][selected].tolist(), 'model': a, 'training_fraction': fraction, 'actual_fit_n': count, 'unlabelled_pool': data['ids'][tr].tolist(), 'no_test_pool': True, 'reference_assay_saving_is_retrospective_simulation': True}
    return ({'regression': pd.concat(records, ignore_index=True)}, audit)

def _permutation(data, task, p, classes):
    tr = np.array(task['train'])
    te = np.array(task['test'])
    seed = task['seed']
    y = labels(data)
    records = []
    audit = {}
    for b in range(-1, p['permutations']):
        yp = y if b == -1 else np.random.RandomState(790001 + b).permutation(y)
        cc = [s for s in clf.catalog(COMBINATIONS['all'], p) if s['family'] in ('logistic', 'svm', 'plsda')]
        m, a = clf.tune(subset(data['X'], tr), yp[tr], classes, cc, p, seed)
        name = 'observed_restricted' if b == -1 else 'permutation_%03d' % b
        records.append(classification_records(data, te, m.predict_proba(subset(data['X'], te)), name, classes, task, yp))
        audit[name] = a
    return ({'classification': pd.concat(records, ignore_index=True)}, audit)

def run(root, data, cfg, profile, mode, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plan = frozen_plan(root, data, cfg, profile, mode, out)
    p = plan['parameters']
    classes = plan['classes']
    for count, task in enumerate(plan['tasks'], 1):
        folder = out / 'tasks' / task['id']
        marker = folder / 'complete.json'
        if marker.exists():
            old = read_json(marker)
            if old.get('plan_hash') != plan['plan_hash']:
                raise ValueError('Checkpoint plan mismatch')
            from .common import sha
            if all(((folder / name).exists() and sha(folder / name) == h for name, h in old['files'].items())):
                print('Resume: %s (%d/%d)' % (task['id'], count, len(plan['tasks'])), flush=True)
                continue
            raise ValueError('Checkpoint damaged; preserve folder and use separate output for recovery: ' + str(folder))
        folder.mkdir(parents=True, exist_ok=True)
        print('Run %s %s (%d/%d)' % (mode, task['id'], count, len(plan['tasks'])), flush=True)
        try:
            if mode == 'core':
                frames, audit = _core_regression(data, task, p) if task['kind'] == 'regression' else _core_classification(data, task, p, classes)
            elif mode == 'advanced':
                frames, audit = _advanced_regression(data, task, p, cfg) if task['kind'] == 'regression' else _advanced_classification(data, task, p, classes)
            elif mode == 'interval':
                frames, audit = _interval(data, task, p, cfg, classes)
            elif mode == 'learning':
                frames, audit = _learning(data, task, p)
            elif mode == 'permutation':
                frames, audit = _permutation(data, task, p, classes)
            elif mode == 'optimized':
                if task['kind'] == 'regression':
                    from .optimized_regression import run_task
                else:
                    from .optimized_classification import run_task
                frames, audit = run_task(data, task, p, dict(cfg, classes=classes))
            elif mode == 'uncertainty':
                from .optimized_uncertainty import run_task
                frames, audit = run_task(data, task, p, dict(cfg, classes=classes))
            else:
                raise ValueError('Unknown experiment ' + mode)
            for name, frame in frames.items():
                save_csv(folder / (name + '.csv'), frame)
            save_json(folder / 'audit.json', audit)
            from .common import sha
            files = {f.name: sha(f) for f in folder.iterdir() if f.suffix in ('.csv', '.json') and f.name not in ('complete.json', 'error.json')}
            save_json(marker, {'status': 'PASS', 'plan_hash': plan['plan_hash'], 'files': files, 'task': task['id']})
        except Exception as e:
            save_json(folder / 'error.json', {'error': repr(e), 'traceback': traceback.format_exc(), 'plan_hash': plan['plan_hash']})
            raise
    from .reporting import summarize, verify
    summarize(out, data)
    receipt = verify(out, data, root)
    if not receipt['complete']:
        raise RuntimeError('Completion validation failed: ' + str(receipt))
    return receipt
