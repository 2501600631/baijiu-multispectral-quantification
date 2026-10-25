import numpy as np
import pandas as pd
from .common import subset, iqr
from .final_ensemble import fit_final_bundle, strict_crossfit_final
from .optimized_uncertainty import ResidualScale, finite_quantile, _audit_quantile
POINT_METHODS = ('v10_pls2_reference', 'v10_v9_unanchored_reference', 'v10_strategy_router')
SCALE_TYPES = ('fixed', 'adaptive')
COVERAGE_TYPES = ('marginal', 'joint')
RETAINED_FRACTIONS = (0.5, 0.7, 0.9, 1.0)

def _partitions(data, task):
    parts = []
    for name in ('train', 'calibration', 'test'):
        raw = np.asarray(task[name])
        if raw.ndim != 1 or not len(raw):
            raise ValueError('Fit, calibration and test partitions must be nonempty')
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError('Partition bottle indices must be integers')
        if len(np.unique(raw)) != len(raw) or np.any(raw < 0) or np.any(raw >= len(data['ids'])):
            raise ValueError('Partition indices are duplicated or outside the dataset')
        parts.append(raw.astype(int))
    if any((set(parts[i]) & set(parts[j]) for i in range(3) for j in range(i))):
        raise ValueError('Fit, calibration and test bottle sets must be disjoint')
    return parts

def _validate_oof_audit(audits, n):
    seen = []
    for audit in audits:
        tr = np.asarray(audit['train'], int)
        va = np.asarray(audit['validation'], int)
        if len(np.unique(tr)) != len(tr) or len(np.unique(va)) != len(va):
            raise ValueError('Repeated crossfit indices')
        if set(tr) & set(va) or set(tr) | set(va) != set(range(n)):
            raise ValueError('Complete-pipeline crossfit must exclude validation bottles from all fitting')
        seen.extend(va.tolist())
    if sorted(seen) != list(range(n)):
        raise ValueError('Complete-pipeline OOF predictions must cover each fit bottle exactly once')

def run_task(data, task, profile, cfg):
    from .engine import regression_records
    if task.get('kind', 'regression') != 'regression':
        raise ValueError('Final coupled interval task supports concentration regression only')
    tr, cal, te = _partitions(data, task)
    X, Xc, Xt = (subset(data['X'], rows) for rows in (tr, cal, te))
    y = np.asarray(data['Y'], float)
    scale = iqr(y[tr])
    alpha = 1.0 - float(cfg['nominal_coverage'])
    if not 0 < alpha < 1:
        raise ValueError('Nominal coverage must lie strictly between zero and one')
    options = dict(cfg.get('ensemble_options', {'ensemble_penalty': 0.1}))
    crossfit_options = dict(options)
    crossfit_options.setdefault('crossfit_folds', int(profile.get('scale_crossfit_folds', profile['inner_folds'])))
    model = fit_final_bundle(X, y[tr], profile, task['seed'], options=options)
    cp = model.predict_methods(Xc, methods=list(POINT_METHODS))
    tp = model.predict_methods(Xt, methods=list(POINT_METHODS))
    oof, crossfit_audits = strict_crossfit_final(X, y[tr], profile, task['seed'] + 1701, methods=list(POINT_METHODS), options=crossfit_options)
    _validate_oof_audit(crossfit_audits, len(tr))
    crossfit_receipt = []
    for entry in crossfit_audits:
        entry = dict(entry)
        entry['fit_bottle_ids'] = np.asarray(data['ids'])[tr[np.asarray(entry['train'], int)]].tolist()
        entry['held_out_bottle_ids'] = np.asarray(data['ids'])[tr[np.asarray(entry['validation'], int)]].tolist()
        crossfit_receipt.append(entry)
    records, calibration, points, risk_rules, risk_predictions = ([], [], [], [], [])
    audit = {'point_model_selection': model.audit, 'fit_partition_ids': np.asarray(data['ids'])[tr].tolist(), 'calibration_partition_ids': np.asarray(data['ids'])[cal].tolist(), 'test_partition_ids': np.asarray(data['ids'])[te].tolist(), 'same_final_point_model_used_for_centers_and_residuals': True, 'full_procedure_oof_includes_candidate_selection_and_weights': True, 'strict_full_pipeline_crossfit': crossfit_receipt, 'calibration_labels_used_only_for_quantiles': True, 'all_point_and_scale_models_use_fit_partition_only': True, 'retained_fractions_predeclared': list(RETAINED_FRACTIONS), 'selective_thresholds_use_fit_partition_only': True, 'conditional_or_selected_subset_coverage_not_guaranteed': True, 'methods': {}}
    ethanol = np.asarray(data['ethanol'], float)
    tails = (ethanol[te] < np.percentile(ethanol[tr], 10)) | (ethanol[te] > np.percentile(ethanol[tr], 90))
    for base in POINT_METHODS:
        for name, predictions, size in [('fit OOF', oof[base], len(tr)), ('calibration', cp[base], len(cal)), ('test', tp[base], len(te))]:
            if np.asarray(predictions).shape != (size, y.shape[1]) or not np.isfinite(predictions).all():
                raise ValueError('Invalid ' + base + ' ' + name + ' point predictions')
        pf = regression_records(data, te, tp[base], base, scale, task)
        pf['base_model'] = base
        points.append(pf)
        residual_model = ResidualScale(X, y[tr], oof[base], list(X), scale)
        adaptive_cal = residual_model.predict(Xc, cp[base])
        adaptive_test = residual_model.predict(Xt, tp[base])
        training_uncertainty = residual_model.predict(X, oof[base]) / scale
        for fraction in RETAINED_FRACTIONS:
            threshold = np.full(y.shape[1], np.inf) if fraction == 1.0 else np.quantile(training_uncertainty, fraction, axis=0)
            for j, target in enumerate(data['target_names']):
                risk_rules.append({'repeat': task['repeat'], 'fold': task['fold'], 'method': base + '__adaptive', 'base_model': base, 'target': target, 'retained_fraction': fraction, 'threshold': float(threshold[j]), 'fit_n': len(tr), 'threshold_source': 'fit_only_normalized_predicted_residual_scale', 'calibration_labels_used': False, 'test_labels_used': False})
            rf = pf.copy()
            rf['method'] = base + '__adaptive'
            rf['retained_fraction'] = fraction
            rf['uncertainty_score'] = (adaptive_test / scale).reshape(-1)
            rf['threshold'] = np.tile(threshold, len(te))
            rf['accepted'] = rf.uncertainty_score <= rf.threshold
            rf['abs_error'] = np.abs(rf.y_pred - rf.y_true)
            rf['normalized_abs_error'] = rf.abs_error / rf.train_iqr
            risk_predictions.append(rf)
        for scale_type, cs, ts in [('fixed', np.broadcast_to(scale, cp[base].shape), np.broadcast_to(scale, tp[base].shape)), ('adaptive', adaptive_cal, adaptive_test)]:
            name = base + '__' + scale_type
            scores = np.abs(cp[base] - y[cal]) / cs
            marginal_q = finite_quantile(scores, alpha)
            joint_q = float(finite_quantile(np.max(scores, axis=1), alpha))
            cf = regression_records(data, cal, cp[base], name, scale, task)
            cf['base_model'] = base
            cf['scale_type'] = scale_type
            cf['interval_scale'] = cs.reshape(-1)
            cf['normalized_abs_residual'] = scores.reshape(-1)
            cf['marginal_quantile'] = np.tile(marginal_q, len(cal))
            cf['joint_quantile'] = joint_q
            cf['cal_n'] = cf['calibration_n'] = len(cal)
            cf['nominal_coverage'] = 1 - alpha
            calibration.append(cf)
            for coverage_type, q in [('marginal', marginal_q), ('joint', np.repeat(joint_q, y.shape[1]))]:
                f = regression_records(data, te, tp[base], name + '__' + coverage_type, scale, task)
                f['base_model'] = base
                f['scale_type'] = scale_type
                f['coverage_type'] = coverage_type
                f['interval_scale'] = ts.reshape(-1)
                f['quantile'] = np.tile(q, len(te))
                f['lower'] = (tp[base] - ts * q).reshape(-1)
                f['upper'] = (tp[base] + ts * q).reshape(-1)
                f['cal_n'] = f['calibration_n'] = len(cal)
                f['nominal_coverage'] = 1 - alpha
                f['above_train_q75'] = (y[te] > np.percentile(y[tr], 75, axis=0)).reshape(-1)
                f['ethanol_tail'] = np.repeat(tails, y.shape[1])
                records.append(f)
            audit['methods'][name] = {'marginal_quantile': _audit_quantile(marginal_q), 'joint_quantile': _audit_quantile(joint_q), 'scale_model': 'constant fit IQR' if scale_type == 'fixed' else 'fit-only PCA/Ridge log-residual scale; complete-procedure OOF residuals', 'calibration_n': len(cal), 'marginal_exchangeability_coverage_only': True}
    return ({'regression': pd.concat(points, ignore_index=True), 'interval_regression': pd.concat(records, ignore_index=True), 'calibration_regression': pd.concat(calibration, ignore_index=True), 'risk_rules': pd.DataFrame(risk_rules), 'risk_predictions': pd.concat(risk_predictions, ignore_index=True)}, audit)

def _same(a, b):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=1e-10, atol=1e-12, equal_nan=False)

def _verify_one(frames, alpha):
    errors = []
    cal, test = (frames['calibration_regression'], frames['interval_regression'])
    if cal.empty or test.empty:
        return ['Missing final model calibration or interval rows']
    if cal.duplicated(['sample_id', 'method', 'target']).any():
        errors.append('Duplicate final calibration keys')
    if test.duplicated(['sample_id', 'method', 'target']).any():
        errors.append('Duplicate final interval keys')
    expected_cal = {base + '__' + s for base in POINT_METHODS for s in SCALE_TYPES}
    if set(cal.method) != expected_cal:
        errors.append('Final model calibration method inventory mismatch')
    if set(test.method) != {m + '__' + c for m in expected_cal for c in COVERAGE_TYPES}:
        errors.append('Final model interval method inventory mismatch')
    cal_ids, test_ids, targets = (set(cal.sample_id), set(test.sample_id), set(cal.target))
    if cal_ids & test_ids:
        errors.append('Calibration and test bottle overlap')
    if set(test.target) != targets:
        errors.append('Calibration/test target inventory mismatch')
    for frame in (cal, test):
        for _, g in frame.groupby(['base_model', 'sample_id', 'target']):
            if not _same(g.y_pred, g.y_pred.iloc[0]) or not _same(g.train_iqr, g.train_iqr.iloc[0]):
                errors.append('Different point centers or IQR within a coupled comparison')
                break
    for method, g in cal.groupby('method'):
        if method not in expected_cal:
            continue
        expected_base = method.rsplit('__', 1)[0]
        expected_scale = method.rsplit('__', 1)[1]
        if set(g.base_model) != {expected_base} or set(g.scale_type) != {expected_scale}:
            errors.append('Calibration method metadata mismatch: ' + method)
        if set(g.target) != targets or any((set(a.sample_id) != cal_ids for _, a in g.groupby('target'))):
            errors.append('Incomplete final calibration grid: ' + method)
            continue
        cs = g.interval_scale.to_numpy(float)
        if not np.isfinite(cs).all() or np.any(cs <= 0):
            errors.append('Invalid final calibration scale: ' + method)
            continue
        scores = np.abs(g.y_true.to_numpy(float) - g.y_pred.to_numpy(float)) / cs
        if not _same(scores, g.normalized_abs_residual):
            errors.append('Final calibration residual equation mismatch: ' + method)
        qj = float(finite_quantile(g.assign(score=scores).groupby('sample_id').score.max().to_numpy(), alpha))
        if not _same(g.joint_quantile, qj):
            errors.append('Final joint quantile mismatch: ' + method)
        for target, a in g.groupby('target'):
            qm = float(finite_quantile(np.abs(a.y_true.to_numpy(float) - a.y_pred.to_numpy(float)) / a.interval_scale, alpha))
            if not _same(a.marginal_quantile, qm):
                errors.append('Final marginal quantile mismatch: ' + method + '/' + target)
            for coverage, q in [('marginal', qm), ('joint', qj)]:
                z = test[(test.method == method + '__' + coverage) & (test.target == target)]
                if z.empty or set(z.sample_id) != test_ids:
                    errors.append('Incomplete final interval grid: ' + method + '/' + target)
                    continue
                ts = z.interval_scale.to_numpy(float)
                if not np.isfinite(ts).all() or np.any(ts <= 0):
                    errors.append('Invalid final test scale: ' + method)
                    continue
                if set(z.base_model) != {expected_base} or set(z.scale_type) != {expected_scale} or set(z.coverage_type) != {coverage}:
                    errors.append('Interval method metadata mismatch: ' + method)
                if not (_same(z['quantile'], q) and _same(z.lower, z.y_pred - q * ts) and _same(z.upper, z.y_pred + q * ts)):
                    errors.append('Final interval quantile/bound equation mismatch: ' + method)
                if not _same(z.cal_n, len(cal_ids)) or not _same(z.nominal_coverage, 1 - alpha):
                    errors.append('Final calibration count/coverage mismatch: ' + method)
                if method.endswith('__fixed') and (not _same(ts, z.train_iqr)):
                    errors.append('Fixed interval scale differs from fit IQR: ' + method)
    if 'regression' in frames:
        points = frames['regression']
        if set(points.method) != set(POINT_METHODS) or points.duplicated(['sample_id', 'method', 'target']).any():
            errors.append('Invalid final point model inventory/duplicates')
        for base, g in points.groupby('method'):
            for method, z in test[test.base_model == base].groupby('method'):
                merged = g.merge(z, on=['sample_id', 'target'], suffixes=('_point', '_interval'), how='outer')
                if merged.y_pred_point.isna().any() or merged.y_pred_interval.isna().any() or (not _same(merged.y_pred_point, merged.y_pred_interval)):
                    errors.append('Final point/interval center mismatch: ' + method)
    if 'risk_rules' in frames or 'risk_predictions' in frames:
        rules, pred = (frames.get('risk_rules', pd.DataFrame()), frames.get('risk_predictions', pd.DataFrame()))
        if rules.empty or pred.empty:
            errors.append('Missing selective risk rules or predictions')
        else:
            keys = ['method', 'target', 'retained_fraction']
            expected_risk = {base + '__adaptive' for base in POINT_METHODS}
            if set(rules.method) != expected_risk or set(pred.method) != expected_risk or set(rules.target) != targets or (set(pred.target) != targets) or (set(rules.retained_fraction) != set(RETAINED_FRACTIONS)) or (set(pred.retained_fraction) != set(RETAINED_FRACTIONS)):
                errors.append('Selective risk inventory mismatch')
            if rules.duplicated(keys).any() or pred.duplicated(keys + ['sample_id']).any():
                errors.append('Duplicate selective risk keys')
            if len(rules) != len(expected_risk) * len(targets) * len(RETAINED_FRACTIONS) or any((set(g.sample_id) != test_ids for _, g in pred.groupby(keys))):
                errors.append('Incomplete selective risk rule/prediction grid')
            matched = pred.merge(rules[keys + ['threshold']], on=keys, suffixes=('', '_rule'), how='left')
            if matched.threshold_rule.isna().any() or not _same(matched.threshold, matched.threshold_rule):
                errors.append('Selective risk thresholds disagree with saved fit-only rules')
            accepted = matched.uncertainty_score.to_numpy(float) <= matched.threshold.to_numpy(float)
            stored = np.asarray([str(v).lower() == 'true' for v in matched.accepted])
            if not np.array_equal(accepted, stored):
                errors.append('Selective acceptance equation mismatch')
            if not _same(matched.abs_error, np.abs(matched.y_pred - matched.y_true)) or not _same(matched.normalized_abs_error, matched.abs_error / matched.train_iqr):
                errors.append('Selective error equation mismatch')
            if not np.isfinite(pred.uncertainty_score.to_numpy(float)).all() or np.any(pred.uncertainty_score.to_numpy(float) <= 0):
                errors.append('Invalid selective risk uncertainty score')
            full = pred[pred.retained_fraction == 1]
            if not np.isposinf(full.threshold).all() or not np.all(full.accepted.astype(str).str.lower() == 'true'):
                errors.append('Full retention rule must accept all test bottles')
            for method, z in pred.groupby('method'):
                base = method.rsplit('__', 1)[0]
                coupled = test[(test.base_model == base) & (test.scale_type == 'adaptive') & (test.coverage_type == 'marginal')]
                merged = z.merge(coupled[['sample_id', 'target', 'y_pred', 'interval_scale', 'train_iqr']], on=['sample_id', 'target'], how='outer', suffixes=('_risk', '_interval'))
                expected_uncertainty = merged.interval_scale / merged.train_iqr_interval
                if not _same(merged.uncertainty_score, expected_uncertainty) or not _same(merged.y_pred_risk, merged.y_pred_interval):
                    errors.append('Selective uncertainty/point model differs from coupled interval system: ' + method)
    return errors

def verify_frames(frames, alpha):
    required = ('calibration_regression', 'interval_regression')
    if any((k not in frames for k in required)):
        return ['Missing final interval/calibration frames']
    errors = []
    try:
        keys = ['repeat', 'fold']
        groups = frames['calibration_regression'][keys].drop_duplicates()
        for repeat, fold in groups.itertuples(index=False, name=None):
            local = {name: a[(a['repeat'] == repeat) & (a['fold'] == fold)].copy() for name, a in frames.items() if isinstance(a, pd.DataFrame) and all((k in a for k in keys))}
            errors.extend(['r%s/f%s: %s' % (repeat, fold, e) for e in _verify_one(local, alpha)])
        if set(map(tuple, frames['interval_regression'][keys].drop_duplicates().to_numpy())) != set(map(tuple, groups.to_numpy())):
            errors.append('Calibration/interval repeat and fold inventory mismatch')
    except Exception as exc:
        errors.append('Final interval verification raised ' + repr(exc))
    return errors
