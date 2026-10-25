import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from .common import COMBINATIONS, subset, iqr
from .design import labels
from .search import candidates, tune, strict_crossfit
from . import classification as clf

def finite_quantile(scores, alpha):
    a = np.asarray(scores, float)
    if not 0 < float(alpha) < 1:
        raise ValueError('alpha must be strictly between zero and one')
    if a.ndim == 0 or len(a) == 0 or (not np.isfinite(a).all()):
        raise ValueError('Calibration scores must be nonempty and finite')
    rank = int(np.ceil((len(a) + 1) * (1 - float(alpha))))
    if rank > len(a):
        return np.full(a.shape[1:] or (), np.inf)
    return np.sort(a, axis=0)[rank - 1]

def class_scores(probabilities, score_type):
    p = np.asarray(probabilities, float)
    if p.ndim != 2 or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('Invalid class scores')
    total = p.sum(axis=1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError('Class-score rows must have positive total')
    p = p / total
    if score_type == 'lac':
        return 1 - p
    if score_type != 'aps':
        raise ValueError('Unknown classification conformal score')
    order = np.argsort(-p, axis=1, kind='mergesort')
    cumulative = np.minimum(1.0, np.cumsum(np.take_along_axis(p, order, axis=1), axis=1))
    cumulative[:, -1] = 1.0
    scores = np.empty_like(p)
    np.put_along_axis(scores, order, cumulative, axis=1)
    return scores

class ResidualScale:

    def __init__(self, X, Y, oof_prediction, modes, training_iqr):
        self.modes = list(modes)
        self.training_iqr = np.asarray(training_iqr, float)
        self.transforms = {}
        parts = []
        for mode in self.modes:
            a = np.asarray(X[mode], float)
            scaler = StandardScaler().fit(a)
            z = scaler.transform(a)
            pc = PCA(n_components=min(6, z.shape[1], len(z) - 1), svd_solver='full').fit(z)
            self.transforms[mode] = (scaler, pc)
            parts.append(pc.transform(z))
        parts.append(np.asarray(oof_prediction) / self.training_iqr)
        self.feature_scaler = StandardScaler().fit(np.column_stack(parts))
        features = self.feature_scaler.transform(np.column_stack(parts))
        residual = np.abs(np.asarray(Y) - np.asarray(oof_prediction)) / self.training_iqr
        self.floor = np.maximum(1e-06, 0.05 * np.median(residual, axis=0))
        log_residual = np.log(np.maximum(residual, self.floor))
        self.center = np.median(log_residual, axis=0)
        self.lower = np.percentile(log_residual, 5, axis=0)
        self.upper = np.percentile(log_residual, 95, axis=0)
        self.regressor = Ridge(alpha=50).fit(features, log_residual)

    def predict(self, X, point_prediction):
        parts = []
        for mode in self.modes:
            scaler, pc = self.transforms[mode]
            parts.append(pc.transform(scaler.transform(np.asarray(X[mode], float))))
        parts.append(np.asarray(point_prediction) / self.training_iqr)
        features = self.feature_scaler.transform(np.column_stack(parts))
        log_scale = 0.5 * self.center + 0.5 * self.regressor.predict(features)
        if log_scale.ndim == 1:
            log_scale = log_scale[:, None]
        value = self.training_iqr * np.exp(np.clip(log_scale, self.lower, self.upper))
        if not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError('Conditional residual scales must be finite and positive')
        return value

def _audit_quantile(q):
    a = np.asarray(q)
    if a.ndim == 0:
        return '+infinity' if np.isposinf(a) else float(a)
    return [_audit_quantile(v) for v in a]

def _regression_task(data, task, profile, cfg):
    from .engine import regression_records
    tr = np.asarray(task['train'], int)
    cal = np.asarray(task['calibration'], int)
    te = np.asarray(task['test'], int)
    Y = np.asarray(data['Y'], float)
    X = subset(data['X'], tr)
    Xc = subset(data['X'], cal)
    Xt = subset(data['X'], te)
    training_iqr = iqr(Y[tr])
    alpha = 1 - float(cfg['nominal_coverage'])
    records, calibration_records, audit = ([], [], {})
    for key in ['nir', 'raman_uv', 'all']:
        catalog = candidates(COMBINATIONS[key], profile, full=False)
        model, selected = tune(X, Y[tr], catalog, profile, task['seed'])
        cp, tp = (model.predict(Xc), model.predict(Xt))
        oof, crossfit_audit = strict_crossfit(X, Y[tr], catalog, profile, task['seed'] + 1701)
        residual_model = ResidualScale(X, Y[tr], oof, COMBINATIONS[key], training_iqr)
        adaptive_cal = residual_model.predict(Xc, cp)
        adaptive_test = residual_model.predict(Xt, tp)
        fixed_cal = np.broadcast_to(training_iqr, cp.shape)
        fixed_test = np.broadcast_to(training_iqr, tp.shape)
        for scale_type, cs, ts in [('fixed', fixed_cal, fixed_test), ('adaptive', adaptive_cal, adaptive_test)]:
            name = 'v8_' + key + '__' + scale_type
            scores = np.abs(cp - Y[cal]) / cs
            marginal_q = finite_quantile(scores, alpha)
            joint_q = float(finite_quantile(np.max(scores, axis=1), alpha))
            cf = regression_records(data, cal, cp, name, training_iqr, task)
            cf['interval_scale'] = cs.reshape(-1)
            cf['normalized_abs_residual'] = scores.reshape(-1)
            cf['marginal_quantile'] = np.tile(marginal_q, len(cal))
            cf['joint_quantile'] = joint_q
            cf['score_type'] = 'normalized_absolute_residual_' + scale_type
            calibration_records.append(cf)
            for coverage_type, q in [('marginal', marginal_q), ('joint', np.repeat(joint_q, Y.shape[1]))]:
                f = regression_records(data, te, tp, name + '__' + coverage_type, training_iqr, task)
                f['interval_scale'] = ts.reshape(-1)
                f['quantile'] = np.tile(q, len(te))
                f['lower'] = (tp - ts * q).reshape(-1)
                f['upper'] = (tp + ts * q).reshape(-1)
                f['calibration_n'] = len(cal)
                f['nominal_coverage'] = cfg['nominal_coverage']
                f['score_type'] = 'normalized_absolute_residual_' + scale_type
                f['coverage_type'] = coverage_type
                f['above_train_q75'] = (Y[te] > np.percentile(Y[tr], 75, axis=0)).reshape(-1)
                ethanol = np.asarray(data['ethanol'])
                tails = (ethanol[te] < np.percentile(ethanol[tr], 10)) | (ethanol[te] > np.percentile(ethanol[tr], 90))
                f['ethanol_tail'] = np.repeat(tails, Y.shape[1])
                records.append(f)
            audit[name] = {'selection': selected, 'marginal_quantile': _audit_quantile(marginal_q), 'joint_quantile': _audit_quantile(joint_q), 'same_point_model_and_calibration_ids': True, 'calibration_labels_used_only_for_quantile': True, 'fit_partition_ids': data['ids'][tr].tolist(), 'calibration_partition_ids': data['ids'][cal].tolist(), 'marginal_exchangeability_coverage_only': True, 'scale_model': 'constant train IQR' if scale_type == 'fixed' else {'family': 'PCA + Ridge on nested OOF log absolute residual', 'alpha': 50, 'shrinkage_to_training_median': 0.5, 'train_log_scale_bounds_percentiles': [5, 95], 'nested_crossfit': crossfit_audit}}
    return ({'interval_regression': pd.concat(records, ignore_index=True), 'calibration_regression': pd.concat(calibration_records, ignore_index=True)}, audit)

def _classification_task(data, task, profile, cfg):
    from .engine import classification_records
    tr = np.asarray(task['train'], int)
    cal = np.asarray(task['calibration'], int)
    te = np.asarray(task['test'], int)
    y = labels(data)
    classes = list(cfg.get('classes', sorted(np.unique(y).tolist())))
    true_index = np.asarray([classes.index(c) for c in y[cal]])
    alpha = 1 - float(cfg['nominal_coverage'])
    X, Xc, Xt = (subset(data['X'], tr), subset(data['X'], cal), subset(data['X'], te))
    records, calibration_records, audit = ([], [], {})
    for key in ['nir', 'uv', 'raman_uv', 'all']:
        model, selected = clf.tune(X, y[tr], classes, clf.catalog(COMBINATIONS[key], profile), profile, task['seed'])
        cp, tp = (model.predict_proba(Xc), model.predict_proba(Xt))
        for score_type in ['lac', 'aps']:
            name = 'v8_' + key + '__' + score_type
            cs, ts = (class_scores(cp, score_type), class_scores(tp, score_type))
            score = cs[np.arange(len(cal)), true_index]
            q = float(finite_quantile(score, alpha))
            cf = classification_records(data, cal, cp, name, classes, task, y)
            cf['nonconformity'] = score
            cf['quantile'] = q
            cf['score_type'] = score_type
            calibration_records.append(cf)
            include = ts <= q
            f = classification_records(data, te, tp, name, classes, task, y)
            f['prediction_set'] = [' | '.join(np.asarray(classes)[row]) for row in include]
            f['set_size'] = include.sum(axis=1)
            f['covered'] = [bool(include[i, classes.index(c)]) for i, c in enumerate(y[te])]
            f['calibration_n'] = len(cal)
            f['nominal_coverage'] = cfg['nominal_coverage']
            f['score_type'] = score_type
            f['quantile'] = q
            records.append(f)
            audit[name] = {'selection': selected, 'score_type': score_type, 'quantile': _audit_quantile(q), 'deterministic_class_order': classes, 'nonrandomized_aps': score_type == 'aps', 'empty_sets_retained': True, 'calibration_labels_used_only_for_quantile': True, 'marginal_only_no_class_conditional_guarantee': True, 'rare_classes_not_merged': True}
    return ({'interval_classification': pd.concat(records, ignore_index=True), 'calibration_classification': pd.concat(calibration_records, ignore_index=True)}, audit)

def run_task(data, task, profile, cfg):
    partitions = [set(task[name]) for name in ['train', 'calibration', 'test']]
    if not partitions[1] or any((partitions[i] & partitions[j] for i in range(3) for j in range(i))):
        raise ValueError('Fit, calibration and test bottle sets must be disjoint; calibration must be nonempty')
    if task['kind'] == 'regression':
        return _regression_task(data, task, profile, cfg)
    if task['kind'] == 'classification':
        return _classification_task(data, task, profile, cfg)
    raise ValueError('Unknown interval task kind')

def verify_frames(frames, alpha):
    errors = []
    if 'calibration_regression' in frames:
        cal, test = (frames['calibration_regression'], frames['interval_regression'])
        if cal.duplicated(['sample_id', 'method', 'target']).any():
            errors.append('Duplicate regression calibration keys')
        vcal = cal[cal.method.astype(str).str.startswith('v8_')]
        vtest = test[test.method.astype(str).str.startswith('v8_')]
        expected = {m + '__' + k for m in vcal.method.unique() for k in ['marginal', 'joint']}
        if set(vtest.method) != expected:
            errors.append('Regression test/calibration method inventory mismatch')
        ids = set(vcal.sample_id)
        targets = set(vtest.target)
        for method, g in vcal.groupby('method'):
            if set(g.target) != targets:
                errors.append('Calibration target inventory mismatch: ' + method)
            for _, part in g.groupby('target'):
                if set(part.sample_id) != ids:
                    errors.append('Incomplete regression calibration bottles: ' + method)
        for method, g in cal.groupby('method'):
            if not str(method).startswith('v8_'):
                continue
            if not np.isfinite(g.interval_scale.to_numpy(float)).all() or np.any(g.interval_scale.to_numpy(float) <= 0):
                errors.append('Invalid calibration interval scale: ' + method)
                continue
            score = np.abs(g.y_pred.to_numpy(float) - g.y_true.to_numpy(float)) / g.interval_scale.to_numpy(float)
            if not np.allclose(score, g.normalized_abs_residual.to_numpy(float)):
                errors.append('Calibration normalized residual mismatch: ' + method)
            qj = float(finite_quantile(g.assign(residual=score).groupby('sample_id').residual.max().to_numpy(), alpha))
            if not np.allclose(g.joint_quantile.to_numpy(float), qj):
                errors.append('Saved joint calibration quantile mismatch: ' + method)
            for target, t in g.groupby('target'):
                qm = float(finite_quantile(np.abs(t.y_pred.to_numpy(float) - t.y_true.to_numpy(float)) / t.interval_scale.to_numpy(float), alpha))
                if not np.allclose(t.marginal_quantile.to_numpy(float), qm):
                    errors.append('Saved marginal calibration quantile mismatch: ' + method)
                for kind, q in [('marginal', qm), ('joint', qj)]:
                    a = test[(test.method == method + '__' + kind) & (test.target == target)]
                    if a.empty:
                        errors.append('Missing interval predictions: ' + method + '/' + target)
                        continue
                    if not np.isfinite(a.interval_scale.to_numpy(float)).all() or np.any(a.interval_scale.to_numpy(float) <= 0):
                        errors.append('Invalid test interval scale: ' + method)
                        continue
                    if not np.allclose(a['quantile'].to_numpy(float), q):
                        errors.append('Test quantile mismatch: ' + method)
                    lower = a.y_pred.to_numpy(float) - q * a.interval_scale.to_numpy(float)
                    upper = a.y_pred.to_numpy(float) + q * a.interval_scale.to_numpy(float)
                    if not (np.allclose(lower, a.lower.to_numpy(float)) and np.allclose(upper, a.upper.to_numpy(float))):
                        errors.append('Interval bound mismatch: ' + method)
    if 'calibration_classification' in frames:
        cal, test = (frames['calibration_classification'], frames['interval_classification'])
        if cal.duplicated(['sample_id', 'method']).any():
            errors.append('Duplicate classification calibration keys')
        vcal = cal[cal.method.astype(str).str.startswith('v8_')]
        vtest = test[test.method.astype(str).str.startswith('v8_')]
        if set(vcal.method) != set(vtest.method):
            errors.append('Classification test/calibration method inventory mismatch')
        ids = set(vcal.sample_id)
        for method, g in vcal.groupby('method'):
            if set(g.sample_id) != ids:
                errors.append('Incomplete classification calibration bottles: ' + method)
        pcols = [c for c in cal.columns if c.startswith('p_')]
        classes = [c[2:] for c in pcols]
        for method, g in cal.groupby('method'):
            if not str(method).startswith('v8_'):
                continue
            probabilities = g[pcols].to_numpy(float)
            if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (not np.allclose(probabilities.sum(axis=1), 1)):
                errors.append('Invalid calibration class scores: ' + method)
                continue
            score_type = str(g.score_type.iloc[0])
            cm = class_scores(g[pcols].to_numpy(float), score_type)
            score = cm[np.arange(len(g)), [classes.index(str(c)) for c in g.y_true]]
            q = float(finite_quantile(score, alpha))
            if not np.allclose(score, g.nonconformity.to_numpy(float)) or not np.allclose(q, g['quantile'].to_numpy(float)):
                errors.append('Class calibration score/quantile mismatch: ' + method)
            t = test[test.method == method]
            if t.empty:
                errors.append('Missing prediction sets: ' + method)
                continue
            if not np.allclose(q, t['quantile'].to_numpy(float)):
                errors.append('Saved test class quantile mismatch: ' + method)
            include = class_scores(t[pcols].to_numpy(float), score_type) <= q
            if not np.array_equal(t.set_size.to_numpy(int), include.sum(axis=1)):
                errors.append('Prediction set size mismatch: ' + method)
            expected_sets = [' | '.join(np.asarray(classes)[row]) for row in include]
            actual_sets = t.prediction_set.fillna('').astype(str).tolist()
            if actual_sets != expected_sets:
                errors.append('Prediction set label mismatch: ' + method)
            coverage = [bool(include[i, classes.index(str(c))]) for i, c in enumerate(t.y_true)]
            stored = [str(v).lower() == 'true' for v in t.covered]
            if stored != coverage:
                errors.append('Prediction set coverage mismatch: ' + method)
    return errors
