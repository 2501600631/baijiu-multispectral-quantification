import copy
import warnings
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from . import classification as legacy
from .common import COMBINATIONS, digest, subset
from .design import labels

class RegularizedClassifier:

    def __init__(self, spec, X, y, classes, seed=0):
        self.spec = copy.deepcopy(spec)
        self.classes_ = np.asarray(classes, str)
        y = np.asarray(y, str)
        self.local = np.unique(y)
        self.feature = legacy.Features(spec, seed)
        self.estimator = None
        family = spec['family']
        params = spec.get('params', {})
        with threadpool_limits(limits=1):
            z = self.feature.fit_transform(X)
            if len(self.local) == 1:
                self.prior = (self.classes_ == self.local[0]).astype(float)
            else:
                weight = params.get('class_weight', 'balanced')
                if family == 'regularized_logistic':
                    model = LogisticRegression(C=params.get('C', 1), class_weight=weight, solver='lbfgs', max_iter=20000, tol=1e-05, random_state=seed)
                elif family == 'linear_svm':
                    model = SVC(C=params.get('C', 1), kernel='linear', class_weight=weight, probability=False, decision_function_shape='ovr', random_state=seed)
                elif family == 'shrinkage_lda':
                    priors = np.full(len(self.local), 1 / len(self.local)) if params.get('uniform_prior', False) else None
                    model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=params.get('shrinkage', 'auto'), priors=priors)
                else:
                    raise ValueError('Unknown regularized classifier: ' + family)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model.fit(z, y)
                if any((issubclass(w.category, ConvergenceWarning) for w in caught)):
                    raise RuntimeError('Regularized classification candidate did not converge')
                self.estimator = model
        self.metadata = {'family': family, 'training_n': len(y), 'input_modes': list(spec['modes']), 'classes': self.classes_.tolist(), 'training_class_counts': {str(c): int(np.sum(y == c)) for c in self.classes_}, 'class_order_assumed': False, 'classes_merged': False, 'measured_chemistry_required': False, 'measured_ethanol_required': False, 'scores_are_calibrated_probabilities': False, 'linear_svm_softmax_is_uncalibrated': family == 'linear_svm', 'feature_preprocessing_fitted_on_training_only': True}

    def predict_proba(self, X):
        n = len(X[self.spec['modes'][0]])
        if self.estimator is None:
            return np.tile(self.prior, (n, 1))
        with threadpool_limits(limits=1):
            z = self.feature.transform(X)
            if hasattr(self.estimator, 'predict_proba'):
                p = self.estimator.predict_proba(z)
            else:
                score = self.estimator.decision_function(z)
                if score.ndim == 1:
                    score = np.column_stack([-score, score])
                p = softmax(score, axis=1)
        answer = np.zeros((n, len(self.classes_)))
        for j, name in enumerate(self.estimator.classes_):
            answer[:, np.flatnonzero(self.classes_ == name)[0]] = p[:, j]
        if not np.isfinite(answer).all() or np.any(answer < 0):
            raise ValueError('Invalid classification scores')
        return answer

def catalog(modes, profile):
    count = min(4, max(1, int(profile['candidates_per_family'])))
    families = ['regularized_logistic', 'linear_svm', 'shrinkage_lda']
    result = []
    for family in families:
        for i in range(count):
            result.append({'family': family, 'modes': list(modes), 'pre': ['raw', 'smooth', 'snv', 'smooth'][i], 'pca_components': [8, 16, 24, 12][i], 'params': {'C': [0.1, 1, 10, 100][i], 'class_weight': ['balanced', None, 'balanced', None][i], 'uniform_prior': [False, True, False, True][i], 'shrinkage': ['auto', 0.1, 0.5, 0.9][i]}})
    if list(modes) == ['uv']:
        for spec in result:
            if spec['pre'] == 'snv':
                spec['pre'] = 'raw'
    return result

def _fit(spec, X, y, classes, seed):
    with threadpool_limits(limits=1):
        if spec['family'] in ('regularized_logistic', 'linear_svm', 'shrinkage_lda'):
            return RegularizedClassifier(spec, X, y, classes, seed)
        return legacy.FittedClassifier(spec, X, y, classes, None, seed)

def _rank(score, spec):
    if spec.get('family') in ('logistic', 'svm', 'plsda'):
        return (-score['macro_f1'], -score['balanced_accuracy'], 0.0, repr(spec))
    return (-score['macro_f1'], -score['balanced_accuracy'], -score['accuracy'], digest(spec))

def tune(X, y, classes, specs, profile, seed):
    y = np.asarray(y, str)
    counts = np.unique(y, return_counts=True)[1]
    k = min(int(profile['inner_folds']), int(counts.min()))
    failures = []
    support = {str(c): int(np.sum(y == c)) for c in classes}
    if k < 2:
        if all((s['family'] in ('logistic', 'svm', 'plsda') for s in specs)):
            spec = {'family': 'logistic', 'modes': list(specs[0]['modes']), 'pre': 'smooth', 'pca_components': 12, 'params': {'C': 1}, 'use_ethanol': specs[0].get('use_ethanol', False)}
        else:
            spec = {'family': 'regularized_logistic', 'modes': list(specs[0]['modes']), 'pre': 'smooth', 'pca_components': 8, 'params': {'C': 1, 'class_weight': 'balanced'}}
        model = _fit(spec, X, y, classes, seed)
        return (model, None, {'selected': spec, 'selected_metrics': None, 'selection': 'predeclared fixed fallback: a training class has fewer than 2 bottles', 'inner_folds': 0, 'training_class_counts': support, 'failures': failures, 'metadata': model.metadata})
    folds = list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    rankings = []
    audit = []
    for spec in specs:
        p = np.zeros((len(y), len(classes)))
        errors = []
        for f, (tr, va) in enumerate(folds):
            try:
                model = _fit(spec, subset(X, tr), y[tr], classes, seed + f)
                p[va] = model.predict_proba(subset(X, va))
            except Exception as exc:
                errors.append({'inner_fold': f, 'error': repr(exc)})
        if errors:
            failures.append({'spec': spec, 'stage': 'inner_fit', 'errors': errors})
            continue
        score = legacy.metrics(y, p, classes)
        audit.append({'spec': spec, 'metrics': score})
        rankings.append((_rank(score, spec), spec, p, score))
    if not rankings:
        raise RuntimeError('Every optimized classification candidate failed: ' + repr(failures))
    for _, spec, p, score in sorted(rankings, key=lambda row: row[0]):
        try:
            model = _fit(spec, X, y, classes, seed)
        except Exception as exc:
            failures.append({'spec': spec, 'stage': 'final_fit', 'error': repr(exc)})
            continue
        return (model, p, {'selected': spec, 'selected_metrics': score, 'selection': 'outer-training stratified inner-CV macro-F1; balanced accuracy tie break', 'inner_folds': k, 'training_class_counts': support, 'candidates': audit, 'failures': failures, 'metadata': model.metadata})
    raise RuntimeError('Every optimized classification refit failed: ' + repr(failures))

def _records(data, idx, scores, method, classes, task, y):
    pred = np.asarray(classes)[np.argmax(scores, axis=1)]
    rows = []
    for k, i in enumerate(idx):
        row = {'sample_id': str(data['ids'][i]), 'repeat': task['repeat'], 'fold': task['fold'], 'method': method, 'y_true': str(y[i]), 'y_pred': str(pred[k]), 'fit_n': len(task['train'])}
        row.update({'p_' + str(c): float(scores[k, j]) for j, c in enumerate(classes)})
        rows.append(row)
    return pd.DataFrame(rows)

def run_task(data, task, profile, cfg):
    tr = np.asarray(task['train'], int)
    te = np.asarray(task['test'], int)
    seed = int(task['seed'])
    y = labels(data)
    classes = list(cfg.get('classes', sorted(np.unique(y).tolist())))
    if not set(y[tr]).issubset(set(classes)):
        raise ValueError('Training labels are outside the frozen original class vocabulary')
    X, Xt = (subset(data['X'], tr), subset(data['X'], te))
    baseline_specs = [s for s in legacy.catalog(['uv'], profile) if s['family'] in ('logistic', 'svm', 'plsda')]
    baseline, bp, ba = tune(X, y[tr], classes, baseline_specs, profile, seed)
    baseline_pred = baseline.predict_proba(Xt)
    frames = [_records(data, te, baseline_pred, 'v8_uv_reference', classes, task, y)]
    audit = {'uv_reference': ba, 'classes': classes, 'class_order_assumed': False, 'classes_merged': False, 'test_input': 'spectra only', 'scores_are_calibrated_probabilities': False, 'test_sensory_labels_used_in_selection': False, 'chemical_reference_values_used_in_selection': False}
    choices = {}
    for key, modes in COMBINATIONS.items():
        model, op, choice = tune(X, y[tr], classes, catalog(modes, profile), profile, seed)
        pp = model.predict_proba(Xt)
        metric = choice['selected_metrics']
        rank = (np.inf, np.inf, np.inf, key) if metric is None else _rank(metric, {'subset': key, 'spec': choice['selected']})
        choices[key] = (rank, model, op, pp, choice)
        audit['regularized_' + key] = choice
    for name, keys in [('v8_best_regularized_single', ['nir', 'raman', 'uv']), ('v8_best_regularized_subset', list(COMBINATIONS))]:
        key = min(keys, key=lambda k: choices[k][0])
        if choices[key][2] is None:
            key = 'uv'
        frames.append(_records(data, te, choices[key][3], name, classes, task, y))
        audit[name] = {'selected_subset': key, 'selection': 'training-only inner macro-F1', 'no_cv_selection_possible': choices[key][2] is None}
    frames.append(_records(data, te, choices['uv'][3], 'v8_uv_regularized', classes, task, y))
    key = min(choices, key=lambda k: choices[k][0])
    candidates_blend = []
    if bp is not None and choices[key][2] is not None:
        for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            scores = (1 - weight) * bp + weight * choices[key][2]
            metric = legacy.metrics(y[tr], scores, classes)
            candidates_blend.append(((-metric['macro_f1'], -metric['balanced_accuracy'], -metric['accuracy'], weight), weight, metric))
        _, weight, blend_metric = min(candidates_blend, key=lambda row: row[0])
    else:
        weight, blend_metric = (0.0, None)
    scores = (1 - weight) * baseline_pred + weight * choices[key][3]
    frames.append(_records(data, te, scores, 'v8_selected_blend', classes, task, y))
    audit['v8_selected_blend'] = {'uv_reference_weight': 1 - weight, 'regularized_subset_weight': weight, 'regularized_subset': key, 'selected_metrics': blend_metric, 'selection': 'prespecified convex grid on outer-training inner-CV scores only', 'same_inner_scores_used_for_model_and_weight_selection': True, 'performance_must_be_assessed_on_outer_fold': True, 'scores_are_calibrated_probabilities': False, 'fixed_uv_fallback_if_no_cv': bp is None, 'candidates': [{'weight': w, 'metrics': s} for _, w, s in candidates_blend]}
    return ({'classification': pd.concat(frames, ignore_index=True)}, audit)
