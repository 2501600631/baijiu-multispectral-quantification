from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from .common import digest, iqr, subset
METHODS = ('v10_pls2_reference', 'v10_v9_unanchored_reference', 'v10_strategy_router')
DEFAULT_NOISE = (0.005, 0.02)
DEFAULT_INTENSITY = (0.98, 1.02)
DEFAULT_OFFSET = (-0.02, 0.02)

def _blocks(X_train, X_test):
    if not X_train or set(X_train) != set(X_test):
        raise ValueError('Training and test modality names must agree')
    train, test = ({}, {})
    for mode in sorted(X_train):
        a, b = (np.asarray(X_train[mode], float), np.asarray(X_test[mode], float))
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or (not a.shape[1]):
            raise ValueError('Wrong spectral block shape: ' + mode)
        if not len(a) or not len(b) or (not np.isfinite(a).all()) or (not np.isfinite(b).all()):
            raise ValueError('Empty or nonfinite spectral block: ' + mode)
        train[mode], test[mode] = (a, b)
    if len({len(a) for a in train.values()}) != 1 or len({len(b) for b in test.values()}) != 1:
        raise ValueError('Bottle counts must agree between modalities')
    return (train, test)

def _scenario_seed(seed, scenario):
    token = ('%d:%s' % (int(seed), scenario)).encode('ascii')
    return int.from_bytes(hashlib.sha256(token).digest()[:4], 'big')

def make_stress_inputs(X_train, X_test, seed, options=None):
    train, test = _blocks(X_train, X_test)
    options = {} if options is None else options
    settings = [('gaussian_noise', options.get('noise_levels', DEFAULT_NOISE)), ('intensity_multiplier', options.get('intensity_factors', DEFAULT_INTENSITY)), ('training_std_offset', options.get('offset_levels', DEFAULT_OFFSET))]
    output = []
    for mode in sorted(train):
        scale = np.std(train[mode], axis=0, ddof=0)
        for kind, values in settings:
            for value in values:
                value = float(value)
                if not np.isfinite(value) or (kind == 'gaussian_noise' and value < 0):
                    raise ValueError('Invalid prespecified stress magnitude')
                if kind == 'intensity_multiplier' and value <= 0:
                    raise ValueError('Intensity factors must be positive')
                scenario = '%s__%s__%s' % (mode, kind, format(value, '.8g'))
                changed = {key: a.copy() for key, a in test.items()}
                if kind == 'gaussian_noise':
                    rng = np.random.RandomState(_scenario_seed(seed, scenario))
                    changed[mode] += value * scale * rng.normal(size=changed[mode].shape)
                elif kind == 'intensity_multiplier':
                    changed[mode] *= value
                else:
                    changed[mode] += value * scale
                output.append((scenario, changed, {'scenario': scenario, 'changed_mode': mode, 'challenge_type': kind, 'magnitude': value, 'scale_source': 'outer_training_only', 'numerical_sensitivity_only': True, 'independent_real_instrument_validation': False}))
    return output

def make_occlusion_inputs(X_train, X_test, axes=None, windows_per_modality=6):
    train, test = _blocks(X_train, X_test)
    if int(windows_per_modality) != windows_per_modality or int(windows_per_modality) < 1:
        raise ValueError('The number of windows must be a positive integer')
    axes = {} if axes is None else axes
    output = []
    for mode in sorted(train):
        columns = train[mode].shape[1]
        axis = np.asarray(axes.get(mode, np.arange(columns)), float)
        if axis.ndim != 1 or len(axis) != columns or (not np.isfinite(axis).all()):
            raise ValueError('Wrong numerical spectral coordinates: ' + mode)
        training_mean = np.mean(train[mode], axis=0)
        for window, indices in enumerate(np.array_split(np.arange(columns), int(windows_per_modality))):
            if not len(indices):
                continue
            changed = {key: a.copy() for key, a in test.items()}
            changed[mode][:, indices] = training_mean[indices]
            scenario = '%s__training_mean_window__%02d' % (mode, window)
            output.append((scenario, changed, {'scenario': scenario, 'changed_mode': mode, 'challenge_type': 'training_mean_window_replacement', 'window_index': window, 'column_start': int(indices[0]), 'column_stop_exclusive': int(indices[-1] + 1), 'coordinate_start': float(axis[indices[0]]), 'coordinate_end': float(axis[indices[-1]]), 'coordinate_units': 'not_inferred', 'scale_source': 'outer_training_only', 'numerical_sensitivity_only': True, 'causal_chemical_assignment': False}))
    return output

def _fit_bundle(X, Y, profile, seed, options):
    from .final_ensemble import fit_final_bundle
    return fit_final_bundle(X, Y, profile, seed, options=options)

def _predictions(bundle, X, expected_shape):
    out = bundle.predict_methods(X, methods=list(METHODS))
    if set(out) != set(METHODS):
        raise ValueError('Required frozen point methods missing from bundle')
    result = {}
    for method in METHODS:
        pred = np.asarray(out[method], float)
        if pred.shape != expected_shape or not np.isfinite(pred).all():
            raise ValueError('Wrong or nonfinite boundary prediction: ' + method)
        result[method] = pred
    return result

def _records(data, task, test, prediction, method, scale, extra=None):
    truth = np.asarray(data['Y'])[test]
    extra = {} if extra is None else extra
    rows = []
    for k, index in enumerate(test):
        for j, target in enumerate(data['target_names']):
            row = {'sample_id': str(data['ids'][index]), 'repeat': task['repeat'], 'fold': task['fold'], 'method': method, 'target': target, 'y_true': float(truth[k, j]), 'y_pred': float(prediction[k, j]), 'train_iqr': float(scale[j]), 'fit_n': len(task['train'])}
            row.update(extra)
            rows.append(row)
    return pd.DataFrame(rows)

def _measured_abv_baseline(ethanol_train, Y_train, ethanol_test, profile, seed):
    e = np.asarray(ethanol_train, float).reshape(-1, 1)
    et = np.asarray(ethanol_test, float).reshape(-1, 1)
    if not np.isfinite(e).all() or not np.isfinite(et).all():
        raise ValueError('Measured ABV comparator requires finite observed ABV')
    folds = min(int(profile.get('inner_folds', 3)), len(e))
    if folds < 2:
        raise ValueError('Measured ABV comparator requires at least two CV folds')
    splits = list(KFold(folds, shuffle=True, random_state=int(seed)).split(e))
    options, outcomes = ([], [])
    for degree in (1, 2, 3):
        for alpha in (0.1, 1.0, 10.0, 100.0):
            loss = []
            for train, valid in splits:
                model = make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), Ridge(alpha=alpha))
                model.fit(e[train], Y_train[train])
                residual = (model.predict(e[valid]) - Y_train[valid]) / iqr(Y_train[train])
                loss.append(float(np.mean(np.sqrt(np.mean(residual ** 2, axis=0)))))
            score = float(np.mean(loss))
            if not np.isfinite(score):
                raise ValueError('Nonfinite measured-ABV comparator score')
            options.append((score, degree, alpha))
            outcomes.append({'degree': degree, 'alpha': alpha, 'inner_macro_nrmse': score})
    score, degree, alpha = min(options)
    model = make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), Ridge(alpha=alpha))
    model.fit(e, Y_train)
    return (model.predict(et), {'method': 'v10_measured_abv_polynomial_ridge', 'requires_measured_test_abv': True, 'spectra_only_method': False, 'selection_scope': 'outer_training_only', 'selected_degree': degree, 'selected_alpha': alpha, 'inner_score': score, 'all_candidate_scores': outcomes})

def _posthoc_diagnostics(data, task, train, test):
    Y, ethanol = (np.asarray(data['Y'], float), np.asarray(data['ethanol'], float))
    q75 = np.percentile(Y[train], 75, axis=0)
    lo, hi = np.percentile(ethanol[train], [10, 90])
    rows = []
    for index in test:
        for j, target in enumerate(data['target_names']):
            rows.append({'sample_id': str(data['ids'][index]), 'repeat': task['repeat'], 'fold': task['fold'], 'target': target, 'training_q75': float(q75[j]), 'y_true': float(Y[index, j]), 'above_training_q75': bool(Y[index, j] > q75[j]), 'measured_abv': float(ethanol[index]), 'training_abv_q10': float(lo), 'training_abv_q90': float(hi), 'outside_training_abv_central80': bool(ethanol[index] < lo or ethanol[index] > hi), 'truth_used_only_for_posthoc_grouping': True, 'concentration_tail_known_without_reference': False, 'abv_group_requires_measured_test_abv': True})
    return pd.DataFrame(rows)

def run_task(data, task, p, cfg, bundle=None):
    train, test = (np.asarray(task['train'], int), np.asarray(task['test'], int))
    if not len(train) or not len(test) or set(train) & set(test):
        raise ValueError('Empty or overlapping outer training/test bottle IDs')
    X, Xt = (subset(data['X'], train), subset(data['X'], test))
    Y = np.asarray(data['Y'], float)[train]
    expected_shape = (len(test), Y.shape[1])
    seed = int(task['seed'])
    options = cfg.get('final_boundaries', {})
    reused = bundle is not None
    if bundle is None:
        bundle = _fit_bundle(X, Y, p, seed, cfg.get('ensemble_options', {}))
    clean = _predictions(bundle, Xt, expected_shape)
    scale = iqr(Y)
    records = [_records(data, task, test, pred, method, scale, {'scenario': 'clean'}) for method, pred in clean.items()]
    abv, abv_audit = _measured_abv_baseline(np.asarray(data['ethanol'])[train], Y, np.asarray(data['ethanol'])[test], p, seed)
    records.append(_records(data, task, test, abv, abv_audit['method'], scale, {'scenario': 'clean', 'requires_measured_test_abv': True}))
    stress, window, challenge_audit = ([], [], [])
    sets = [('stress', make_stress_inputs(X, Xt, seed, options)), ('occlusion', make_occlusion_inputs(X, Xt, data.get('axes'), options.get('windows_per_modality', 6)))]
    for kind, challenges in sets:
        for scenario, changed, metadata in challenges:
            pred = _predictions(bundle, changed, expected_shape)
            target = stress if kind == 'stress' else window
            for method, values in pred.items():
                target.append(_records(data, task, test, values, method, scale, metadata))
            challenge_audit.append(metadata)
    audit = {'bundle_reused': reused, 'bundle': bundle.audit, 'abv_comparator': abv_audit, 'spectral_refits_inside_challenges': 0, 'challenge_count': len(challenge_audit), 'challenges': challenge_audit, 'training_spectral_moment_digest': digest({k: {'mean': np.mean(v, axis=0), 'std_ddof0': np.std(v, axis=0, ddof=0)} for k, v in X.items()}), 'pairing': 'identical_perturbed_input_for_reference_and_anchored_ensemble', 'real_instrument_drift_validated': False, 'causal_peak_assignment': False, 'whole_missing_modality_evaluation': 'not_retrained_here_preserve_V8_specialist_comparisons', 'tail_groups': 'training_cutoffs_test_reference_used_only_for_posthoc_reporting', 'stability': 'aggregate_outer_fold_training_weights_not_independent_bootstrap_samples'}
    frames = {'regression': pd.concat(records, ignore_index=True), 'stress': pd.concat(stress, ignore_index=True) if stress else pd.DataFrame(), 'occlusion': pd.concat(window, ignore_index=True) if window else pd.DataFrame(), 'diagnostics': _posthoc_diagnostics(data, task, train, test)}
    return (frames, audit)
