from __future__ import annotations
import copy
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from .common import digest, iqr, subset
from .optimized_regression import _ABVResidual
SPECTRAL_METHODS = ('v10_strategy_router', 'v10_v9_unanchored_reference')
ABV_METHODS = ('v10_measured_abv_only', 'v10_measured_abv_residual', 'v10_measured_abv_guarded')
POINT_METHODS = SPECTRAL_METHODS + ABV_METHODS
POINTMETHODS = POINT_METHODS
LEGACY_METHOD = SPECTRAL_METHODS[1]

def build_residual_catalog():
    modes = [('nir', 'raman', 'uv'), ('uv',), ('raman', 'uv'), ('nir', 'uv')]
    return [{'family': 'pls2', 'modes': list(view), 'pre': 'snv' if n == 8 else 'smooth', 'representation': 'direct', 'params': {'n_components': n, 'pca_components': {2: 16, 4: 24, 8: 32}[n]}} for view in modes for n in (2, 4, 8)]

class _ABVOnly:

    def __init__(self, Y, ethanol):
        self.poly = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), StandardScaler())
        self.regressor = Ridge(alpha=1.0).fit(self.poly.fit_transform(np.asarray(ethanol, float).reshape(-1, 1)), np.asarray(Y, float))

    def predict(self, ethanol):
        e = np.asarray(ethanol, float).reshape(-1, 1)
        if not np.isfinite(e).all():
            raise ValueError('Finite separately measured ethanol is required')
        return np.asarray(self.regressor.predict(self.poly.transform(e)))

def _prediction(values, shape, label):
    out = np.asarray(values, float)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError('Invalid finite prediction shape for ' + label)
    return out

def _validate_splits(receipts, n):
    splits, visits = ([], np.zeros(n, dtype=int))
    for receipt in receipts:
        train = np.asarray(receipt['train'], dtype=int)
        validation = np.asarray(receipt['validation'], dtype=int)
        if not len(train) or not len(validation) or np.any(train < 0) or np.any(train >= n) or np.any(validation < 0) or np.any(validation >= n) or (len(set(train)) != len(train)) or (len(set(validation)) != len(validation)) or set(train).intersection(validation) or (len(train) + len(validation) != n):
            raise ValueError('Invalid strict-procedure training/validation partition')
        visits[validation] += 1
        splits.append((train, validation))
    if len(splits) < 2 or not np.all(visits == 1):
        raise ValueError('Incomplete strict-procedure OOF membership')
    return splits

def _baseline_oof(bundle, X, Y, p, seed, cfg):
    strategy = bundle.audit.get('strategy_router', {})
    cached = strategy.get('strict_procedure_oof', {}).get(LEGACY_METHOD)
    if cached is not None:
        prediction = _prediction(cached, Y.shape, LEGACY_METHOD + ' strict OOF')
        splits = _validate_splits(strategy.get('crossfit', []), len(Y))
        return (prediction, splits, {'source': 'cached strict whole-procedure OOF', 'additional_spectral_model_fitting': False, 'strict_membership': copy.deepcopy(strategy['crossfit'])})
    from .final_ensemble import strict_crossfit_final
    options = dict(cfg.get('ensemble_options', {}), enable_strategy_router=False)
    options['crossfit_folds'] = int(p.get('strategy_folds', p['inner_folds']))
    predictions, receipts = strict_crossfit_final(X, Y, p, int(seed) + 3901, methods=[LEGACY_METHOD], options=options)
    splits = _validate_splits(receipts, len(Y))
    return (_prediction(predictions[LEGACY_METHOD], Y.shape, LEGACY_METHOD + ' strict OOF'), splits, {'source': 'new strict whole-procedure baseline OOF', 'additional_spectral_model_fitting': True, 'strict_membership': receipts})

def _records(data, task, method, prediction, scale):
    rows = []
    for row, i in enumerate(task['test']):
        for j, target in enumerate(data['target_names']):
            rows.append({'sample_id': str(data['ids'][i]), 'repeat': task['repeat'], 'fold': task['fold'], 'method': method, 'target': target, 'y_true': float(data['Y'][i, j]), 'y_pred': float(prediction[row, j]), 'train_iqr': float(scale[j]), 'fit_n': len(task['train']), 'requires_measured_test_abv': method in ABV_METHODS, 'input_requirement': 'spectra_and_measured_ethanol' if method in ABV_METHODS[1:] else 'measured_ethanol_only' if method == ABV_METHODS[0] else 'spectra_only', 'evaluation_context': 'full_outer_training', 'prediction_intervals_available_in_this_mode': False})
    return pd.DataFrame(rows)

def run_task(data, task, p, cfg, bundle):
    if bundle is None:
        raise ValueError('A spectra-only bundle fitted on the supplied train partition is required')
    train, test = (np.asarray(task['train'], int), np.asarray(task['test'], int))
    if not len(train) or not len(test) or set(train).intersection(test):
        raise ValueError('Nonempty disjoint train and test bottle indices are required')
    if len(set(train)) != len(train) or len(set(test)) != len(test):
        raise ValueError('Duplicate train/test indices are not permitted')
    if 'ethanol' not in data:
        raise ValueError('Separately measured ethanol is required by ABV-assisted comparators')
    ethanol = np.asarray(data['ethanol'], float)
    if ethanol.ndim != 1 or len(ethanol) != len(data['Y']) or (not np.isfinite(ethanol[np.r_[train, test]]).all()):
        raise ValueError('Finite separately measured ethanol must match train and test bottles')
    Y = np.asarray(data['Y'], float)[train]
    if Y.ndim != 2 or not np.isfinite(Y).all():
        raise ValueError('Training reference concentrations must be finite matrices')
    X, Xt = (subset(data['X'], train), subset(data['X'], test))
    E, Et = (ethanol[train], ethanol[test])
    scale = iqr(Y)
    seed = int(task['seed'])
    spectral = bundle.predict_methods(Xt, methods=list(SPECTRAL_METHODS))
    for name in SPECTRAL_METHODS:
        spectral[name] = _prediction(spectral[name], (len(test), Y.shape[1]), name)
    baseline_oof, splits, baseline_audit = _baseline_oof(bundle, X, Y, p, seed, cfg)
    only_oof = np.full_like(Y, np.nan)
    memberships = []
    for fold, (tr, va) in enumerate(splits):
        only_oof[va] = _ABVOnly(Y[tr], E[tr]).predict(E[va])
        memberships.append({'fold': fold, 'train': tr.tolist(), 'validation': va.tolist(), 'global_train_bottle_indices': train[tr].tolist(), 'global_validation_bottle_indices': train[va].tolist(), 'nuisance_and_spectral_learning_excludes_validation': True})
    _prediction(only_oof, Y.shape, 'ethanol baseline OOF')
    catalog = build_residual_catalog()
    accepted, failures = ([], [])
    for spec in catalog:
        candidate_oof, fold_errors = (np.full_like(Y, np.nan), [])
        for fold, (tr, va) in enumerate(splits):
            try:
                model = _ABVResidual(spec, subset(X, tr), Y[tr], E[tr], seed + 5101 + fold)
                candidate_oof[va] = _prediction(model.predict(subset(X, va), E[va]), Y[va].shape, 'residual candidate OOF')
            except Exception as exc:
                fold_errors.append({'fold': fold, 'error': repr(exc)})
        if fold_errors:
            failures.append({'id': digest(spec), 'spec': spec, 'stage': 'strict_candidate_oof', 'errors': fold_errors})
            continue
        try:
            fitted = _ABVResidual(spec, X, Y, E, seed)
            _prediction(fitted.predict(X, E), Y.shape, 'residual candidate full-training check')
        except Exception as exc:
            failures.append({'id': digest(spec), 'spec': spec, 'stage': 'full_training_refit', 'error': repr(exc)})
            continue
        accepted.append({'id': digest(spec), 'spec': spec, 'oof': candidate_oof, 'model': fitted, 'score_targetwise': np.mean(((candidate_oof - Y) / scale) ** 2, axis=0)})
    only_model = _ABVOnly(Y, E)
    only_prediction = _prediction(only_model.predict(Et), (len(test), Y.shape[1]), 'ethanol-only')
    if accepted:
        routes = [min(range(len(accepted)), key=lambda k: (round(float(accepted[k]['score_targetwise'][j]), 12), accepted[k]['id'])) for j in range(Y.shape[1])]
        residual_oof = np.column_stack([accepted[k]['oof'][:, j] for j, k in enumerate(routes)])
        used = sorted(set(routes))
        candidate_predictions = {k: _prediction(accepted[k]['model'].predict(Xt, Et), (len(test), Y.shape[1]), 'frozen residual test prediction') for k in used}
        residual_prediction = np.column_stack([candidate_predictions[k][:, j] for j, k in enumerate(routes)])
    else:
        routes = [-1] * Y.shape[1]
        residual_oof, residual_prediction = (only_oof.copy(), only_prediction.copy())
    guard_oof = [baseline_oof, only_oof, residual_oof]
    guard_order = [LEGACY_METHOD, ABV_METHODS[0], ABV_METHODS[1]]
    guard_scores = np.column_stack([np.mean(((value - Y) / scale) ** 2, axis=0) for value in guard_oof])
    guard_routes = [min(range(3), key=lambda k: (round(float(guard_scores[j, k]), 12), k)) for j in range(Y.shape[1])]
    guard_pool = [spectral[LEGACY_METHOD], only_prediction, residual_prediction]
    guarded = np.column_stack([guard_pool[k][:, j] for j, k in enumerate(guard_routes)])
    predictions = dict(spectral)
    predictions.update({ABV_METHODS[0]: only_prediction, ABV_METHODS[1]: residual_prediction, ABV_METHODS[2]: guarded})
    frames = {'regression': pd.concat([_records(data, task, name, predictions[name], scale) for name in POINT_METHODS], ignore_index=True)}
    measured_audit = {'requires_measured_ethanol': True, 'training_bottles': len(train), 'training_sample_ids': [str(data['ids'][i]) for i in train], 'target_names': list(data['target_names']), 'degree': 2, 'nuisance_ridge_alpha': 1.0, 'both_X_and_Y_residualized': True, 'causal_interpretation': False, 'residual_catalog_size': len(catalog), 'fixed_residual_catalog': catalog, 'candidates': [{'id': a['id'], 'spec': a['spec'], 'score_targetwise': a['score_targetwise'].tolist(), 'metadata': a['model'].metadata} for a in accepted], 'candidate_failures': failures, 'residual_routes': routes, 'residual_candidate_ids_by_target': [accepted[k]['id'] if k >= 0 else None for k in routes], 'residual_all_failed_fallback_to_abv_only': not bool(accepted), 'guard_route_order': guard_order, 'guard_route_indices': guard_routes, 'guard_routes_by_target': [guard_order[k] for k in guard_routes], 'guard_scores_targetwise': guard_scores.tolist(), 'baseline_strict_procedure': baseline_audit, 'inner_partitions': memberships, 'no_test_reference_in_any_learning': True, 'no_test_spectra_in_model_selection': True, 'outer_test_ethanol_is_prediction_input_only': True, 'selection_oof_is_not_independent_performance': True, 'measured_abv_intervals_provided': False, 'spectral_predictions_depend_on_test_ethanol': False, 'residual_spec_selection_and_guard_learning': 'training OOF selection; whole guarded procedure evaluated only by outer test', 'prediction_clipping': False}
    return (frames, {'bundle': bundle.audit, 'measured_abv': measured_audit, 'spectral_bundle_reused': True, 'validation_tier': 'retrospective_internal_only'})
