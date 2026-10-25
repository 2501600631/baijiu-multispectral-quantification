from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from .common import digest, iqr, subset
from .integrated_models import fit_model
from .integrated_catalog import build_integrated_catalog
from .optimized_regression import build_catalog as build_legacy_catalog, _routes, _simplex_weights, _validated_prediction
BASE_METHODS = ('v10_pls2_reference', 'v10_shared_catalog_best', 'v10_target_adaptive', 'v10_conservative_adaptive', 'v10_equal_weight', 'v10_unanchored_ensemble', 'v10_anchored_ensemble')
LEGACY_METHOD = 'v10_v9_unanchored_reference'
LATE_METHOD = 'v10_single_view_fusion'
ROUTER_METHOD = 'v10_strategy_router'
STRATEGY_METHODS = BASE_METHODS + (LEGACY_METHOD, LATE_METHOD)
FINAL_METHODS = STRATEGY_METHODS + (ROUTER_METHOD,)
ANCHOR_PENALTY = 0.1

def _fixed_candidate_oof(X, Y, catalog, splits, seed):
    accepted, failures = ([], [])
    membership = np.full(len(Y), -1, int)
    for fold, (_, va) in enumerate(splits):
        membership[va] = fold
    for spec in catalog:
        oof = np.full_like(Y, np.nan)
        losses = []
        errors = []
        for fold, (tr, va) in enumerate(splits):
            try:
                model = fit_model(spec, subset(X, tr), Y[tr], seed=seed + fold)
                pred = _validated_prediction(model, subset(X, va), Y[va].shape)
                oof[va] = pred
                losses.append(np.sqrt(np.mean(((pred - Y[va]) / iqr(Y[tr])) ** 2, axis=0)))
            except Exception as exc:
                errors.append({'fold': fold, 'stage': 'inner_fit_or_predict', 'error': repr(exc)})
        if errors:
            failures.append({'spec': spec, 'errors': errors})
        else:
            accepted.append({'id': digest(spec), 'spec': spec, 'oof': oof, 'fold_scores': np.asarray(losses)})
    return (accepted, failures, membership)

def _requested_methods(methods):
    selected = list(FINAL_METHODS) if methods is None else list(methods)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError('Methods must be a nonempty list without duplicates')
    unknown = set(selected) - set(FINAL_METHODS)
    if unknown:
        raise ValueError('Unknown final-system methods: ' + repr(sorted(unknown)))
    return selected

def _training_arrays(X, Y):
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.ndim != 2 or not np.isfinite(Y).all() or (not X):
        raise ValueError('Finite two-dimensional targets and spectral blocks are required')
    arrays = {mode: np.asarray(values, dtype=float) for mode, values in X.items()}
    if any((a.ndim != 2 or len(a) != len(Y) or (not np.isfinite(a).all()) for a in arrays.values())):
        raise ValueError('Spectral blocks must be finite matrices with matching rows')
    iqr(Y)
    return (arrays, Y)

class FinalBundle:
    method_names = FINAL_METHODS

    def predict_methods(self, X, methods=None):
        wanted = _requested_methods(methods)
        n = len(next(iter(X.values())))
        cached = {i: _validated_prediction(model, X, (n, self.n_targets)) for i, model in self.models.items()}
        predictions = np.stack([cached[i] for i in range(len(self.candidate_ids))], axis=-1)

        def routed(indices):
            return np.column_stack([cached[int(i)][:, j] for j, i in enumerate(indices)])
        results = {'v10_pls2_reference': cached[self.reference], 'v10_shared_catalog_best': cached[self.shared_best], 'v10_target_adaptive': routed(self.minimum), 'v10_conservative_adaptive': routed(self.guarded), 'v10_equal_weight': predictions.mean(axis=-1), 'v10_unanchored_ensemble': np.einsum('nmc,mc->nm', predictions, self.unanchored_weights), 'v10_anchored_ensemble': np.einsum('nmc,mc->nm', predictions, self.anchored_weights)}
        results[LEGACY_METHOD] = np.einsum('nmc,mc->nm', predictions, self.legacy_weights)
        results[LATE_METHOD] = sum((weight * cached[index] for index, weight in zip(self.single_view_indices, self.single_view_weights)))
        results[ROUTER_METHOD] = np.column_stack([results[name][:, j] for j, name in enumerate(self.strategy_routes)])
        return {name: results[name] for name in wanted}

    def predict(self, X):
        return self.predict_methods(X, [ROUTER_METHOD])[ROUTER_METHOD]

def fit_final_bundle(X, Y, profile, seed, options=None):
    options = {} if options is None else dict(options)
    penalty = float(options.get('ensemble_penalty', ANCHOR_PENALTY))
    if not np.isfinite(penalty) or abs(penalty - ANCHOR_PENALTY) > 1e-15:
        raise ValueError('V10 anchor penalty is frozen at 0.1; do not tune it on test/calibration data')
    X, Y = _training_arrays(X, Y)
    k = min(int(profile['inner_folds']), len(Y) // 4)
    if k < 2:
        raise ValueError('Too few fitting bottles for inner candidate cross-validation')
    splits = list(KFold(k, shuffle=True, random_state=int(seed)).split(Y))
    catalog, reference_ids, library_manifest = build_integrated_catalog(profile, options)
    oof_success, failures, membership = _fixed_candidate_oof(X, Y, catalog, splits, int(seed))
    if not oof_success:
        raise RuntimeError('All final-system inner candidates failed: ' + repr(failures))
    accepted, fitted_models, full_status = ([], [], [])
    for row in oof_success:
        try:
            model = fit_model(row['spec'], X, Y, seed=int(seed))
            _validated_prediction(model, X, Y.shape)
            accepted.append(row)
            fitted_models.append(model)
            full_status.append({'id': row['id'], 'status': 'accepted'})
        except Exception as exc:
            failure = {'id': row['id'], 'spec': row['spec'], 'stage': 'full_training_refit', 'error': repr(exc)}
            failures.append(failure)
            full_status.append({'id': row['id'], 'status': 'rejected', 'error': repr(exc)})
    if not accepted:
        raise RuntimeError('All final-system full-training refits failed: ' + repr(failures))
    reference, minimum, guarded, anchored, rule, anchored_solver = _routes(accepted, Y, reference_ids, ANCHOR_PENALTY)
    shared_best = min(range(len(accepted)), key=lambda i: (round(float(accepted[i]['fold_scores'].mean()), 12), accepted[i]['id']))
    oof_predictions = np.stack([row['oof'] for row in accepted], axis=-1)
    unanchored, unanchored_solver = _simplex_weights(oof_predictions, Y, reference, 0.0)
    candidate_ids = [row['id'] for row in accepted]
    old_catalog, _ = build_legacy_catalog(profile)
    old_ids = {digest(s) for s in old_catalog}
    old_indices = [i for i, a in enumerate(accepted) if a['id'] in old_ids]
    if not old_indices:
        raise RuntimeError('Original V9 candidate pool entirely failed')
    old_reference = old_indices.index(reference)
    old_weights, _old_solver = _simplex_weights(oof_predictions[:, :, old_indices], Y, old_reference, 0.0)
    legacy_weights = np.zeros_like(anchored)
    legacy_weights[:, old_indices] = old_weights
    equal_weights = np.full_like(anchored, 1.0 / len(accepted))
    bundle = FinalBundle()
    bundle.reference, bundle.shared_best = (reference, shared_best)
    bundle.minimum, bundle.guarded = (minimum, guarded)
    bundle.models = dict(enumerate(fitted_models))
    bundle.candidate_ids, bundle.n_targets = (candidate_ids, Y.shape[1])
    bundle.unanchored_weights, bundle.anchored_weights = (unanchored, anchored)
    bundle.legacy_weights = legacy_weights
    single_indices = []
    single_modes = []
    for mode in ('nir', 'raman', 'uv'):
        eligible = [i for i, a in enumerate(accepted) if a['spec']['modes'] == [mode]]
        if eligible:
            single_indices.append(min(eligible, key=lambda i: (round(float(accepted[i]['fold_scores'].mean()), 12), accepted[i]['id'])))
            single_modes.append(mode)
    if not single_indices:
        single_indices = [reference]
        single_modes = ['reference_fallback']
    d = (oof_predictions[:, :, single_indices] - Y[:, :, None]) / iqr(Y)[None, :, None]
    gram = d.reshape(-1, len(single_indices)).T @ d.reshape(-1, len(single_indices)) / Y.size
    initial = np.ones(len(single_indices)) / len(single_indices)
    result = minimize(lambda w: float(w @ gram @ w), initial, jac=lambda w: 2 * gram @ w, method='SLSQP', bounds=[(0.0, 1.0)] * len(initial), constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1.0}, options={'maxiter': 500, 'ftol': 1e-10})
    weights = np.maximum(result.x, 0.0) if result.success and np.isfinite(result.x).all() else initial
    weights = weights / weights.sum()
    bundle.single_view_indices = single_indices
    bundle.single_view_weights = weights
    bundle.strategy_routes = ['v10_pls2_reference'] * Y.shape[1]
    bundle.weights = anchored
    successful = {row['id'] for row in accepted}

    def candidate_audit(row):
        return {'id': row['id'], 'spec': row['spec'], 'fold_scores_targetwise': row['fold_scores'].tolist(), 'score': float(row['fold_scores'].mean()), 'oof_predictions': row['oof'].tolist(), 'final_pool_accepted': row['id'] in successful}
    bundle.metadata = {'procedure': 'V10 matched-library fixed-penalty ensemble ablation', 'family': 'convex_ensemble', 'training_bottles': len(Y), 'train_n': len(Y), 'n_targets': Y.shape[1], 'catalog_size': len(catalog), 'final_candidate_count': len(accepted), 'anchor_penalty': ANCHOR_PENALTY, 'default_prediction': ROUTER_METHOD, 'requires_measured_ethanol': False, 'methods': list(FINAL_METHODS)}
    bundle.audit = {'version': '10.0.0', 'training_bottles': len(Y), 'inner_folds': k, 'target_count': Y.shape[1], 'catalog_size': len(catalog), 'catalog_specifications': [{'id': digest(s), 'spec': s} for s in catalog], 'candidate_order': candidate_ids, 'common_final_candidate_ids': candidate_ids, 'final_candidate_count': len(accepted), 'all_three_ensemble_methods_share_identical_candidate_pool': True, 'reference_definition': 'Unchanged V8 all-view PLS2 fixed specifications; inner joint train-IQR NRMSE selection', 'reference_eligible_candidate_ids': [i for i in candidate_ids if i in reference_ids], 'reference_candidate': accepted[reference]['id'], 'shared_catalog_candidate': accepted[shared_best]['id'], 'shared_catalog_score': float(accepted[shared_best]['fold_scores'].mean()), 'minimum_routes': [accepted[i]['id'] for i in minimum], 'conservative_routes': [accepted[i]['id'] for i in guarded], 'conservative_rule': rule, 'penalties_by_method': {'v10_unanchored_ensemble': 0.0, 'v10_anchored_ensemble': ANCHOR_PENALTY}, 'weights_by_method': {'v10_equal_weight': equal_weights.tolist(), 'v10_unanchored_ensemble': unanchored.tolist(), 'v10_anchored_ensemble': anchored.tolist()}, 'simplex_penalty': ANCHOR_PENALTY, 'simplex_weights_target_by_candidate': anchored.tolist(), 'anchored_weights_target_by_candidate': anchored.tolist(), 'unanchored_weights_target_by_candidate': unanchored.tolist(), 'simplex_solver': anchored_solver, 'unanchored_solver': unanchored_solver, 'solver_by_method': {'v10_unanchored_ensemble': unanchored_solver, 'v10_anchored_ensemble': anchored_solver}, 'oof_membership_in_supplied_training_order': membership.tolist(), 'inner_splits': [{'fit': tr.tolist(), 'validation': va.tolist()} for tr, va in splits], 'candidates': [candidate_audit(row) for row in accepted], 'excluded_after_successful_inner_oof': [candidate_audit(row) for row in oof_success if row['id'] not in successful], 'full_training_refit_status': full_status, 'failures': failures, 'fitted_model_metadata': {row['id']: model.metadata for row, model in zip(accepted, fitted_models)}, 'fit_budget': {'catalog_candidate_count': len(catalog), 'inner_fit_attempts': len(catalog) * k, 'full_fit_attempts': len(oof_success), 'full_fit_successes': len(accepted), 'full_fit_cache_reused_across_methods': True, 'full_fit_repetitions_per_candidate': 1}, 'selection_values_are_performance_estimates': False, 'no_outer_test_rows_used_in_fit_or_selection': True, 'no_calibration_rows_used_in_fit_or_selection': True, 'baseline_eligible_does_not_guarantee_outer_improvement': True, 'inner_oof_weight_training_predictions_are_not_strict_system_oof': True, 'simplex_penalties_selected_from_outer_test_or_calibration': False}
    bundle.audit['integrated_library'] = library_manifest
    bundle.audit['legacy_comparator'] = {'method': LEGACY_METHOD, 'candidate_ids': [candidate_ids[i] for i in old_indices], 'weights': legacy_weights.tolist(), 'solver': _old_solver, 'same_original_V9_specs_and_inner_seed': True}
    bundle.audit['single_view_fusion'] = {'method': LATE_METHOD, 'modes': single_modes, 'selected_candidates': [candidate_ids[i] for i in single_indices], 'weights': weights.tolist(), 'solver_success': bool(result.success), 'training_data_only': True, 'candidate_oof_is_weight_learning_data_not_system_performance': True, 'exact_historical_V7_strict_specialist_meta_crossfit_reproduced': False, 'interpretation': 'Single-view fusion concept; full procedure strict crossfit used for strategy selection and intervals'}
    if options.get('enable_strategy_router', True):
        _learn_strategy_router(bundle, X, Y, profile, seed, options)
    else:
        bundle.audit['strategy_router'] = {'enabled': False, 'routes': bundle.strategy_routes, 'reason': 'internal fixed-procedure crossfit; no recursive strategy selection'}
    return bundle

def _learn_strategy_router(bundle, X, Y, profile, seed, options):
    k = min(int(profile.get('strategy_folds', options.get('strategy_folds', 3))), len(Y) // 8)
    if k < 2:
        bundle.audit['strategy_router'] = {'enabled': True, 'fallback': True, 'routes': bundle.strategy_routes, 'reason': 'fewer than16fitting bottles; fixed PLS2 fallback before observing test labels'}
        return
    predictions = {name: np.full_like(Y, np.nan) for name in STRATEGY_METHODS}
    receipts = []
    nested_options = dict(options, enable_strategy_router=False)
    for fold, (tr, va) in enumerate(KFold(k, shuffle=True, random_state=int(seed) + 3901).split(Y)):
        fitted = fit_final_bundle(subset(X, tr), Y[tr], profile, int(seed) + 4001 + fold * 101, options=nested_options)
        values = fitted.predict_methods(subset(X, va), methods=list(STRATEGY_METHODS))
        for name in STRATEGY_METHODS:
            predictions[name][va] = values[name]
        receipts.append({'fold': fold, 'train': tr.tolist(), 'validation': va.tolist(), 'training_bottles': len(tr), 'all_selection_and_weights_exclude_validation': True, 'inner_splits': fitted.audit['inner_splits'], 'candidate_order': fitted.candidate_ids, 'selected_reference': fitted.audit['reference_candidate'], 'training_audit_digest': digest(fitted.audit)})
    scores = np.column_stack([np.mean(((predictions[name] - Y) / iqr(Y)) ** 2, axis=0) for name in STRATEGY_METHODS])
    if not np.isfinite(scores).all():
        raise RuntimeError('Incomplete full-procedure strategy OOF')
    priority = ['v10_pls2_reference', LEGACY_METHOD, LATE_METHOD, 'v10_equal_weight', 'v10_unanchored_ensemble', 'v10_anchored_ensemble', 'v10_conservative_adaptive', 'v10_target_adaptive', 'v10_shared_catalog_best']
    bundle.strategy_routes = [min(priority, key=lambda name: (round(float(scores[j, STRATEGY_METHODS.index(name)]), 12), priority.index(name))) for j in range(Y.shape[1])]
    bundle.audit['strategy_router'] = {'enabled': True, 'fallback': False, 'methods': list(STRATEGY_METHODS), 'routes': bundle.strategy_routes, 'scores_targetwise': scores.tolist(), 'strict_procedure_oof': {name: a.tolist() for name, a in predictions.items()}, 'crossfit': receipts, 'metric': 'fit-IQR-normalized MSE of held-out entire fixed procedures; selection data only', 'route_winners_chosen_using_outer_test': False, 'history_metric_splicing': False}

def strict_crossfit_final(X, Y, profile, seed, methods=None, options=None):
    options = {} if options is None else dict(options)
    methods = _requested_methods(methods)
    X, Y = _training_arrays(X, Y)
    requested = int(options.get('crossfit_folds', profile['inner_folds']))
    k = min(requested, len(Y) // 8)
    if k < 2:
        raise ValueError('Strict whole-system cross-fitting requires at least 16 fitting bottles')
    result = {method: np.full(Y.shape, np.nan) for method in methods}
    audits, visits = ([], np.zeros(len(Y), dtype=int))
    splits = KFold(k, shuffle=True, random_state=int(seed))
    for fold, (tr, va) in enumerate(splits.split(Y)):
        fit_seed = int(seed) + 1009 * (fold + 1)
        model = fit_final_bundle(subset(X, tr), Y[tr], profile, fit_seed, options=options)
        predictions = model.predict_methods(subset(X, va), methods=methods)
        for method in methods:
            result[method][va] = predictions[method]
        visits[va] += 1
        audits.append({'fold': fold, 'seed': fit_seed, 'train': tr.tolist(), 'validation': va.tolist(), 'fit': tr.tolist(), 'training_indices_in_supplied_row_order': tr.tolist(), 'validation_indices_in_supplied_row_order': va.tolist(), 'entire_model_selection_and_weights_refitted': True, 'validation_rows_excluded_from_all_learning': True, 'nested_audit_indices_are_local_to_this_training_subset': True, 'audit': model.audit})
    if not np.all(visits == 1) or any((not np.isfinite(a).all() for a in result.values())):
        raise RuntimeError('Incomplete strict whole-system OOF partition')
    return (result, audits)
