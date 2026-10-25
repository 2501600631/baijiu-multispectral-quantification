from __future__ import annotations
import copy
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from .common import COMBINATIONS, digest, iqr, subset
from .models import fit_model
from .search import candidates
SPECTRAL_METHODS = ('v8_shared_pls2_reference', 'v8_shared_catalog_best', 'v8_target_adaptive', 'v8_conservative_target_adaptive', 'v8_shrunk_ensemble')
ABV_METHODS = ('v8_measured_abv_only', 'v8_measured_abv_residual', 'v8_measured_abv_guarded')

def build_catalog(profile):
    smoke = bool(profile.get('is_smoke'))
    catalog = []
    reference = [s for s in candidates(COMBINATIONS['all'], profile, full=False) if s['family'] == 'pls2']
    catalog.extend(reference)
    keys = ['all', 'uv'] if smoke else list(COMBINATIONS)
    for key in keys:
        options = candidates(COMBINATIONS[key], profile, full=False)
        for s in options:
            limit = 1 if smoke else profile['candidates_per_family'] if s['family'] == 'pls2' else 2
            if s['family'] in ('pls2', 'ridge', 'kernel_ridge'):
                same = [x for x in catalog if x['family'] == s['family'] and x['modes'] == s['modes']]
                if len(same) < limit:
                    catalog.append(s)
    if not smoke:
        for s in reference:
            independent = copy.deepcopy(s)
            independent['family'] = 'pls1'
            catalog.append(independent)
        for key in ['all', 'raman_uv', 'nir_uv']:
            for nc, pre in [(4, {'nir': 'smooth_snv', 'raman': 'derivative', 'uv': 'smooth'}), (8, {'nir': 'snv_derivative', 'raman': 'smooth_snv', 'uv': 'raw'}), (12, {'nir': 'derivative', 'raman': 'raw', 'uv': 'smooth'})]:
                s = copy.deepcopy(reference[0])
                s['modes'] = COMBINATIONS[key]
                s['pre'] = pre
                s['params']['n_components'] = nc
                catalog.append(s)
    unique = {digest(s): s for s in catalog}
    reference_ids = {digest(s) for s in reference}
    return ([unique[k] for k in sorted(unique)], reference_ids)

def _validated_prediction(model, X, shape, ethanol=None):
    pred = np.asarray(model.predict(X, ethanol), float)
    if pred.shape != shape or not np.isfinite(pred).all():
        raise ValueError('Wrong regression prediction shape or nonfinite value')
    return pred

def _normalized_loss(pred, actual, training):
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        loss = np.sqrt(np.mean(((pred - actual) / iqr(training)) ** 2, axis=0))
    if not np.isfinite(loss).all():
        raise ValueError('Nonfinite normalized regression loss')
    return loss

def _fixed_candidate_oof(X, Y, catalog, splits, seed):
    accepted, failures = ([], [])
    membership = np.full(len(Y), -1, int)
    for fold, (_, va) in enumerate(splits):
        membership[va] = fold
    for spec in catalog:
        oof = np.empty_like(Y, dtype=float)
        losses, errors = ([], [])
        for fold, (tr, va) in enumerate(splits):
            try:
                model = fit_model(spec, subset(X, tr), Y[tr], seed=seed + fold)
                pred = _validated_prediction(model, subset(X, va), Y[va].shape)
                oof[va] = pred
                losses.append(_normalized_loss(pred, Y[va], Y[tr]))
            except Exception as exc:
                errors.append({'fold': fold, 'stage': 'inner_fit_or_predict', 'error': repr(exc)})
        if errors:
            failures.append({'spec': spec, 'errors': errors})
        else:
            accepted.append({'id': digest(spec), 'spec': spec, 'oof': oof, 'fold_scores': np.asarray(losses)})
    return (accepted, failures, membership)

def _simplex_weights(predictions, Y, reference_index, penalty):
    n, m, c = predictions.shape
    scale = iqr(Y)
    anchor = np.zeros(c)
    anchor[reference_index] = 1.0
    weights = np.zeros((m, c))
    outcomes = []
    for j in range(m):
        design = (predictions[:, j, :] - Y[:, j, None]) / scale[j]
        gram = design.T @ design / n

        def objective(w):
            d = w - anchor
            return float(w @ gram @ w + penalty * (d @ d))

        def gradient(w):
            return 2.0 * (gram @ w + penalty * (w - anchor))
        result = minimize(objective, anchor.copy(), jac=gradient, method='SLSQP', bounds=[(0.0, 1.0)] * c, constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1.0, 'jac': lambda w: np.ones_like(w)}, options={'maxiter': 500, 'ftol': 1e-10})
        good = result.success and np.isfinite(result.x).all() and (result.x.min() >= -1e-08) and (abs(result.x.sum() - 1.0) <= 1e-07)
        if good:
            w = np.maximum(result.x, 0.0)
            w[w < 1e-07] = 0.0
            w /= w.sum()
        else:
            w = anchor.copy()
        weights[j] = w
        outcomes.append({'success': bool(good), 'message': str(result.message), 'iterations': int(result.nit), 'objective': objective(w), 'reference_objective': objective(anchor), 'numerical_reference_fallback': not bool(good)})
    return (weights, outcomes)

def _routes(accepted, Y, reference_ids, penalty):
    joint = np.array([a['fold_scores'].mean() for a in accepted])
    choices = [i for i, a in enumerate(accepted) if a['id'] in reference_ids]
    if not choices:
        raise RuntimeError('All required shared PLS2 reference candidates failed')
    ref = min(choices, key=lambda i: (round(float(joint[i]), 12), accepted[i]['id']))
    losses = np.stack([a['fold_scores'] for a in accepted], axis=-1)
    scores = losses.mean(axis=0)
    minimum = np.array([min(range(len(accepted)), key=lambda i: (round(float(scores[j, i]), 12), accepted[i]['id'])) for j in range(Y.shape[1])], int)
    guarded, rule = (minimum.copy(), [])
    for j, selected in enumerate(minimum):
        improvement = losses[:, j, ref] - losses[:, j, selected]
        se = float(improvement.std(ddof=1) / np.sqrt(len(improvement)))
        use = float(improvement.mean()) > se + 1e-12
        if not use:
            guarded[j] = ref
        rule.append({'target_index': j, 'inner_mean_normalized_rmse_improvement': float(improvement.mean()), 'paired_fold_standard_error': se, 'used_alternative': bool(use), 'interpretation': 'fixed conservative selection heuristic, not a significance test'})
    oof = np.stack([a['oof'] for a in accepted], axis=-1)
    weights, solver = _simplex_weights(oof, Y, ref, penalty)
    return (ref, minimum, guarded, weights, rule, solver)

class _ABVResidual:

    def __init__(self, spec, X, Y, ethanol, seed):
        self.spec = copy.deepcopy(spec)
        e = np.asarray(ethanol, float).reshape(-1, 1)
        if len(e) != len(Y) or not np.isfinite(e).all():
            raise ValueError('Measured ethanol must be finite and match training rows')
        self.poly = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), StandardScaler())
        q = self.poly.fit_transform(e)
        self.target_nuisance = Ridge(alpha=1.0).fit(q, Y)
        self.spectral_nuisance = {mode: Ridge(alpha=1.0).fit(q, a) for mode, a in X.items()}
        residual_X = {mode: a - self.spectral_nuisance[mode].predict(q) for mode, a in X.items()}
        residual_Y = Y - self.target_nuisance.predict(q)
        self.model = fit_model(spec, residual_X, residual_Y, seed=seed)
        self.metadata = {'requires_measured_ethanol': True, 'degree': 2, 'nuisance_ridge_alpha': 1.0, 'training_bottles': len(Y), 'both_X_and_Y_residualized': True, 'causal_interpretation': False, 'residual_model': self.model.metadata}

    def nuisance(self, ethanol):
        if ethanol is None:
            raise ValueError('Separately measured ethanol is required by the ABV model')
        q = self.poly.transform(np.asarray(ethanol, float).reshape(-1, 1))
        return (q, np.asarray(self.target_nuisance.predict(q)))

    def predict(self, X, ethanol=None):
        q, base = self.nuisance(ethanol)
        residual_X = {mode: np.asarray(a) - self.spectral_nuisance[mode].predict(q) for mode, a in X.items()}
        return base + self.model.predict(residual_X)

class FittedBundle:

    def predict_methods(self, X, ethanol=None):
        cached = {i: _validated_prediction(model, X, (len(next(iter(X.values()))), self.n_targets)) for i, model in self.models.items()}
        n = len(next(iter(X.values())))

        def routed(indices):
            return np.column_stack([cached[int(i)][:, j] for j, i in enumerate(indices)])
        results = {'v8_shared_pls2_reference': cached[self.reference], 'v8_shared_catalog_best': cached[self.shared_best], 'v8_target_adaptive': routed(self.minimum), 'v8_conservative_target_adaptive': routed(self.guarded)}
        ensemble = np.zeros((n, self.n_targets))
        for i, pred in cached.items():
            ensemble += pred * self.weights[:, i][None, :]
        results['v8_shrunk_ensemble'] = ensemble
        if self.abv_model is not None and ethanol is not None:
            _, base = self.abv_model.nuisance(ethanol)
            residual = self.abv_model.predict(X, ethanol)
            results['v8_measured_abv_only'] = base
            results['v8_measured_abv_residual'] = residual
            pool = [base, residual, results['v8_shared_pls2_reference']]
            results['v8_measured_abv_guarded'] = np.column_stack([pool[int(i)][:, j] for j, i in enumerate(self.abv_routes)])
        return results

    def predict(self, X, ethanol=None):
        return self.predict_methods(X, ethanol)['v8_shrunk_ensemble']

def fit_bundle(X, Y, profile, seed, ethanol=None, options=None):
    options = {} if options is None else dict(options)
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    iqr(Y)
    k = min(int(profile['inner_folds']), len(Y) // 4)
    if k < 2:
        raise ValueError('Too few training bottles for target-adaptive inner CV')
    penalty = float(options.get('ensemble_penalty', 0.1))
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError('Simplex reference penalty must be finite and nonnegative')
    splits = list(KFold(k, shuffle=True, random_state=seed).split(Y))
    catalog, reference_ids = build_catalog(profile)
    accepted, failures, membership = _fixed_candidate_oof(X, Y, catalog, splits, seed)
    if not accepted:
        raise RuntimeError('All optimized regression candidates failed: ' + repr(failures))
    models, final_cache = ({}, {})
    while True:
        ref, minimum, guarded, weights, rule, solver = _routes(accepted, Y, reference_ids, penalty)
        shared_best = min(range(len(accepted)), key=lambda i: (round(float(accepted[i]['fold_scores'].mean()), 12), accepted[i]['id']))
        needed = set([ref, shared_best] + minimum.tolist() + guarded.tolist() + np.flatnonzero(weights.max(axis=0) > 0).tolist())
        broken = []
        for i in sorted(needed):
            try:
                candidate_id = accepted[i]['id']
                if candidate_id in final_cache:
                    model = final_cache[candidate_id]
                else:
                    model = fit_model(accepted[i]['spec'], X, Y, seed=seed)
                    _validated_prediction(model, X, Y.shape)
                    final_cache[candidate_id] = model
                models[i] = model
            except Exception as exc:
                failures.append({'spec': accepted[i]['spec'], 'stage': 'full_training_refit', 'error': repr(exc)})
                broken.append(i)
        if not broken:
            break
        accepted = [a for i, a in enumerate(accepted) if i not in broken]
        models = {}
    bundle = FittedBundle()
    bundle.reference, bundle.minimum, bundle.guarded = (ref, minimum, guarded)
    bundle.shared_best = shared_best
    bundle.weights, bundle.models, bundle.n_targets = (weights, models, Y.shape[1])
    bundle.abv_model, bundle.abv_routes = (None, None)
    audit = {'reference_definition': 'Exact V7 all-view PLS2 catalog and inner splits; joint train-IQR NRMSE selection', 'reference_candidate': accepted[ref]['id'], 'shared_catalog_candidate': accepted[shared_best]['id'], 'shared_catalog_score': float(accepted[shared_best]['fold_scores'].mean()), 'shared_catalog_control': 'One joint model selected from the exact same accepted fixed candidate specifications as target adaptation', 'minimum_routes': [accepted[i]['id'] for i in minimum], 'conservative_routes': [accepted[i]['id'] for i in guarded], 'conservative_rule': rule, 'simplex_penalty': penalty, 'simplex_solver': solver, 'simplex_weights_target_by_candidate': weights.tolist(), 'candidate_order': [a['id'] for a in accepted], 'oof_membership_in_supplied_training_order': membership.tolist(), 'inner_splits': [{'fit': tr.tolist(), 'validation': va.tolist()} for tr, va in splits], 'candidates': [{'id': a['id'], 'spec': a['spec'], 'fold_scores_targetwise': a['fold_scores'].tolist(), 'score': float(a['fold_scores'].mean()), 'oof_predictions': a['oof'].tolist()} for a in accepted], 'failures': failures, 'fitted_model_metadata': {accepted[i]['id']: m.metadata for i, m in models.items()}, 'selection_values_are_performance_estimates': False, 'no_outer_test_rows_used_in_fit_or_selection': True, 'baseline_eligible_does_not_guarantee_outer_improvement': True, 'fit_cost_upper_bound_spectral': len(catalog) * k + len(catalog), 'catalog_size': len(catalog), 'training_bottles': len(Y), 'inner_folds': k}
    if options.get('include_abv', True) and ethanol is not None:
        e = np.asarray(ethanol, float)
        residual_catalog = [s for s in catalog if digest(s) in reference_ids]
        residual_valid, residual_failures = ([], [])
        base_oof = np.empty_like(Y)
        for spec in residual_catalog:
            pred = np.empty_like(Y)
            losses = []
            errors = []
            for fold, (tr, va) in enumerate(splits):
                try:
                    model = _ABVResidual(spec, subset(X, tr), Y[tr], e[tr], seed + fold)
                    pred[va] = _validated_prediction(model, subset(X, va), Y[va].shape, e[va])
                    _, base_oof[va] = model.nuisance(e[va])
                    losses.append(_normalized_loss(pred[va], Y[va], Y[tr]))
                except Exception as exc:
                    errors.append({'fold': fold, 'error': repr(exc)})
            if errors:
                residual_failures.append({'spec': spec, 'errors': errors})
            else:
                residual_valid.append({'spec': spec, 'oof': pred, 'loss': np.asarray(losses)})
        for row in sorted(residual_valid, key=lambda a: (round(float(a['loss'].mean()), 12), digest(a['spec']))):
            try:
                final_abv = _ABVResidual(row['spec'], X, Y, e, seed)
                _validated_prediction(final_abv, X, Y.shape, e)
                residual_oof = row['oof']
                pool = [base_oof, residual_oof, accepted[ref]['oof']]
                target_scores = np.zeros((Y.shape[1], 3))
                for tr, va in splits:
                    for c, pred in enumerate(pool):
                        target_scores[:, c] += _normalized_loss(pred[va], Y[va], Y[tr]) / k
                bundle.abv_routes = np.array([min([2, 0, 1], key=lambda c: (round(float(target_scores[j, c]), 12), [2, 0, 1].index(c))) for j in range(Y.shape[1])])
                bundle.abv_model = final_abv
                audit['measured_abv'] = {'requires_measured_ethanol': True, 'selected_spec': row['spec'], 'target_route_indices': bundle.abv_routes.tolist(), 'route_order': ['abv_only', 'abv_residual', 'spectra_reference'], 'targetwise_inner_scores': target_scores.tolist(), 'metadata': bundle.abv_model.metadata, 'candidates': [{'spec': a['spec'], 'fold_scores_targetwise': a['loss'].tolist(), 'oof_predictions': a['oof'].tolist()} for a in residual_valid], 'abv_only_oof_predictions': base_oof.tolist(), 'failures': residual_failures, 'confounding_eliminated': False, 'causal_interpretation': False}
                break
            except Exception as exc:
                residual_failures.append({'spec': row['spec'], 'stage': 'full_training_refit', 'error': repr(exc)})
        if bundle.abv_model is None:
            raise RuntimeError('All required measured-ABV residual candidates failed: ' + repr(residual_failures))
    bundle.audit = audit
    return bundle

def run_task(data, task, profile, cfg):
    from .engine import regression_records
    tr, te = (np.asarray(task['train'], int), np.asarray(task['test'], int))
    options = cfg.get('optimized_regression', {})
    bundle = fit_bundle(subset(data['X'], tr), np.asarray(data['Y'])[tr], profile, task['seed'], ethanol=np.asarray(data['ethanol'])[tr], options=options)
    predictions = bundle.predict_methods(subset(data['X'], te), np.asarray(data['ethanol'])[te])
    frames = [regression_records(data, te, pred, method, iqr(np.asarray(data['Y'])[tr]), task) for method, pred in predictions.items()]
    bundle.audit['training_sample_ids'] = [str(data['ids'][i]) for i in tr]
    bundle.audit['target_names'] = list(data['target_names'])
    return ({'regression': pd.concat(frames, ignore_index=True)}, bundle.audit)
