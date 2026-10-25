from pathlib import Path
import json
import numpy as np
import pandas as pd
from .common import csv_read, digest, iqr, read_json, save_csv, save_json, sha
from .final_ensemble import FINAL_METHODS, STRATEGY_METHODS, LEGACY_METHOD, LATE_METHOD, ROUTER_METHOD
from .final_interval import POINT_METHODS, verify_frames
from .integrated_abv import POINT_METHODS as ABV_POINT_METHODS

def _bool(values):
    return np.asarray([str(v).strip().lower() == 'true' for v in values])

def _bundle(audit):
    if isinstance(audit, dict):
        if 'candidate_order' in audit and 'weights_by_method' in audit:
            return audit
        for key in ('bundle', 'point_model_selection', 'model_selection', 'audit'):
            if key in audit:
                found = _bundle(audit[key])
                if found is not None:
                    return found
    return None

def _collect(out, plan):
    frames, audits, errors = ({}, [], [])
    for task in plan['tasks']:
        checkpoint = Path(out) / 'checkpoints' / task['id']
        try:
            receipt = read_json(checkpoint / 'receipt.json')
            if receipt.get('task_id') != task['id'] or receipt.get('plan_hash') != plan['plan_hash']:
                raise ValueError('Task receipt plan or task identity mismatch')
            listed = receipt['files']
            actual = {p.name for p in checkpoint.glob('*.csv')} | {'audit.json'}
            if not actual.issubset(set(listed)):
                raise ValueError('Unreceipted checkpoint CSV or audit')
            for name, expected in listed.items():
                if Path(name).name != name or not (checkpoint / name).is_file() or sha(checkpoint / name) != expected:
                    raise ValueError('Checkpoint file digest mismatch: ' + str(name))
            local = {p.stem: csv_read(p, dtype={'sample_id': str}) for p in sorted(checkpoint.glob('*.csv'))}
            for name, frame in local.items():
                frames.setdefault(name, []).append(frame)
            audits.append((task, read_json(checkpoint / 'audit.json'), local))
        except Exception as exc:
            errors.append(task['id'] + ': ' + repr(exc))
    return ({name: pd.concat(parts, ignore_index=True) for name, parts in frames.items()}, audits, errors)

def _verify_local(data, task, local, mode, alpha):
    errors = []
    ids = np.asarray(data['ids']).astype(str)
    targets = list(data['target_names'])
    required = {'ablation': {'regression'}, 'ethanol': {'regression'}, 'measured_abv': {'regression'}, 'interval': {'regression', 'calibration_regression', 'interval_regression', 'risk_rules', 'risk_predictions'}, 'boundaries': {'regression', 'stress', 'occlusion', 'diagnostics'}}[mode]
    if not required.issubset(local):
        errors.append('Missing required frames: ' + repr(sorted(required - set(local))))
    tr, te = (np.asarray(task['train'], int), np.asarray(task['test'], int))
    cal = np.asarray(task.get('calibration', []), int)
    parts = [a for a in (tr, cal, te) if len(a)]
    if any((len(set(a)) != len(a) or np.any(a < 0) or np.any(a >= len(ids)) for a in parts)):
        return errors + ['Invalid or repeated partition indices']
    if any((set(parts[i]) & set(parts[j]) for i in range(len(parts)) for j in range(i))):
        errors.append('Overlapping fitting/calibration/test bottles')
    lookup = {(str(ids[i]), target): float(data['Y'][i, j]) for i in range(len(ids)) for j, target in enumerate(targets)}
    train_scale = dict(zip(targets, iqr(np.asarray(data['Y'])[tr])))
    for name, frame in local.items():
        if frame.empty:
            errors.append('Empty frame: ' + name)
            continue
        for key in ('repeat', 'fold'):
            if key not in frame or set(frame[key]) != {task[key]}:
                errors.append('Repeat/fold identity mismatch: ' + name)
        if 'fit_n' in frame and (not np.all(frame.fit_n == len(tr))):
            errors.append('Fitting bottle count mismatch: ' + name)
        if 'sample_id' not in frame:
            continue
        expected_ids = set(ids[cal if name == 'calibration_regression' else te])
        if set(frame.sample_id) != expected_ids or set(frame.target) != set(targets):
            errors.append('Bottle or target inventory mismatch: ' + name)
            continue
        keys = ['sample_id', 'target'] + [k for k in ('method', 'scenario', 'retained_fraction') if k in frame]
        if frame.duplicated(keys).any():
            errors.append('Duplicate evaluation keys: ' + name)
        groups = [k for k in ('method', 'scenario', 'retained_fraction', 'target') if k in frame]
        for _, group in frame.groupby(groups):
            if set(group.sample_id) != expected_ids:
                errors.append('Incomplete bottle grid: ' + name)
                break
        expected_truth = np.asarray([lookup.get((str(s), t), np.nan) for s, t in zip(frame.sample_id, frame.target)])
        if 'y_true' in frame and (not np.array_equal(frame.y_true.to_numpy(float), expected_truth)):
            errors.append('Reference values differ from raw matched data: ' + name)
        if 'y_pred' in frame and (not np.isfinite(frame.y_pred.to_numpy(float)).all()):
            errors.append('Nonfinite point prediction: ' + name)
        if 'train_iqr' in frame:
            expected_scale = np.asarray([train_scale[t] for t in frame.target])
            if not np.array_equal(frame.train_iqr.to_numpy(float), expected_scale):
                errors.append('Training IQR differs from exact fit-only calculation: ' + name)
        if 'fit_n' in frame and (not np.all(frame.fit_n == len(tr))):
            errors.append('Fitting bottle count mismatch: ' + name)
    if 'regression' in local:
        expected = set(FINAL_METHODS) if mode in ('ablation', 'ethanol') else set(ABV_POINT_METHODS) if mode == 'measured_abv' else set(POINT_METHODS)
        if mode == 'boundaries':
            expected.add('v10_measured_abv_polynomial_ridge')
        if set(local['regression'].method) != expected:
            errors.append('Point method inventory mismatch')
    if mode == 'interval' and required.issubset(local):
        errors.extend(verify_frames(local, alpha))
    for name in ('stress', 'occlusion'):
        if name in local and set(local[name].method) != set(POINT_METHODS):
            errors.append('Boundary method inventory mismatch: ' + name)
    return errors

def _verify_router(data, task, bundle, internal=False):
    errors = []
    n, m = (len(task['train']), len(data['target_names']))
    router = bundle.get('strategy_router', {})
    routes = router.get('routes', [])
    if len(routes) != m or not set(routes).issubset(set(STRATEGY_METHODS)):
        return ['Invalid per-target strategy route inventory']
    if router.get('enabled') is False:
        if not internal and n >= 16:
            errors.append('Outer strategy selection disabled despite sufficient fitting bottles')
        if set(routes) != {'v10_pls2_reference'}:
            errors.append('Disabled internal strategy must use fixed PLS2 route')
        return errors
    if router.get('enabled') is not True:
        return ['Missing strategy-selection audit']
    if router.get('fallback'):
        if n >= 16 or set(routes) != {'v10_pls2_reference'}:
            errors.append('Strategy fallback is valid only below16 fitting bottles with fixed PLS2')
        return errors
    if set(router.get('methods', [])) != set(STRATEGY_METHODS) or len(router.get('methods', [])) != len(STRATEGY_METHODS):
        return ['Strict strategy OOF method inventory mismatch']
    values = router.get('strict_procedure_oof', {})
    if set(values) != set(STRATEGY_METHODS):
        return ['Missing strict full-procedure strategy OOF predictions']
    Y = np.asarray(data['Y'], float)[np.asarray(task['train'], int)]
    computed = []
    for name in router['methods']:
        pred = np.asarray(values[name], float)
        if pred.shape != (n, m) or not np.isfinite(pred).all():
            return ['Invalid strict full-procedure strategy OOF array: ' + name]
        computed.append(np.mean(((pred - Y) / iqr(Y)) ** 2, axis=0))
    computed = np.column_stack(computed)
    saved = np.asarray(router.get('scores_targetwise'), float)
    if saved.shape != computed.shape or not np.allclose(saved, computed, rtol=1e-12, atol=1e-14):
        errors.append('Strategy selection scores differ from exact fit-only OOF calculation')
    priority = ['v10_pls2_reference', LEGACY_METHOD, LATE_METHOD, 'v10_equal_weight', 'v10_unanchored_ensemble', 'v10_anchored_ensemble', 'v10_conservative_adaptive', 'v10_target_adaptive', 'v10_shared_catalog_best']
    expected = [min(priority, key=lambda name: (round(float(computed[j, router['methods'].index(name)]), 12), priority.index(name))) for j in range(m)]
    if routes != expected:
        errors.append('Strategy routes differ from fixed training-OOF selection rule')
    visits = []
    for entry in router.get('crossfit', []):
        fit, va = (entry.get('train', []), entry.get('validation', []))
        if len(set(fit)) != len(fit) or len(set(va)) != len(va) or set(fit) & set(va) or (set(fit) | set(va) != set(range(n))):
            errors.append('Strategy whole-procedure OOF fold exclusion mismatch')
            continue
        if entry.get('training_bottles') != len(fit) or not entry.get('all_selection_and_weights_exclude_validation'):
            errors.append('Strategy whole-procedure fitting-count/exclusion audit mismatch')
        visits.extend(va)
        nested_seen = []
        for split in entry.get('inner_splits', []):
            innerfit, innerva = (split.get('fit', []), split.get('validation', []))
            if set(innerfit) & set(innerva) or set(innerfit) | set(innerva) != set(range(len(fit))):
                errors.append('Strategy inner selection includes OOF validation rows')
            nested_seen.extend(innerva)
        if sorted(nested_seen) != list(range(len(fit))):
            errors.append('Strategy inner selection membership incomplete')
        if not entry.get('training_audit_digest') or entry.get('selected_reference') not in entry.get('candidate_order', []):
            errors.append('Missing strategy nested-model provenance')
    if sorted(visits) != list(range(n)):
        errors.append('Strict strategy OOF does not cover fitting bottles once')
    if router.get('route_winners_chosen_using_outer_test') is not False or router.get('history_metric_splicing') is not False:
        errors.append('Strategy selection does not explicitly exclude historical/test score splicing')
    return errors

def _verify_audit(data, task, audit, mode, local=None, internal=False):
    errors = []
    bundle = _bundle(audit)
    if bundle is None:
        return ['Missing final point selection audit']
    if bundle.get('training_bottles') != len(task['train']):
        errors.append('Selection audit fitting count mismatch')
    errors.extend(_verify_router(data, task, bundle, internal=internal))
    order = bundle.get('candidate_order', [])
    if not order or len(order) != len(set(order)):
        errors.append('Empty/duplicate accepted candidate pool')
    if set(bundle.get('weights_by_method', {})) != {'v10_equal_weight', 'v10_unanchored_ensemble', 'v10_anchored_ensemble'}:
        errors.append('Missing matched-ensemble weight audit inventory')
    if bundle.get('penalties_by_method') != {'v10_unanchored_ensemble': 0.0, 'v10_anchored_ensemble': 0.1}:
        errors.append('Frozen simplex penalty audit mismatch')
    if bundle.get('reference_candidate') not in order or bundle.get('shared_catalog_candidate') not in order:
        errors.append('Selected reference/shared candidate absent from fitted pool')
    for key in ('minimum_routes', 'conservative_routes'):
        if len(bundle.get(key, [])) != len(data['target_names']) or not set(bundle.get(key, [])).issubset(order):
            errors.append('Target route audit inventory mismatch: ' + key)
    for name, raw in bundle.get('weights_by_method', {}).items():
        weights = np.asarray(raw, float)
        if weights.shape != (len(data['target_names']), len(order)) or not np.isfinite(weights).all() or np.any(weights < 0) or (not np.allclose(weights.sum(axis=1), 1.0, atol=1e-12, rtol=0)):
            errors.append('Invalid saved ensemble weights: ' + name)
        if name == 'v10_equal_weight' and (not np.allclose(weights, 1.0 / len(order), atol=1e-12, rtol=0)):
            errors.append('Equal-weight control is not uniform across the common pool')
    candidates = bundle.get('candidates', [])
    if len(candidates) != len(order) or {a['id'] for a in candidates} != set(order):
        errors.append('Candidate audit and accepted pool inventory mismatch')
    seen = []
    membership = np.full(len(task['train']), -1, int)
    splits = bundle.get('inner_splits', [])
    for fold, split in enumerate(splits):
        fit, va = (split['fit'], split['validation'])
        if len(set(fit)) != len(fit) or len(set(va)) != len(va) or set(fit) & set(va) or (set(fit) | set(va) != set(range(len(membership)))):
            errors.append('Candidate inner CV partition exclusion mismatch')
        if any((i < 0 or i >= len(membership) for i in va)):
            errors.append('Candidate inner CV validation index outside fitting rows')
            continue
        membership[va] = fold
        seen.extend(va)
    if sorted(seen) != list(range(len(membership))) or bundle.get('oof_membership_in_supplied_training_order') != membership.tolist():
        errors.append('Candidate inner OOF membership mismatch')
    for row in bundle.get('candidates', []):
        oof = np.asarray(row.get('oof_predictions'), float)
        if row['id'] not in order or oof.shape != (len(task['train']), len(data['target_names'])) or (not np.isfinite(oof).all()):
            errors.append('Invalid candidate OOF audit')
        scores = np.asarray(row.get('fold_scores_targetwise'), float)
        if scores.shape != (len(splits), len(data['target_names'])) or not np.isfinite(scores).all():
            errors.append('Invalid candidate fold-score audit')
    if mode == 'interval':
        tr = np.asarray(task['train'], int)
        n = len(tr)
        outer_seen = []
        for entry in audit.get('strict_full_pipeline_crossfit', []):
            fit, validation = (entry['train'], entry['validation'])
            if set(fit) & set(validation) or set(fit) | set(validation) != set(range(n)):
                errors.append('Whole-procedure OOF fold exclusion mismatch')
            outer_seen.extend(validation)
            nested = entry.get('audit', {})
            if nested.get('training_bottles') != len(fit):
                errors.append('Whole-procedure OOF fitting audit count mismatch')
            nested_task = dict(task, train=np.asarray(task['train'])[fit].tolist())
            errors.extend(('strict crossfit: ' + e for e in _verify_audit(data, nested_task, nested, 'ablation', internal=True)))
        if sorted(outer_seen) != list(range(n)):
            errors.append('Whole-procedure OOF does not cover fit bottles once')
        for key, partition in [('fit_partition_ids', 'train'), ('calibration_partition_ids', 'calibration'), ('test_partition_ids', 'test')]:
            if set(map(str, audit.get(key, []))) != set(map(str, np.asarray(data['ids'])[task[partition]])):
                errors.append('Interval audit partition ID mismatch: ' + key)
    if mode == 'boundaries' and local is not None:
        challenges = audit.get('challenges', [])
        if len(challenges) != audit.get('challenge_count'):
            errors.append('Boundary challenge audit count mismatch')
        for frame_name, window in [('occlusion', True), ('stress', False)]:
            expected = {a['scenario'] for a in challenges if (a['challenge_type'] == 'training_mean_window_replacement') == window}
            frame = local.get(frame_name, pd.DataFrame())
            if not expected or 'scenario' not in frame or set(frame.scenario) != expected:
                errors.append('Boundary scenario inventory mismatch: ' + frame_name)
    if mode == 'measured_abv':
        measured = audit.get('measured_abv', {})
        expected_ids = [str(data['ids'][i]) for i in task['train']]
        if measured.get('training_sample_ids') != expected_ids or measured.get('training_bottles') != len(expected_ids):
            errors.append('Measured-ABV training provenance mismatch')
        if measured.get('requires_measured_ethanol') is not True or measured.get('measured_abv_intervals_provided') is not False:
            errors.append('Measured-ABV input/interval boundary mismatch')
        if measured.get('degree') != 2 or measured.get('nuisance_ridge_alpha') != 1.0:
            errors.append('Frozen measured-ABV nuisance specification mismatch')
        order = [LEGACY_METHOD, 'v10_measured_abv_only', 'v10_measured_abv_residual']
        scores = np.asarray(measured.get('guard_scores_targetwise'), float)
        routes = measured.get('guard_route_indices', [])
        if scores.shape != (len(data['target_names']), 3) or not np.isfinite(scores).all():
            errors.append('Measured-ABV guard selection-score inventory mismatch')
        else:
            expected_routes = [min(range(3), key=lambda k: (round(float(scores[j, k]), 12), k)) for j in range(len(data['target_names']))]
            if routes != expected_routes or measured.get('guard_route_order') != order:
                errors.append('Measured-ABV guard does not follow fixed training selection rule')
        if measured.get('no_test_reference_in_any_learning') is not True:
            errors.append('Measured-ABV learning/test boundary not recorded')
    return errors

def verify(root, out, data, cfg, plan, check_report=True):
    frames, audits, errors = _collect(out, plan)
    unhashed = dict(plan)
    unhashed.pop('plan_hash', None)
    if digest(unhashed) != plan['plan_hash']:
        errors.append('Frozen plan digest mismatch')
    if 'data_ids' in plan and list(map(str, plan['data_ids'])) != list(map(str, data['ids'])):
        errors.append('Frozen dataset bottle order mismatch')
    if 'target_names' in plan and list(plan['target_names']) != list(data['target_names']):
        errors.append('Frozen target names mismatch')
    for name, expected in plan.get('code_hashes', {}).items():
        file = Path(root) / name
        if not file.is_file() or sha(file) != expected:
            errors.append('Frozen source/config fingerprint mismatch: ' + name)
    for task, audit, local in audits:
        try:
            errors.extend((task['id'] + ': ' + e for e in _verify_local(data, task, local, plan['mode'], 1.0 - float(cfg['nominal_coverage']))))
            errors.extend((task['id'] + ': ' + e for e in _verify_audit(data, task, audit, plan['mode'], local)))
        except Exception as exc:
            errors.append(task['id'] + ': verification raised ' + repr(exc))
    if check_report:
        try:
            receipt = read_json(Path(out) / 'report_receipt.json')
            if receipt['plan_hash'] != plan['plan_hash']:
                raise ValueError('Report receipt plan mismatch')
            expected_tasks = {task['id'] for task in plan['tasks']}
            if set(receipt['source_receipt_hashes']) != expected_tasks:
                raise ValueError('Report source task inventory mismatch')
            for task_id, expected in receipt['source_receipt_hashes'].items():
                if sha(Path(out) / 'checkpoints' / task_id / 'receipt.json') != expected:
                    raise ValueError('Report source task receipt changed: ' + task_id)
            listed = receipt['tables']
            actual = {p.name for p in (Path(out) / 'tables').glob('*.csv')}
            if set(listed) != actual or not listed:
                raise ValueError('Report table inventory mismatch')
            for name, expected in listed.items():
                if Path(name).name != name or sha(Path(out) / 'tables' / name) != expected:
                    raise ValueError('Report table digest mismatch: ' + name)
        except Exception as exc:
            errors.append('Saved report verification: ' + repr(exc))
    result = {'status': 'PASS' if not errors else 'FAIL', 'ok': not errors, 'complete': len(audits) == len(plan['tasks']) and (not errors), 'errors': errors, 'expected_tasks': len(plan['tasks']), 'completed_tasks': len(audits), 'rows': sum((len(a) for a in frames.values())), 'mode': plan['mode'], 'model_fitting': False, 'reference_values_checked': True, 'round_trip_float_reading': True, 'validation_tier': plan.get('evidence_tier', 'retrospective_internal_only'), 'computational_consistency_check_only': True, 'plan_hash': plan['plan_hash']}
    save_json(Path(out) / 'completion_check.json', result)
    return result

def _regression_summary(frame, extra=None):
    keys = ['method', 'target'] + ([] if extra is None else list(extra))
    rows = []
    for key, group in frame.groupby(keys, dropna=False):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        e = group.y_pred.to_numpy(float) - group.y_true.to_numpy(float)
        y = group.y_true.to_numpy(float)
        denominator = np.sum((y - y.mean()) ** 2)
        row.update(RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))), R2=float(1 - np.sum(e ** 2) / denominator) if denominator > 0 else np.nan, nRMSE=float(np.sqrt(np.mean((e / group.train_iqr.to_numpy(float)) ** 2))), unique_bottles=group.sample_id.nunique(), records=len(group), concentration_unit='source_unit_unconfirmed' if row['target'] == 'ethanol' else 'g/L_as_provided')
        rows.append(row)
    return pd.DataFrame(rows)

def _paired(frame, repeats, seed):
    rows, methods = ([], sorted(set(frame.method)))
    references = [m for m in ('v10_pls2_reference', LEGACY_METHOD, 'v10_equal_weight', 'v10_unanchored_ensemble') if m in methods]
    rng = np.random.RandomState(seed)
    keys = ['repeat', 'fold', 'sample_id', 'target']
    for reference in references:
        a = frame[frame.method == reference][keys + ['y_true', 'y_pred', 'train_iqr']]
        for method in methods:
            if method == reference:
                continue
            b = frame[frame.method == method][keys + ['y_pred']]
            joined = a.merge(b, on=keys, how='outer', suffixes=('_reference', '_method'), validate='one_to_one')
            if joined.isna().any().any():
                raise ValueError('Unpaired prediction rows in comparison')
            joined['reference_mse'] = (joined.y_pred_reference - joined.y_true) ** 2
            joined['method_mse'] = (joined.y_pred_method - joined.y_true) ** 2
            joined['reference_nmse'] = joined.reference_mse / joined.train_iqr ** 2
            joined['method_nmse'] = joined.method_mse / joined.train_iqr ** 2
            by_bottle = joined.groupby(['sample_id', 'target'])[['reference_mse', 'method_mse', 'reference_nmse', 'method_nmse']].mean().reset_index()
            for target in list(sorted(set(joined.target))) + ['__macro__']:
                if target == '__macro__':
                    rv = by_bottle.pivot(index='sample_id', columns='target', values='reference_nmse').to_numpy()
                    mv = by_bottle.pivot(index='sample_id', columns='target', values='method_nmse').to_numpy()
                    metric = 'equal_target_macro_train_IQR_NRMSE'
                else:
                    g = by_bottle[by_bottle.target == target]
                    rv, mv = (g.reference_mse.to_numpy()[:, None], g.method_mse.to_numpy()[:, None])
                    metric = 'RMSE_source_unit' if target == 'ethanol' else 'RMSE_g_L'
                if np.isnan(rv).any() or np.isnan(mv).any():
                    raise ValueError('Incomplete paired bottle/target matrix')
                r, m = (float(np.sqrt(rv.mean(axis=0)).mean()), float(np.sqrt(mv.mean(axis=0)).mean()))
                sample = rng.randint(0, len(rv), size=(int(repeats), len(rv)))
                rb, mb = (np.sqrt(rv[sample].mean(axis=1)).mean(axis=1), np.sqrt(mv[sample].mean(axis=1)).mean(axis=1))
                gain = 100.0 * (1.0 - mb / rb)
                delta = (mv - rv).mean(axis=1)
                db = delta[sample].mean(axis=1)
                lo, hi = np.percentile(gain, [2.5, 97.5])
                dlo, dhi = np.percentile(db, [2.5, 97.5])
                rows.append({'method': method, 'reference': reference, 'target': target, 'metric': metric, 'reference_error': r, 'method_error': m, 'relative_RMSE_gain_pct': 100.0 * (1.0 - m / r), 'gain_ci_low_pct': lo, 'gain_ci_high_pct': hi, 'delta_MSE': float(delta.mean()), 'delta_ci_low': dlo, 'delta_ci_high': dhi, 'unique_bottles': len(rv), 'bootstrap_repetitions': int(repeats), 'uncertainty_scope': 'bottle_cluster_bootstrap_conditional_on_saved_fits_and_splits', 'new_external_validation': False, 'multiple_comparisons_adjusted': False})
    return pd.DataFrame(rows)

def _stability(audits, target_names):
    weights, routes = ([], [])
    for task, audit, _ in audits:
        bundle = _bundle(audit)
        if bundle is None:
            continue
        order = bundle['candidate_order']
        specs = {row['id']: row['spec'] for row in bundle['candidates']}
        for method, raw in bundle['weights_by_method'].items():
            for j, target in enumerate(target_names):
                for c, candidate in enumerate(order):
                    spec = specs[candidate]
                    weights.append({'method': method, 'target': target, 'candidate_id': candidate, 'weight': float(raw[j][c]), 'repeat': task['repeat'], 'fold': task['fold'], 'family': spec['family'], 'modes': '+'.join(spec['modes'])})
        for name in ('minimum_routes', 'conservative_routes'):
            for target, candidate in zip(target_names, bundle[name]):
                spec = specs[candidate]
                routes.append({'route': name, 'target': target, 'candidate_id': candidate, 'family': spec['family'], 'modes': '+'.join(spec['modes'])})
    wf, rf = (pd.DataFrame(weights), pd.DataFrame(routes))
    if len(wf):
        summary = []
        for key, g in wf.groupby(['method', 'target', 'candidate_id', 'family', 'modes']):
            values = np.r_[g.weight.to_numpy(), np.zeros(len(audits) - len(g))]
            row = dict(zip(['method', 'target', 'candidate_id', 'family', 'modes'], key))
            row.update(mean_weight=values.mean(), std_weight=values.std(ddof=1) if len(values) > 1 else 0.0, nonzero_fraction=np.mean(values > 1e-07), fitted_tasks_present=len(g), total_tasks=len(audits), interpretation='descriptive_dependent_outer_fold_stability')
            summary.append(row)
        ws = pd.DataFrame(summary)
    else:
        ws = pd.DataFrame()
    if len(rf):
        rs = rf.groupby(['route', 'target', 'candidate_id', 'family', 'modes']).size().reset_index(name='count')
        rs['fraction'] = rs['count'] / len(audits)
    else:
        rs = pd.DataFrame()
    return (wf, ws, rs)

def _strategy_tables(audits, target_names):
    selected, definitions, mappings = ([], [], [])
    for task, audit, _ in audits:
        bundle = _bundle(audit)
        if bundle is None:
            continue
        specs = {row['id']: row['spec'] for row in bundle['candidates']}
        order = bundle['candidate_order']
        router = bundle.get('strategy_router', {})
        methods = router.get('methods', [])
        scores = np.asarray(router.get('scores_targetwise', []), float)
        for candidate, spec in specs.items():
            definitions.append({'repeat': task['repeat'], 'fold': task['fold'], 'candidate_id': candidate, 'family': spec['family'], 'modes': '+'.join(spec['modes']), 'preprocessing': spec.get('pre', 'raw'), 'specification_json': json.dumps(spec, ensure_ascii=False, sort_keys=True), 'candidate_accepted_in_training': True})
        for j, target in enumerate(target_names):
            choice = router.get('routes', ['v10_pls2_reference'] * len(target_names))[j]
            score = float(scores[j, methods.index(choice)]) if scores.shape == (len(target_names), len(methods)) and choice in methods else np.nan
            selected.append({'repeat': task['repeat'], 'fold': task['fold'], 'target': target, 'selected_procedure': choice, 'training_selection_score': score, 'training_selection_score_is_test_performance': False, 'strict_procedure_folds': len(router.get('crossfit', [])), 'small_sample_fixed_fallback': bool(router.get('fallback', False)), 'training_bottles': len(task['train']), 'descriptive_dependent_fold_stability': True})

            def coefficients(method):
                if method == 'v10_pls2_reference':
                    return {bundle['reference_candidate']: 1.0}
                if method == 'v10_shared_catalog_best':
                    return {bundle['shared_catalog_candidate']: 1.0}
                if method in ('v10_target_adaptive', 'v10_conservative_adaptive'):
                    key = 'minimum_routes' if method == 'v10_target_adaptive' else 'conservative_routes'
                    return {bundle[key][j]: 1.0}
                if method in bundle['weights_by_method']:
                    return {candidate: float(bundle['weights_by_method'][method][j][k]) for k, candidate in enumerate(order)}
                if method == LEGACY_METHOD:
                    raw = bundle.get('legacy_comparator', {}).get('weights')
                    return {} if raw is None else {candidate: float(raw[j][k]) for k, candidate in enumerate(order)}
                if method == LATE_METHOD:
                    entry = bundle.get('single_view_fusion', {})
                    return dict(zip(entry.get('selected_candidates', []), entry.get('weights', [])))
                return {}
            for method in FINAL_METHODS:
                resolved = choice if method == ROUTER_METHOD else method
                weights = coefficients(resolved)
                included = [candidate for candidate, weight in weights.items() if weight > 0]
                mappings.append({'repeat': task['repeat'], 'fold': task['fold'], 'target': target, 'method': method, 'resolved_training_procedure': resolved, 'selected_candidate_count': len(included), 'candidate_ids_json': json.dumps(included), 'candidate_weights_json': json.dumps({k: weights[k] for k in included}, sort_keys=True), 'algorithm_specifications_json': json.dumps({k: specs[k] for k in included}, ensure_ascii=False, sort_keys=True), 'historical_best_scores_spliced': False, 'selection_uses_outer_test_reference': False})
    sf, df, mf = (pd.DataFrame(selected), pd.DataFrame(definitions), pd.DataFrame(mappings))
    if len(sf):
        stability = sf.groupby(['target', 'selected_procedure']).size().reset_index(name='selection_count')
        stability['evaluated_outer_fits'] = len(audits)
        stability['selection_fraction'] = stability.selection_count / len(audits)
        stability['interpretation'] = 'descriptive_overlapping_outer_training_folds_not_independent_replication'
    else:
        stability = pd.DataFrame()
    return (sf, stability, df, mf)

def generate(root, out, data, cfg, plan):
    out = Path(out)
    check = verify(root, out, data, cfg, plan, check_report=False)
    if not check['ok']:
        return check
    try:
        frames, audits, errors = _collect(out, plan)
        if errors:
            raise ValueError(repr(errors))
        tables = out / 'tables'
        tables.mkdir(exist_ok=True)
        for name, frame in frames.items():
            save_csv(tables / ('records_' + name + '.csv'), frame)
        point = frames['regression']
        summary = _regression_summary(point)
        save_csv(tables / 'regression_summary.csv', summary)
        macro = summary.groupby('method', as_index=False).nRMSE.mean().rename(columns={'nRMSE': 'macro_nRMSE'})
        save_csv(tables / 'macro_regression_summary.csv', macro)
        by_repeat = _regression_summary(point, ['repeat'])
        save_csv(tables / 'regression_by_repeat.csv', by_repeat)
        macro_repeat = by_repeat.groupby(['method', 'repeat'], as_index=False).nRMSE.mean().rename(columns={'nRMSE': 'macro_nRMSE'})
        save_csv(tables / 'macro_by_repeat.csv', macro_repeat)
        repeat_gains = []
        for reference in ('v10_pls2_reference', LEGACY_METHOD, 'v10_equal_weight', 'v10_unanchored_ensemble'):
            baseline = by_repeat[by_repeat.method == reference][['target', 'repeat', 'RMSE', 'nRMSE']]
            if len(baseline):
                paired = by_repeat.merge(baseline, on=['target', 'repeat'], suffixes=('', '_reference'))
                paired['reference'] = reference
                paired['relative_RMSE_gain_pct'] = 100.0 * (1.0 - paired.RMSE / paired.RMSE_reference)
                paired['relative_nRMSE_gain_pct'] = 100.0 * (1.0 - paired.nRMSE / paired.nRMSE_reference)
                repeat_gains.append(paired)
        if repeat_gains:
            save_csv(tables / 'paired_gain_by_repeat.csv', pd.concat(repeat_gains, ignore_index=True))
        p = plan.get('parameters', {})
        boot = int(p.get('bootstrap', cfg.get('bootstrap_repetitions', 2000)))
        save_csv(tables / 'paired_comparisons.csv', _paired(point, boot, int(cfg.get('seed', 20261020)) + 993))
        tolerance = []
        for (method, target), g in point.groupby(['method', 'target']):
            error = np.abs(g.y_pred - g.y_true)
            for threshold in np.linspace(0.05, 1.0, 20):
                tolerance.append({'method': method, 'target': target, 'threshold_type': 'training_IQR_fraction_illustrative', 'threshold': threshold, 'within_tolerance_fraction': float(np.mean(error <= threshold * g.train_iqr)), 'industrial_acceptance_limit': False})
            allowed = None if target == 'ethanol' else (cfg.get('allowable_absolute_errors_g_L') or {}).get(target)
            if allowed is not None:
                allowed = float(allowed)
                if not np.isfinite(allowed) or allowed <= 0:
                    raise ValueError('Author-supplied allowable errors must be finite and positive')
                tolerance.append({'method': method, 'target': target, 'threshold_type': 'author_supplied_g_L', 'threshold': allowed, 'within_tolerance_fraction': float(np.mean(error <= allowed)), 'industrial_acceptance_limit': False})
        save_csv(tables / 'tolerance_curves.csv', pd.DataFrame(tolerance))
        raw_weights, weight_summary, route_summary = _stability(audits, data['target_names'])
        for filename, frame in [('weights_by_fold', raw_weights), ('weight_stability', weight_summary), ('route_stability', route_summary)]:
            save_csv(tables / (filename + '.csv'), frame)
        strategy, strategy_stability, candidate_list, mappings = _strategy_tables(audits, data['target_names'])
        for filename, frame in [('strategy_selection_by_fold', strategy), ('strategy_selection_stability', strategy_stability), ('candidate_library_by_fold', candidate_list), ('real_winning_algorithm_mappings', mappings)]:
            save_csv(tables / (filename + '.csv'), frame)
        if 'interval_regression' in frames:
            interval = frames['interval_regression'].copy()
            interval['covered'] = (interval.y_true >= interval.lower) & (interval.y_true <= interval.upper)
            interval['width'] = interval.upper - interval.lower
            alpha = 1.0 - float(cfg['nominal_coverage'])
            interval['winkler'] = interval.width + 2.0 / alpha * np.maximum(interval.lower - interval.y_true, 0) + 2.0 / alpha * np.maximum(interval.y_true - interval.upper, 0)
            rows = []
            for (method, target), g in interval.groupby(['method', 'target']):
                rows.append({'method': method, 'target': target, 'coverage': g.covered.mean(), 'mean_width': g.width.mean(), 'mean_width_iqr': np.mean(g.width / g.train_iqr), 'finite_mean_width_iqr': np.mean((g.width / g.train_iqr)[np.isfinite(g.width)]) if np.isfinite(g.width).any() else np.nan, 'infinite_width_count': int(np.isinf(g.width).sum()), 'Winkler': g.winkler.mean(), 'unique_bottles': g.sample_id.nunique(), 'records': len(g)})
            save_csv(tables / 'interval_summary.csv', pd.DataFrame(rows))
            joint = []
            for method, g in interval.groupby('method'):
                joint.append({'method': method, 'joint_coverage': g.groupby(['repeat', 'fold', 'sample_id']).covered.all().mean(), 'mean_width_iqr': np.mean(g.width / g.train_iqr), 'finite_mean_width_iqr': np.mean((g.width / g.train_iqr)[np.isfinite(g.width)]) if np.isfinite(g.width).any() else np.nan, 'mean_component_coverage': g.covered.mean(), 'infinite_width_count': int(np.isinf(g.width).sum())})
            save_csv(tables / 'interval_joint_summary.csv', pd.DataFrame(joint))
            boundaries = []
            for flag in ('above_train_q75', 'ethanol_tail'):
                for (method, target, marked), g in interval.assign(_flag=_bool(interval[flag])).groupby(['method', 'target', '_flag']):
                    boundaries.append({'method': method, 'target': target, 'grouping': flag, 'flag': marked, 'coverage': g.covered.mean(), 'mean_width_iqr': np.mean(g.width / g.train_iqr), 'unique_bottles': g.sample_id.nunique(), 'records': len(g), 'conditional_coverage_guaranteed': False})
            save_csv(tables / 'interval_boundary_summary.csv', pd.DataFrame(boundaries))
        if 'risk_predictions' in frames:
            risk = []
            for (method, target, fraction), g in frames['risk_predictions'].groupby(['method', 'target', 'retained_fraction']):
                accepted = _bool(g.accepted)
                selected = g[accepted]
                rejected = g[~accepted]
                error = selected.y_pred - selected.y_true
                rejected_error = rejected.y_pred - rejected.y_true
                all_error = g.y_pred - g.y_true
                risk.append({'method': method, 'target': target, 'retained_fraction': fraction, 'actual_retention': accepted.mean(), 'accepted_records': len(selected), 'total_records': len(g), 'RMSE': float(np.sqrt(np.mean(error ** 2))) if len(selected) else np.nan, 'nRMSE': float(np.sqrt(np.mean((error / selected.train_iqr) ** 2))) if len(selected) else np.nan, 'rejected_records': len(rejected), 'rejected_RMSE': float(np.sqrt(np.mean(rejected_error ** 2))) if len(rejected) else np.nan, 'rejected_nRMSE': float(np.sqrt(np.mean((rejected_error / rejected.train_iqr) ** 2))) if len(rejected) else np.nan, 'all_RMSE': float(np.sqrt(np.mean(all_error ** 2))), 'all_nRMSE': float(np.sqrt(np.mean((all_error / g.train_iqr) ** 2))), 'accepted_unique_bottles': selected.sample_id.nunique(), 'conditional_coverage_guaranteed': False})
            save_csv(tables / 'risk_summary.csv', pd.DataFrame(risk))
        clean = summary[['method', 'target', 'RMSE']].rename(columns={'RMSE': 'clean_RMSE'})
        for name in ('stress', 'occlusion'):
            if name in frames:
                frame = frames[name]
                s = _regression_summary(frame, ['scenario']).merge(clean, on=['method', 'target'], validate='many_to_one')
                s['delta_RMSE_pct_vs_clean'] = 100.0 * (s.RMSE / s.clean_RMSE - 1.0)
                meta = [k for k in ('scenario', 'changed_mode', 'challenge_type', 'magnitude', 'window_index', 'coordinate_start', 'coordinate_end') if k in frame]
                s = s.merge(frame[meta].drop_duplicates(), on='scenario', validate='many_to_one')
                save_csv(tables / (name + '_summary.csv'), s)
        if 'diagnostics' in frames:
            joined = point.merge(frames['diagnostics'], on=['repeat', 'fold', 'sample_id', 'target'], suffixes=('', '_diagnostic'), validate='many_to_one')
            group_rows = []
            for flag in ('above_training_q75', 'outside_training_abv_central80'):
                g = joined.copy()
                g['group_flag'] = _bool(g[flag])
                s = _regression_summary(g, ['group_flag'])
                s['grouping'] = flag
                group_rows.append(s)
            save_csv(tables / 'boundary_groups_summary.csv', pd.concat(group_rows, ignore_index=True))
        narrative = 'These summaries use saved bottle-level predictions. The procedure was developed retrospectively after historical analyses of the same 139 bottles. Candidate discovery and new external validation are distinct. All routes, preprocessing, candidate selection and weights are selected within training partitions. Measured-alcohol-strength procedures require an additional reference input and are reported separately from spectra-only methods. Alcohol-strength units remain unconfirmed. Bottle-cluster intervals condition on saved fits and retain all repeated records of each bottle; they exclude refitting and redevelopment uncertainty. Interval coverage, width, unbounded intervals and Winkler scores are reported together. Selected-subset coverage has no additional guarantee. Error tolerance curves are illustrative unless author-confirmed limits are supplied. Route frequencies and weights describe overlapping training partitions and do not establish chemical causality. Target nRMSE pools squared errors normalized by the corresponding training IQR before the square root; the macro metric then averages targets equally. Completion checks establish consistency of computational records only.\n'
        (out / 'result_interpretation.md').write_text(narrative, encoding='utf-8')
        save_json(out / 'report_receipt.json', {'plan_hash': plan['plan_hash'], 'model_fitting': False, 'source_receipt_hashes': {task['id']: sha(out / 'checkpoints' / task['id'] / 'receipt.json') for task in plan['tasks']}, 'tables': {p.name: sha(p) for p in sorted(tables.glob('*.csv'))}, 'conditional_bottle_bootstrap': True, 'bootstrap_repetitions': boot})
        return verify(root, out, data, cfg, plan, check_report=True)
    except Exception as exc:
        failure = {'status': 'FAIL', 'ok': False, 'complete': False, 'model_fitting': False, 'errors': ['Reporting raised ' + repr(exc)]}
        save_json(out / 'reporting_error.json', failure)
        return failure
