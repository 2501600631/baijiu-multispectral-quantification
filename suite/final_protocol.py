from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from .common import digest, source_hashes, read_json, save_json
FINAL_MODES = ('ablation', 'interval', 'boundaries', 'measured_abv')
OPTIONAL_MODES = ('ethanol',)

def input_fingerprint(data):
    if data['data_audit'].get('fingerprint'):
        return data['data_audit']['fingerprint']
    return digest({'ids': data['ids'], 'Y': data['Y'], 'ethanol': data['ethanol'], 'X': data['X'], 'axes': data['axes']})

def build_plan(root, data, cfg, profile_name, mode):
    if mode not in FINAL_MODES + OPTIONAL_MODES:
        raise ValueError('Unknown V10 mode: ' + mode)
    p = dict(cfg['profiles'][profile_name], is_smoke=profile_name == 'smoke')
    tasks = []
    n = len(data['ids'])
    seed = int(cfg['seed'])
    for repeat in range(p['regression_repeats']):
        outer = KFold(p['regression_folds'], shuffle=True, random_state=seed + repeat)
        for fold, (train, test) in enumerate(outer.split(np.zeros(n))):
            fit, cal = (np.sort(train), np.array([], int))
            if mode == 'interval':
                fit, cal = tuple((np.sort(a) for a in train_test_split(train, test_size=float(cfg['calibration_fraction']), random_state=seed + 700 + repeat * 31 + fold)))
            tasks.append({'id': 'reg_r%02d_f%02d' % (repeat, fold), 'kind': 'regression', 'repeat': repeat, 'fold': fold, 'train': fit.tolist(), 'calibration': cal.tolist(), 'test': test.tolist(), 'outer_train': np.sort(train).tolist(), 'seed': seed + repeat * 1000 + fold * 13})
    plan = {'version': cfg['version'], 'mode': mode, 'profile': profile_name, 'parameters': p, 'configuration': cfg, 'data_ids': data['ids'].tolist(), 'target_names': data['target_names'], 'data_fingerprint': input_fingerprint(data), 'code_hashes': source_hashes(root), 'tasks': tasks, 'synthetic': bool(data['data_audit'].get('synthetic', False)), 'evidence_tier': 'synthetic_only' if p['is_smoke'] else cfg['evidence_tier'], 'redesigned_after_viewing_V9': True, 'new_external_validation': False, 'calibration_membership': 'label-independent random; same outer bottles as V8; no class quotas', 'evaluation_context': 'disjoint_fit_calibration' if mode == 'interval' else 'full_outer_training', 'fixed_primary_model': cfg['primary_model'], 'fixed_primary_comparisons': cfg['primary_comparisons']}
    plan['plan_hash'] = digest(plan)
    return plan

def frozen_plan(root, data, cfg, profile, mode, out):
    expected = build_plan(root, data, cfg, profile, mode)
    file = Path(out) / 'plan.json'
    if file.exists():
        actual = read_json(file)
        actual_payload = dict(actual)
        actual_payload.pop('plan_hash', None)
        if digest(actual_payload) != actual.get('plan_hash') or actual.get('plan_hash') != expected['plan_hash']:
            raise ValueError('Frozen scientific inputs/code/config changed. Preserve this output; use a new --out directory.')
        return actual
    save_json(file, expected)
    return expected
