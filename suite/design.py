from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from .common import digest, source_hashes, save_json, read_json

def labels(data, name='sensory_grade'):
    frame = data['chemistry_all'].set_index('sample_id')
    frame.index = frame.index.astype(str)
    return frame.loc[data['ids'], name].astype(str).to_numpy()

def calibration_partition(train, y, seed, classification=False):
    return tuple((np.sort(a) for a in train_test_split(train, test_size=0.25, random_state=seed)))

def build_plan(root, data, cfg, profile_name, mode):
    p = cfg['profiles'][profile_name].copy()
    p['is_smoke'] = profile_name == 'smoke'
    n = len(data['ids'])
    y = labels(data)
    tasks = []
    seed = cfg['seed']
    do_reg = mode in ('core', 'advanced', 'interval', 'learning', 'optimized', 'uncertainty')
    do_clf = mode in ('core', 'advanced', 'interval', 'permutation', 'optimized', 'uncertainty')
    if do_reg:
        for rep in range(p['regression_repeats']):
            for fold, (tr, te) in enumerate(KFold(p['regression_folds'], shuffle=True, random_state=seed + rep).split(np.zeros(n))):
                fit, cal = calibration_partition(tr, y, seed + 700 + rep * 31 + fold) if mode in ('interval', 'uncertainty') else (tr, np.array([], int))
                tasks.append({'id': 'reg_r%02d_f%02d' % (rep, fold), 'kind': 'regression', 'repeat': rep, 'fold': fold, 'train': fit.tolist(), 'calibration': cal.tolist(), 'test': te.tolist(), 'seed': seed + rep * 1000 + fold * 13})
    if do_clf:
        min_count = int(np.unique(y, return_counts=True)[1].min())
        k = min(p['classification_folds'], min_count)
        if k < 2:
            raise ValueError('Original sensory labels cannot support stratified outer validation')
        for rep in range(p['classification_repeats']):
            splitter = KFold(k, shuffle=True, random_state=seed + rep).split(np.zeros(n)) if mode in ('permutation', 'interval', 'uncertainty') else StratifiedKFold(k, shuffle=True, random_state=seed + rep).split(np.zeros(n), y)
            for fold, (tr, te) in enumerate(splitter):
                fit, cal = calibration_partition(tr, y, seed + 900 + rep * 31 + fold, True) if mode in ('interval', 'uncertainty') else (tr, np.array([], int))
                tasks.append({'id': 'clf_r%02d_f%02d' % (rep, fold), 'kind': 'classification', 'repeat': rep, 'fold': fold, 'train': fit.tolist(), 'calibration': cal.tolist(), 'test': te.tolist(), 'seed': seed + 50000 + rep * 1000 + fold * 13})
    plan = {'version': cfg['version'], 'mode': mode, 'profile': profile_name, 'parameters': p, 'configuration': cfg, 'data_ids': data['ids'].tolist(), 'data_fingerprint': data['data_audit'].get('fingerprint', digest({'ids': data['ids'], 'Y': data['Y'], 'ethanol': data['ethanol']})), 'sensory_labels_hash': digest(y), 'classes': sorted(np.unique(y).tolist()), 'target_names': data['target_names'], 'code_hashes': source_hashes(root), 'tasks': tasks, 'synthetic': bool(data['data_audit'].get('synthetic', False)), 'evidence_tier': 'synthetic_only' if profile_name == 'smoke' else cfg['evidence_tier'], 'new_external_validation': False, 'grade_order_assumed': False}
    plan['classification_outer_stratified'] = mode not in ('interval', 'permutation', 'uncertainty')
    plan['calibration_sampling'] = 'label-independent random split; no per-class quota'
    plan['plan_hash'] = digest(plan)
    return plan

def frozen_plan(root, data, cfg, profile, mode, out):
    plan = build_plan(root, data, cfg, profile, mode)
    file = Path(out) / 'plan.json'
    if file.exists():
        old = read_json(file)
        if old['plan_hash'] != plan['plan_hash']:
            raise ValueError('Frozen plan/input/code changed. Use another --out directory; do not overwrite old checkpoints.')
        return old
    save_json(file, plan)
    return plan
