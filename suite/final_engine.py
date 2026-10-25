from pathlib import Path
import os, shutil, traceback
import numpy as np
import pandas as pd
from .common import sha, read_json, save_json, save_csv, csv_read, runtime, subset, iqr
from .final_protocol import frozen_plan
from .final_cache import get_bundle

def _checkpoint_valid(directory, plan_hash, task_id):
    receipt = directory / 'receipt.json'
    if not receipt.exists():
        return False
    r = read_json(receipt)
    if r.get('plan_hash') != plan_hash or r.get('task_id') != task_id:
        raise ValueError('Checkpoint identity changed: ' + str(directory))
    for name, expected in r['files'].items():
        f = directory / name
        if not f.exists() or sha(f) != expected:
            raise ValueError('Checkpoint bytes changed: ' + str(f))
    return True

def _task(root, out, data, cfg, p, mode, task):
    if mode == 'interval':
        from .final_interval import run_task
        return run_task(data, task, p, cfg)
    bundle, cache = get_bundle(root, data, task, p, cfg, out.parent / '_bundle_cache')
    if mode == 'measured_abv':
        from .integrated_abv import run_task
        frames, audit = run_task(data, task, p, cfg, bundle=bundle)
        audit['cache'] = cache
        return (frames, audit)
    if mode == 'boundaries':
        from .final_boundaries import run_task
        frames, audit = run_task(data, task, p, cfg, bundle=bundle)
        audit['cache'] = cache
        return (frames, audit)
    from .engine import regression_records
    tr = np.asarray(task['train'], int)
    te = np.asarray(task['test'], int)
    preds = bundle.predict_methods(subset(data['X'], te))
    frames = {'regression': pd.concat([regression_records(data, te, pred, name, iqr(data['Y'][tr]), task) for name, pred in preds.items()], ignore_index=True)}
    return (frames, {'bundle': bundle.audit, 'cache': cache, 'metadata': bundle.metadata, 'same_candidate_pool_for_all_seven_methods': True})

def run(root, out, data, cfg, profile_name, mode, make_report=True):
    root, out = (Path(root), Path(out))
    out.mkdir(parents=True, exist_ok=True)
    plan = frozen_plan(root, data, cfg, profile_name, mode, out)
    p = plan['parameters']
    save_json(out / 'runtime.json', runtime())
    completed = 0
    reused = 0
    for task in plan['tasks']:
        directory = out / 'checkpoints' / task['id']
        if _checkpoint_valid(directory, plan['plan_hash'], task['id']):
            completed += 1
            reused += 1
            continue
        try:
            print('%s %s: fitting training partitions only' % (mode, task['id']), flush=True)
            frames, audit = _task(root, out, data, cfg, p, mode, task)
            temporary = directory.with_name(directory.name + '.pending')
            temporary.mkdir(parents=True, exist_ok=True)
            expected = {'audit.json'} | {name + '.csv' for name in frames}
            for old in temporary.iterdir():
                if old.is_file() and old.name not in expected:
                    old.unlink()
            for name, frame in frames.items():
                save_csv(temporary / (name + '.csv'), frame)
            save_json(temporary / 'audit.json', audit)
            receipt = {'plan_hash': plan['plan_hash'], 'task_id': task['id'], 'files': {name: sha(temporary / name) for name in sorted(expected)}}
            save_json(temporary / 'receipt.json', receipt)
            if directory.exists():
                raise ValueError('Incomplete checkpoint directory exists; preserve it and use a new --out: ' + str(directory))
            os.replace(str(temporary), str(directory))
            completed += 1
            save_json(out / 'progress.json', {'completed_tasks': completed, 'expected_tasks': len(plan['tasks']), 'last_task': task['id'], 'model_fitting': True})
            print('Completed %s (%d/%d)' % (task['id'], completed, len(plan['tasks'])), flush=True)
        except Exception:
            save_json(out / 'errors' / (task['id'] + '.json'), {'task': task['id'], 'traceback': traceback.format_exc(), 'completed_checkpoints_preserved': True})
            raise
    if make_report:
        from .final_reporting import generate, verify
        try:
            report = generate(root, out, data, cfg, plan)
            if not report.get('ok'):
                raise RuntimeError('Saved checkpoint reporting failed: ' + str(report))
            result = verify(root, out, data, cfg, plan)
            if not result.get('complete'):
                raise RuntimeError('Scientific completion checks failed: ' + str(result))
            result['checkpoints_reused'] = reused
            return result
        except Exception:
            save_json(out / 'errors/reporting.json', {'traceback': traceback.format_exc(), 'completed_checkpoints_preserved': True, 'retry_without_fitting': 'run.py report --mode ' + mode})
            raise
    return {'completed_tasks': completed, 'expected_tasks': len(plan['tasks']), 'checkpoints_reused': reused, 'model_fitting': True, 'report_generated': False}
