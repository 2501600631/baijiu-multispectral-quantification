import sys
sys.dont_write_bytecode = True
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'LOKY_MAX_CPU_COUNT']:
    os.environ.setdefault(key, '1')
from pathlib import Path
import argparse, json, sys, traceback, math
from suite.common import read_json, save_json, normalize, runtime
from suite.data import load_data, synthetic_data
from suite.final_protocol import FINAL_MODES, OPTIONAL_MODES, frozen_plan
ROOT = Path(__file__).resolve().parent

def _read_plan(out, root, data, cfg, profile, mode):
    if not (out / 'plan.json').exists():
        raise ValueError('No frozen plan found: ' + str(out))
    return frozen_plan(root, data, cfg, profile, mode, out)

def main():
    ap = argparse.ArgumentParser(description='V10 targeted finishing: matched fusion, same-model intervals and boundaries')
    ap.add_argument('action', choices=['check', 'plan', 'run', 'smoke', 'report', 'report-all', 'verify', 'verify-all', 'plot'])
    ap.add_argument('--mode', choices=FINAL_MODES + OPTIONAL_MODES, default='ablation')
    ap.add_argument('--profile', choices=['smoke', 'pilot', 'main'], default='main')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    cfg = read_json(ROOT / 'config.json')
    if args.action == 'smoke':
        args.profile = 'smoke'
    if cfg['ensemble_options']['ensemble_penalty'] != 0.1:
        raise ValueError('V10 fixed anchor penalty must remain 0.1')
    tolerance = cfg.get('allowable_absolute_errors_g_L')
    if tolerance is not None and (not isinstance(tolerance, dict) or any((not math.isfinite(float(v)) or float(v) <= 0 for v in tolerance.values()))):
        raise ValueError('Use null or author-confirmed positive finite absolute tolerances')
    data = synthetic_data(n=73, m=11) if args.profile == 'smoke' else load_data(ROOT, verify=True)
    if args.mode == 'ethanol' and args.action != 'smoke':
        from suite.integrated_data import as_ethanol_target
        data = as_ethanol_target(data)
        cfg = dict(cfg, allowable_absolute_errors_g_L=None)
        tolerance = None
    if tolerance is not None and (not set(tolerance) <= set(data['target_names'])):
        raise ValueError('Unknown tolerance target')
    out = args.out or ROOT / 'results' / args.profile / args.mode
    if args.action == 'check':
        from suite.integrated_catalog import build_integrated_catalog
        _, _, library = build_integrated_catalog(dict(cfg['profiles'][args.profile], is_smoke=args.profile == 'smoke'), cfg['ensemble_options'])
        result = {'status': 'PASS', 'model_fitting': False, 'independent_bottles': len(data['ids']), 'targets': data['target_names'], 'spectra_shapes': {k: list(v.shape) for k, v in data['X'].items()}, 'source_audit': data['data_audit'], 'runtime': runtime(), 'new_external_validation': False, 'fixed_algorithm_library': library, 'required_modes': list(FINAL_MODES), 'optional_modes': list(OPTIONAL_MODES)}
        save_json(ROOT / 'verification/data_check.json', result)
    elif args.action == 'plan':
        plan = frozen_plan(ROOT, data, cfg, args.profile, args.mode, out)
        result = {'status': 'FROZEN', 'expected_tasks': len(plan['tasks']), 'plan_hash': plan['plan_hash'], 'out': str(out), 'model_fitting': False}
    elif args.action in ['run', 'smoke']:
        from suite.final_engine import run
        if args.action == 'smoke':
            base = args.out or ROOT / 'results/smoke'
            result = {m: run(ROOT, base / m, data, cfg, 'smoke', m) for m in FINAL_MODES}
            result['synthetic_only'] = True
            result['real_model_fitting'] = False
        else:
            result = run(ROOT, out, data, cfg, args.profile, args.mode)
    elif args.action in ['report', 'verify', 'report-all', 'verify-all']:
        from suite.final_reporting import generate, verify
        modes = FINAL_MODES if args.action.endswith('-all') else [args.mode]
        base = args.out or ROOT / 'results' / args.profile
        results = {}
        for mode in modes:
            directory = base / mode if args.action.endswith('-all') else out
            if not (directory / 'plan.json').exists():
                results[mode] = {'status': 'NOT_RUN', 'complete': False, 'model_fitting': False}
                continue
            plan = _read_plan(directory, ROOT, data, cfg, args.profile, mode)
            if args.action.startswith('report'):
                report = generate(ROOT, directory, data, cfg, plan)
                if not report.get('ok'):
                    results[mode] = report
                    continue
            results[mode] = verify(ROOT, directory, data, cfg, plan)
        result = {'experiments': results, 'all_required_experiments_complete': all((v.get('complete') for v in results.values())), 'model_fitting': False, 'computational_consistency_check_only': True}
        save_json(ROOT / 'verification' / ('completion_' + args.profile + '.json'), result)
        if not result['all_required_experiments_complete']:
            print(json.dumps(normalize(result), ensure_ascii=False, indent=2))
            return 1
    else:
        from suite.final_plotting import draw_all
        if args.out:
            result = draw_all(out)
        else:
            result = {m: draw_all(ROOT / 'results' / args.profile / m) for m in FINAL_MODES}
    print(json.dumps(normalize(result), ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Interrupted. Finished checkpoints are preserved; run the same launcher to resume.')
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
