import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from suite.data import load_data

ROOT = Path(__file__).resolve().parent

def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()

def main():
    errors = []
    manifest = ROOT / 'CHECKSUMS.sha256'
    checked_files = 0
    if not manifest.exists():
        raise ValueError('Missing distribution checksum manifest')
    for line in manifest.read_text(encoding='ascii').splitlines():
        expected, name = line.split('  ', 1)
        file = ROOT / name
        if not file.is_file() or checksum(file) != expected:
            errors.append('File integrity mismatch: ' + name)
        checked_files += 1
    data = load_data(ROOT, verify=True)
    samples = pd.read_csv(ROOT / 'data/aligned/samples.csv', dtype={'sample_id': str}, encoding='utf-8-sig', float_precision='round_trip')
    references = samples.set_index('sample_id')
    checked_records = 0
    checked_plans = 0
    for folder in sorted((ROOT / 'archives/main').iterdir()):
        plan = json.loads((folder / 'plan.json').read_text(encoding='utf-8'))
        ids = plan['data_ids']
        if ids != data['ids'].tolist():
            errors.append('Archive bottle order differs: ' + folder.name)
        task_map = {(task['repeat'], task['fold']): task for task in plan['tasks']}
        for task in plan['tasks']:
            fitting, calibration, test = map(set, (task['train'], task['calibration'], task['test']))
            if fitting & calibration or fitting & test or calibration & test:
                errors.append('Overlapping split: ' + folder.name + '/' + task['id'])
            if fitting | calibration | test != set(range(len(ids))):
                errors.append('Incomplete split: ' + folder.name + '/' + task['id'])
        checked_plans += 1
        for file in sorted((folder / 'tables').glob('records*.csv.gz')):
            columns = pd.read_csv(file, nrows=0, encoding='utf-8-sig').columns
            if not {'sample_id', 'target', 'y_true'}.issubset(columns):
                continue
            for frame in pd.read_csv(file, chunksize=150000, dtype={'sample_id': str}, encoding='utf-8-sig', low_memory=False, float_precision='round_trip'):
                for target, part in frame.groupby('target'):
                    if target not in references:
                        errors.append('Unknown reference target: ' + str(target))
                        continue
                    expected = part.sample_id.map(references[target]).to_numpy(float)
                    if not np.array_equal(expected, part.y_true.to_numpy(float)):
                        errors.append('Reference concentration differs: ' + folder.name + '/' + file.name + '/' + target)
                if file.name in ['records_regression.csv.gz', 'records_interval_regression.csv.gz', 'records_risk_predictions.csv.gz']:
                    for key, part in frame.groupby(['repeat', 'fold']):
                        allowed = {ids[index] for index in task_map[key]['test']}
                        if not set(part.sample_id).issubset(allowed):
                            errors.append('Saved test membership differs: ' + folder.name + '/' + file.name)
                checked_records += len(frame)
    result = {'status': 'PASS' if not errors else 'FAIL', 'ok': not errors, 'model_fitting': False,
              'independent_bottles': len(data['ids']), 'checked_distribution_files': checked_files,
              'checked_archived_plans': checked_plans, 'checked_reference_records': checked_records,
              'workbook_and_aligned_values_equal': True, 'errors': errors,
              'validation_tier': 'retrospective_internal_only'}
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1

if __name__ == '__main__':
    sys.exit(main())
