from pathlib import Path
import hashlib
import json
import os
import platform
import sys
import numpy as np
import pandas as pd
MODES = ['nir', 'raman', 'uv']
COMBINATIONS = {'nir': ['nir'], 'raman': ['raman'], 'uv': ['uv'], 'nir_raman': ['nir', 'raman'], 'nir_uv': ['nir', 'uv'], 'raman_uv': ['raman', 'uv'], 'all': ['nir', 'raman', 'uv']}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def normalize(v):
    if isinstance(v, dict):
        return {str(k): normalize(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [normalize(x) for x in v]
    if isinstance(v, np.ndarray):
        return normalize(v.tolist())
    if isinstance(v, np.generic):
        return normalize(v.item())
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, float) and (not np.isfinite(v)):
        return None
    return v

def digest(v):
    return hashlib.sha256(json.dumps(normalize(v), sort_keys=True, ensure_ascii=True, allow_nan=False).encode()).hexdigest()

def save_json(path, v):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(normalize(v), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(str(tmp), str(path))

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def csv_read(path, **kwargs):
    options = {'encoding': 'utf-8-sig', 'low_memory': False, 'float_precision': 'round_trip'}
    options.update(kwargs)
    if options.get('float_precision') != 'round_trip':
        raise ValueError('Scientific CSV reads require round_trip floating precision')
    return pd.read_csv(path, **options)

def save_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    frame.to_csv(temp, index=False, encoding='utf-8-sig')
    os.replace(str(temp), str(path))

def subset(x, idx):
    return {k: np.asarray(v)[idx] for k, v in x.items()}

def iqr(y):
    a = np.percentile(y, 75, axis=0) - np.percentile(y, 25, axis=0)
    if np.any(a <= 1e-12) or not np.isfinite(a).all():
        raise ValueError('Training target IQR is zero or invalid; inspect the target/split.')
    return a

def source_hashes(root):
    root = Path(root)
    files = [root / 'config.json', root / 'run.py'] + [f for f in sorted((root / 'suite').glob('*.py')) if f.name not in ('plotting.py', 'final_plotting.py')]
    return {str(f.relative_to(root)).replace('\\', '/'): sha(f) for f in files if f.exists()}

def runtime():
    import sklearn, scipy
    return {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scikit_learn': sklearn.__version__, 'scipy': scipy.__version__}
