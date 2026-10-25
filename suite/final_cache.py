from pathlib import Path
import hashlib, os
import numpy as np
import joblib
from .common import digest, sha, read_json, save_json, runtime, source_hashes, subset

def _array_digest(values):
    values = np.ascontiguousarray(values)
    h = hashlib.sha256(str((values.shape, str(values.dtype))).encode('ascii'))
    h.update(values.tobytes())
    return h.hexdigest()

def cache_identity(root, data, task, profile, cfg):
    tr = np.asarray(task['train'], int)
    return {'version': cfg['version'], 'training_ids': data['ids'][tr].tolist(), 'seed': task['seed'], 'profile': profile, 'ensemble_options': cfg['ensemble_options'], 'sources': source_hashes(root), 'training_X': {k: _array_digest(v[tr]) for k, v in sorted(data['X'].items())}, 'training_Y': _array_digest(data['Y'][tr]), 'runtime': runtime()}

def get_bundle(root, data, task, profile, cfg, cache_dir):
    from .final_ensemble import fit_final_bundle
    identity = cache_identity(root, data, task, profile, cfg)
    key = digest(identity)
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / (key + '.joblib')
    receipt = directory / (key + '.json')
    if receipt.exists():
        proof = read_json(receipt)
        if proof.get('key') != key or proof.get('identity') != identity or (not model.exists()) or (sha(model) != proof.get('model_sha256')):
            raise ValueError('Local fitted cache changed. Preserve it and select a fresh output directory.')
        return (joblib.load(model), {'reused': True, 'cache_key': key, 'model_sha256': proof['model_sha256']})
    tr = np.asarray(task['train'], int)
    fitted = fit_final_bundle(subset(data['X'], tr), data['Y'][tr], profile, task['seed'], options=cfg['ensemble_options'])
    temporary = model.with_suffix('.joblib.tmp')
    joblib.dump(fitted, temporary, compress=3)
    os.replace(str(temporary), str(model))
    proof = {'key': key, 'identity': identity, 'model_sha256': sha(model), 'trained_in_this_version': True}
    save_json(receipt, proof)
    return (fitted, {'reused': False, 'cache_key': key, 'model_sha256': proof['model_sha256']})
