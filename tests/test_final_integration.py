from pathlib import Path
import copy
import io
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suite.common import csv_read, digest, read_json, save_json
from suite.data import synthetic_data
from suite.final_protocol import build_plan, frozen_plan
from suite.final_cache import cache_identity, get_bundle

class LocalCacheFixture:

    def __init__(self, y):
        self.center = np.mean(y, axis=0)
        self.audit = {'synthetic_test_fixture': True}

class FinalIntegrationTests(unittest.TestCase):

    def setUp(self):
        self.data = synthetic_data(139, 3, seed=1411)
        self.cfg = {'version': '9.test', 'seed': 20261020, 'evidence_tier': 'retrospective_internal_only', 'calibration_fraction': 0.25, 'ensemble_options': {'ensemble_penalty': 0.1}, 'primary_model': 'v10_anchored_ensemble', 'primary_comparisons': ['v10_pls2_reference', 'v10_equal_weight', 'v10_unanchored_ensemble'], 'profiles': {'main': {'regression_repeats': 2, 'regression_folds': 5, 'inner_folds': 3}}}
        self.profile = self.cfg['profiles']['main']
        self.temp = tempfile.TemporaryDirectory(dir=str(Path(__file__).resolve().parent))
        self.root = Path(self.temp.name)
        self.source_patch = patch('suite.final_protocol.source_hashes', return_value={'frozen.py': 'abc'})
        self.cache_source_patch = patch('suite.final_cache.source_hashes', return_value={'frozen.py': 'abc'})
        self.runtime_patch = patch('suite.final_cache.runtime', return_value={'synthetic_runtime': 'fixed'})
        self.source_patch.start()
        self.cache_source_patch.start()
        self.runtime_patch.start()
        self.task = build_plan(self.root, self.data, self.cfg, 'main', 'ablation')['tasks'][0]

    def tearDown(self):
        self.runtime_patch.stop()
        self.cache_source_patch.stop()
        self.source_patch.stop()
        self.temp.cleanup()

    def test_ablation_boundary_share_folds_seeds_and_uneven_bottle_counts(self):
        a = build_plan(self.root, self.data, self.cfg, 'main', 'ablation')
        b = build_plan(self.root, self.data, self.cfg, 'main', 'boundaries')
        self.assertEqual(a['tasks'], b['tasks'])
        self.assertEqual(set((len(t['train']) for t in a['tasks'])), {111, 112})
        for repeat in range(2):
            tests = [index for t in a['tasks'] if t['repeat'] == repeat for index in t['test']]
            self.assertEqual(sorted(tests), list(range(139)))

    def test_interval_preserves_outer_test_and_partitions_training_without_labels(self):
        a = build_plan(self.root, self.data, self.cfg, 'main', 'ablation')
        b = build_plan(self.root, self.data, self.cfg, 'main', 'interval')
        changed = copy.deepcopy(self.data)
        changed['Y'] *= 19
        c = build_plan(self.root, changed, self.cfg, 'main', 'interval')
        self.assertEqual(b['tasks'], c['tasks'])
        for original, interval in zip(a['tasks'], b['tasks']):
            self.assertEqual(original['test'], interval['test'])
            self.assertEqual(original['seed'], interval['seed'])
            fit, calibration, test = map(set, (interval['train'], interval['calibration'], interval['test']))
            self.assertFalse(fit & calibration or fit & test or calibration & test)
            self.assertEqual(fit | calibration, set(original['train']))
            self.assertEqual(fit | calibration | test, set(range(139)))

    def test_frozen_plan_refuses_changed_configuration_and_responses(self):
        out = self.root / 'frozen'
        first = frozen_plan(self.root, self.data, self.cfg, 'main', 'ablation', out)
        self.assertEqual(first, frozen_plan(self.root, self.data, self.cfg, 'main', 'ablation', out))
        changed = copy.deepcopy(self.cfg)
        changed['ensemble_options']['ensemble_penalty'] = 0.2
        with self.assertRaisesRegex(ValueError, 'Frozen'):
            frozen_plan(self.root, self.data, changed, 'main', 'ablation', out)
        changed_data = copy.deepcopy(self.data)
        changed_data['Y'][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'Frozen'):
            frozen_plan(self.root, changed_data, self.cfg, 'main', 'ablation', out)

    def test_frozen_plan_rejects_edited_task_with_untouched_claimed_hash(self):
        out = self.root / 'tampered_plan'
        frozen_plan(self.root, self.data, self.cfg, 'main', 'ablation', out)
        altered = read_json(out / 'plan.json')
        altered['tasks'][0]['seed'] += 1
        save_json(out / 'plan.json', altered)
        with self.assertRaisesRegex(ValueError, 'Frozen'):
            frozen_plan(self.root, self.data, self.cfg, 'main', 'ablation', out)

    def test_reporting_failure_cannot_become_checkpoint_only_success(self):
        from suite.final_engine import run
        plan = {'parameters': {}, 'tasks': [], 'plan_hash': 'test_only'}
        failure = {'ok': False, 'complete': False, 'errors': ['injected reporting failure']}
        with patch('suite.final_engine.frozen_plan', return_value=plan), patch('suite.final_reporting.generate', return_value=failure), patch('suite.final_reporting.verify', return_value={'ok': True, 'complete': True}) as verify:
            with self.assertRaisesRegex(RuntimeError, 'reporting failed'):
                run(self.root, self.root / 'failed_report', self.data, self.cfg, 'main', 'ablation')
        verify.assert_not_called()
        self.assertTrue((self.root / 'failed_report' / 'errors' / 'reporting.json').exists())

    def test_local_cache_key_depends_only_on_the_supplied_training_problem(self):
        identity = cache_identity(self.root, self.data, self.task, self.profile, self.cfg)
        test_changed = copy.deepcopy(self.data)
        test_changed['Y'][self.task['test']] += 100
        for mode in test_changed['X']:
            test_changed['X'][mode][self.task['test']] += 10
        self.assertEqual(identity, cache_identity(self.root, test_changed, self.task, self.profile, self.cfg))
        changes = []
        y_changed = copy.deepcopy(self.data)
        y_changed['Y'][self.task['train'][0], 0] += 1
        changes.append(cache_identity(self.root, y_changed, self.task, self.profile, self.cfg))
        x_changed = copy.deepcopy(self.data)
        x_changed['X']['uv'][self.task['train'][0], 0] += 1
        changes.append(cache_identity(self.root, x_changed, self.task, self.profile, self.cfg))
        cfg_changed = copy.deepcopy(self.cfg)
        cfg_changed['ensemble_options']['ensemble_penalty'] = 0.2
        changes.append(cache_identity(self.root, self.data, self.task, self.profile, cfg_changed))
        changes.append(cache_identity(self.root, self.data, self.task, dict(self.profile, inner_folds=2), self.cfg))
        changes.append(cache_identity(self.root, self.data, dict(self.task, seed=self.task['seed'] + 1), self.profile, self.cfg))
        self.assertTrue(all((digest(other) != digest(identity) for other in changes)))

    def test_cache_reuses_local_training_fit_but_rejects_modified_bytes(self):
        fitted = LocalCacheFixture(self.data['Y'][self.task['train']])
        with patch('suite.final_ensemble.fit_final_bundle', return_value=fitted) as fit:
            _, first = get_bundle(self.root, self.data, self.task, self.profile, self.cfg, self.root / 'cache')
            changed = copy.deepcopy(self.data)
            changed['Y'][self.task['test']] += 100
            loaded, second = get_bundle(self.root, changed, self.task, self.profile, self.cfg, self.root / 'cache')
        self.assertFalse(first['reused'])
        self.assertTrue(second['reused'])
        self.assertEqual(fit.call_count, 1)
        np.testing.assert_array_equal(loaded.center, fitted.center)
        file = next((self.root / 'cache').glob('*.joblib'))
        with file.open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'cache changed'):
            get_bundle(self.root, self.data, self.task, self.profile, self.cfg, self.root / 'cache')

    def test_roundtrip_exact_conformal_boundary_and_infinity(self):
        numbers = np.array([0.0, np.nextafter(1.0, 0.0), 1.0, np.nextafter(1.0, 2.0), np.inf, -np.inf])
        file = self.root / 'boundary.csv'
        pd.DataFrame({'score': numbers}).to_csv(file, index=False)
        actual = csv_read(file).score.to_numpy()
        np.testing.assert_array_equal(actual, numbers)
        np.testing.assert_array_equal(actual <= 1.0, numbers <= 1.0)
        with self.assertRaisesRegex(ValueError, 'round_trip'):
            csv_read(file, float_precision=None)
if __name__ == '__main__':
    unittest.main()
