import copy
import unittest
from unittest import mock
import numpy as np
from suite.final_boundaries import make_stress_inputs, make_occlusion_inputs, run_task, METHODS

class _Bundle:

    def __init__(self, X, Y):
        self.offset = Y.mean(axis=0)
        self.audit = {'synthetic_test_bundle': True}

    def predict_methods(self, X, methods=None):
        signal = sum((np.mean(value, axis=1) for value in X.values()))
        base = self.offset[None, :] + signal[:, None] * np.array([0.1, 0.2])[None, :]
        return {method: base.copy() + (0 if method == METHODS[0] else 0.01) for method in (METHODS if methods is None else methods)}

class BoundariesTest(unittest.TestCase):

    def setUp(self):
        r = np.random.RandomState(749)
        self.train = {name: r.normal(size=(25, 13)) for name in ['nir', 'raman', 'uv']}
        self.test = {name: r.normal(size=(7, 13)) for name in self.train}

    def test_noise_is_deterministic_and_paired_with_training_scale(self):
        first = make_stress_inputs(self.train, self.test, 112)
        again = make_stress_inputs(self.train, self.test, 112)
        self.assertEqual(len(first), 18)
        for (name, a, meta), (same, b, _) in zip(first, again):
            self.assertEqual(name, same)
            for mode in a:
                np.testing.assert_array_equal(a[mode], b[mode])
                if mode != meta['changed_mode']:
                    np.testing.assert_array_equal(a[mode], self.test[mode])
        offset = next((item for item in first if item[2]['challenge_type'] == 'training_std_offset'))
        _, X, meta = offset
        mode = meta['changed_mode']
        expected = meta['magnitude'] * self.train[mode].std(axis=0)
        np.testing.assert_allclose(X[mode] - self.test[mode], np.tile(expected, (7, 1)), atol=1e-15)
        self.assertFalse(np.shares_memory(X[mode], self.test[mode]))

    def test_windows_partition_the_axis_and_replace_only_training_mean(self):
        axis = {name: np.arange(13) * 2.5 + 7 for name in self.train}
        windows = make_occlusion_inputs(self.train, self.test, axis, 6)
        self.assertEqual(len(windows), 18)
        for mode in self.train:
            seen = []
            for _, changed, meta in windows:
                if meta['changed_mode'] != mode:
                    continue
                cols = np.arange(meta['column_start'], meta['column_stop_exclusive'])
                seen.extend(cols)
                expected = self.test[mode].copy()
                expected[:, cols] = self.train[mode].mean(axis=0)[cols]
                np.testing.assert_array_equal(expected, changed[mode])
                self.assertEqual(meta['coordinate_start'], axis[mode][cols[0]])
                self.assertEqual(meta['coordinate_end'], axis[mode][cols[-1]])
            self.assertEqual(seen, list(range(13)))

    def test_constant_training_column_gets_no_gaussian_or_offset_change(self):
        train, test = (copy.deepcopy(self.train), copy.deepcopy(self.test))
        train['uv'][:, 0], test['uv'][:, 0] = (4.0, 9.0)
        for _, X, meta in make_stress_inputs(train, test, 101):
            if meta['changed_mode'] == 'uv' and meta['challenge_type'] != 'intensity_multiplier':
                np.testing.assert_array_equal(X['uv'][:, 0], test['uv'][:, 0])

    def test_heldout_reference_changes_do_not_change_any_prediction(self):
        r = np.random.RandomState(201)
        X = {k: np.vstack([self.train[k], self.test[k]]) for k in self.train}
        Y = np.column_stack([2 + np.arange(32) * 0.02, r.uniform(1, 3, 32)])
        data = {'X': X, 'Y': Y, 'ethanol': r.uniform(50, 65, 32), 'ids': [str(i) for i in range(32)], 'target_names': ['a', 'b']}
        task = {'train': list(range(25)), 'test': list(range(25, 32)), 'repeat': 0, 'fold': 0, 'seed': 293}
        profile = {'inner_folds': 2}
        with mock.patch('suite.final_boundaries._fit_bundle', side_effect=lambda x, y, p, seed, options: _Bundle(x, y)) as fit:
            first, audit = run_task(data, task, profile, {})
            self.assertEqual(fit.call_count, 1)
            changed = dict(data, Y=data['Y'].copy())
            changed['Y'][25:] += 1000
            second, _ = run_task(changed, task, profile, {})
        for frame in ['regression', 'stress', 'occlusion']:
            np.testing.assert_array_equal(first[frame]['y_pred'], second[frame]['y_pred'])
        self.assertEqual(audit['spectral_refits_inside_challenges'], 0)
        self.assertEqual(audit['challenge_count'], 36)
        self.assertTrue(first['diagnostics']['truth_used_only_for_posthoc_grouping'].all())
        self.assertFalse(first['diagnostics']['concentration_tail_known_without_reference'].any())
        self.assertTrue(first['diagnostics']['abv_group_requires_measured_test_abv'].all())

    def test_bundle_reuse_does_not_trigger_second_spectral_fit(self):
        r = np.random.RandomState(301)
        X = {k: np.vstack([self.train[k], self.test[k]]) for k in self.train}
        Y = r.uniform(1, 4, (32, 2))
        data = {'X': X, 'Y': Y, 'ethanol': r.uniform(50, 65, 32), 'ids': [str(i) for i in range(32)], 'target_names': ['a', 'b']}
        task = {'train': list(range(25)), 'test': list(range(25, 32)), 'repeat': 0, 'fold': 0, 'seed': 293}
        bundle = _Bundle(self.train, Y[:25])
        with mock.patch('suite.final_boundaries._fit_bundle', side_effect=AssertionError('Refitted bundle')):
            frames, audit = run_task(data, task, {'inner_folds': 2}, {}, bundle=bundle)
        self.assertTrue(audit['bundle_reused'])
        self.assertEqual(set(frames['regression']['method']), set(METHODS) | {'v10_measured_abv_polynomial_ridge'})

    def test_overlapping_outer_split_rejected(self):
        data = {'X': self.train, 'Y': np.ones((25, 2)), 'ethanol': np.ones(25), 'ids': np.arange(25), 'target_names': ['a', 'b']}
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            run_task(data, {'train': [0, 1], 'test': [1, 2], 'repeat': 0, 'fold': 0, 'seed': 9}, {}, {})
if __name__ == '__main__':
    unittest.main()
