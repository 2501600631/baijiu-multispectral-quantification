import copy
import unittest
from unittest import mock
import numpy as np
from sklearn.model_selection import KFold
from suite import integrated_abv as abv

class _Bundle:

    def __init__(self, X, Y):
        self.mean = Y.mean(axis=0)
        prediction = np.full_like(Y, np.nan)
        receipts = []
        for fold, (tr, va) in enumerate(KFold(2, shuffle=True, random_state=77).split(Y)):
            prediction[va] = Y[tr].mean(axis=0)
            receipts.append({'fold': fold, 'train': tr.tolist(), 'validation': va.tolist(), 'all_selection_and_weights_exclude_validation': True})
        self.audit = {'strategy_router': {'strict_procedure_oof': {abv.LEGACY_METHOD: prediction.tolist()}, 'crossfit': receipts}}

    def predict_methods(self, X, methods=None):
        signal = np.mean(X['uv'], axis=1)[:, None]
        base = self.mean[None, :] + signal * np.array([0.03, 0.02])[None, :]
        return {name: base + (0 if name == abv.LEGACY_METHOD else 0.001) for name in (abv.SPECTRAL_METHODS if methods is None else methods)}

class IntegratedABVTest(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(581)
        n = 36
        ethanol = rng.uniform(48, 70, n)
        latent = rng.normal(size=(n, 2))
        X = {name: latent @ rng.normal(size=(2, 11)) + rng.normal(size=(n, 11)) * 0.08 + ethanol[:, None] * 0.02 for name in ['nir', 'raman', 'uv']}
        for values in X.values():
            values[:, 0] = np.arange(n)
        Y = 1 + ethanol[:, None] * np.array([0.02, 0.006])[None, :]
        Y = Y + latent * np.array([0.03, 0.08])[None, :]
        self.data = {'X': X, 'Y': Y, 'ethanol': ethanol, 'ids': ['b' + str(i) for i in range(n)], 'target_names': ['a', 'b']}
        self.task = {'train': list(range(28)), 'test': list(range(28, n)), 'repeat': 0, 'fold': 0, 'seed': 419}
        self.profile = {'inner_folds': 2, 'strategy_folds': 2}
        self.bundle = _Bundle({k: v[:28] for k, v in X.items()}, Y[:28])

    def run_task(self, data=None):
        return abv.run_task(self.data if data is None else data, self.task, self.profile, {}, self.bundle)

    def test_catalog_is_fixed_bounded_and_preserves_uv_amplitude(self):
        catalog = abv.build_residual_catalog()
        self.assertEqual(len(catalog), 12)
        self.assertEqual({s['params']['n_components'] for s in catalog}, {2, 4, 8})
        self.assertEqual(len({tuple(s['modes']) for s in catalog}), 4)
        from suite.models import _row_preprocess
        raw_uv = np.tile(np.arange(11, dtype=float), (2, 1)) * np.array([1, 3])[:, None]
        transformed, pre = _row_preprocess(raw_uv, 'snv', 'uv')
        self.assertEqual(pre, 'smooth')
        np.testing.assert_allclose(transformed[1], transformed[0] * 3, atol=1e-12)

    def test_heldout_reference_changes_only_output_truth(self):
        original, audit = self.run_task()
        changed = copy.deepcopy(self.data)
        changed['Y'][self.task['test']] += 37
        after, again = self.run_task(changed)
        np.testing.assert_array_equal(original['regression'].y_pred, after['regression'].y_pred)
        self.assertFalse(np.array_equal(original['regression'].y_true, after['regression'].y_true))
        self.assertEqual(audit, again)
        self.assertEqual(set(original['regression'].method), set(abv.POINT_METHODS))
        self.assertFalse(audit['measured_abv']['baseline_strict_procedure']['additional_spectral_model_fitting'])

    def test_test_ethanol_cannot_change_spectra_only_methods(self):
        original, audit = self.run_task()
        changed = copy.deepcopy(self.data)
        changed['ethanol'][self.task['test']] += 7
        after, again = self.run_task(changed)
        for method in abv.SPECTRAL_METHODS:
            np.testing.assert_array_equal(original['regression'].query('method == @method').y_pred, after['regression'].query('method == @method').y_pred)
        method = abv.ABV_METHODS[0]
        self.assertFalse(np.array_equal(original['regression'].query('method == @method').y_pred, after['regression'].query('method == @method').y_pred))
        self.assertEqual(audit, again)
        self.assertTrue(original['regression'].query('method == @method').requires_measured_test_abv.all())

    def test_all_nuisance_and_spectral_fits_exclude_outer_test_and_oof_validation(self):
        real = abv._ABVResidual
        fitted_rows = []

        def spy(spec, X, Y, E, seed):
            fitted_rows.append(set(np.asarray(X['nir'][:, 0], int).tolist()))
            return real(spec, X, Y, E, seed)
        with mock.patch.object(abv, '_ABVResidual', side_effect=spy):
            _, audit = self.run_task()
        train_set = set(self.task['train'])
        allowed = [train_set] + [set(x['global_train_bottle_indices']) for x in audit['measured_abv']['inner_partitions']]
        self.assertEqual(len(fitted_rows), 12 * 3)
        for rows in fitted_rows:
            self.assertTrue(any((rows == permitted for permitted in allowed)))
            self.assertFalse(rows.intersection(self.task['test']))
        for receipt in audit['measured_abv']['inner_partitions']:
            self.assertFalse(set(receipt['global_train_bottle_indices']).intersection(receipt['global_validation_bottle_indices']))

    def test_missing_ethanol_and_corrupt_strict_membership_fail_before_training(self):
        changed = copy.deepcopy(self.data)
        del changed['ethanol']
        with mock.patch.object(abv, '_ABVResidual') as fitted:
            with self.assertRaisesRegex(ValueError, 'ethanol'):
                self.run_task(changed)
            fitted.assert_not_called()
        self.bundle.audit['strategy_router']['crossfit'][0]['train'].append(self.bundle.audit['strategy_router']['crossfit'][0]['validation'][0])
        with self.assertRaisesRegex(ValueError, 'partition'):
            self.run_task()

    def test_failed_residual_library_is_exposed_with_frozen_abv_fallback(self):
        with mock.patch.object(abv, '_ABVResidual', side_effect=RuntimeError('synthetic failure')):
            frames, audit = self.run_task()
        self.assertTrue(audit['measured_abv']['residual_all_failed_fallback_to_abv_only'])
        self.assertEqual(len(audit['measured_abv']['candidate_failures']), 12)
        values = frames['regression']
        first, second = abv.ABV_METHODS[:2]
        np.testing.assert_array_equal(values.query('method == @first').y_pred, values.query('method == @second').y_pred)
if __name__ == '__main__':
    unittest.main()
