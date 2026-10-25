import copy
import unittest
import warnings
from unittest.mock import patch
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from suite.common import digest, subset
from suite.integrated_catalog import build_integrated_catalog
from suite.integrated_models import fit_model, validate_spec
from suite.optimized_regression import build_catalog as v9_catalog

class IntegratedModelTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(102326)
        z = rng.normal(size=(49, 2))
        cls.X = {m: z @ rng.normal(size=(2, p)) + 0.03 * rng.normal(size=(49, p)) for m, p in [('nir', 11), ('raman', 9), ('uv', 7)]}
        cls.Y = np.column_stack([5.0 + 0.4 * z[:, 0] + 0.2 * z[:, 1], 3.0 + 0.2 * z[:, 0] - 0.3 * z[:, 1]])
        cls.train, cls.test = (np.arange(37), np.arange(37, 49))
        cls.main = {'is_smoke': False, 'inner_folds': 3, 'candidates_per_family': 4, 'trees': 160}

    def spec(self, family, **params):
        return {'family': family, 'modes': ['nir'], 'pre': 'raw', 'params': dict({'pca_components': 4}, **params)}

    def fit(self, spec):
        return fit_model(spec, subset(self.X, self.train), self.Y[self.train], seed=17)

    def test_main_catalog_exactly_preserves_v9_and_is_bounded(self):
        catalog, references, manifest = build_integrated_catalog(self.main)
        old, old_references = v9_catalog(self.main)
        ids = {digest(s) for s in catalog}
        self.assertTrue({digest(s) for s in old}.issubset(ids))
        self.assertEqual(references, old_references)
        self.assertEqual(len(catalog), 138)
        self.assertEqual(len(ids), 138)
        self.assertEqual(len(manifest['families']), 20)
        self.assertEqual(manifest['family_counts']['weighted_block_kernel'], 17)
        self.assertEqual(manifest['family_counts']['mlp'], 4)
        self.assertFalse(manifest['library_changes_when_optional_package_installed'])
        self.assertFalse(manifest['measured_ethanol_in_spectral_candidate_pool'])
        self.assertEqual(manifest['library_fingerprint'], digest(catalog))
        again, _, again_manifest = build_integrated_catalog(self.main)
        self.assertEqual(catalog, again)
        self.assertEqual(manifest, again_manifest)

    def test_explicit_historical_specs_deduplicate_and_cannot_execute_code(self):
        extra = self.spec('knn', n_neighbors=5)
        catalog, _, manifest = build_integrated_catalog(self.main, {'historical_specs': [extra, extra]})
        self.assertEqual(len(catalog), 139)
        self.assertEqual(manifest['historical_specs_explicitly_added'], 2)
        for bad in [dict(extra, custom=lambda x: x), dict(extra, family='xgboost'), dict(extra, family='catboost'), dict(extra, use_ethanol=True), dict(extra, params={'alpha': float('nan')})]:
            with self.assertRaises(ValueError):
                build_integrated_catalog(self.main, {'historical_specs': [bad]})
        with self.assertRaises(ValueError):
            build_integrated_catalog(self.main, {'historical_specs': 'not-a-list'})

    def test_all_missing_families_fit_two_targets_and_single_target(self):
        specifications = [self.spec('elasticnet', alpha=0.001, max_iter=20000), self.spec('gpr', length_scale=1.0, noise_level=0.05), self.spec('knn', n_neighbors=3), self.spec('mlp', hidden_layer_sizes=[4], alpha=0.1, max_iter=5000, max_fun=100000)]
        for spec in specifications:
            with self.subTest(family=spec['family']):
                model = self.fit(spec)
                prediction = model.predict(subset(self.X, self.test))
                self.assertEqual(prediction.shape, (12, 2))
                self.assertTrue(np.isfinite(prediction).all())
                self.assertEqual(model.metadata['training_bottles'], 37)
                self.assertFalse(model.metadata['input_requirements']['measured_ethanol_required'])
                single = fit_model(spec, subset(self.X, self.train), self.Y[self.train, 0], seed=17)
                self.assertEqual(single.predict(subset(self.X, self.test)).shape, (12, 1))

    def test_weighted_kernel_linear_formula_and_rbf_are_finite(self):
        spec = {'family': 'weighted_block_kernel', 'modes': ['nir', 'uv'], 'pre': 'raw', 'params': {'kernel': 'linear', 'alpha': 0.1, 'weights': [0.7, 0.3]}}
        model = self.fit(spec)
        test = subset(self.X, self.test)
        blocks = model._blocks(test)
        kernel = sum((w * (blocks[m] @ model.train_blocks[m].T) for w, m in zip(model.weights, model.modes)))
        expected = model.target_scaler.inverse_transform(kernel @ model.dual)
        np.testing.assert_allclose(model.predict(test), expected, atol=1e-12)
        self.assertEqual(model.metadata['kernel_alpha_times_train_n'], 3.7)
        self.assertFalse(model.metadata['identical_to_old_V3_preprocessing'])
        rbf_spec = copy.deepcopy(spec)
        rbf_spec['params']['kernel'] = 'rbf'
        self.assertTrue(np.isfinite(self.fit(rbf_spec).predict(test)).all())
        for weights in [[1.0, 1.0], [-0.1, 1.1], [1.0]]:
            invalid = copy.deepcopy(spec)
            invalid['params']['weights'] = weights
            with self.assertRaises(ValueError):
                self.fit(invalid)

    def test_train_only_states_and_batch_invariance(self):
        for family in ['gpr', 'weighted_block_kernel']:
            spec = self.spec(family)
            model = self.fit(spec)
            expected = model.predict(subset(self.X, self.test))
            changed = {m: a.copy() for m, a in self.X.items()}
            for values in changed.values():
                values[self.test[1:]] += 1000.0
            actual = model.predict(subset(changed, self.test))
            np.testing.assert_allclose(expected[0], actual[0], rtol=1e-11, atol=1e-11)
            if family == 'gpr':
                self.assertEqual(model.features.blocks['nir'].scaler.n_samples_seen_, 37)
            else:
                self.assertEqual(model.blocks['nir'].scaler.n_samples_seen_, 37)

    def test_log1p_wrapper_is_exact_inverse_of_fitted_transformed_prediction(self):
        specs = [dict(self.spec('ridge', alpha=1.0), target_transform='log1p'), dict(self.spec('gpr'), target_transform='log1p'), dict(self.spec('weighted_block_kernel'), target_transform='log1p')]
        for spec in specs:
            with self.subTest(family=spec['family']):
                model = self.fit(spec)
                self.assertTrue(np.isfinite(model.predict(subset(self.X, self.test))).all())
                self.assertEqual(model.metadata['target_transform'], 'log1p')
                negative = self.Y[self.train].copy()
                negative[0, 0] = -0.01
                with self.assertRaises(ValueError):
                    fit_model(spec, subset(self.X, self.train), negative, seed=17)

    def test_uv_amplitude_preprocessing_remains_preserved(self):
        spec = {'family': 'weighted_block_kernel', 'modes': ['uv'], 'pre': 'snv', 'params': {'alpha': 0.1}}
        model = self.fit(spec)
        self.assertEqual(model.blocks['uv'].effective_pre, 'smooth')

    def test_mlp_convergence_is_a_failure_and_not_hidden(self):

        class WarnRegressor:

            def fit(self, features, targets):
                warnings.warn('synthetic deliberate nonconvergence', ConvergenceWarning)
                return self
        with patch('suite.integrated_models._extra_regressor', return_value=WarnRegressor()):
            with self.assertRaisesRegex(RuntimeError, 'did not converge'):
                self.fit(self.spec('mlp'))

    def test_shapes_nonfinite_and_input_requirements_are_checked(self):
        spec = self.spec('gpr')
        missing = {'uv': self.X['uv'][self.train]}
        with self.assertRaises(ValueError):
            fit_model(spec, missing, self.Y[self.train])
        bad = self.Y[self.train].copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            fit_model(spec, subset(self.X, self.train), bad)
        with self.assertRaises(ValueError):
            validate_spec(dict(spec, modes=['nir', 'nir']))
        with self.assertRaises(ValueError):
            validate_spec(dict(spec, params={'callback': object()}))
        abv = {'family': 'abv_only', 'modes': [], 'pre': 'raw', 'params': {'alpha': 1.0}}
        with self.assertRaises(ValueError):
            validate_spec(abv)
        self.assertEqual(validate_spec(abv, allow_measured_ethanol=True), abv)

    def test_band_selection_is_train_local_and_metadata_retains_windows(self):
        spec = {'family': 'band_svr', 'modes': ['raman', 'uv'], 'pre': 'smooth', 'params': {'n_components': 2, 'windows_per_mode': 4, 'selected_windows': 2, 'pca_components': 4, 'C': 10.0, 'gamma': 0.1}}
        model = self.fit(spec)
        expected = model.predict(subset(self.X, self.test))
        self.assertEqual(expected.shape, (12, 2))
        self.assertTrue(np.isfinite(expected).all())
        state = model.metadata['input_requirements']
        self.assertTrue(state['band_selection_uses_supplied_training_targets_only'])
        self.assertFalse(state['historical_band_selection_exactly_reproduced'])
        for mode in ['raman', 'uv']:
            self.assertEqual(len(state['band_selection'][mode]['selected_windows']), 2)
            self.assertEqual(model.features.blocks[mode].scaler.n_samples_seen_, 37)
        changed = copy.deepcopy(self.X)
        for values in changed.values():
            values[self.test[1:]] += 1000.0
        np.testing.assert_allclose(expected[0], model.predict(subset(changed, self.test))[0])
        same = self.fit(spec)
        self.assertEqual(state['band_selection'], same.metadata['input_requirements']['band_selection'])

    def test_distilled_nir_uses_only_nir_at_prediction_and_strict_teacher_partitions(self):
        spec = self.spec('distilled_nir', teacher_modes=['nir', 'uv'], distillation_strength=0.25)
        model = self.fit(spec)
        test = subset(self.X, self.test)
        expected = model.predict(test)
        np.testing.assert_array_equal(expected, model.predict({'nir': test['nir']}))
        self.assertEqual(expected.shape, (12, 2))
        self.assertEqual(model.metadata['input_requirements']['prediction_modalities'], ['nir'])
        self.assertEqual(model.metadata['teacher_inner_folds'], 3)
        partitions = model.metadata['teacher_fold_partitions']
        visits = np.zeros(len(self.train), dtype=int)
        for fold in partitions:
            self.assertFalse(set(fold['fit']) & set(fold['validation']))
            visits[fold['validation']] += 1
        np.testing.assert_array_equal(visits, np.ones(len(self.train), dtype=int))
        first = partitions[0]['validation']
        changed_y = self.Y[self.train].copy()
        changed_y[first] += 10.0
        changed = fit_model(spec, subset(self.X, self.train), changed_y, seed=17)
        a = np.asarray(model.metadata['teacher_oof_pseudo_targets'])
        b = np.asarray(changed.metadata['teacher_oof_pseudo_targets'])
        np.testing.assert_array_equal(a[first], b[first])
        self.assertFalse(model.metadata['pseudo_targets_are_performance_estimates'])
        with self.assertRaises(ValueError):
            model.predict({'uv': test['uv']})
if __name__ == '__main__':
    unittest.main()
