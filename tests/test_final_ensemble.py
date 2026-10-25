import copy
import unittest
from unittest.mock import patch
import numpy as np
from suite.common import digest, subset
from suite import final_ensemble as final
from suite.final_ensemble import FINAL_METHODS, fit_final_bundle, strict_crossfit_final
from suite.optimized_regression import build_catalog, fit_bundle

class FinalEnsembleTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(260123)
        n = 43
        z = rng.normal(size=(n, 3))
        cls.X = {mode: z @ rng.normal(size=(3, p)) + 0.3 * rng.normal(size=(n, p)) for mode, p in [('nir', 17), ('raman', 15), ('uv', 13)]}
        cls.Y = 5.0 + z @ rng.normal(size=(3, 3)) + 0.2 * rng.normal(size=(n, 3))
        cls.profile = {'is_smoke': True, 'inner_folds': 2, 'candidates_per_family': 1, 'trees': 8}
        cls.train, cls.test = (np.arange(31), np.arange(31, n))
        cls.bundle = fit_final_bundle(subset(cls.X, cls.train), cls.Y[cls.train], cls.profile, 501)

    def test_methods_same_complete_pool_and_expected_predictions(self):
        bundle = self.bundle
        test = subset(self.X, self.test)
        predictions = bundle.predict_methods(test)
        self.assertEqual(tuple(predictions), FINAL_METHODS)
        self.assertEqual(len(bundle.models), len(bundle.audit['candidate_order']))
        self.assertEqual(bundle.audit['fit_budget']['full_fit_repetitions_per_candidate'], 1)
        self.assertEqual(bundle.audit['fit_budget']['full_fit_attempts'], len(bundle.audit['candidates']))
        self.assertEqual(bundle.audit['common_final_candidate_ids'], bundle.candidate_ids)
        for a in predictions.values():
            self.assertEqual(a.shape, (12, 3))
            self.assertTrue(np.isfinite(a).all())
        independent = np.stack([bundle.models[i].predict(test) for i in range(len(bundle.models))], axis=-1)
        np.testing.assert_allclose(predictions['v10_equal_weight'], independent.mean(axis=-1))
        np.testing.assert_allclose(bundle.predict(test), predictions['v10_strategy_router'])
        self.assertEqual(list(bundle.predict_methods(test, ['v10_pls2_reference'])), ['v10_pls2_reference'])

    def test_fixed_penalties_and_nonnegative_unit_sum_weights(self):
        audit = self.bundle.audit
        self.assertEqual(audit['penalties_by_method'], {'v10_unanchored_ensemble': 0.0, 'v10_anchored_ensemble': 0.1})
        for name, raw in audit['weights_by_method'].items():
            weights = np.asarray(raw)
            self.assertEqual(weights.shape, (3, len(audit['candidate_order'])))
            self.assertTrue((weights >= 0).all(), name)
            np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-12)
        for penalty in [0.0, 0.01, 1.0, np.nan]:
            with self.assertRaises(ValueError):
                fit_final_bundle(subset(self.X, self.train), self.Y[self.train], self.profile, 501, {'ensemble_penalty': penalty})

    def test_unchanged_v8_reference_and_anchor_when_all_full_fits_succeed(self):
        old = fit_bundle(subset(self.X, self.train), self.Y[self.train], self.profile, 501, options={'include_abv': False, 'ensemble_penalty': 0.0})
        test = subset(self.X, self.test)
        a, b = (old.predict_methods(test), self.bundle.predict_methods(test))
        np.testing.assert_allclose(a['v8_shared_pls2_reference'], b['v10_pls2_reference'], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(a['v8_shrunk_ensemble'], b['v10_v9_unanchored_reference'], rtol=1e-12, atol=1e-12)
        self.assertEqual(old.audit['reference_candidate'], self.bundle.audit['reference_candidate'])

    def test_main_frozen_catalog_has_69_unique_specifications(self):
        catalog, references = build_catalog(dict(self.profile, is_smoke=False, candidates_per_family=4))
        self.assertEqual(len(catalog), 69)
        self.assertEqual(len({digest(s) for s in catalog}), 69)
        self.assertEqual(len(references), 4)

    def test_held_out_values_do_not_change_fit_or_other_predictions(self):
        altered = self.Y.copy()
        altered[self.test] += 10000.0
        model = fit_final_bundle(subset(self.X, self.train), altered[self.train], self.profile, 501)
        self.assertEqual(digest(model.audit), digest(self.bundle.audit))
        a = self.bundle.predict_methods(subset(self.X, self.test))
        b = model.predict_methods(subset(self.X, self.test))
        for method in FINAL_METHODS:
            np.testing.assert_array_equal(a[method], b[method])
        changed_X = copy.deepcopy(self.X)
        for values in changed_X.values():
            values[self.test[1:]] += 100.0
        changed = self.bundle.predict_methods(subset(changed_X, self.test))
        for method in FINAL_METHODS:
            np.testing.assert_allclose(a[method][0], changed[method][0], rtol=1e-11, atol=1e-11)

    def test_full_refit_failure_removes_candidate_for_every_method(self):
        n = len(self.train)
        bad_id = next((i for i in self.bundle.candidate_ids if i != self.bundle.audit['reference_candidate']))
        original = final.fit_model
        full_attempts = []

        def failing_fit(spec, X, Y, *args, **kwargs):
            if len(Y) == n:
                full_attempts.append(digest(spec))
                if digest(spec) == bad_id:
                    raise RuntimeError('synthetic full-fit failure')
            return original(spec, X, Y, *args, **kwargs)
        with patch('suite.final_ensemble.fit_model', side_effect=failing_fit):
            fitted = fit_final_bundle(subset(self.X, self.train), self.Y[self.train], self.profile, 501)
        self.assertNotIn(bad_id, fitted.candidate_ids)
        self.assertEqual(len(full_attempts), len(set(full_attempts)))
        self.assertEqual(len(full_attempts), len(self.bundle.candidate_ids))
        self.assertEqual(fitted.audit['excluded_after_successful_inner_oof'][0]['id'], bad_id)
        self.assertTrue(any((row.get('stage') == 'full_training_refit' for row in fitted.audit['failures'])))
        for raw in fitted.audit['weights_by_method'].values():
            self.assertEqual(np.asarray(raw).shape[1], len(fitted.candidate_ids))
        self.assertTrue(np.isfinite(fitted.predict(subset(self.X, self.test))).all())

    def test_strict_system_crossfit_excludes_whole_validation_fold(self):
        X, Y = (subset(self.X, np.arange(25)), self.Y[:25])
        methods = ['v10_pls2_reference', 'v10_anchored_ensemble']
        a, audits_a = strict_crossfit_final(X, Y, self.profile, 1501, methods=methods)
        self.assertEqual(set(a), set(methods))
        seen = []
        for audit in audits_a:
            self.assertFalse(set(audit['train']) & set(audit['validation']))
            self.assertEqual(len(audit['audit']['oof_membership_in_supplied_training_order']), len(audit['train']))
            self.assertTrue(audit['entire_model_selection_and_weights_refitted'])
            seen.extend(audit['validation'])
        self.assertEqual(sorted(seen), list(range(25)))
        changed = Y.copy()
        held = np.asarray(audits_a[0]['validation'])
        changed[held] += 5000.0
        b, audits_b = strict_crossfit_final(X, changed, self.profile, 1501, methods=methods)
        self.assertEqual(digest(audits_a[0]), digest(audits_b[0]))
        for method in methods:
            np.testing.assert_allclose(a[method][held], b[method][held], rtol=0, atol=0)
            self.assertTrue(np.isfinite(a[method]).all())

    def test_reject_unknown_method_and_undersized_strict_fit(self):
        with self.assertRaises(ValueError):
            self.bundle.predict_methods(subset(self.X, self.test), ['unknown'])
        with self.assertRaises(ValueError):
            strict_crossfit_final(subset(self.X, np.arange(15)), self.Y[:15], self.profile, 5)
if __name__ == '__main__':
    unittest.main()
