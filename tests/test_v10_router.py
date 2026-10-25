import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
from suite.common import digest, iqr, subset
from suite import final_ensemble as final
from suite.final_cache import get_bundle
from suite.optimized_regression import fit_bundle
PRIORITY = ['v10_pls2_reference', final.LEGACY_METHOD, final.LATE_METHOD, 'v10_equal_weight', 'v10_unanchored_ensemble', 'v10_anchored_ensemble', 'v10_conservative_adaptive', 'v10_target_adaptive', 'v10_shared_catalog_best']

class RouterTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(261023)
        z = rng.normal(size=(41, 3))
        cls.X = {mode: z @ rng.normal(size=(3, p)) + 0.15 * rng.normal(size=(41, p)) for mode, p in [('nir', 15), ('raman', 13), ('uv', 11)]}
        cls.Y = 10.0 + z @ np.array([[0.5, -0.2], [0.1, 0.7], [-0.3, 0.2]]) + 0.08 * rng.normal(size=(41, 2))
        cls.train = np.arange(33)
        cls.test = np.arange(33, 41)
        cls.profile = {'is_smoke': True, 'inner_folds': 2, 'strategy_folds': 2, 'candidates_per_family': 1, 'trees': 8}
        cls.options = {'ensemble_penalty': 0.1}
        cls.bundle = final.fit_final_bundle(subset(cls.X, cls.train), cls.Y[cls.train], cls.profile, 2901, cls.options)

    def test_frozen_v9_unanchored_unchanged_when_expanded_candidate_present(self):
        legacy = fit_bundle(subset(self.X, self.train), self.Y[self.train], self.profile, 2901, options={'include_abv': False, 'ensemble_penalty': 0.0})
        prediction = self.bundle.predict_methods(subset(self.X, self.test))
        np.testing.assert_allclose(legacy.predict(subset(self.X, self.test)), prediction[final.LEGACY_METHOD], rtol=1e-11, atol=1e-11)
        old = set(legacy.audit['candidate_order'])
        self.assertEqual(old, set(self.bundle.audit['legacy_comparator']['candidate_ids']))
        self.assertGreater(len(self.bundle.candidate_ids), len(old))
        for index, candidate in enumerate(self.bundle.candidate_ids):
            if candidate not in old:
                np.testing.assert_array_equal(self.bundle.legacy_weights[:, index], np.zeros(2))

    def test_routes_are_argmin_of_actual_strict_procedure_oof(self):
        audit = self.bundle.audit['strategy_router']
        predictions = audit['strict_procedure_oof']
        y = self.Y[self.train]
        scores = np.column_stack([np.mean(((np.asarray(predictions[name]) - y) / iqr(y)) ** 2, axis=0) for name in final.STRATEGY_METHODS])
        np.testing.assert_allclose(scores, audit['scores_targetwise'], rtol=0, atol=0)
        expected = [min(PRIORITY, key=lambda name: (round(float(scores[j, final.STRATEGY_METHODS.index(name)]), 12), PRIORITY.index(name))) for j in range(2)]
        self.assertEqual(self.bundle.strategy_routes, expected)
        self.assertFalse(audit['history_metric_splicing'])
        self.assertFalse(audit['route_winners_chosen_using_outer_test'])
        visited = []
        for fold in audit['crossfit']:
            tr, va = (fold['train'], fold['validation'])
            self.assertFalse(set(tr) & set(va))
            visited.extend(va)
            self.assertEqual(fold['training_bottles'], len(tr))
            for inner in fold['inner_splits']:
                self.assertLess(max(inner['fit'] + inner['validation']), len(tr))
        self.assertEqual(sorted(visited), list(range(33)))

    def test_own_strategy_fold_predictions_and_audit_exclude_heldout_labels(self):
        first = self.bundle.audit['strategy_router']['crossfit'][0]
        held = np.asarray(first['validation'])
        altered = self.Y[self.train].copy()
        altered[held] += 500.0
        changed = final.fit_final_bundle(subset(self.X, self.train), altered, self.profile, 2901, self.options)
        after = changed.audit['strategy_router']['crossfit'][0]
        self.assertEqual(first['training_audit_digest'], after['training_audit_digest'])
        self.assertEqual(digest(first), digest(after))
        for name in final.STRATEGY_METHODS:
            a = np.asarray(self.bundle.audit['strategy_router']['strict_procedure_oof'][name])
            b = np.asarray(changed.audit['strategy_router']['strict_procedure_oof'][name])
            np.testing.assert_array_equal(a[held], b[held])

    def test_outer_test_targets_cannot_change_training_or_predictions(self):
        altered = self.Y.copy()
        altered[self.test] = 1000000.0
        changed = final.fit_final_bundle(subset(self.X, self.train), altered[self.train], self.profile, 2901, self.options)
        self.assertEqual(digest(changed.audit), digest(self.bundle.audit))
        before = self.bundle.predict_methods(subset(self.X, self.test))
        after = changed.predict_methods(subset(self.X, self.test))
        for name in final.FINAL_METHODS:
            np.testing.assert_array_equal(before[name], after[name])

    def test_default_prediction_and_joblib_roundtrip_preserve_router(self):
        x = subset(self.X, self.test)
        allpred = self.bundle.predict_methods(x)
        expected = np.column_stack([allpred[name][:, j] for j, name in enumerate(self.bundle.strategy_routes)])
        np.testing.assert_array_equal(self.bundle.predict(x), expected)
        stream = io.BytesIO()
        joblib.dump(self.bundle, stream)
        stream.seek(0)
        restored = joblib.load(stream)
        self.assertEqual(digest(restored.audit), digest(self.bundle.audit))
        np.testing.assert_array_equal(restored.predict(x), self.bundle.predict(x))

    def test_strategy_ties_prefer_frozen_reference(self):

        class FixedProcedure:
            candidate_ids = ['fixed']
            audit = {'inner_splits': [], 'reference_candidate': 'fixed'}

            def predict_methods(self, x, methods):
                return {name: np.ones((len(next(iter(x.values()))), 2)) * 7.0 for name in methods}

        class EmptyBundle:
            pass
        bundle = EmptyBundle()
        bundle.strategy_routes = ['v10_pls2_reference'] * 2
        bundle.audit = {}
        with patch('suite.final_ensemble.fit_final_bundle', return_value=FixedProcedure()):
            final._learn_strategy_router(bundle, subset(self.X, self.train), self.Y[self.train], self.profile, 2901, self.options)
        self.assertEqual(bundle.strategy_routes, ['v10_pls2_reference'] * 2)
        scores = np.asarray(bundle.audit['strategy_router']['scores_targetwise'])
        np.testing.assert_array_equal(scores, np.repeat(scores[:, :1], len(final.STRATEGY_METHODS), axis=1))

    def test_fewer_than_sixteen_fit_bottles_have_audited_fixed_fallback(self):
        model = final.fit_final_bundle(subset(self.X, np.arange(15)), self.Y[:15], self.profile, 2901, self.options)
        audit = model.audit['strategy_router']
        self.assertTrue(audit['fallback'])
        self.assertIn('fewer than16', audit['reason'])
        self.assertEqual(model.strategy_routes, ['v10_pls2_reference'] * 2)
        x = subset(self.X, self.test)
        np.testing.assert_array_equal(model.predict(x), model.predict_methods(x, ['v10_pls2_reference'])['v10_pls2_reference'])

    def test_local_cache_roundtrip_reuses_exact_router_without_refitting(self):
        data = {'ids': np.asarray([str(i) for i in range(41)]), 'X': self.X, 'Y': self.Y}
        task = {'train': self.train.tolist(), 'seed': 2901}
        cfg = {'version': '10.0.0', 'ensemble_options': self.options}
        with tempfile.TemporaryDirectory(prefix='router_fixture_', dir=str(Path(__file__).resolve().parents[1])) as directory:
            root = Path(directory)
            (root / 'config.json').write_text('{}', encoding='utf8')
            (root / 'run.py').write_text('# synthetic fixture only\n', encoding='utf8')
            cache = root / 'cache'
            with patch('suite.final_ensemble.fit_final_bundle', return_value=self.bundle) as fitted:
                first, a = get_bundle(root, data, task, self.profile, cfg, cache)
                self.assertEqual(fitted.call_count, 1)
            with patch('suite.final_ensemble.fit_final_bundle', side_effect=AssertionError('cache must not train')):
                second, b = get_bundle(root, data, task, self.profile, cfg, cache)
                changed = copy.deepcopy(data)
                changed['Y'][self.test] = 100000.0
                third, c = get_bundle(root, changed, task, self.profile, cfg, cache)
            self.assertFalse(a['reused'])
            self.assertTrue(b['reused'])
            self.assertTrue(c['reused'])
            self.assertEqual(a['cache_key'], b['cache_key'])
            self.assertEqual(b['cache_key'], c['cache_key'])
            np.testing.assert_array_equal(first.predict(subset(self.X, self.test)), second.predict(subset(self.X, self.test)))
            np.testing.assert_array_equal(third.predict(subset(self.X, self.test)), second.predict(subset(self.X, self.test)))
if __name__ == '__main__':
    unittest.main()
